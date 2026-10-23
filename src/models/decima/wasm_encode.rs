#[cfg(feature = "int8")]
use super::cpu::Linear;
use super::{D, Model, Scratch, cpu, specialization::CpuPlan};
use anyhow::{Result, ensure};
use std::sync::Arc;

#[cfg(feature = "decima-wasm-threads")]
#[derive(Default)]
pub(super) struct AttentionScratch {
    output: Vec<f32>,
    scores: Vec<f32>,
    work: Vec<f32>,
}

impl Model {
    #[cfg(feature = "kernel-checks")]
    pub(super) fn check_encoder_segments(&self) -> Result<serde_json::Value> {
        #[cfg(feature = "decima-wasm-threads")]
        if super::wasm_threads::count() > 1 && rayon::current_thread_index().is_none() {
            return rayon::scope(|_| self.check_encoder_segments());
        }
        let mut checked = 0;
        let mut cases = 0;
        for lengths in [
            vec![1, 7, 63],
            vec![31, 32, 65],
            vec![128, 13, 5],
            vec![512, 64, 17],
            vec![512, 64, 64, 64, 64, 64, 64, 64, 64],
            vec![32; 17],
            vec![512, 512, 64],
            vec![64; 17],
            std::iter::once(512)
                .chain(std::iter::repeat_n(13, 255))
                .collect(),
        ] {
            let tokens: Vec<Vec<u32>> = lengths
                .iter()
                .enumerate()
                .map(|(segment, &length)| {
                    (0..length)
                        .map(|i| 3 + ((i * 1741 + segment * 1193) % 249000) as u32)
                        .collect()
                })
                .collect();
            let sequences: Vec<_> = tokens.iter().map(Vec::as_slice).collect();
            for tile in [1, 3, 4] {
                let mut plans = vec![
                    super::specialization::default_plan(super::Kernel::WasmSimd);
                    sequences.len()
                ];
                for (i, plan) in plans.iter_mut().enumerate() {
                    plan.int8_projection_rows = Some(tile);
                    plan.attention_scores = 512 * (i + 1);
                }
                let actual =
                    self.encode_segment_groups(&sequences, &plans, &mut Scratch::default())?;
                ensure!(
                    actual.len() == sequences.len(),
                    "grouped encoder changed sequence count"
                );
                let mut scratch = Scratch::default();
                for ((ids, plan), actual) in sequences.iter().zip(&plans).zip(actual) {
                    let expected = self.encode(ids, &mut scratch, *plan)?;
                    ensure!(
                        actual.len() == expected.len()
                            && actual
                                .iter()
                                .zip(expected.iter())
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "batched encoder changed hidden states at lengths={lengths:?}, tile={tile}"
                    );
                    checked += actual.len();
                }
                cases += 1;
            }
        }
        Ok(serde_json::json!({"cases":cases,"hidden_values":checked,"bitwise_equal":true}))
    }

    pub(super) fn encode_segment_groups(
        &self,
        sequences: &[&[u32]],
        plans: &[CpuPlan],
        scratch: &mut Scratch,
    ) -> Result<Vec<Arc<[f32]>>> {
        ensure!(
            !sequences.is_empty() && sequences.len() == plans.len(),
            "invalid encoder segments"
        );
        if sequences.iter().map(|ids| ids.len()).sum::<usize>() <= 1024 {
            return self.encode_segments(sequences, plans, scratch);
        }
        let mut output = Vec::with_capacity(sequences.len());
        let mut first = 0;
        while first < sequences.len() {
            let mut end = first;
            let mut rows = 0;
            while end < sequences.len() && rows + sequences[end].len() <= 1024 {
                rows += sequences[end].len();
                end += 1;
            }
            ensure!(end > first, "encoder sequence exceeds the group limit");
            output.extend(self.encode_segments(
                &sequences[first..end],
                &plans[first..end],
                scratch,
            )?);
            first = end;
        }
        Ok(output)
    }

    pub(super) fn encode_segments(
        &self,
        sequences: &[&[u32]],
        plans: &[CpuPlan],
        s: &mut Scratch,
    ) -> Result<Vec<Arc<[f32]>>> {
        ensure!(
            !sequences.is_empty() && sequences.len() == plans.len(),
            "invalid encoder segments"
        );
        let plan = plans[0];
        ensure!(
            plans
                .iter()
                .all(|p| p.projection_rows == plan.projection_rows
                    && p.int8_projection_rows == plan.int8_projection_rows),
            "incompatible segment projections"
        );
        let mut offsets = Vec::with_capacity(sequences.len() + 1);
        offsets.push(0);
        for ids in sequences {
            offsets.push(offsets.last().unwrap() + ids.len() * D);
        }
        let total = offsets[sequences.len()] / D;
        ensure!(total <= 1024, "encoder batch exceeds scratch bound");
        s.hidden.resize(total * D, 0.);
        let mut base = 0;
        for ids in sequences {
            ensure!(
                !ids.is_empty() && ids.len() <= 512 && ids.iter().all(|id| (*id as usize) < 250037),
                "invalid encoder token sequence"
            );
            for (position, &id) in ids.iter().enumerate() {
                let row = &mut s.hidden[(base + position) * D..][..D];
                self.embeddings.row(id as usize, row);
                for i in 0..D {
                    row[i] = (row[i] + self.types[i]) + self.positions[position * D + i];
                }
            }
            base += ids.len();
        }
        self.embedding_norm.in_place(&mut s.hidden);
        #[cfg(feature = "decima-wasm-threads")]
        if s.segment_attention.len() < sequences.len() {
            s.segment_attention
                .resize_with(sequences.len(), Default::default);
        }
        for layer in &self.encoder {
            #[cfg(not(feature = "int8"))]
            let shared = false;
            #[cfg(feature = "int8")]
            let shared = Linear::apply_shared_int8(
                [&layer.q, &layer.k, &layer.v],
                &s.hidden,
                [&mut s.q, &mut s.k, &mut s.v],
                &mut s.work,
                plan,
            );
            if !shared {
                layer
                    .q
                    .apply_with_plan(&s.hidden, &mut s.q, &mut s.work, plan);
                layer
                    .k
                    .apply_with_plan(&s.hidden, &mut s.k, &mut s.work, plan);
                layer
                    .v
                    .apply_with_plan(&s.hidden, &mut s.v, &mut s.work, plan);
            }
            s.ctx.clear();
            #[cfg(feature = "decima-wasm-threads")]
            let parallel_attention = super::wasm_threads::count() > 1;
            #[cfg(not(feature = "decima-wasm-threads"))]
            let parallel_attention = false;
            #[cfg(feature = "decima-wasm-threads")]
            if parallel_attention {
                use rayon::prelude::*;
                s.segment_attention[..sequences.len()]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(index, scratch)| {
                        let start = offsets[index];
                        let end = offsets[index + 1];
                        cpu::attention_with_tile(
                            &s.q[start..end],
                            &s.k[start..end],
                            &s.v[start..end],
                            D,
                            12,
                            &mut scratch.output,
                            &mut scratch.scores,
                            &mut scratch.work,
                            plans[index].query_tile(sequences[index].len()),
                            self.avx2,
                        );
                    });
                for scratch in &s.segment_attention[..sequences.len()] {
                    s.ctx.extend_from_slice(&scratch.output);
                }
            }
            if !parallel_attention {
                for (index, (ids, segment_plan)) in sequences.iter().zip(plans).enumerate() {
                    let base = offsets[index];
                    let end = offsets[index + 1];
                    cpu::attention_with_tile(
                        &s.q[base..end],
                        &s.k[base..end],
                        &s.v[base..end],
                        D,
                        12,
                        &mut s.tmp,
                        &mut s.scores,
                        &mut s.heads,
                        segment_plan.query_tile(ids.len()),
                        self.avx2,
                    );
                    s.ctx.extend_from_slice(&s.tmp);
                }
            }
            layer
                .o
                .apply_with_plan(&s.ctx, &mut s.tmp, &mut s.work, plan);
            cpu::add(&mut s.hidden, &s.tmp);
            layer.n1.in_place(&mut s.hidden);
            #[cfg(feature = "int8")]
            if Linear::try_encoder_ffn(
                &layer.ff1,
                &layer.ff2,
                &layer.n2,
                &mut s.hidden,
                plan,
                &mut s.ffn,
            ) {
                continue;
            }
            layer
                .ff1
                .apply_with_plan(&s.hidden, &mut s.ff, &mut s.work, plan);
            cpu::gelu(&mut s.ff);
            layer
                .ff2
                .apply_with_plan(&s.ff, &mut s.tmp, &mut s.work, plan);
            cpu::add(&mut s.hidden, &s.tmp);
            layer.n2.in_place(&mut s.hidden);
        }
        Ok(offsets
            .windows(2)
            .map(|span| Arc::from(&s.hidden[span[0]..span[1]]))
            .collect())
    }
}
