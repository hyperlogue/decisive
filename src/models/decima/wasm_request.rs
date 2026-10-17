use super::{CachedChoice, D, Kind, Model, Precision, Question, Session, cpu, input};
#[cfg(feature = "decima-wasm-threads")]
use super::{wasm_score, wasm_threads};
use anyhow::{Result, ensure};
use std::sync::Arc;

impl Model {
    pub(super) fn decide_inner(
        &self,
        state: &str,
        q: &Question,
        s: &mut Session,
    ) -> Result<Vec<f32>> {
        ensure!((2..=255).contains(&q.choices.len()), "need 2–255 choices");
        if q.kind == Kind::Score {
            ensure!(q.choices.len() <= 10, "score allows at most 10 levels");
        }
        if !s
            .identity
            .as_ref()
            .is_some_and(|id| Arc::ptr_eq(id, &self.identity))
        {
            s.clear_cache();
            s.identity = Some(self.identity.clone());
        }
        let body = if self.config.question_in_state {
            format!("{}\n{state}", q.text)
        } else {
            state.into()
        };
        let text = format!(
            "{}{}",
            self.config.state_prefix,
            input::normalize(&body, &q.lang)
        );
        let ids = self.input.ids(&text)?;
        ensure!(
            ids.len() <= self.input.max_state,
            "STATE_TRUNCATED: state + question is {} tokens; maximum {}",
            ids.len(),
            self.input.max_state
        );
        let state_plan = s.plan(ids.len(), self.options.kernel);
        let rows = self.input.choices(
            q,
            &self.config.choice_prefix,
            self.config.question_in_choices,
            self.config.choice_fit,
        )?;
        let batched = {
            let precision = matches!(
                self.options.precision,
                Precision::Int8 | Precision::Int8Full
            );
            #[cfg(feature = "decima-wasm-threads")]
            let precision = precision || wasm_threads::count() > 1;
            let eligible = precision
                && rows.len() >= 2
                && rows.iter().enumerate().all(|(i, row)| {
                    !s.cache.iter().any(|(key, _)| key == row) && !rows[..i].contains(row)
                });
            if eligible {
                let plans: Vec<_> = std::iter::once(state_plan)
                    .chain(rows.iter().map(|r| s.plan(r.len(), self.options.kernel)))
                    .collect();
                if plans.iter().all(|p| {
                    p.projection_rows == state_plan.projection_rows
                        && p.int8_projection_rows == state_plan.int8_projection_rows
                }) {
                    let sequences: Vec<_> = std::iter::once(ids.as_slice())
                        .chain(rows.iter().map(Vec::as_slice))
                        .collect();
                    Some(self.encode_segment_groups(&sequences, &plans, &mut s.encoder)?)
                } else {
                    None
                }
            } else {
                None
            }
        };
        let state_h = if let Some(values) = &batched {
            values[0].clone()
        } else {
            self.encode(&ids, &mut s.encoder, state_plan)?
        };
        s.stats.state_tokens += ids.len() as u64;
        let state_mean = cpu::mean(&state_h, D);
        let norm = |v: &[f32]| v.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let state_norm = norm(&state_mean);
        let cosine = |choice_mean: &[f32], choice_norm: f32| {
            state_mean
                .iter()
                .zip(choice_mean)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                / (state_norm * choice_norm)
        };
        s.memory.resize_with(self.scorer.len(), Default::default);
        for (l, (k, v)) in self.scorer.iter().zip(&mut s.memory) {
            l.cross_attn
                .k
                .apply_with_plan(&state_h, k, &mut s.scorer.work, state_plan);
            l.cross_attn
                .v
                .apply_with_plan(&state_h, v, &mut s.scorer.work, state_plan);
        }
        #[cfg(feature = "decima-wasm-threads")]
        let parallel_choices = wasm_threads::count() > 1 && rows.len() >= 2;
        #[cfg(feature = "decima-wasm-threads")]
        let choice_count = rows.len();
        #[cfg(feature = "decima-wasm-threads")]
        if parallel_choices && s.choices.len() < choice_count {
            s.choices.resize_with(choice_count, Default::default);
        }
        #[cfg(feature = "decima-wasm-threads")]
        let parallel_prefix = if parallel_choices && choice_count >= 8 {
            if let Some(encoded) = &batched {
                let plans: Vec<_> = rows
                    .iter()
                    .map(|ids| s.plan(ids.len(), self.options.kernel))
                    .collect();
                wasm_score::prepare(self, &mut s.choices[..choice_count], &encoded[1..], &plans);
                true
            } else {
                false
            }
        } else {
            false
        };
        let mut logits = vec![];
        let mut zs = vec![];
        for (choice_index, ids) in rows.into_iter().enumerate() {
            let plan = s.plan(ids.len(), self.options.kernel);
            #[cfg(feature = "decima-wasm-threads")]
            let x = if parallel_choices {
                &mut s.choices[choice_index].scratch
            } else {
                &mut s.scorer
            };
            #[cfg(not(feature = "decima-wasm-threads"))]
            let x = &mut s.scorer;
            let similarity;
            if let Some(index) = s.cache.iter().position(|(key, _)| key == &ids) {
                let entry = s.cache.remove(index).unwrap();
                x.hidden.clear();
                x.hidden.extend_from_slice(&entry.1.hidden);
                x.q.clear();
                x.q.extend_from_slice(&entry.1.query);
                similarity = cosine(&entry.1.mean, entry.1.norm);
                s.cache.push_back(entry);
                s.stats.choice_cache_hits += 1;
            } else {
                let h = if let Some(values) = &batched {
                    values[choice_index + 1].clone()
                } else {
                    self.encode(&ids, &mut s.encoder, plan)?
                };
                s.stats.choice_tokens += ids.len() as u64;
                let mean = cpu::mean(&h, D);
                let choice_norm = norm(&mean);
                similarity = cosine(&mean, choice_norm);
                #[cfg(feature = "decima-wasm-threads")]
                let prepared = parallel_prefix;
                #[cfg(not(feature = "decima-wasm-threads"))]
                let prepared = false;
                if !prepared {
                    x.hidden.clear();
                    x.hidden.extend_from_slice(&h);
                    self.scorer[0].prepare_query(x, plan, self.avx2);
                }
                let bytes = (ids.len() + x.hidden.len() + x.q.len() + mean.len() + 1) * 4;
                if bytes <= s.cache_limit {
                    while s.cache_bytes + bytes > s.cache_limit {
                        if let Some((key, old)) = s.cache.pop_front() {
                            s.cache_bytes -= old.bytes(key.len());
                        }
                    }
                    s.cache_bytes += bytes;
                    s.cache.push_back((
                        ids,
                        CachedChoice {
                            hidden: x.hidden.as_slice().into(),
                            query: x.q.as_slice().into(),
                            mean: mean.into_boxed_slice(),
                            norm: choice_norm,
                        },
                    ));
                }
            }
            #[cfg(feature = "decima-wasm-threads")]
            if parallel_choices {
                let choice = &mut s.choices[choice_index];
                choice.plan = plan;
                choice.similarity = similarity;
                continue;
            }
            for (index, (l, (mk, mv))) in self.scorer.iter().zip(&s.memory).enumerate() {
                if index > 0 {
                    l.prepare_query(x, plan, self.avx2);
                }
                cpu::attention_with_tile(
                    &x.q,
                    mk,
                    mv,
                    D,
                    6,
                    &mut x.ctx,
                    &mut x.scores,
                    &mut x.heads,
                    state_plan.query_tile(state_h.len() / D),
                    self.avx2,
                );
                l.cross_attn
                    .o
                    .apply_with_plan(&x.ctx, &mut x.tmp, &mut x.work, plan);
                cpu::add(&mut x.hidden, &x.tmp);
                l.n3.apply(&x.hidden, &mut x.norm);
                l.ff1.apply_with_plan(&x.norm, &mut x.ff, &mut x.work, plan);
                cpu::gelu(&mut x.ff);
                l.ff2.apply_with_plan(&x.ff, &mut x.tmp, &mut x.work, plan);
                cpu::add(&mut x.hidden, &x.tmp);
            }
            self.out_norm.in_place(&mut x.hidden);
            let z = cpu::mean(&x.hidden, D);
            self.score.apply(&z, &mut x.tmp, &mut x.work);
            logits.push(
                (x.tmp[0] + self.sim_scale * (similarity - self.sim_center))
                    / self.config.temperature,
            );
            zs.extend(z);
        }
        #[cfg(feature = "decima-wasm-threads")]
        if parallel_choices {
            wasm_score::finish(
                self,
                &mut s.choices[..choice_count],
                &s.memory,
                state_plan.query_tile(state_h.len() / D),
            );
            for choice in &s.choices[..choice_count] {
                logits.push(choice.logit);
                zs.extend_from_slice(&choice.mean);
            }
        }
        self.finish_probabilities(logits, &zs, q.kind, &mut s.scorer)
    }
}
