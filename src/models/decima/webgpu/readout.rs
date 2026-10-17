use super::super::{D, Kind, cpu};
use anyhow::{Result, ensure};

pub(super) struct Readout {
    score: Projection,
    ord_g: Projection,
    ord_gap: Projection,
    sim_scale: f32,
    sim_center: f32,
}

struct Projection {
    weight: Vec<f32>,
    bias: f32,
}
impl Projection {
    fn from_cpu(source: cpu::Linear) -> Result<Self> {
        let (input, output, weight, bias) = source.into_float_parts()?;
        ensure!(
            input == D && output == 1,
            "invalid Decima readout projection"
        );
        Ok(Self {
            weight,
            bias: bias[0],
        })
    }
    fn apply(&self, x: &[f32], out: &mut Vec<f32>) {
        assert_eq!(x.len() % D, 0);
        out.resize(x.len() / D, 0.);
        for (i, dst) in out.chunks_mut(32).enumerate() {
            unsafe {
                matrixmultiply::sgemm(
                    dst.len(),
                    D,
                    1,
                    1.,
                    x.as_ptr().add(i * 32 * D),
                    D as isize,
                    1,
                    self.weight.as_ptr(),
                    1,
                    D as isize,
                    0.,
                    dst.as_mut_ptr(),
                    1,
                    1,
                );
            }
            for value in dst {
                *value += self.bias;
            }
        }
    }
}
impl Readout {
    pub(super) fn new(
        score: cpu::Linear,
        ord_g: cpu::Linear,
        ord_gap: cpu::Linear,
        sim_scale: f32,
        sim_center: f32,
    ) -> Result<Self> {
        Ok(Self {
            score: Projection::from_cpu(score)?,
            ord_g: Projection::from_cpu(ord_g)?,
            ord_gap: Projection::from_cpu(ord_gap)?,
            sim_scale,
            sim_center,
        })
    }
    pub(super) fn probabilities(
        &self,
        pooled: &[f32],
        kind: Kind,
        temperature: f32,
    ) -> Result<Vec<f32>> {
        ensure!(
            pooled.len() >= 5 * D && (pooled.len() - D) % (2 * D) == 0,
            "invalid Decima pooled readout"
        );
        let state_mean = &pooled[..D];
        let mut logits = Vec::new();
        let mut zs = Vec::new();
        let mut tmp = Vec::new();
        let norm = |v: &[f32]| v.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for pair in pooled[D..].chunks_exact(2 * D) {
            let original = &pair[..D];
            let z = &pair[D..];
            self.score.apply(z, &mut tmp);
            let dot = state_mean
                .iter()
                .zip(original)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            logits.push(
                (tmp[0]
                    + self.sim_scale
                        * (dot / (norm(state_mean) * norm(original)) - self.sim_center))
                    / temperature,
            );
            zs.extend_from_slice(z);
        }
        self.finish_probabilities(logits, &zs, kind, &mut tmp)
    }
    fn finish_probabilities(
        &self,
        mut logits: Vec<f32>,
        zs: &[f32],
        kind: Kind,
        tmp: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite Decima scores"
        );
        match kind {
            Kind::Score => {
                softmax(&mut logits);
                let expected = logits
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f32 * p)
                    .sum::<f32>()
                    - (logits.len() - 1) as f32 / 2.;
                let zm = mean(zs);
                self.ord_g.apply(&zm, tmp);
                let g = tmp[0] + expected;
                self.ord_gap.apply(zs, tmp);
                let gaps: Vec<_> = tmp
                    .iter()
                    .map(|v| {
                        if *v > 20. {
                            *v + 1e-3
                        } else {
                            v.exp().ln_1p() + 1e-3
                        }
                    })
                    .collect();
                let center = gaps.iter().sum::<f32>() / 2.;
                let mut theta = 0.;
                let mut previous = 1.;
                for i in 0..logits.len() - 1 {
                    theta += gaps[i];
                    let sig = 1. / (1. + (-(g - theta + center)).exp());
                    logits[i] = (previous - sig).max(1e-7);
                    previous = sig;
                }
                *logits.last_mut().unwrap() = previous.max(1e-7);
                let sum = logits.iter().sum::<f32>();
                for v in &mut logits {
                    *v /= sum;
                }
            }
            Kind::Rank => {
                for v in &mut logits {
                    *v = 1. / (1. + (-*v).exp());
                }
            }
            _ => softmax(&mut logits),
        }
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite Decima probabilities"
        );
        Ok(logits)
    }
}
fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.;
    for v in &mut *values {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in values {
        *v /= sum;
    }
}
fn mean(values: &[f32]) -> Vec<f32> {
    let mut out = vec![0.; D];
    for row in values.chunks_exact(D) {
        for (a, b) in out.iter_mut().zip(row) {
            *a += b;
        }
    }
    let n = (values.len() / D) as f32;
    for v in &mut out {
        *v /= n;
    }
    out
}
