//! Cache first-layer state-to-state attention only; question attention and later layers still run.
use crate::laya::cpu::{self, AttentionScratch, SoftmaxState};
use std::sync::{Arc, Weak};

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StateCacheStats {
    pub hits: u64,
    pub builds: u64,
    pub skipped_requests: u64,
    pub reused_state_queries: u64,
    pub model_resets: u64,
    pub resident_state_tokens: usize,
    pub retained_bytes: usize,
}
#[derive(Default)]
pub(super) struct StateCache {
    capacity: usize,
    model: Weak<()>,
    qkv: Vec<f32>,
    context: Vec<f32>,
    states: Vec<SoftmaxState>,
    prefix_states: Vec<SoftmaxState>,
    new_states: Vec<SoftmaxState>,
    shape: (usize, usize, u32),
    stats: StateCacheStats,
}
impl StateCache {
    pub fn enabled(&self) -> bool {
        self.capacity > 0
    }
    pub fn configure(&mut self, capacity: usize) {
        if self.capacity != capacity {
            *self = Self {
                capacity,
                ..Self::default()
            };
        }
    }
    pub fn invalidate(&mut self) {
        *self = Self {
            capacity: self.capacity,
            ..Self::default()
        };
    }
    pub fn clear(&mut self) {
        self.configure(0);
    }
    pub fn reset_stats(&mut self) {
        self.stats = StateCacheStats::default();
    }
    pub fn stats(&self) -> StateCacheStats {
        let mut stats = self.stats.clone();
        stats.resident_state_tokens = if self.shape.0 * self.shape.1 == 0 {
            0
        } else {
            self.context.len() / (self.shape.0 * self.shape.1)
        };
        stats.retained_bytes = (self.qkv.capacity() + self.context.capacity()) * 4
            + (self.states.capacity() + self.prefix_states.capacity() + self.new_states.capacity())
                * std::mem::size_of::<SoftmaxState>();
        stats
    }
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        model: &Arc<()>,
        qkv: &mut [f32],
        out: &mut [f32],
        heads: usize,
        dim: usize,
        theta: f32,
        state_start: usize,
        tile: usize,
        attention: &mut AttentionScratch,
    ) -> bool {
        let d = heads * dim;
        let n = qkv.len() / (3 * d);
        assert!(state_start <= n && out.len() == n * d);
        let state_tokens = n - state_start;
        if !self.enabled() || state_start == 0 || state_tokens < 128 || state_tokens > self.capacity
        {
            self.stats.skipped_requests += 1;
            return false;
        }
        if !self.model.ptr_eq(&Arc::downgrade(model)) {
            self.qkv.clear();
            self.context.clear();
            self.states.clear();
            self.model = Arc::downgrade(model);
            self.stats.model_resets += 1;
        }
        attention.rotate_offset(qkv, heads, dim, theta, state_start);
        let current = &qkv[state_start * 3 * d..];
        let hit = self.shape == (heads, dim, theta.to_bits())
            && current.len() == self.qkv.len()
            && current
                .iter()
                .zip(&self.qkv)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if hit {
            self.stats.hits += 1;
            self.stats.reused_state_queries += state_tokens as u64;
        } else {
            if self.context.capacity() < state_tokens * d {
                self.context
                    .reserve_exact(state_tokens * d - self.context.len());
            }
            self.context.resize(state_tokens * d, 0.);
            cpu::attention_block(
                qkv,
                &mut self.context,
                &mut self.states,
                heads,
                dim,
                state_start..n,
                state_start..n,
                tile,
                attention,
            );
            if self.qkv.capacity() < current.len() {
                self.qkv.reserve_exact(current.len() - self.qkv.len());
            }
            self.qkv.clear();
            self.qkv.extend_from_slice(current);
            self.shape = (heads, dim, theta.to_bits());
            self.stats.builds += 1;
        }
        cpu::attention_block(
            qkv,
            &mut out[..state_start * d],
            &mut self.prefix_states,
            heads,
            dim,
            0..state_start,
            0..n,
            tile,
            attention,
        );
        cpu::attention_block(
            qkv,
            &mut out[state_start * d..],
            &mut self.new_states,
            heads,
            dim,
            state_start..n,
            0..state_start,
            tile,
            attention,
        );
        merge(
            &mut out[state_start * d..],
            &self.new_states,
            &self.context,
            &self.states,
            heads,
            dim,
        );
        true
    }
}

fn merge(
    new: &mut [f32],
    new_states: &[SoftmaxState],
    cached: &[f32],
    old_states: &[SoftmaxState],
    heads: usize,
    dim: usize,
) {
    assert_eq!(new.len(), cached.len());
    assert_eq!(new_states.len(), old_states.len());
    assert_eq!(new_states.len() * dim, new.len());
    assert!(heads > 0);
    for (((dst, old), a), b) in new
        .chunks_exact_mut(dim)
        .zip(cached.chunks_exact(dim))
        .zip(new_states)
        .zip(old_states)
    {
        let max = a.maximum.max(b.maximum) as f64;
        let wa = a.denominator as f64 * (a.maximum as f64 - max).exp();
        let wb = b.denominator as f64 * (b.maximum as f64 - max).exp();
        let total = wa + wb;
        for (x, &y) in dst.iter_mut().zip(old) {
            *x = ((*x as f64 * wa + y as f64 * wb) / total) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_merge_is_stable_with_large_logit_offsets() {
        let mut a = vec![2., 4., -3., 5.];
        let b = vec![6., 8., 7., -9.];
        let states_a = [
            SoftmaxState {
                maximum: 1000.,
                denominator: 2.,
            },
            SoftmaxState {
                maximum: -1000.,
                denominator: 3.,
            },
        ];
        let states_b = [
            SoftmaxState {
                maximum: 1000.,
                denominator: 6.,
            },
            SoftmaxState {
                maximum: 1000.,
                denominator: 1.,
            },
        ];
        merge(&mut a, &states_a, &b, &states_b, 1, 2);
        assert_eq!(a, [5., 7., 7., -9.]);
    }
    #[test]
    fn state_cache_survives_question_length_changes_and_invalidates_changed_state() {
        for dim in [8, 64] {
            let heads = 2;
            let d = heads * dim;
            let mut cache = StateCache::default();
            cache.configure(512);
            let mut identity = Arc::new(());
            let mut attention = AttentionScratch::default();
            attention.set_packed_avx2(cpu::packed_supported());
            attention.set_long_attention(true);
            attention.set_math(cpu::AttentionMath::TableExp);
            for (case, start) in [20, 33, 20, 21, 21].into_iter().enumerate() {
                let n = start + 160;
                let raw: Vec<_> = (0..n * 3 * d)
                    .map(|i| {
                        let row = i / (3 * d);
                        let col = i % (3 * d);
                        let token = if row < start {
                            row + case * 73
                        } else {
                            row - start + 1000
                        };
                        ((token * 3 * d + col) as f32 * 0.017).sin() * 0.5
                    })
                    .collect();
                let mut qkv = raw.clone();
                if case >= 3 {
                    qkv[start * 3 * d + 2 * d] += 0.25;
                }
                let mut expected_qkv = qkv.clone();
                attention.rotate_offset(&mut expected_qkv, heads, dim, 160000., start);
                let mut expected = vec![0.; n * d];
                cpu::attention_with_scratch(
                    &mut expected_qkv,
                    &mut expected,
                    heads,
                    dim,
                    None,
                    None,
                    64,
                    &mut attention,
                    true,
                );
                let mut actual = vec![f32::NAN; n * d];
                if case == 4 {
                    identity = Arc::new(());
                }
                assert!(cache.apply(
                    &identity,
                    &mut qkv,
                    &mut actual,
                    heads,
                    dim,
                    160000.,
                    start,
                    64,
                    &mut attention
                ));
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| (a - b).abs() <= 2e-5),
                    "dim={dim} case={case}"
                );
            }
            assert_eq!(cache.stats().hits, 2);
            assert_eq!(cache.stats().builds, 3);
            assert_eq!(cache.stats().model_resets, 2);
            assert_eq!(cache.stats().resident_state_tokens, 160);
            cache.clear();
            assert_eq!(cache.stats().retained_bytes, 0);
            assert!(!cache.enabled());
        }
    }
}
