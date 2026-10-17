//! Cache unrotated first-layer Q/K/V; changing questions can shift RoPE positions.
use crate::laya::{
    cpu::{Linear, Norm, ProjectionStrategy},
    weights::Matrix,
};
use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct TokenCacheStats {
    pub cached_tokens: u64,
    pub projected_tokens: u64,
    pub repeated_tokens_in_request: u64,
    pub evictions: u64,
    pub model_resets: u64,
    pub resident_tokens: usize,
    pub retained_bytes: usize,
}

#[derive(Default)]
pub(super) struct TokenCache {
    capacity: usize,
    model: Weak<()>,
    index: HashMap<u32, usize>,
    ids: Vec<u32>,
    rows: Vec<f32>,
    next: usize,
    pending: HashMap<u32, usize>,
    missing_ids: Vec<u32>,
    missing_rows: Vec<(usize, usize)>,
    embedding: Vec<f32>,
    normalized: Vec<f32>,
    projected: Vec<f32>,
    stats: TokenCacheStats,
}
impl TokenCache {
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
    pub fn stats(&self) -> TokenCacheStats {
        let mut stats = self.stats.clone();
        stats.resident_tokens = self.ids.len();
        stats.retained_bytes = (self.rows.capacity()
            + self.embedding.capacity()
            + self.normalized.capacity()
            + self.projected.capacity())
            * 4;
        stats
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
        self.stats = TokenCacheStats::default();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        model: &Arc<()>,
        ids: impl Iterator<Item = u32>,
        embedding: &Matrix,
        norm: &Norm,
        qkv: &Linear,
        hidden: &mut [f32],
        output: &mut Vec<f32>,
        workspace: &mut Vec<f32>,
        group: usize,
        projection: ProjectionStrategy,
    ) {
        assert!(self.enabled());
        let d = qkv.input;
        assert_eq!(qkv.output, d * 3);
        assert_eq!(hidden.len() % d, 0);
        if !self.model.ptr_eq(&Arc::downgrade(model)) {
            self.index.clear();
            self.ids.clear();
            self.rows.clear();
            self.next = 0;
            self.model = Arc::downgrade(model);
            self.stats.model_resets += 1;
        }
        output.resize(hidden.len() * 3, 0.);
        self.pending.clear();
        self.missing_ids.clear();
        self.missing_rows.clear();
        let mut count = 0;
        for (row, id) in ids.enumerate() {
            count += 1;
            if let Some(&slot) = self.index.get(&id) {
                let values = &self.rows[slot * 4 * d..][..4 * d];
                hidden[row * d..][..d].copy_from_slice(&values[..d]);
                output[row * 3 * d..][..3 * d].copy_from_slice(&values[d..]);
                self.stats.cached_tokens += 1;
            } else {
                let unique = if let Some(&unique) = self.pending.get(&id) {
                    self.stats.repeated_tokens_in_request += 1;
                    unique
                } else {
                    let unique = self.missing_ids.len();
                    self.pending.insert(id, unique);
                    self.missing_ids.push(id);
                    unique
                };
                self.missing_rows.push((row, unique));
            }
        }
        assert_eq!(count * d, hidden.len());
        let missing = self.missing_ids.len();
        if missing == 0 {
            return;
        }
        self.embedding.resize(missing * d, 0.);
        for (row, &id) in self.embedding.chunks_exact_mut(d).zip(&self.missing_ids) {
            embedding.copy_row(id as usize, row);
        }
        norm.apply(&self.embedding, &mut self.normalized);
        qkv.apply_with_workspace(
            &self.normalized,
            &mut self.projected,
            group,
            projection,
            workspace,
        );
        self.stats.projected_tokens += missing as u64;
        for &(row, unique) in &self.missing_rows {
            hidden[row * d..][..d].copy_from_slice(&self.normalized[unique * d..][..d]);
            output[row * 3 * d..][..3 * d]
                .copy_from_slice(&self.projected[unique * 3 * d..][..3 * d]);
        }
        // Copy every requested row before eviction can invalidate hits.
        let wanted = (self.ids.len() + missing).min(self.capacity) * 4 * d;
        if self.rows.capacity() < wanted {
            self.rows.reserve_exact(wanted - self.rows.len());
        }
        for (unique, &id) in self.missing_ids.iter().enumerate() {
            let slot = if self.ids.len() < self.capacity {
                let slot = self.ids.len();
                self.ids.push(id);
                self.rows.resize((slot + 1) * 4 * d, 0.);
                slot
            } else {
                let slot = self.next;
                self.next = (self.next + 1) % self.capacity;
                self.index.remove(&self.ids[slot]);
                self.ids[slot] = id;
                self.stats.evictions += 1;
                slot
            };
            self.index.insert(id, slot);
            let values = &mut self.rows[slot * 4 * d..][..4 * d];
            values[..d].copy_from_slice(&self.normalized[unique * d..][..d]);
            values[d..].copy_from_slice(&self.projected[unique * 3 * d..][..3 * d]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unrotated_token_reuse_handles_reordering_eviction_and_model_changes() {
        let d = 8;
        let embedding = Matrix::F32((0..16 * d).map(|i| (i as f32 * 0.07).sin()).collect());
        let norm = Norm {
            weight: vec![1.; d],
            bias: None,
            eps: 1e-5,
        };
        let qkv = Linear::from_parts(
            d,
            3 * d,
            Matrix::F32((0..3 * d * d).map(|i| (i as f32 * 0.017).sin()).collect()),
            Some(vec![0.1; 3 * d]),
        );
        let mut cache = TokenCache::default();
        cache.configure(3);
        let mut identity = Arc::new(());
        for ids in [
            vec![1, 2, 1, 2],
            vec![2, 1, 2, 1],
            vec![3, 4, 1, 2, 5],
            vec![5, 4, 2, 1],
            vec![3],
            vec![3, 3],
        ] {
            let mut hidden = vec![f32::NAN; ids.len() * d];
            let mut got = Vec::new();
            cache.prepare(
                &identity,
                ids.iter().copied(),
                &embedding,
                &norm,
                &qkv,
                &mut hidden,
                &mut got,
                &mut Vec::new(),
                1,
                ProjectionStrategy::Default,
            );
            let mut raw = vec![0.; ids.len() * d];
            for (row, &id) in raw.chunks_exact_mut(d).zip(&ids) {
                embedding.copy_row(id as usize, row);
            }
            let mut normalized = Vec::new();
            norm.apply(&raw, &mut normalized);
            let mut expected = Vec::new();
            qkv.apply(&normalized, &mut expected);
            assert_eq!(hidden, normalized);
            assert!(got.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-6));
            assert!(cache.stats().resident_tokens <= 3);
            if ids.len() == 1 {
                identity = Arc::new(());
            }
        }
        assert!(cache.stats().cached_tokens > 0);
        assert!(cache.stats().evictions > 0);
        assert_eq!(cache.stats().model_resets, 2);
        cache.reset_stats();
        assert_eq!(cache.stats().projected_tokens, 0);
        cache.clear();
        assert!(!cache.enabled());
        assert_eq!(cache.stats().retained_bytes, 0);
    }
}
