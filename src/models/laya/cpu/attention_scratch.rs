#[derive(Clone, Copy, Default)]
pub enum AttentionMath {
    #[default]
    Scalar,
    ExactSimd,
    TableExp,
    TableExpSum,
    TableExpRows,
    PolynomialExpSum,
}
#[derive(Default)]
pub struct AttentionScratch {
    tables: Vec<RopeTable>,
    pub(super) packed_avx2: bool,
    pub(super) long_attention: bool,
    pub(super) math: AttentionMath,
    pub(super) packed_normalization: bool,
    pub(super) row_parallel_avx2: bool,
    pub(super) key_tile: usize,
    pub(super) heads: Vec<HeadScratch>,
}
struct RopeTable {
    theta: u32,
    dim: usize,
    inv: Vec<f32>,
    angles: Vec<(f32, f32)>,
}
#[derive(Default)]
pub(super) struct HeadScratch {
    pub scores: Vec<f32>,
    pub values: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub normalizers: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub online: Vec<f64>,
}
impl AttentionScratch {
    pub fn set_packed_avx2(&mut self, enabled: bool) {
        self.packed_avx2 = enabled;
    }
    pub fn set_long_attention(&mut self, enabled: bool) {
        self.long_attention = enabled;
    }
    pub fn set_simd_softmax(&mut self, enabled: bool) {
        self.math = if enabled {
            AttentionMath::ExactSimd
        } else {
            AttentionMath::Scalar
        };
    }
    pub fn set_math(&mut self, math: AttentionMath) {
        self.math = math;
    }
    pub fn set_packed_normalization(&mut self, enabled: bool) {
        self.packed_normalization = enabled;
    }
    pub fn set_row_parallel_attention(&mut self, enabled: bool) {
        self.row_parallel_avx2 = enabled;
    }
    pub fn set_key_tile(&mut self, rows: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            rows == 0 || (rows.is_power_of_two() && (64..=1024).contains(&rows)),
            "attention key tile must be zero or a power of two in 64..=1024"
        );
        self.key_tile = rows;
        Ok(())
    }
    pub fn retained_bytes(&self) -> usize {
        let tables = self
            .tables
            .iter()
            .map(|t| t.inv.capacity() * 4 + t.angles.capacity() * 8)
            .sum::<usize>();
        tables
            + self
                .heads
                .iter()
                .map(|h| {
                    (h.scores.capacity()
                        + h.values.capacity()
                        + h.query.capacity()
                        + h.key.capacity()
                        + h.value.capacity()
                        + h.normalizers.capacity()
                        + h.probabilities.capacity())
                        * 4
                        + h.online.capacity() * 8
                })
                .sum::<usize>()
    }
    pub(super) fn rotate(&mut self, qkv: &mut [f32], heads: usize, dim: usize, theta: f32) {
        self.rotate_offset(qkv, heads, dim, theta, 0);
    }
    pub(crate) fn rotate_offset(
        &mut self,
        qkv: &mut [f32],
        heads: usize,
        dim: usize,
        theta: f32,
        offset: usize,
    ) {
        let half = dim / 2;
        let d = heads * dim;
        let n = qkv.len() / (3 * d);
        assert!(offset <= n);
        let index = self
            .tables
            .iter()
            .position(|t| t.theta == theta.to_bits() && t.dim == dim)
            .unwrap_or_else(|| {
                self.tables.push(RopeTable {
                    theta: theta.to_bits(),
                    dim,
                    inv: (0..half)
                        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
                        .collect(),
                    angles: Vec::new(),
                });
                self.tables.len() - 1
            });
        let table = &mut self.tables[index];
        if half == 0 {
            return;
        }
        for t in table.angles.len() / half..n.max(offset + 1) {
            table
                .angles
                .extend(table.inv.iter().map(|&v| (t as f32 * v).sin_cos()));
        }
        for t in 0..n {
            let position = t.abs_diff(offset);
            for head in 0..heads {
                for which in 0..2 {
                    let v = &mut qkv[t * 3 * d + which * d + head * dim..][..dim];
                    for i in 0..half {
                        let (mut s, c) = table.angles[position * half + i];
                        if t < offset {
                            s = -s;
                        }
                        let (a, b) = (v[i], v[i + half]);
                        v[i] = a * c - b * s;
                        v[i + half] = b * c + a * s;
                    }
                }
            }
        }
    }
}
