use crate::laya::cpu::ProjectionStrategy;
use crate::length::{Band, Rules};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuPlan {
    pub attention_query_tile: usize,
    pub projection: ProjectionStrategy,
}
impl CpuPlan {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.attention_query_tile, 16 | 32 | 64 | 128 | 256),
            "invalid Laya attention query tile"
        );
        match self.projection {
            ProjectionStrategy::WasmRows(rows) => ensure!(
                (1..=6).contains(&rows),
                "Wasm projection rows must be 1..=6"
            ),
            ProjectionStrategy::PackedCache { rows, reduction } => {
                ensure!(
                    rows > 0 && rows.is_multiple_of(6) && reduction > 0,
                    "packed projection rows must be a positive multiple of six, with nonzero reduction"
                )
            }
            ProjectionStrategy::LongRows(rows) => {
                ensure!(
                    rows > 0 && rows.is_multiple_of(6),
                    "long projection rows must be a positive multiple of six"
                )
            }
            ProjectionStrategy::ReductionBlocks(rows) => {
                ensure!(rows > 0, "empty projection tile")
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reject_plans_that_would_panic_in_packed_dispatch() {
        for projection in [
            ProjectionStrategy::PackedCache {
                rows: 7,
                reduction: 64,
            },
            ProjectionStrategy::PackedCache {
                rows: 6,
                reduction: 0,
            },
            ProjectionStrategy::LongRows(1),
            ProjectionStrategy::WasmRows(7),
        ] {
            assert!(
                CpuPlan {
                    attention_query_tile: 32,
                    projection
                }
                .validate()
                .is_err()
            );
        }
        for projection in [
            ProjectionStrategy::PackedCache {
                rows: 24,
                reduction: usize::MAX,
            },
            ProjectionStrategy::LongRows(6),
            ProjectionStrategy::WasmRows(1),
        ] {
            assert!(
                CpuPlan {
                    attention_query_tile: 32,
                    projection
                }
                .validate()
                .is_ok()
            );
        }
    }
}
pub fn wasm() -> Rules<CpuPlan> {
    Rules::new(
        vec![Band {
            through: 64,
            settings: CpuPlan {
                attention_query_tile: 16,
                projection: ProjectionStrategy::WasmRows(4),
            },
        }],
        CpuPlan {
            attention_query_tile: 32,
            projection: ProjectionStrategy::WasmRows(4),
        },
    )
    .unwrap()
}
pub fn avx2() -> Rules<CpuPlan> {
    Rules::new(
        vec![Band {
            through: 512,
            settings: CpuPlan {
                attention_query_tile: 64,
                projection: ProjectionStrategy::PackedInput,
            },
        }],
        CpuPlan {
            attention_query_tile: 64,
            projection: ProjectionStrategy::LongRows(24),
        },
    )
    .unwrap()
}
pub fn portable() -> Rules<CpuPlan> {
    Rules::new(
        vec![],
        CpuPlan {
            attention_query_tile: 64,
            projection: ProjectionStrategy::Default,
        },
    )
    .unwrap()
}
