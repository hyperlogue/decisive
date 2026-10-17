use crate::length::Rules;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuPlan {
    pub projection_rows: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub int8_projection_rows: Option<usize>,
    pub attention_scores: usize,
}
impl CpuPlan {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=32).contains(&self.projection_rows),
            "projection rows must be 1..=32"
        );
        ensure!(
            self.int8_projection_rows
                .is_none_or(|n| (1..=4).contains(&n)),
            "INT8 projection rows must be 1..=4"
        );
        ensure!(
            (512..=262144).contains(&self.attention_scores),
            "attention score budget must be 512..=262144 elements"
        );
        Ok(())
    }
    pub fn query_tile(self, keys: usize) -> usize {
        (self.attention_scores / keys.max(1)).max(1)
    }
}
pub fn portable() -> Rules<CpuPlan> {
    Rules::new(
        vec![],
        CpuPlan {
            projection_rows: 32,
            int8_projection_rows: None,
            attention_scores: 16384,
        },
    )
    .unwrap()
}
pub fn avx2() -> Rules<CpuPlan> {
    Rules::new(
        vec![],
        CpuPlan {
            projection_rows: 6,
            int8_projection_rows: None,
            attention_scores: 16384,
        },
    )
    .unwrap()
}
pub fn wasm() -> Rules<CpuPlan> {
    Rules::new(
        vec![],
        CpuPlan {
            projection_rows: if cfg!(feature = "decima-wasm-fma") {
                4
            } else {
                3
            },
            int8_projection_rows: Some(4),
            attention_scores: 4096,
        },
    )
    .unwrap()
}
pub(super) fn default_plan(kernel: super::Kernel) -> CpuPlan {
    match kernel {
        super::Kernel::WasmSimd => CpuPlan {
            projection_rows: if cfg!(feature = "decima-wasm-fma") {
                4
            } else {
                3
            },
            int8_projection_rows: Some(4),
            attention_scores: 4096,
        },
        super::Kernel::Avx2 => CpuPlan {
            projection_rows: 6,
            int8_projection_rows: None,
            attention_scores: 16384,
        },
        _ => CpuPlan {
            projection_rows: 32,
            int8_projection_rows: None,
            attention_scores: 16384,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_int8_budget_is_optional_and_bounded() -> Result<()> {
        let p: CpuPlan = serde_json::from_str(r#"{"projection_rows":3,"attention_scores":4096}"#)?;
        assert!(p.int8_projection_rows.is_none());
        for n in [0, 5, usize::MAX] {
            assert!(
                CpuPlan {
                    int8_projection_rows: Some(n),
                    ..p
                }
                .validate()
                .is_err()
            );
        }
        CpuPlan {
            int8_projection_rows: Some(4),
            ..p
        }
        .validate()
    }
}
