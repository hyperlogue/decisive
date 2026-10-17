#![allow(dead_code)]

#[cfg(all(feature = "bench", feature = "kev", feature = "jev-style"))]
#[path = "../benches/causal_bench.rs"]
mod causal_bench;
#[cfg(all(feature = "bench", feature = "laya"))]
#[path = "../benches/laya_vm.rs"]
mod laya_vm;
#[cfg(all(feature = "cli", feature = "laya"))]
#[path = "../benches/profile_summary.rs"]
mod profile_summary;
#[cfg(all(feature = "cli", feature = "laya"))]
#[path = "../benches/serving_matrix.rs"]
mod serving_matrix;
