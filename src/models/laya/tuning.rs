use super::benchmark;
use anyhow::{Result, ensure};
use clap::{Args, ValueEnum};
use decisive::laya::{Model, Scratch};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Instant};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Objective {
    P50,
    P95,
    Throughput,
}

#[derive(Args)]
pub struct Tune {
    #[arg(long)]
    input: PathBuf,
    #[arg(help = "Outputs from the pinned SDK for this exact checkpoint and corpus")]
    #[arg(long)]
    expected: PathBuf,
    #[arg(help = "Worker counts to try, bounded by the global --threads budget")]
    #[arg(long, value_delimiter = ',')]
    candidates: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "32,64,128")]
    tiles: Vec<usize>,
    #[arg(long, value_enum, default_value = "p50")]
    objective: Objective,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 6)]
    rounds: usize,
}

fn candidates(max: usize) -> Vec<usize> {
    let mut result = vec![1, max, (max / 2).max(1)];
    let mut n = 2;
    while n < max {
        result.push(n);
        let Some(next) = n.checked_mul(2) else { break };
        n = next;
    }
    result.sort_unstable();
    result.dedup();
    result
}

fn rank(v: &Value, objective: Objective) -> f64 {
    match objective {
        Objective::P50 => v["timing"]["p50_ms"].as_f64().unwrap(),
        Objective::P95 => v["timing"]["p95_ms"].as_f64().unwrap(),
        Objective::Throughput => -v["timing"]["requests_per_second"].as_f64().unwrap(),
    }
}

fn winner(results: &[Value], objective: Objective) -> Option<&Value> {
    results
        .iter()
        .filter(|v| v["accuracy_passed"] == true)
        .min_by(|a, b| rank(a, objective).total_cmp(&rank(b, objective)))
}

struct Candidate {
    pool: usize,
    tile: usize,
    scratch: Scratch,
    times: Vec<f64>,
    outputs: Vec<Value>,
    errors: Vec<Value>,
    max_error: f64,
}

impl Tune {
    pub fn run(
        &self,
        model: &Model,
        max_threads: usize,
        kernel: &str,
        storage: &str,
    ) -> Result<Value> {
        ensure!(
            self.rounds >= 2 && self.warmup > 0,
            "tuning needs at least two measured rounds and one warmup"
        );
        let read = |path: &PathBuf| -> Result<Vec<Value>> {
            std::fs::read_to_string(path)?
                .lines()
                .filter(|s| !s.trim().is_empty())
                .map(|s| Ok(serde_json::from_str(s)?))
                .collect()
        };
        let requests = read(&self.input)?;
        let expected = read(&self.expected)?;
        ensure!(
            !requests.is_empty() && requests.len() == expected.len(),
            "corpus and reference must have equal, nonzero lengths"
        );
        let mut counts = if self.candidates.is_empty() {
            candidates(max_threads)
        } else {
            self.candidates.clone()
        };
        counts.sort_unstable();
        counts.dedup();
        ensure!(
            counts.iter().all(|&t| t > 0 && t <= max_threads),
            "candidate exceeds --threads budget or is zero"
        );
        let mut tiles = self.tiles.clone();
        tiles.sort_unstable();
        tiles.dedup();
        ensure!(!tiles.is_empty(), "empty tile list");
        for &tile in &tiles {
            Scratch::default().set_attention_tile(tile)?;
        }
        let pools = counts
            .iter()
            .map(|&threads| rayon::ThreadPoolBuilder::new().num_threads(threads).build())
            .collect::<Result<Vec<_>, _>>()?;
        let mut cases = Vec::new();
        for pool in 0..pools.len() {
            for &tile in &tiles {
                let mut scratch = Scratch::default();
                scratch.set_attention_tile(tile)?;
                cases.push(Candidate {
                    pool,
                    tile,
                    scratch,
                    times: Vec::new(),
                    outputs: Vec::new(),
                    errors: Vec::new(),
                    max_error: 0.0,
                });
            }
        }
        let mut samples = Vec::new();
        for round in 0..self.warmup + self.rounds {
            let mut order: Vec<_> = (0..cases.len())
                .map(|i| (i + round) % cases.len())
                .collect();
            if round % 2 == 1 {
                order.reverse();
            }
            for ci in order {
                let case = &mut cases[ci];
                for j in 0..requests.len() {
                    let ri = (j + round) % requests.len();
                    let request = requests[ri].clone();
                    let start = Instant::now();
                    let output = pools[case.pool]
                        .install(|| model.predict_json(request, &mut case.scratch))?;
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    match benchmark::compare_outputs(&expected[ri], &output, 1e-4) {
                        Ok(error) => case.max_error = case.max_error.max(error),
                        Err(e) => case.errors.push(json!({"round_including_warmup":round,"request":ri,"error":e.to_string()})),
                    }
                    if round == 0 {
                        case.outputs.push(output);
                    }
                    if round >= self.warmup {
                        case.times.push(ms);
                        samples.push(json!({"candidate":ci,"round":round - self.warmup,"request":ri,"ms":ms}));
                    }
                }
            }
            eprintln!(
                "tuning round {}/{} complete",
                round + 1,
                self.warmup + self.rounds
            );
        }
        let reference_quality = benchmark::quality(&requests, &expected)?;
        let results: Vec<_> = cases.iter().enumerate().map(|(i,c)| -> Result<Value> {
            let quality = benchmark::quality(&requests, &c.outputs)?;
            Ok(json!({"candidate":i,"threads":counts[c.pool],"attention_tile":c.tile,
                "accuracy_passed":c.errors.is_empty() && quality["correct"] == reference_quality["correct"],
                "max_output_error_on_matching_requests":c.max_error,"mismatches":c.errors,
                "quality":quality,"timing":benchmark::summarize(&c.times)}))
        }).collect::<Result<_>>()?;
        let recommendation = winner(&results, self.objective).map(|w| json!({
            "threads":w["threads"], "attention_tile":w["attention_tile"], "kernel":kernel, "storage":storage,
            "flags":["--kernel",kernel,"--storage",storage,"--threads",w["threads"].to_string().as_str(),
                "--attention-tile",w["attention_tile"].to_string().as_str()],
            "candidate":w["candidate"]
        }));
        Ok(
            json!({"hardware":decisive::laya::cpu::hardware::detect(), "kernel":kernel, "storage":storage,
            "objective":format!("{:?}",self.objective).to_lowercase(),"warmup_rounds":self.warmup,"rounds":self.rounds,
            "corpus_requests":requests.len(),"thread_budget":max_threads,"reference_quality":reference_quality,
            "candidates":results,"samples":samples,"recommendation":recommendation,
            "scope":"Warm predict_json plus thread-pool dispatch; includes tokenization. Excludes model loading, validation, HTTP and queue wait. One candidate active at a time; model shared; per-candidate scratch. Recommendations apply only to this corpus, checkpoint, precision and environment. Repeat under representative load before deployment; no settings are persisted or applied automatically."}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn budgets_and_accuracy_gate() {
        assert_eq!(candidates(1), vec![1]);
        assert_eq!(candidates(6), vec![1, 2, 3, 4, 6]);
        assert!(candidates(usize::MAX).iter().all(|&v| v > 0));
        let results = vec![
            json!({"accuracy_passed":false,"timing":{"p50_ms":1.,"p95_ms":1.,"requests_per_second":1000.}}),
            json!({"accuracy_passed":true,"timing":{"p50_ms":10.,"p95_ms":40.,"requests_per_second":40.}}),
            json!({"accuracy_passed":true,"timing":{"p50_ms":20.,"p95_ms":30.,"requests_per_second":45.}}),
        ];
        assert_eq!(winner(&results, Objective::P50), Some(&results[1]));
        assert_eq!(winner(&results, Objective::P95), Some(&results[2]));
        assert_eq!(winner(&results, Objective::Throughput), Some(&results[2]));
        assert!(winner(&results[..1], Objective::P50).is_none());
        for tile in [0, 1, 63, 512] {
            assert!(Scratch::default().set_attention_tile(tile).is_err());
        }
    }
}
