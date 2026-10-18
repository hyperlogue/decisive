use super::*;
use serde_json::{Value, json};
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 19) % 251) as f32 - 125.) / 127.)
        .collect()
}
fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}
fn rounded(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect()
}
fn compare(actual: &[f32], expected: &[f32], name: &str) -> Result<Value> {
    ensure!(actual.len() == expected.len(), "{name}: shape mismatch");
    ensure!(
        !actual.is_empty() && actual.iter().chain(expected).all(|v| v.is_finite()),
        "{name}: empty or non-finite values"
    );
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    ensure!(error <= 1e-4, "{name}: max error {error}");
    Ok(json!({"name":name,"values":actual.len(),"max_error":error}))
}
pub async fn run() -> Result<Value> {
    let gpu = Gpu::new().await?;
    let d = &gpu.device;
    let mut checks = Vec::new();
    for (m, k, n) in [(1, 18, 34), (17, 30, 18), (33, 66, 34)] {
        let x = values(m * k, 1);
        let w = rounded(&values(n * k, 2));
        let xb = d.buffer(&x, "check input")?;
        let wb = d.bytes(&bf16(&w), "check weights")?;
        let mut e = d.encoder();
        let out = gpu.linear(&mut e, &xb, &wb, m, k, n)?;
        let actual = d.read(e, &out, m * n).await?;
        let mut expected = vec![0.; m * n];
        for r in 0..m {
            for c in 0..n {
                for i in 0..k {
                    expected[r * n + c] += x[r * k + i] * w[c * k + i];
                }
            }
        }
        checks.push(compare(&actual, &expected, "BF16 GEMM edge tiles")?);
    }
    let n = 3;
    let x = values(n * D, 16);
    let weight = rounded(&values(D, 17));
    let xb = d.buffer(&x, "RMS input")?;
    let wb = d.bytes(&bf16(&weight), "RMS weight")?;
    let mut e = d.encoder();
    let out = gpu.norm(&mut e, &xb, &wb, n)?;
    let actual = d.read(e, &out, n * D).await?;
    let mut expected = x.clone();
    for row in expected.chunks_exact_mut(D) {
        let scale = 1. / (row.iter().map(|v| v * v).sum::<f32>() / D as f32 + 1e-6).sqrt();
        for (v, w) in row.iter_mut().zip(&weight) {
            *v = (*v * scale) * (1. + w);
        }
    }
    checks.push(compare(&actual, &expected, "residual RMS normalization")?);

    let gate = values(n * D, 18);
    let gb = d.buffer(&gate, "MLP gate")?;
    let mut e = d.encoder();
    let out = gpu.point(&mut e, &xb, &gb, n * D, 1)?;
    let actual = d.read(e, &out, n * D).await?;
    let expected: Vec<_> = x
        .iter()
        .zip(&gate)
        .map(|(x, g)| (x / (1. + (-x).exp())) * g)
        .collect();
    checks.push(compare(&actual, &expected, "MLP SiLU product")?);

    let decay = rounded(&values(32, 19));
    let dt = rounded(&values(32, 20));
    let db = d.bytes(&bf16(&decay), "BF16 log decay")?;
    let tb = d.bytes(&bf16(&dt), "BF16 dt bias")?;
    let out = gpu.empty(64)?;
    let mut e = d.encoder();
    gpu.run(
        &mut e,
        "delta_constants",
        [&gpu.dummy, &db, &tb, &out],
        &[],
        32,
    )?;
    let actual = d.read(e, &out, 64).await?;
    let expected: Vec<_> = decay
        .iter()
        .map(|v| v.exp())
        .chain(dt.iter().copied())
        .collect();
    checks.push(compare(&actual, &expected, "BF16 delta constants")?);

    let x = values(n * 4096, 21);
    let gate = values(n * 4096, 22);
    let weight = rounded(&values(128, 23));
    let xb = d.buffer(&x, "delta output")?;
    let gb = d.buffer(&gate, "delta gate")?;
    let wb = d.bytes(&bf16(&weight), "delta RMS weight")?;
    let out = gpu.empty(n * 4096)?;
    let mut e = d.encoder();
    gpu.run(&mut e, "delta_gate", [&xb, &wb, &gb, &out], &[n], n * 32)?;
    let actual = d.read(e, &out, n * 4096).await?;
    let mut expected = x.clone();
    for (h, row) in expected.chunks_exact_mut(128).enumerate() {
        let scale = 1. / (row.iter().map(|v| v * v).sum::<f32>() / 128. + 1e-6).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            let z = gate[h * 128 + i];
            *v = ((*v * scale) * weight[i]) * (z / (1. + (-z).exp()));
        }
    }
    checks.push(compare(&actual, &expected, "delta RMS SiLU gate")?);
    gpu.discard_scratch();
    for n in [1, 3, 17] {
        let raw = values(n * 8192, 3);
        let weight = rounded(&values(8192 * 4, 4));
        let a = values(n * 32, 5);
        let b = values(n * 32, 6);
        let decay: Vec<_> = values(32, 7).iter().map(|v| v.exp()).collect();
        let dt = values(32, 8);
        let mut constants = decay.clone();
        constants.extend(&dt);
        let rawb = d.buffer(&raw, "qkv")?;
        let wb = d.bytes(&bf16(&weight), "conv")?;
        let ab = d.buffer(&a, "a")?;
        let bb = d.buffer(&b, "b")?;
        let cb = d.buffer(&constants, "params")?;
        let conv = gpu.empty(n * 8192)?;
        let qk = gpu.empty(n * 8192)?;
        let gates = gpu.empty(n * 64)?;
        let out = gpu.empty(n * 4096)?;
        let mut e = d.encoder();
        gpu.run(
            &mut e,
            "conv",
            [&rawb, &wb, &gpu.dummy, &conv],
            &[n],
            n * 8192,
        )?;
        gpu.run(
            &mut e,
            "delta_qk",
            [&conv, &gpu.dummy, &gpu.dummy, &qk],
            &[n],
            n * 64,
        )?;
        gpu.run(
            &mut e,
            "delta_params",
            [&ab, &bb, &cb, &gates],
            &[n],
            n * 32,
        )?;
        gpu.run(
            &mut e,
            "delta_scan",
            [&qk, &gates, &gpu.dummy, &out],
            &[n],
            4096,
        )?;
        let actual = d.read(e, &out, n * 4096).await?;
        let mut x = vec![0.; n * 8192];
        for t in 0..n {
            for c in 0..8192 {
                let mut sum = 0.;
                for tap in 0..4 {
                    if t + tap >= 3 {
                        sum += raw[(t + tap - 3) * 8192 + c] * weight[c * 4 + tap];
                    }
                }
                x[t * 8192 + c] = sum / (1. + (-sum).exp());
            }
        }
        let mut expected = vec![0.; n * 4096];
        for h in 0..32 {
            let mut state = vec![0.; 128 * 128];
            for t in 0..n {
                let q0 = t * 8192 + (h / 2) * 128;
                let k0 = q0 + 2048;
                let qi = 1. / (x[q0..q0 + 128].iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
                let ki = 1. / (x[k0..k0 + 128].iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
                let z = a[t * 32 + h] + dt[h];
                let g = (-decay[h] * z.exp().ln_1p()).exp();
                let beta = 1. / (1. + (-b[t * 32 + h]).exp());
                for c in 0..128 {
                    let mut memory = 0.;
                    for i in 0..128 {
                        state[i * 128 + c] *= g;
                        memory += state[i * 128 + c] * (x[k0 + i] * ki);
                    }
                    let delta = (x[t * 8192 + 4096 + h * 128 + c] - memory) * beta;
                    let mut sum = 0.;
                    for i in 0..128 {
                        state[i * 128 + c] += (x[k0 + i] * ki) * delta;
                        sum += state[i * 128 + c] * ((x[q0 + i] * qi) / 128f32.sqrt());
                    }
                    expected[t * 4096 + h * 128 + c] = sum;
                }
            }
        }
        checks.push(compare(
            &actual,
            &expected,
            "causal convolution + delta recurrence",
        )?);
    }
    let positions = [2, 18, 2047];
    let x = values(3 * 16 * 512, 14);
    let weight = rounded(&values(256, 15));
    let xb = d.buffer(&x, "q + gate")?;
    let wb = d.bytes(&bf16(&weight), "q norm")?;
    let rope = rope_table(&positions);
    let rb = d.buffer(&rope, "RoPE")?;
    let out = gpu.empty(3 * 4096)?;
    let mut e = d.encoder();
    gpu.run(
        &mut e,
        "attention_norm",
        [&xb, &wb, &rb, &out],
        &[3, 16, 512],
        48,
    )?;
    let actual = d.read(e, &out, 3 * 4096).await?;
    let mut expected = vec![0.; 3 * 4096];
    for h in 0..48 {
        let scale = 1.
            / (x[h * 512..h * 512 + 256].iter().map(|v| v * v).sum::<f32>() / 256. + 1e-6).sqrt();
        let row = &mut expected[h * 256..(h + 1) * 256];
        for i in 0..256 {
            row[i] = (x[h * 512 + i] * scale) * (1. + weight[i]);
        }
        for i in 0..32 {
            let a = row[i];
            let b = row[i + 32];
            let c = rope[(h / 16) * 64 + i];
            let s = rope[(h / 16) * 64 + 32 + i];
            row[i] = a * c - b * s;
            row[i + 32] = b * c + a * s;
        }
    }
    checks.push(compare(
        &actual,
        &expected,
        "Q RMS + partial RoPE at selected positions",
    )?);
    // A non-contiguous pair of decision rows must still see every key/value row.
    let n = 19;
    let positions = [2, 18];
    let query = values(2 * 4096, 9);
    let key = values(n * 1024, 10);
    let value = values(n * 1024, 11);
    let gate = values(2 * 8192, 12);
    let q = d.buffer(&query, "query")?;
    let k = d.buffer(&key, "key")?;
    let v = d.buffer(&value, "value")?;
    let z = d.buffer(&gate, "gate")?;
    let scores = gpu.empty(2 * 16 * n)?;
    let probs = gpu.empty(2 * 16 * n)?;
    let out = gpu.empty(2 * 4096)?;
    let mut e = d.encoder();
    gpu.run(
        &mut e,
        "scores",
        [&q, &k, &gpu.dummy, &scores],
        &[n, 2, 0],
        2 * 16 * n,
    )?;
    gpu.run(
        &mut e,
        "softmax",
        [&scores, &gpu.dummy, &gpu.dummy, &probs],
        &[32, n],
        32,
    )?;
    gpu.run(
        &mut e,
        "values",
        [&probs, &v, &z, &out],
        &[n, 2, 0],
        2 * 4096,
    )?;
    let actual = d.read(e, &out, 2 * 4096).await?;
    let mut expected = vec![0.; 2 * 4096];
    for r in 0..2 {
        for h in 0..16 {
            let mut score = vec![0.; n];
            for t in 0..n {
                for c in 0..256 {
                    score[t] += query[r * 4096 + h * 256 + c] * key[t * 1024 + (h / 4) * 256 + c];
                }
                score[t] /= 16.;
            }
            let max = score.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.;
            for s in &mut score {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in &mut score {
                *s /= sum;
            }
            for c in 0..256 {
                let mut sum = 0.;
                for t in 0..n {
                    sum += score[t] * value[t * 1024 + (h / 4) * 256 + c];
                }
                expected[r * 4096 + h * 256 + c] =
                    sum / (1. + (-gate[(r * 16 + h) * 512 + 256 + c]).exp());
            }
        }
    }
    checks.push(compare(&actual, &expected, "bidirectional GQA + gate")?);
    let x = values(n * D, 13);
    let xb = d.buffer(&x, "residual")?;
    let mut e = d.encoder();
    let selected = gpu.gather(&mut e, &xb, &positions)?;
    let actual = d.read(e, &selected, 2 * D).await?;
    let expected: Vec<_> = positions
        .iter()
        .flat_map(|&i| x[i * D..(i + 1) * D].iter().copied())
        .collect();
    checks.push(compare(
        &actual,
        &expected,
        "non-contiguous marker residual",
    )?);
    let count = 65535 * 64 + 65;
    let x = vec![0.25; count];
    let xb = d.buffer(&x, "large dispatch")?;
    let mut e = d.encoder();
    let y = gpu.point(&mut e, &xb, &xb, count, 0)?;
    let actual = d.read(e, &y, count).await?;
    checks.push(compare(
        &actual,
        &vec![0.5; count],
        "dispatch beyond one dimension",
    )?);
    Ok(json!({"adapter":gpu.info(),"checks":checks,"bound":1e-4}))
}
