use anyhow::{Result, ensure};
use rayon::prelude::*;

const GROUP: usize = 64;

#[derive(Default)]
pub(super) struct Scratch {
    values: Vec<i8>,
    scales: Vec<f32>,
}

pub(super) struct Matrix {
    input: usize,
    output: usize,
    data: Vec<i8>,
    scales: Vec<f32>,
    avx2: bool,
}

impl Matrix {
    pub fn from_bf16(bytes: &[u8], input: usize, output: usize, avx2: bool) -> Result<Self> {
        ensure!(
            !avx2 || super::cpu::avx2_available(),
            "INT8 AVX2/FMA unavailable"
        );
        ensure!(
            input > 0
                && output > 0
                && input.checked_mul(output).and_then(|n| n.checked_mul(2)) == Some(bytes.len()),
            "invalid INT8 matrix shape"
        );
        let groups = input.div_ceil(GROUP);
        let mut data = vec![0; output.div_ceil(8) * groups * GROUP * 8];
        let mut scales = vec![0.; output.div_ceil(8) * groups * 8];
        data.par_chunks_mut(groups * GROUP * 8)
            .zip(scales.par_chunks_mut(groups * 8))
            .enumerate()
            .try_for_each(|(panel, (data, scales))| -> Result<()> {
                for c in 0..8.min(output - panel * 8) {
                    for g in 0..groups {
                        let count = GROUP.min(input - g * GROUP);
                        let mut values = [0.; GROUP];
                        for (k, v) in values[..count].iter_mut().enumerate() {
                            let at = ((panel * 8 + c) * input + g * GROUP + k) * 2;
                            *v = f32::from_bits(
                                (u16::from_le_bytes([bytes[at], bytes[at + 1]]) as u32) << 16,
                            );
                        }
                        ensure!(
                            values.iter().all(|v| v.is_finite()),
                            "non-finite INT8 weight"
                        );
                        let s = group_scale(&values, avx2);
                        scales[g * 8 + c] = s;
                        let mut codes = [0; GROUP];
                        pack(&values, &mut codes, s, avx2);
                        for k in (0..GROUP).step_by(4) {
                            let at = g * GROUP * 8 + k / 4 * 32 + c * 4;
                            data[at..at + 4].copy_from_slice(&codes[k..k + 4]);
                        }
                    }
                }
                Ok(())
            })?;
        Ok(Self {
            input,
            output,
            data,
            scales,
            avx2,
        })
    }

    pub fn bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    pub fn apply(&self, x: &[f32], y: &mut Vec<f32>, scratch: &mut Scratch) {
        assert_eq!(x.len() % self.input, 0);
        let tokens = x.len() / self.input;
        let groups = self.input.div_ceil(GROUP);
        y.resize(tokens * self.output, 0.);
        if !x.iter().all(|v| v.is_finite()) {
            y.fill(f32::NAN);
            return;
        }
        scratch.values.resize(tokens * groups * GROUP, 0);
        scratch.scales.resize(tokens * groups, 0.);
        scratch
            .values
            .par_chunks_mut(groups * GROUP)
            .zip(scratch.scales.par_chunks_mut(groups))
            .zip(x.par_chunks(self.input))
            .for_each(|((out, scales), row)| {
                for (g, values) in row.chunks(GROUP).enumerate() {
                    let s = group_scale(values, self.avx2);
                    scales[g] = s;
                    let out = &mut out[g * GROUP..(g + 1) * GROUP];
                    out.fill(0);
                    pack(values, out, s, self.avx2);
                }
            });
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            let dst = Columns(y.as_mut_ptr());
            (0..self.output.div_ceil(32))
                .into_par_iter()
                .for_each(|tile| {
                    let first = tile * 4;
                    unsafe {
                        project::<4>(
                            self,
                            scratch,
                            dst.pointer(),
                            tokens,
                            first,
                            (first + 4).min(self.output.div_ceil(8)),
                        );
                    }
                });
            return;
        }
        let _ = self.avx2;
        y.par_chunks_mut(self.output)
            .enumerate()
            .for_each(|(r, out)| {
                for (c, out) in out.iter_mut().enumerate() {
                    let mut sum = 0.;
                    for g in 0..groups {
                        let mut dot = 0_i32;
                        for k in 0..GROUP {
                            dot += scratch.values[(r * groups + g) * GROUP + k] as i32
                                * self.data[(c / 8 * groups + g) * GROUP * 8
                                    + k / 4 * 32
                                    + c % 8 * 4
                                    + k % 4] as i32;
                        }
                        sum += dot as f32
                            * (scratch.scales[r * groups + g]
                                * self.scales[(c / 8 * groups + g) * 8 + c % 8]);
                    }
                    *out = sum;
                }
            });
    }
}

fn scale(x: &[f32]) -> f32 {
    (x.iter().fold(0_f32, |a, v| a.max(v.abs())) / 127.).max(f32::MIN_POSITIVE)
}
fn quantize(x: f32, scale: f32) -> i8 {
    (x * scale.recip()).round_ties_even().clamp(-127., 127.) as i8
}

fn group_scale(x: &[f32], avx2: bool) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        return unsafe { scale_avx2(x) };
    }
    let _ = avx2;
    scale(x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let end = x.len() / 8 * 8;
        let mut maximum = _mm256_setzero_ps();
        for k in (0..end).step_by(8) {
            maximum = _mm256_max_ps(
                maximum,
                _mm256_and_ps(_mm256_loadu_ps(x.as_ptr().add(k)), mask),
            );
        }
        let half = _mm_max_ps(
            _mm256_castps256_ps128(maximum),
            _mm256_extractf128_ps::<1>(maximum),
        );
        let pairs = _mm_max_ps(half, _mm_movehl_ps(half, half));
        let mut maximum = _mm_cvtss_f32(_mm_max_ss(pairs, _mm_shuffle_ps::<0x55>(pairs, pairs)));
        for &v in &x[end..] {
            maximum = maximum.max(v.abs());
        }
        (maximum / 127.).max(f32::MIN_POSITIVE)
    }
}

fn pack(x: &[f32], out: &mut [i8], scale: f32, avx2: bool) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        unsafe {
            pack_avx2(x, out, scale);
        }
        return;
    }
    let _ = avx2;
    for (out, &v) in out.iter_mut().zip(x) {
        *out = quantize(v, scale);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_avx2(x: &[f32], out: &mut [i8], scale: f32) {
    use std::arch::x86_64::*;
    unsafe {
        let inverse = _mm256_set1_ps(scale.recip());
        let end = x.len() / 8 * 8;
        for k in (0..end).step_by(8) {
            let value = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(k)), inverse);
            let value = _mm256_min_ps(
                _mm256_max_ps(value, _mm256_set1_ps(-127.)),
                _mm256_set1_ps(127.),
            );
            let value = _mm256_cvttps_epi32(_mm256_round_ps::<
                { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
            >(value));
            let half = _mm_packs_epi32(
                _mm256_castsi256_si128(value),
                _mm256_extracti128_si256::<1>(value),
            );
            let bytes = _mm_packs_epi16(half, _mm_setzero_si128());
            _mm_storel_epi64(out.as_mut_ptr().add(k).cast(), bytes);
        }
        for k in end..x.len() {
            out[k] = quantize(x[k], scale);
        }
    }
}

// Each job owns disjoint output columns and joins before the mutable borrow ends.
#[cfg(target_arch = "x86_64")]
struct Columns(*mut f32);
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for Columns {}
#[cfg(target_arch = "x86_64")]
impl Columns {
    fn pointer(&self) -> *mut f32 {
        self.0
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn project<const R: usize>(
    w: &Matrix,
    x: &Scratch,
    out: *mut f32,
    tokens: usize,
    start: usize,
    end: usize,
) {
    unsafe {
        for panel in start..end {
            let full = tokens / R * R;
            for r in (0..full).step_by(R) {
                rows::<R>(w, x, out, r, panel);
            }
            match tokens - full {
                0 => {}
                1 => rows::<1>(w, x, out, full, panel),
                2 => rows::<2>(w, x, out, full, panel),
                3 => rows::<3>(w, x, out, full, panel),
                4 => rows::<4>(w, x, out, full, panel),
                5 => rows::<5>(w, x, out, full, panel),
                6 => rows::<6>(w, x, out, full, panel),
                7 => rows::<7>(w, x, out, full, panel),
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows<const R: usize>(w: &Matrix, x: &Scratch, out: *mut f32, first: usize, panel: usize) {
    use std::arch::x86_64::*;
    unsafe {
        let groups = w.input.div_ceil(GROUP);
        let one = _mm256_set1_epi16(1);
        let mut sums = [_mm256_setzero_ps(); R];
        for g in 0..groups {
            let mut acc = [_mm256_setzero_si256(); R];
            let base = (panel * groups + g) * GROUP * 8;
            for k in (0..GROUP).step_by(4) {
                let b = _mm256_loadu_si256(w.data.as_ptr().add(base + k * 8).cast());
                let magnitude = _mm256_abs_epi8(b);
                for r in 0..R {
                    let a = _mm256_set1_epi32(
                        x.values
                            .as_ptr()
                            .add(((first + r) * groups + g) * GROUP + k)
                            .cast::<i32>()
                            .read_unaligned(),
                    );
                    // Both codes are in [-127,127]: each pair fits i16, including extrema.
                    let pair = _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(a, b));
                    acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(pair, one));
                }
            }
            let sw = _mm256_loadu_ps(w.scales.as_ptr().add((panel * groups + g) * 8));
            for r in 0..R {
                let s = _mm256_mul_ps(sw, _mm256_set1_ps(x.scales[(first + r) * groups + g]));
                sums[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[r]), s, sums[r]);
            }
        }
        for r in 0..R {
            let dst = out.add((first + r) * w.output + panel * 8);
            if panel * 8 + 8 <= w.output {
                _mm256_storeu_ps(dst, sums[r]);
            } else {
                let mut tail = [0.; 8];
                _mm256_storeu_ps(tail.as_mut_ptr(), sums[r]);
                std::ptr::copy_nonoverlapping(tail.as_ptr(), dst, w.output - panel * 8);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn activation_rounding_matches_portable_at_boundaries() {
        if !super::super::cpu::avx2_available() {
            return;
        }
        let mut values = vec![0., -0., f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
        for i in -127..127 {
            let half = i as f32 + 0.5;
            values.extend([half.next_down(), half, half.next_up()]);
        }
        for s in [1., 0.0037, f32::MIN_POSITIVE, 1e30] {
            let scaled: Vec<_> = values.iter().map(|v| v * s).collect();
            for n in [1, 7, 8, 9, 63, 64, 65, scaled.len()] {
                let mut a = vec![0; n];
                let mut b = vec![0; n];
                assert_eq!(
                    group_scale(&scaled[..n], true),
                    group_scale(&scaled[..n], false)
                );
                pack(&scaled[..n], &mut a, s, false);
                pack(&scaled[..n], &mut b, s, true);
                assert_eq!(a, b, "scale={s}, n={n}");
                assert!(a.iter().all(|&v| v != -128));
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "manual AVX2 projection benchmark; excludes model quality and weight load time"]
    fn projection_benchmark() -> Result<()> {
        use super::super::{
            cpu::{Linear, ProjectionScratch},
            weights::Weights,
        };
        use safetensors::{Dtype, tensor::TensorView};
        use std::{hint::black_box, time::Instant};
        if !super::super::cpu::avx2_available() {
            return Ok(());
        }
        let m = std::env::var("DECISIVE_PPLX_BENCH_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(117);
        for (k, n) in [(2560, 9216), (9216, 2560)] {
            let bytes: Vec<_> = (0..k * n)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * 17 % 113) as f32 - 56.) / 256.)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect();
            let w = Matrix::from_bf16(&bytes, k, n, true)?;
            let weights = Weights(std::collections::HashMap::from([(
                "p.weight".into(),
                TensorView::new(Dtype::BF16, vec![n, k], &bytes)?,
            )]));
            let bf16 = Linear::load(&weights, "p", k, n, true)?;
            let x: Vec<_> = (0..m * k)
                .map(|i| ((i * 23 % 127) as f32 - 63.) / 64.)
                .collect();
            for threads in [1, 3] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()?;
                pool.install(|| {
                    let mut scratch = Scratch::default(); let mut y = Vec::new();
                    w.apply(&x, &mut y, &mut scratch);
                    let expected=y.clone();
                    for tile_rows in [4,5,6,8] {
                        let mut times = Vec::new();
                        for _ in 0..5 {
                            let dst = Columns(y.as_mut_ptr());
                            let start = Instant::now();
                            (0..n.div_ceil(32)).into_par_iter().for_each(|tile| unsafe {
                                let first=tile*4; let end=(first+4).min(n.div_ceil(8));
                                match tile_rows {
                                    2=>project::<2>(&w,&scratch,dst.pointer(),m,first,end),
                                    3=>project::<3>(&w,&scratch,dst.pointer(),m,first,end),
                                    4=>project::<4>(&w,&scratch,dst.pointer(),m,first,end),
                                    5=>project::<5>(&w,&scratch,dst.pointer(),m,first,end),
                                    6=>project::<6>(&w,&scratch,dst.pointer(),m,first,end),
                                    8=>project::<8>(&w,&scratch,dst.pointer(),m,first,end),
                                    _=>unreachable!(),
                                }
                            });
                            times.push(start.elapsed().as_secs_f64()*1000.);
                            assert_eq!(y,expected,"projection tile variant {tile_rows}");
                            black_box(&y);
                        }
                        println!("{}", serde_json::json!({"kind":"int8-kernel", "m":m,"k":k,"n":n,"threads":threads,"rows":tile_rows,"ms":times}));
                    }
                    let mut scratch = ProjectionScratch::default();
                    let mut times = Vec::new();
                    for _ in 0..5 {
                        let start=Instant::now();bf16.apply(&x,&mut y,6,&mut scratch);
                        times.push(start.elapsed().as_secs_f64()*1000.);black_box(&y);
                    }
                    println!("{}", serde_json::json!({"kind":"bf16-projection", "m":m,"k":k,"n":n,"threads":threads,"ms":times}));
                });
            }
        }
        Ok(())
    }
    #[test]
    fn integer_extrema_tails_and_threads_match_scalar() -> Result<()> {
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?;
            for (m, k, n) in [
                (0, 1, 1),
                (1, 7, 1),
                (3, 64, 8),
                (4, 65, 9),
                (7, 127, 33),
                (13, 256, 65),
                (17, 1025, 9),
            ] {
                let weights: Vec<_> = (0..k * n)
                    .map(|i| match i % 5 {
                        0 => -127.,
                        1 => 127.,
                        2 => 0.,
                        _ => (i % 31) as f32 / 8.,
                    })
                    .collect();
                let bytes: Vec<_> = weights
                    .iter()
                    .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                    .collect();
                let x: Vec<_> = (0..m * k)
                    .map(|i| match i % 7 {
                        0 => 127.,
                        1 => -127.,
                        2 => 0.,
                        _ => (i % 13) as f32 - 6.,
                    })
                    .collect();
                let mut w = Matrix::from_bf16(&bytes, k, n, false)?;
                let mut a = Vec::new();
                pool.install(|| w.apply(&x, &mut a, &mut Scratch::default()));
                for r in 0..m {
                    for c in 0..n {
                        let expected: f64 = (0..k.div_ceil(GROUP))
                            .map(|g| {
                                let end = (g * GROUP + GROUP).min(k);
                                let xv = &x[r * k + g * GROUP..r * k + end];
                                let wv = &weights[c * k + g * GROUP..c * k + end];
                                let sx = scale(xv);
                                let sw = scale(wv);
                                xv.iter()
                                    .zip(wv)
                                    .map(|(&x, &w)| quantize(x, sx) as f64 * quantize(w, sw) as f64)
                                    .sum::<f64>()
                                    * (sx * sw) as f64
                            })
                            .sum();
                        assert!(
                            (a[r * n + c] as f64 - expected).abs() <= 0.01 + expected.abs() * 1e-6
                        );
                    }
                }
                if super::super::cpu::avx2_available() {
                    w.avx2 = true;
                    let mut b = Vec::new();
                    pool.install(|| w.apply(&x, &mut b, &mut Scratch::default()));
                    for (a, b) in a.iter().zip(b) {
                        assert!(
                            (a - b).abs() <= 0.02 + a.abs() * 1e-6,
                            "{m}x{k}x{n}: {a} != {b}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn invalid_weights_and_nonfinite_activations_are_rejected() -> Result<()> {
        assert!(Matrix::from_bf16(&[], 0, 1, false).is_err());
        assert!(Matrix::from_bf16(&half::bf16::NAN.to_bits().to_le_bytes(), 1, 1, false).is_err());
        let w = Matrix::from_bf16(&[0, 0], 1, 1, false)?;
        let mut y = Vec::new();
        w.apply(&[f32::NAN], &mut y, &mut Scratch::default());
        assert!(y[0].is_nan());
        w.apply(&[0.], &mut y, &mut Scratch::default());
        assert_eq!(y, [0.]);
        Ok(())
    }
}
