use super::Matrix;
use rayon::prelude::*;
use std::arch::x86_64::*;

pub(super) fn apply(
    w: &Matrix,
    x: &[f32],
    out: &mut [f32],
    bias: Option<&[f32]>,
    rows: usize,
    work: &mut Vec<f32>,
) {
    let m = x.len() / w.cols;
    let block_rows = if m <= 128 {
        m.div_ceil(rayon::current_num_threads())
            .div_ceil(rows)
            .max(1)
            * rows
    } else {
        (m.div_ceil(rayon::current_num_threads() * 2).min(32) / rows).max(1) * rows
    };
    let scratch = w.cols * 16;
    work.resize(m.div_ceil(block_rows) * scratch, 0.);
    out.par_chunks_mut(block_rows * w.rows)
        .zip(work.par_chunks_mut(scratch))
        .enumerate()
        .for_each(|(block, (out, work))| {
            let first = block * block_rows;
            for panel in 0..w.rows.div_ceil(16) {
                unsafe {
                    decode(w, panel, work);
                }
                for (row, dst) in out.chunks_mut(rows * w.rows).enumerate() {
                    let first = first + row * rows;
                    unsafe {
                        match dst.len() / w.rows {
                            1 => kernel::<1>(x, work, dst, w.cols, w.rows, first, panel, bias),
                            2 => kernel::<2>(x, work, dst, w.cols, w.rows, first, panel, bias),
                            3 => kernel::<3>(x, work, dst, w.cols, w.rows, first, panel, bias),
                            4 => kernel::<4>(x, work, dst, w.cols, w.rows, first, panel, bias),
                            _ => unreachable!(),
                        }
                    }
                }
            }
        });
}

#[target_feature(enable = "avx2,fma")]
unsafe fn decode(w: &Matrix, panel: usize, out: &mut [f32]) {
    unsafe {
        let groups = w.groups();
        for half in 0..2 {
            let source_panel = panel * 2 + half;
            if source_panel >= w.rows.div_ceil(8) {
                for j in 0..w.cols {
                    out[j * 16 + half * 8..j * 16 + half * 8 + 8].fill(0.);
                }
                continue;
            }
            for g in 0..groups {
                let source = w
                    .data
                    .as_ptr()
                    .add((source_panel * groups + g) * w.block * 8);
                let scale = _mm256_loadu_ps(w.scales.as_ptr().add((source_panel * groups + g) * 8));
                for j in 0..w.block.min(w.cols - g * w.block) {
                    let q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        source.add(j * 8).cast(),
                    )));
                    _mm256_storeu_ps(
                        out.as_mut_ptr().add((g * w.block + j) * 16 + half * 8),
                        _mm256_mul_ps(q, scale),
                    );
                }
            }
        }
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel<const R: usize>(
    x: &[f32],
    weights: &[f32],
    out: &mut [f32],
    k: usize,
    n: usize,
    first: usize,
    panel: usize,
    bias: Option<&[f32]>,
) {
    unsafe {
        let mut sum = [[_mm256_setzero_ps(); 2]; R];
        for j in 0..k {
            let b = weights.as_ptr().add(j * 16);
            let weights = [_mm256_loadu_ps(b), _mm256_loadu_ps(b.add(8))];
            for r in 0..R {
                let a = _mm256_set1_ps(*x.get_unchecked((first + r) * k + j));
                for p in 0..2 {
                    sum[r][p] = _mm256_fmadd_ps(a, weights[p], sum[r][p]);
                }
            }
        }
        for r in 0..R {
            let mut values = [0.; 16];
            _mm256_storeu_ps(values.as_mut_ptr(), sum[r][0]);
            _mm256_storeu_ps(values.as_mut_ptr().add(8), sum[r][1]);
            for c in 0..16.min(n - panel * 16) {
                let col = panel * 16 + c;
                out[r * n + col] = values[c] + bias.map_or(0., |b| b[col]);
            }
        }
    }
}
