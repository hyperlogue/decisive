use crate::laya::weights::{Matrix, Storage, Weights};
use anyhow::Result;
mod attention_scratch;
mod block_attention;
pub use attention_scratch::{AttentionMath, AttentionScratch};
pub(crate) use block_attention::{SoftmaxState, attention_block};
#[cfg(feature = "cpu")]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod activation_wasm;
#[cfg(feature = "cpu")]
pub mod hardware;
#[cfg(feature = "int8")]
pub mod int8;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod softmax_wasm;
#[cfg(feature = "cpu")]
mod tiled_attention;
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
mod vector;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd;
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
mod x86_avx2;

pub fn packed_supported() -> bool {
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    return is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c");
    #[cfg(not(all(feature = "cpu", target_arch = "x86_64")))]
    false
}

#[cfg(feature = "cpu")]
type Sgemm = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f32,
    *const f32,
    i32,
    *const f32,
    i32,
    f32,
    *mut f32,
    i32,
);
#[cfg(feature = "cpu")]
pub struct Blas {
    _library: libloading::Library,
    sgemm: Sgemm,
    threads: i32,
    openmp: Option<OpenMp>,
}
#[cfg(feature = "cpu")]
#[derive(Clone, Copy)]
struct OpenMp {
    get_max: unsafe extern "C" fn() -> i32,
    set_num: unsafe extern "C" fn(i32),
}
#[cfg(feature = "cpu")]
struct OpenMpLimit {
    runtime: OpenMp,
    restore: Option<i32>,
}
#[cfg(feature = "cpu")]
impl OpenMp {
    fn limit(self, threads: i32) -> OpenMpLimit {
        let previous = unsafe { (self.get_max)() };
        let restore = (previous != threads).then_some(previous);
        if restore.is_some() {
            unsafe {
                (self.set_num)(threads);
            }
        }
        OpenMpLimit {
            runtime: self,
            restore,
        }
    }
}
#[cfg(feature = "cpu")]
impl Drop for OpenMpLimit {
    fn drop(&mut self) {
        if let Some(previous) = self.restore {
            unsafe {
                (self.runtime.set_num)(previous);
            }
        }
    }
}
#[cfg(feature = "cpu")]
impl Blas {
    pub fn load(path: &std::path::Path, threads: usize) -> Result<Self> {
        anyhow::ensure!(
            threads > 0 && threads <= i32::MAX as usize,
            "invalid BLAS thread count"
        );
        unsafe {
            let library = libloading::Library::new(path)?;
            let sgemm = *library.get::<Sgemm>(b"cblas_sgemm\0")?;
            if let Ok(set_threads) =
                library.get::<unsafe extern "C" fn(i32)>(b"openblas_set_num_threads\0")
            {
                set_threads(threads as i32);
            }
            let openmp = if library
                .get::<unsafe extern "C" fn() -> i32>(b"openblas_get_parallel\0")
                .is_ok_and(|get| get() == 2)
            {
                Some(OpenMp {
                    get_max: *library.get(b"omp_get_max_threads\0")?,
                    set_num: *library.get(b"omp_set_num_threads\0")?,
                })
            } else {
                None
            };
            Ok(Self {
                _library: library,
                sgemm,
                threads: threads as i32,
                openmp,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Portable,
    X86Avx2,
    WasmSimd,
}
impl Kernel {
    pub fn detect() -> Self {
        if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
            Self::WasmSimd
        } else if packed_supported() {
            Self::X86Avx2
        } else {
            Self::Portable
        }
    }
    pub fn validate(self) -> Result<()> {
        anyhow::ensure!(
            self != Self::WasmSimd || cfg!(all(target_arch = "wasm32", target_feature = "simd128")),
            "Wasm SIMD128 requires a simd128 build"
        );
        anyhow::ensure!(
            self != Self::X86Avx2 || packed_supported(),
            "AVX2 kernel requires native x86-64 AVX2/FMA/F16C"
        );
        Ok(())
    }
}
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectionStrategy {
    #[default]
    Default,
    TiledTails,
    InlineTiles,
    PackedInput,
    PackedCache {
        rows: usize,
        reduction: usize,
    },
    LongRows(usize),
    ReductionBlocks(usize),
    WasmRows(usize),
}
impl ProjectionStrategy {
    fn for_shape(self, rows: usize, reduction: usize) -> Self {
        match self {
            Self::LongRows(tile)
                if rows >= 1024 && reduction >= 2048 && rows > (3 * 1024 * 1024) / reduction =>
            {
                Self::PackedCache {
                    rows: tile,
                    reduction: usize::MAX,
                }
            }
            Self::LongRows(_) => Self::PackedInput,
            other => other,
        }
    }
}
pub struct Linear {
    pub input: usize,
    pub output: usize,
    pub weight: Matrix,
    pub bias: Option<Vec<f32>>,
    #[cfg(feature = "cpu")]
    blas: Option<std::sync::Arc<Blas>>,
}
impl Linear {
    pub fn from_parts(input: usize, output: usize, weight: Matrix, bias: Option<Vec<f32>>) -> Self {
        Self {
            input,
            output,
            weight,
            bias,
            #[cfg(feature = "cpu")]
            blas: None,
        }
    }
    #[cfg(feature = "cpu")]
    pub fn set_blas(&mut self, blas: Option<std::sync::Arc<Blas>>) -> Result<()> {
        anyhow::ensure!(
            blas.is_none() || self.weight.is_unpacked(),
            "BLAS requires unpacked float projection weights"
        );
        self.blas = blas;
        Ok(())
    }
    #[cfg(all(feature = "cpu", feature = "int8"))]
    pub(crate) fn uses_blas(&self) -> bool {
        self.blas.is_some()
    }
    pub fn load(
        w: &Weights<'_>,
        prefix: &str,
        input: usize,
        output: usize,
        storage: Storage,
    ) -> Result<Self> {
        Self::load_with_kernel(w, prefix, input, output, storage, Kernel::Portable)
    }
    pub fn load_with_kernel(
        w: &Weights<'_>,
        prefix: &str,
        input: usize,
        output: usize,
        storage: Storage,
        kernel: Kernel,
    ) -> Result<Self> {
        Self::load_named_with_kernel(
            w,
            &format!("{prefix}.weight"),
            &format!("{prefix}.bias"),
            input,
            output,
            storage,
            kernel,
        )
    }
    pub fn load_named(
        w: &Weights<'_>,
        weight: &str,
        bias: &str,
        input: usize,
        output: usize,
        storage: Storage,
    ) -> Result<Self> {
        Self::load_named_with_kernel(w, weight, bias, input, output, storage, Kernel::Portable)
    }
    pub fn load_named_with_kernel(
        w: &Weights<'_>,
        weight: &str,
        bias: &str,
        input: usize,
        output: usize,
        storage: Storage,
        kernel: Kernel,
    ) -> Result<Self> {
        kernel.validate()?;
        let matrix = w.matrix_layout(weight, output, input, storage, kernel == Kernel::X86Avx2)?;
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let matrix = if kernel == Kernel::WasmSimd {
            match matrix {
                #[cfg(feature = "fp16")]
                Matrix::F16(data) => Matrix::WasmF16 {
                    data: wasm_simd::pack_half(&data, input, output),
                    rows: output,
                    cols: input,
                },
                other => Matrix::WasmF32 {
                    data: wasm_simd::pack(&other.values(), input, output),
                    rows: output,
                    cols: input,
                },
            }
        } else {
            matrix
        };
        Ok(Self::from_parts(
            input,
            output,
            matrix,
            if w.contains(bias) {
                Some(w.vector(bias, output)?)
            } else {
                None
            },
        ))
    }
    #[cfg(feature = "int8")]
    pub fn quantize(&mut self, mode: int8::Mode, block: usize, avx2: bool) -> Result<usize> {
        #[cfg(feature = "cpu")]
        anyhow::ensure!(self.blas.is_none(), "disable BLAS before quantizing");
        let weight = int8::Matrix::new(
            &self.weight.values(),
            self.output,
            self.input,
            block,
            mode,
            avx2,
        )?;
        let bytes = weight.bytes();
        self.weight = Matrix::Int8(weight);
        Ok(bytes)
    }
    pub fn apply(&self, x: &[f32], y: &mut Vec<f32>) {
        self.apply_grouped(x, y, 1);
    }
    pub fn apply_gated(&self, x: &[f32], y: &mut Vec<f32>, temporary: &mut Vec<f32>, group: usize) {
        assert_eq!(self.output % 2, 0);
        assert_eq!(x.len() % self.input, 0);
        let width = self.output / 2;
        y.resize(x.len() / self.input * width, 0.0);
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if width.is_multiple_of(16) {
            match &self.weight {
                #[cfg(feature = "fp16")]
                Matrix::PackedF16 { data, .. } => {
                    x86_avx2::gated::<true>(
                        x,
                        bytemuck::cast_slice(data),
                        y,
                        self.input,
                        width,
                        self.bias.as_deref(),
                        group,
                    );
                    return;
                }
                Matrix::PackedF32 { data, .. } => {
                    x86_avx2::gated::<false>(
                        x,
                        bytemuck::cast_slice(data),
                        y,
                        self.input,
                        width,
                        self.bias.as_deref(),
                        group,
                    );
                    return;
                }
                _ => {}
            }
        }
        self.apply_grouped(x, temporary, group);
        gated_gelu(temporary, y, width);
    }
    pub fn apply_with_strategy(
        &self,
        x: &[f32],
        y: &mut Vec<f32>,
        group: usize,
        strategy: ProjectionStrategy,
    ) {
        self.apply_with_workspace(x, y, group, strategy, &mut Vec::new());
    }
    pub fn apply_with_workspace(
        &self,
        x: &[f32],
        y: &mut Vec<f32>,
        group: usize,
        strategy: ProjectionStrategy,
        workspace: &mut Vec<f32>,
    ) {
        #[cfg(feature = "int8")]
        if let Matrix::Int8(weight) = &self.weight {
            let rows = if let ProjectionStrategy::WasmRows(rows) = strategy {
                rows.min(4)
            } else {
                4
            };
            weight.apply_with_rows(x, y, self.bias.as_deref(), workspace, rows);
            return;
        }
        let strategy = strategy.for_shape(x.len() / self.input, self.input);
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        #[cfg(feature = "fp16")]
        if let Matrix::WasmF16 { data, .. } = &self.weight {
            y.resize(x.len() / self.input * self.output, 0.);
            let rows = if let ProjectionStrategy::WasmRows(rows) = strategy {
                rows
            } else {
                4
            };
            wasm_simd::multiply_half(
                x,
                data,
                y,
                self.input,
                self.output,
                self.bias.as_deref(),
                rows,
                workspace,
            );
            return;
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if let Matrix::WasmF32 { data, .. } = &self.weight {
            y.resize(x.len() / self.input * self.output, 0.);
            let rows = if let ProjectionStrategy::WasmRows(rows) = strategy {
                rows
            } else {
                4
            };
            wasm_simd::multiply(
                x,
                data,
                y,
                self.input,
                self.output,
                self.bias.as_deref(),
                rows,
            );
            return;
        }
        #[cfg(not(all(feature = "cpu", target_arch = "x86_64")))]
        let _ = workspace;
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if matches!(
            strategy,
            ProjectionStrategy::PackedInput | ProjectionStrategy::PackedCache { .. }
        ) && self.output.is_multiple_of(16)
            && self.weight.is_packed()
        {
            assert!(group > 0 && self.input > 0 && x.len() % self.input == 0);
            let m = x.len() / self.input;
            let packed_len = x86_avx2::prepare_input(x, self.input, workspace, None);
            y.resize(m * self.output, 0.);
            self.apply_packed_input(&workspace[..packed_len], y, group, strategy);
            return;
        }
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if matches!(
            strategy,
            ProjectionStrategy::InlineTiles
                | ProjectionStrategy::PackedInput
                | ProjectionStrategy::PackedCache { .. }
        ) {
            assert!(group > 0 && self.input > 0 && x.len() % self.input == 0);
            y.resize(x.len() / self.input * self.output, 0.);
            macro_rules! run {
                ($half:literal, $data:expr) => {
                    x86_avx2::inline_tiles::<$half, false>(
                        x,
                        bytemuck::cast_slice($data),
                        y,
                        self.input,
                        self.output,
                        self.bias.as_deref(),
                        group,
                    );
                };
            }
            match &self.weight {
                #[cfg(feature = "fp16")]
                Matrix::PackedF16 { data, .. } => {
                    run!(true, data);
                    return;
                }
                Matrix::PackedF32 { data, .. } => {
                    run!(false, data);
                    return;
                }
                _ => {}
            }
        }
        if let ProjectionStrategy::ReductionBlocks(k) = strategy {
            return self.apply_blocked(x, y, group, k);
        }
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if strategy == ProjectionStrategy::TiledTails {
            assert!(group > 0 && self.input > 0 && x.len() % self.input == 0);
            y.resize(x.len() / self.input * self.output, 0.);
            match &self.weight {
                #[cfg(feature = "fp16")]
                Matrix::PackedF16 { data, .. } => {
                    x86_avx2::tiled_tails::<true>(
                        x,
                        bytemuck::cast_slice(data),
                        y,
                        self.input,
                        self.output,
                        self.bias.as_deref(),
                        group,
                    );
                    return;
                }
                Matrix::PackedF32 { data, .. } => {
                    x86_avx2::tiled_tails::<false>(
                        x,
                        bytemuck::cast_slice(data),
                        y,
                        self.input,
                        self.output,
                        self.bias.as_deref(),
                        group,
                    );
                    return;
                }
                _ => {}
            }
        }
        self.apply_grouped(x, y, group);
    }
    pub fn apply_normalized(
        &self,
        x: &[f32],
        norm: &Norm,
        y: &mut Vec<f32>,
        temporary: &mut Vec<f32>,
        workspace: &mut Vec<f32>,
        group: usize,
        strategy: ProjectionStrategy,
    ) {
        assert_eq!(norm.weight.len(), self.input);
        let strategy = strategy.for_shape(x.len() / self.input, self.input);
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if matches!(
            strategy,
            ProjectionStrategy::PackedInput | ProjectionStrategy::PackedCache { .. }
        ) && self.output.is_multiple_of(16)
            && self.weight.is_packed()
        {
            assert!(self.input > 0 && x.len() % self.input == 0);
            let packed_len = norm.pack_six_rows(x, workspace);
            y.resize(x.len() / self.input * self.output, 0.);
            self.apply_packed_input(&workspace[..packed_len], y, group, strategy);
            return;
        }
        norm.apply(x, temporary);
        self.apply_with_workspace(temporary, y, group, strategy, workspace);
    }
    #[allow(clippy::too_many_arguments)]
    pub fn apply_activated(
        &self,
        x: &[f32],
        gated: bool,
        y: &mut Vec<f32>,
        temporary: &mut Vec<f32>,
        workspace: &mut Vec<f32>,
        group: usize,
        strategy: ProjectionStrategy,
    ) {
        let inputs = if gated { 2 } else { 1 };
        assert!(self.input > 0 && x.len() % (inputs * self.input) == 0);
        let rows = x.len() / (inputs * self.input);
        temporary.resize(rows * self.input, 0.);
        let strategy = strategy.for_shape(rows, self.input);
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if matches!(
            strategy,
            ProjectionStrategy::PackedInput | ProjectionStrategy::PackedCache { .. }
        ) && self.output.is_multiple_of(16)
            && self.weight.is_packed()
        {
            let len = x86_avx2::activate_pack(x, temporary, workspace, self.input, gated);
            y.resize(rows * self.output, 0.);
            self.apply_packed_input(&workspace[..len], y, group, strategy);
            return;
        }
        if gated {
            gated_gelu_simd(x, temporary, self.input);
        } else {
            for (out, &value) in temporary.iter_mut().zip(x) {
                *out = value.max(0.);
            }
        }
        self.apply_with_workspace(temporary, y, group, strategy, workspace);
    }
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    fn apply_packed_input(
        &self,
        packed: &[f32],
        y: &mut [f32],
        group: usize,
        strategy: ProjectionStrategy,
    ) {
        assert!(group > 0);
        macro_rules! run {
            ($half:literal, $data:expr) => {
                if let ProjectionStrategy::PackedCache { rows, reduction } = strategy {
                    x86_avx2::cache_tiles::<$half>(
                        packed,
                        bytemuck::cast_slice($data),
                        y,
                        self.input,
                        self.output,
                        self.bias.as_deref(),
                        rows,
                        reduction,
                    );
                } else {
                    x86_avx2::inline_tiles::<$half, true>(
                        packed,
                        bytemuck::cast_slice($data),
                        y,
                        self.input,
                        self.output,
                        self.bias.as_deref(),
                        group,
                    );
                }
            };
        }
        match &self.weight {
            #[cfg(feature = "fp16")]
            Matrix::PackedF16 { data, .. } => run!(true, data),
            Matrix::PackedF32 { data, .. } => run!(false, data),
            _ => unreachable!(),
        }
    }
    pub fn apply_blocked(&self, x: &[f32], y: &mut Vec<f32>, group: usize, block: usize) {
        assert!(group > 0);
        assert_eq!(x.len() % self.input, 0);
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if block > 0 {
            let half = match &self.weight {
                #[cfg(feature = "fp16")]
                Matrix::PackedF16 { .. } => Some(true),
                Matrix::PackedF32 { .. } => Some(false),
                _ => None,
            };
            if let Some(half) = half {
                y.resize(x.len() / self.input * self.output, 0.0);
                match (&self.weight, half) {
                    #[cfg(feature = "fp16")]
                    (Matrix::PackedF16 { data, .. }, true) => x86_avx2::blocked::<true>(
                        x,
                        bytemuck::cast_slice(data),
                        y,
                        self.input,
                        self.output,
                        self.bias.as_deref(),
                        group,
                        block,
                    ),
                    (Matrix::PackedF32 { data, .. }, false) => x86_avx2::blocked::<false>(
                        x,
                        bytemuck::cast_slice(data),
                        y,
                        self.input,
                        self.output,
                        self.bias.as_deref(),
                        group,
                        block,
                    ),
                    _ => unreachable!(),
                }
                return;
            }
        }
        #[cfg(not(all(feature = "cpu", target_arch = "x86_64")))]
        let _ = block;
        self.apply_grouped(x, y, group);
    }
    pub fn apply_grouped(&self, x: &[f32], y: &mut Vec<f32>, group: usize) {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        #[cfg(feature = "fp16")]
        if let Matrix::WasmF16 { data, .. } = &self.weight {
            y.resize(x.len() / self.input * self.output, 0.);
            wasm_simd::multiply_half(
                x,
                data,
                y,
                self.input,
                self.output,
                self.bias.as_deref(),
                4,
                &mut Vec::new(),
            );
            return;
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if let Matrix::WasmF32 { data, .. } = &self.weight {
            y.resize(x.len() / self.input * self.output, 0.);
            wasm_simd::multiply(x, data, y, self.input, self.output, self.bias.as_deref(), 4);
            return;
        }
        #[cfg(feature = "int8")]
        if let Matrix::Int8(weight) = &self.weight {
            weight.apply(x, y, self.bias.as_deref(), &mut Vec::new());
            return;
        }
        assert!(group > 0);
        assert_eq!(x.len() % self.input, 0);
        let rows = x.len() / self.input;
        y.resize(rows * self.output, 0.0);
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        if let Matrix::PackedF32 { data, .. } = &self.weight {
            x86_avx2::multiply(
                x,
                data,
                y,
                rows,
                self.input,
                self.output,
                self.bias.as_deref(),
                group,
            );
            return;
        }
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        #[cfg(feature = "fp16")]
        if let Matrix::PackedF16 { data, .. } = &self.weight {
            x86_avx2::multiply_half(
                x,
                data,
                y,
                rows,
                self.input,
                self.output,
                self.bias.as_deref(),
                group,
            );
            return;
        }
        let w = self.weight.values();
        #[cfg(feature = "cpu")]
        if let Some(blas) = &self.blas {
            assert!(
                rows <= i32::MAX as usize
                    && self.input <= i32::MAX as usize
                    && self.output <= i32::MAX as usize
            );
            let _thread_limit = blas.openmp.map(|runtime| runtime.limit(blas.threads));
            unsafe {
                (blas.sgemm)(
                    101,
                    111,
                    112,
                    rows as i32,
                    self.output as i32,
                    self.input as i32,
                    1.0,
                    x.as_ptr(),
                    self.input as i32,
                    w.as_ptr(),
                    self.input as i32,
                    0.0,
                    y.as_mut_ptr(),
                    self.output as i32,
                );
            }
            if let Some(bias) = &self.bias {
                for row in y.chunks_exact_mut(self.output) {
                    for (a, b) in row.iter_mut().zip(bias) {
                        *a += b;
                    }
                }
            }
            return;
        }
        let run = |start: usize, dst: &mut [f32]| {
            let m = dst.len() / self.output;
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    self.input,
                    self.output,
                    1.0,
                    x.as_ptr().add(start * self.input),
                    self.input as isize,
                    1,
                    w.as_ptr(),
                    1,
                    self.input as isize,
                    0.0,
                    dst.as_mut_ptr(),
                    self.output as isize,
                    1,
                );
            }
            if let Some(bias) = &self.bias {
                for row in dst.chunks_exact_mut(self.output) {
                    for (a, b) in row.iter_mut().zip(bias) {
                        *a += b;
                    }
                }
            }
        };
        #[cfg(feature = "cpu")]
        {
            let threads = rayon::current_num_threads();
            let chunk_rows = rows.div_ceil(threads).max(8);
            y.par_chunks_mut(chunk_rows * self.output)
                .enumerate()
                .for_each(|(i, dst)| run(i * chunk_rows, dst));
        }
        #[cfg(not(feature = "cpu"))]
        run(0, y);
    }
}

pub struct Norm {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub eps: f32,
}
impl Norm {
    pub fn load(w: &Weights<'_>, prefix: &str, size: usize, eps: f32) -> Result<Self> {
        let b = format!("{prefix}.bias");
        Ok(Self {
            weight: w.vector(&format!("{prefix}.weight"), size)?,
            bias: if w.contains(&b) {
                Some(w.vector(&b, size)?)
            } else {
                None
            },
            eps,
        })
    }
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    fn pack_six_rows(&self, x: &[f32], y: &mut Vec<f32>) -> usize {
        if packed_supported() {
            return x86_avx2::prepare_input(x, self.weight.len(), y, Some(self));
        }
        let d = self.weight.len();
        assert!(d > 0 && x.len() % d == 0);
        let rows = x.len() / d;
        let len = rows.div_ceil(6) * 6 * d;
        if y.len() < len {
            y.resize(len, 0.);
        }
        for (r, src) in x.chunks_exact(d).enumerate() {
            let mean = src.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = src.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let scale = (var as f32 + self.eps).sqrt().recip();
            for i in 0..d {
                y[r / 6 * 6 * d + i * 6 + r % 6] = (src[i] - mean as f32) * scale * self.weight[i]
                    + self.bias.as_ref().map_or(0., |b| b[i]);
            }
        }
        len
    }
    pub fn apply(&self, x: &[f32], y: &mut Vec<f32>) {
        y.resize(x.len(), 0.0);
        let d = self.weight.len();
        let run = |(src, dst): (&[f32], &mut [f32])| {
            let mean = src.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = src.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let scale = (var as f32 + self.eps).sqrt().recip();
            for i in 0..d {
                dst[i] = (src[i] - mean as f32) * scale * self.weight[i]
                    + self.bias.as_ref().map_or(0.0, |b| b[i]);
            }
        };
        #[cfg(feature = "cpu")]
        if x.len() >= 16 * d {
            x.par_chunks_exact(d)
                .zip(y.par_chunks_exact_mut(d))
                .for_each(run);
            return;
        }
        x.chunks_exact(d).zip(y.chunks_exact_mut(d)).for_each(run);
    }
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub fn gated_gelu(x: &[f32], out: &mut [f32], width: usize) {
    let run = |(src, dst): (&[f32], &mut [f32])| {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        activation_wasm::gated(&src[..width], &src[width..], dst);
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        for i in 0..width {
            dst[i] = gelu(src[i]) * src[width + i];
        }
    };
    #[cfg(feature = "cpu")]
    x.par_chunks_exact(2 * width)
        .zip(out.par_chunks_exact_mut(width))
        .for_each(run);
    #[cfg(not(feature = "cpu"))]
    x.chunks_exact(2 * width)
        .zip(out.chunks_exact_mut(width))
        .for_each(run);
}

pub fn gated_gelu_simd(x: &[f32], out: &mut [f32], width: usize) {
    assert_eq!(x.len(), out.len() * 2);
    assert!(width > 0 && out.len() % width == 0);
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        x.par_chunks_exact(2 * width)
            .zip(out.par_chunks_exact_mut(width))
            .for_each(|(src, dst)| {
                unsafe { x86_avx2::gated_activation(&src[..width], &src[width..], dst) };
            });
        return;
    }
    gated_gelu(x, out, width);
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x {
        *v /= sum;
    }
}

pub fn residual(x: &mut [f32], update: &[f32]) {
    assert_eq!(x.len(), update.len());
    for (x, y) in x.iter_mut().zip(update) {
        *x += y;
    }
}

pub fn attention(
    qkv: &mut [f32],
    out: &mut Vec<f32>,
    heads: usize,
    dim: usize,
    window: Option<usize>,
    rope: Option<f32>,
) {
    out.resize(qkv.len() / 3, 0.0);
    attention_into(qkv, out, heads, dim, window, rope);
}

pub fn attention_into(
    qkv: &mut [f32],
    out: &mut [f32],
    heads: usize,
    dim: usize,
    window: Option<usize>,
    rope: Option<f32>,
) {
    attention_into_tiled(qkv, out, heads, dim, window, rope, 64);
}

pub fn attention_into_tiled(
    qkv: &mut [f32],
    out: &mut [f32],
    heads: usize,
    dim: usize,
    window: Option<usize>,
    rope: Option<f32>,
    query_tile: usize,
) {
    attention_with_scratch(
        qkv,
        out,
        heads,
        dim,
        window,
        rope,
        query_tile,
        &mut AttentionScratch::default(),
        false,
    );
}

pub fn attention_selected(
    qkv: &[f32],
    out: &mut [f32],
    heads: usize,
    dim: usize,
    queries: &[usize],
    query_tile: usize,
    scratch: &mut AttentionScratch,
    reuse: bool,
) {
    let d = heads * dim;
    assert!(d > 0 && qkv.len().is_multiple_of(3 * d));
    let n = qkv.len() / (3 * d);
    assert_eq!(out.len(), queries.len() * d);
    assert!(queries.iter().all(|&q| q < n));
    assert!((16..=256).contains(&query_tile) && query_tile.is_power_of_two());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        let _ = reuse;
        wasm_simd::attention(
            qkv,
            out,
            heads,
            dim,
            None,
            query_tile,
            &mut scratch.heads,
            Some(queries),
        );
        return;
    }
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    if scratch.packed_avx2
        && dim == 64
        && n >= 16
        && (n <= 512 || scratch.long_attention)
        && packed_supported()
    {
        x86_avx2::packed_attention_selected(
            qkv,
            out,
            heads,
            query_tile,
            &mut scratch.heads,
            scratch.math,
            scratch.packed_normalization,
            queries,
        );
        return;
    }
    #[cfg(feature = "cpu")]
    if n >= 16 {
        if reuse {
            tiled_attention::apply_selected(
                qkv,
                out,
                heads,
                dim,
                query_tile,
                &mut scratch.heads,
                queries,
            );
        } else {
            tiled_attention::apply_selected(
                qkv,
                out,
                heads,
                dim,
                query_tile,
                &mut Vec::new(),
                queries,
            );
        }
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut source = qkv.to_vec();
        let mut full = vec![0.; n * d];
        attention_with_scratch(
            &mut source,
            &mut full,
            heads,
            dim,
            None,
            None,
            query_tile,
            scratch,
            reuse,
        );
        for (row, &q) in out.chunks_exact_mut(d).zip(queries) {
            row.copy_from_slice(&full[q * d..][..d]);
        }
    }
}

pub fn attention_with_scratch(
    qkv: &mut [f32],
    out: &mut [f32],
    heads: usize,
    dim: usize,
    window: Option<usize>,
    rope: Option<f32>,
    query_tile: usize,
    scratch: &mut AttentionScratch,
    reuse: bool,
) {
    assert!((16..=256).contains(&query_tile) && query_tile.is_power_of_two());
    let d = heads * dim;
    let n = qkv.len() / (3 * d);
    assert_eq!(out.len(), n * d);
    if let Some(theta) = rope {
        if reuse {
            scratch.rotate(qkv, heads, dim, theta);
        } else {
            let half = dim / 2;
            let inv: Vec<f32> = (0..half)
                .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
                .collect();
            for t in 0..n {
                let angles: Vec<_> = inv.iter().map(|&v| (t as f32 * v).sin_cos()).collect();
                for head in 0..heads {
                    for which in 0..2 {
                        let v = &mut qkv[t * 3 * d + which * d + head * dim..][..dim];
                        for i in 0..half {
                            let (s, c) = angles[i];
                            let (a, b) = (v[i], v[i + half]);
                            v[i] = a * c - b * s;
                            v[i + half] = b * c + a * s;
                        }
                    }
                }
            }
        }
    }
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    if scratch.packed_avx2
        && dim == 64
        && n >= 16
        && (n <= 512 || scratch.long_attention)
        && window.is_none_or(|w| w.is_multiple_of(16))
        && packed_supported()
    {
        if scratch.key_tile > 0 && window.is_none() && n > scratch.key_tile {
            x86_avx2::online_attention(
                qkv,
                out,
                heads,
                query_tile,
                scratch.key_tile,
                &mut scratch.heads,
                scratch.math,
            );
            return;
        }
        if scratch.row_parallel_avx2
            && window.is_none()
            && matches!(scratch.math, AttentionMath::TableExp)
        {
            x86_avx2::row_attention(qkv, out, heads, query_tile, &mut scratch.heads);
            return;
        }
        x86_avx2::packed_attention(
            qkv,
            out,
            heads,
            window,
            query_tile,
            &mut scratch.heads,
            scratch.long_attention,
            scratch.math,
            scratch.packed_normalization,
        );
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm_simd::attention(
            qkv,
            out,
            heads,
            dim,
            window,
            query_tile,
            &mut scratch.heads,
            None,
        );
        return;
    }
    #[cfg(feature = "cpu")]
    if n >= 16 {
        if reuse {
            tiled_attention::apply(qkv, out, heads, dim, window, query_tile, &mut scratch.heads);
        } else {
            tiled_attention::apply(qkv, out, heads, dim, window, query_tile, &mut Vec::new());
        }
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let scale = (dim as f32).sqrt().recip();
        #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
        let simd = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        let run = |t: usize, row: &mut [f32]| {
            let lo = window.map_or(0, |w| t.saturating_sub(w));
            let hi = window.map_or(n, |w| (t + w + 1).min(n));
            let mut scores = vec![0.0; hi - lo];
            for head in 0..heads {
                let q = &qkv[t * 3 * d + head * dim..][..dim];
                for (j, score) in scores.iter_mut().enumerate() {
                    let k = &qkv[(lo + j) * 3 * d + d + head * dim..][..dim];
                    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
                    if simd {
                        *score = unsafe { vector::dot(q, k) } * scale;
                        continue;
                    }
                    *score = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                softmax(&mut scores);
                let dst = &mut row[head * dim..][..dim];
                dst.fill(0.0);
                for (j, &p) in scores.iter().enumerate() {
                    let v = &qkv[(lo + j) * 3 * d + 2 * d + head * dim..][..dim];
                    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
                    if simd {
                        unsafe { vector::add_scaled(dst, v, p) };
                        continue;
                    }
                    for (a, b) in dst.iter_mut().zip(v) {
                        *a += p * b;
                    }
                }
            }
        };
        #[cfg(feature = "cpu")]
        out.par_chunks_mut(d)
            .enumerate()
            .for_each(|(t, row)| run(t, row));
        #[cfg(not(feature = "cpu"))]
        out.chunks_mut(d)
            .enumerate()
            .for_each(|(t, row)| run(t, row));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn activation_producer_packing_matches_separate_operations_and_tails() {
        for format in 0..if cfg!(feature = "fp16") { 3 } else { 2 } {
            let packed = format > 0;
            if packed && !packed_supported() {
                continue;
            }
            for width in [7, 64, 2624, 4096] {
                let output = 32;
                let values: Vec<f32> = (0..width * output)
                    .map(|i| ((i * 13 % 61) as f32 - 30.) / 256.)
                    .collect();
                let weight = match format {
                    #[cfg(feature = "fp16")]
                    2 => Matrix::PackedF16 {
                        data: values.into_iter().map(half::f16::from_f32).collect(),
                        rows: output,
                        cols: width,
                    },
                    1 => Matrix::PackedF32 {
                        data: values,
                        rows: output,
                        cols: width,
                    },
                    _ => Matrix::F32(values),
                };
                let layer = Linear::from_parts(width, output, weight, Some(vec![0.125; output]));
                for rows in [0, 1, 5, 6, 7, 97, 128] {
                    for gated in [false, true] {
                        let inputs = if gated { 2 } else { 1 };
                        let x: Vec<f32> = (0..rows * width * inputs)
                            .map(|i| ((i * 19 % 271) as f32 - 135.) / 32.)
                            .collect();
                        let mut activated = vec![0.; rows * width];
                        if gated {
                            gated_gelu_simd(&x, &mut activated, width);
                        } else {
                            for (a, &v) in activated.iter_mut().zip(&x) {
                                *a = v.max(0.);
                            }
                        }
                        let mut expected = Vec::new();
                        layer.apply_with_workspace(
                            &activated,
                            &mut expected,
                            1,
                            ProjectionStrategy::PackedInput,
                            &mut Vec::new(),
                        );
                        let mut actual = Vec::new();
                        let mut temporary = Vec::new();
                        layer.apply_activated(
                            &x,
                            gated,
                            &mut actual,
                            &mut temporary,
                            &mut Vec::new(),
                            1,
                            ProjectionStrategy::PackedInput,
                        );
                        assert_eq!(activated, temporary, "activation changed");
                        assert!(
                            expected
                                .iter()
                                .zip(&actual)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "packed={packed} width={width} rows={rows} gated={gated}"
                        );
                        assert_eq!(expected.len(), actual.len());
                    }
                }
            }
        }
    }
    #[test]
    #[cfg(feature = "cpu")]
    fn openmp_limit_is_scoped_to_the_calling_thread_and_restores_nested_limits() {
        thread_local! { static LIMIT: std::cell::Cell<i32> = const { std::cell::Cell::new(6) }; }
        unsafe extern "C" fn get() -> i32 {
            LIMIT.with(|v| v.get())
        }
        unsafe extern "C" fn set(n: i32) {
            LIMIT.with(|v| v.set(n));
        }
        let runtime = OpenMp {
            get_max: get,
            set_num: set,
        };
        {
            let _outer = runtime.limit(1);
            assert_eq!(unsafe { get() }, 1);
            std::thread::spawn(move || {
                assert_eq!(unsafe { get() }, 6);
                let _worker = runtime.limit(2);
                assert_eq!(unsafe { get() }, 2);
            })
            .join()
            .unwrap();
            {
                let _inner = runtime.limit(3);
                assert_eq!(unsafe { get() }, 3);
            }
            assert_eq!(unsafe { get() }, 1);
        }
        assert_eq!(unsafe { get() }, 6);
    }
    #[test]
    fn selected_queries_match_full_global_attention_including_duplicates() {
        for n in [1, 7, 17, 63, 513, 2048] {
            for dim in [7, 64] {
                let heads = 2;
                let d = heads * dim;
                let qkv: Vec<_> = (0..n * 3 * d)
                    .map(|i| (i as f32 * 0.017).sin() * 0.7)
                    .collect();
                let queries = [n - 1, 0, n / 2, 0];
                let mut full = vec![0.; n * d];
                attention_with_scratch(
                    &mut qkv.clone(),
                    &mut full,
                    heads,
                    dim,
                    None,
                    None,
                    64,
                    &mut AttentionScratch::default(),
                    false,
                );
                for packed in [false, true] {
                    let mut scratch = AttentionScratch::default();
                    scratch.set_packed_avx2(packed);
                    scratch.set_long_attention(true);
                    let mut out = vec![f32::NAN; queries.len() * d + 2];
                    out[0] = -17.;
                    out[queries.len() * d + 1] = 31.;
                    attention_selected(
                        &qkv,
                        &mut out[1..queries.len() * d + 1],
                        heads,
                        dim,
                        &queries,
                        64,
                        &mut scratch,
                        true,
                    );
                    for (row, &query) in out[1..queries.len() * d + 1].chunks_exact(d).zip(&queries)
                    {
                        assert!(
                            row.iter()
                                .zip(&full[query * d..][..d])
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "n={n} dim={dim} packed={packed}"
                        );
                    }
                    assert_eq!(out[0], -17.);
                    assert_eq!(out[queries.len() * d + 1], 31.);
                }
            }
        }
    }
    #[test]
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    fn normalized_input_tiles_match_separate_operations() {
        for d in [7, 64, 1024] {
            for m in [1, 5, 6, 7, 16, 31] {
                for biased in [false, true] {
                    let norm = Norm {
                        weight: (0..d).map(|i| 1. + i as f32 * 0.01).collect(),
                        bias: biased.then(|| (0..d).map(|i| i as f32 * 0.02).collect()),
                        eps: 1e-5,
                    };
                    let x: Vec<_> = (0..m * d).map(|i| (i as f32 * 0.71).sin()).collect();
                    let mut expected = Vec::new();
                    norm.apply(&x, &mut expected);
                    let mut packed = Vec::new();
                    norm.pack_six_rows(&x, &mut packed);
                    for r in 0..m {
                        for i in 0..d {
                            assert_eq!(
                                expected[r * d + i].to_bits(),
                                packed[r / 6 * 6 * d + i * 6 + r % 6].to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn cached_attention_handles_theta_dimension_and_length_changes() {
        let mut scratch = AttentionScratch::default();
        for (n, dim, theta, window) in [
            (33, 16, 10000., Some(4)),
            (65, 16, 160000., None),
            (17, 16, 10000., None),
            (33, 8, 10000., Some(0)),
            (65, 16, 160000., Some(16)),
            (7, 16, 10000., None),
        ] {
            let heads = 3;
            let input: Vec<_> = (0..n * 3 * heads * dim)
                .map(|i| (i as f32 * 0.017).sin())
                .collect();
            let mut direct = input.clone();
            let mut cached = input;
            let mut a = vec![0.; n * heads * dim];
            let mut b = a.clone();
            attention_into_tiled(&mut direct, &mut a, heads, dim, window, Some(theta), 32);
            attention_with_scratch(
                &mut cached,
                &mut b,
                heads,
                dim,
                window,
                Some(theta),
                32,
                &mut scratch,
                true,
            );
            assert_eq!(direct, cached, "rotary values");
            assert_eq!(a, b, "attention output");
        }
    }
    #[test]
    fn gated_portable_fallback_accepts_irregular_width() {
        let linear = Linear::from_parts(3, 14, Matrix::F32(vec![0.125; 42]), Some(vec![0.25; 14]));
        let mut full = Vec::new();
        let mut actual = Vec::new();
        let a = vec![0.7; 15];
        linear.apply(&a, &mut full);
        let mut expected = vec![0.; 35];
        gated_gelu(&full, &mut expected, 7);
        linear.apply_gated(&a, &mut actual, &mut Vec::new(), 4);
        assert_eq!(actual, expected);
    }
    #[test]
    fn linear_matches_scalar_on_irregular_shapes() {
        let (m, k, n) = (19, 13, 23);
        let a: Vec<_> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let w: Vec<_> = (0..n * k).map(|i| (i as f32 * 0.17).cos()).collect();
        let b = vec![0.23; n];
        let linear = Linear::from_parts(k, n, Matrix::F32(w.clone()), Some(b.clone()));
        let mut out = Vec::new();
        linear.apply(&a, &mut out);
        for i in 0..m {
            for j in 0..n {
                let expected = (0..k).map(|t| a[i * k + t] * w[j * k + t]).sum::<f32>() + b[j];
                assert!((out[i * n + j] - expected).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn local_attention_excludes_distant_values() {
        let mut qkv = vec![0.0; 5 * 3 * 2];
        for t in 0..5 {
            qkv[t * 6 + 4] = t as f32;
            qkv[t * 6 + 5] = 10.0;
        }
        let mut y = Vec::new();
        attention(&mut qkv, &mut y, 1, 2, Some(1), None);
        assert_eq!(y[0], 0.5);
        assert_eq!(y[4], 2.0);
        assert_eq!(y[8], 3.5);
    }
}

#[cfg(test)]
mod kernel_selection_tests {
    use super::*;

    #[test]
    #[cfg(all(feature = "cpu", unix))]
    fn native_library_selection_is_owned_by_each_projection() {
        unsafe extern "C" fn marker(
            _: i32,
            _: i32,
            _: i32,
            m: i32,
            n: i32,
            _: i32,
            _: f32,
            _: *const f32,
            _: i32,
            _: *const f32,
            _: i32,
            _: f32,
            y: *mut f32,
            _: i32,
        ) {
            unsafe { std::slice::from_raw_parts_mut(y, (m * n) as usize) }.fill(7.);
        }
        let blas = std::sync::Arc::new(Blas {
            _library: libloading::os::unix::Library::this().into(),
            sgemm: marker,
            threads: 1,
            openmp: None,
        });
        let make = || {
            Linear::from_parts(
                3,
                2,
                Matrix::F32(vec![1., 2., 3., 4., 5., 6.]),
                Some(vec![0.5, 1.]),
            )
        };
        let mut native = make();
        let rust = make();
        native.set_blas(Some(blas.clone())).unwrap();
        drop(blas);
        let x = [0.25, 0.5, 1.];
        let mut y = Vec::new();
        native.apply(&x, &mut y);
        assert_eq!(y, [7.5, 8.]);
        rust.apply(&x, &mut y);
        assert_eq!(y, [4.75, 10.5]);
        #[cfg(feature = "int8")]
        assert!(native.quantize(int8::Mode::Weights, 32, false).is_err());
        native.set_blas(None).unwrap();
        native.apply(&x, &mut y);
        assert_eq!(y, [4.75, 10.5]);
        assert!(
            Linear::from_parts(
                1,
                1,
                Matrix::PackedF32 {
                    data: vec![1.; 16],
                    rows: 1,
                    cols: 1
                },
                None
            )
            .set_blas(Some(std::sync::Arc::new(Blas {
                _library: libloading::os::unix::Library::this().into(),
                sgemm: marker,
                threads: 1,
                openmp: None,
            })))
            .is_err()
        );
    }

    #[test]
    #[cfg(feature = "fp16")]
    fn explicit_kernel_owns_its_weight_layout() {
        let header = serde_json::to_vec(
            &serde_json::json!({"p.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}),
        )
        .unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        for v in [1_f32, 2., 3., 4., 5., 6.] {
            bytes.extend(v.to_le_bytes());
        }
        let w = Weights::new(&bytes).unwrap();
        let portable =
            Linear::load_with_kernel(&w, "p", 3, 2, Storage::F16, Kernel::Portable).unwrap();
        assert!(matches!(&portable.weight, Matrix::F16(_)));
        let mut expected = Vec::new();
        portable.apply(&[0.25, 0.5, 1.], &mut expected);
        if packed_supported() {
            let packed =
                Linear::load_with_kernel(&w, "p", 3, 2, Storage::F16, Kernel::X86Avx2).unwrap();
            assert!(matches!(&packed.weight, Matrix::PackedF16 { .. }));
            let mut actual = Vec::new();
            packed.apply(&[0.25, 0.5, 1.], &mut actual);
            assert_eq!(expected, actual);
            portable.apply_with_strategy(
                &[0.25, 0.5, 1.],
                &mut actual,
                1,
                ProjectionStrategy::TiledTails,
            );
            assert_eq!(expected, actual);
        }
    }
}

#[cfg(all(
    feature = "kernel-checks",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
pub(crate) fn wasm_kernel_checks() -> anyhow::Result<serde_json::Value> {
    #[cfg_attr(not(feature = "int8"), allow(unused_mut))]
    let mut checks = serde_json::json!({"projection_values":wasm_simd::check()?,"softmax":softmax_wasm::check()?,"gelu_values":activation_wasm::check()?});
    #[cfg(feature = "int8")]
    {
        checks["int8_values"] = serde_json::json!(int8::wasm_check()?);
    }
    Ok(checks)
}
