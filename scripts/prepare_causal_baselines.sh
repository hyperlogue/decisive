#!/usr/bin/env bash
set -euo pipefail
bench_root=${1:-.cache/causal-baselines}
mkdir -p "$bench_root"
bench_root=$(cd "$bench_root" && pwd)
checkout() {
  local url=$1 path=$2 revision=$3
  mkdir -p "$(dirname "$path")"
  if [ ! -d "$path/.git" ]; then git clone --filter=blob:none --no-checkout "$url" "$path"; fi
  git -C "$path" fetch --depth 1 origin "$revision"
  git -C "$path" checkout --detach "$revision"
}
checkout https://github.com/ggml-org/llama.cpp "$bench_root/llama" 1b70350c92563449576bcbbf552994b1d1ed847d
checkout https://github.com/jaredpalmer/kev "$bench_root/sources/kev-source" 85ccf56a0afb952323c621055b02876e8498d60c
checkout https://github.com/lawrence3699/jev-style "$bench_root/sources/jev-source" ef86def93c4b8c775523bea6812784a36b2b6d8a
if [ ! -x "$bench_root/venv/bin/python" ]; then uv venv --python 3.13 "$bench_root/venv"; fi
uv pip install --python "$bench_root/venv/bin/python" --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0'
uv pip install --python "$bench_root/venv/bin/python" 'transformers==5.17.0' 'peft==0.21.0' 'accelerate==1.15.0' 'pydantic==2.14.1' numpy sentencepiece protobuf mistral-common
cargo build --release --features bench --bin decisive
bash scripts/build_bench.sh causal_bench --features bench
target/release/decisive kev fetch
target/release/decisive jev-style fetch
"$bench_root/venv/bin/python" scripts/prepare_causal_assets.py --sources "$bench_root/sources" --llama "$bench_root/llama"
cmake -S "$bench_root/llama" -B "$bench_root/llama/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF -DGGML_OPENMP=OFF \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build "$bench_root/llama/build" --target llama-server -j "${BUILD_JOBS:-4}"
sh "$bench_root/sources/jev-gguf/build_jev_score.sh" "$bench_root/llama"

mkdir -p "$bench_root/sources/jev-gguf-f32kv"
cp "$bench_root/sources/jev-gguf/jev_score.cpp" "$bench_root/sources/jev-gguf-f32kv/jev_score.cpp"
cp "$bench_root/sources/jev-gguf/build_jev_score.sh" "$bench_root/sources/jev-gguf-f32kv/build_jev_score.sh"
patch -d "$bench_root/sources/jev-gguf-f32kv" -p1 < scripts/jev_score_f32_kv.patch
sh "$bench_root/sources/jev-gguf-f32kv/build_jev_score.sh" "$bench_root/llama"
