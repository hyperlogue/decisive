"""Embed the production Pplx WebGPU shaders in the opportunities benchmark page.

The shaders are produced by the Rust builders in src/models/pplx_flash/webgpu,
compiled here as a tiny standalone program so the page uses identical WGSL.
"""
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src/models/pplx_flash/webgpu'
PAGE = ROOT / 'examples/browser/pplx_flash_opportunities.html'

PROGRAM = r'''
#![allow(dead_code)]
mod shaders { include!(concat!(env!("WEBGPU"), "/shaders.rs")); }
mod grouped { include!("grouped.rs"); }
fn main() {
    use shaders::*;
    let mut m = std::collections::BTreeMap::new();
    for (k, v) in [("norm", NORM), ("point", POINT), ("conv", CONV), ("delta_qk", DELTA_QK),
        ("delta_params", DELTA_PARAMS), ("delta_constants", DELTA_CONSTANTS), ("delta_gate", DELTA_GATE),
        ("attention_norm", ATTENTION_NORM), ("softmax", SOFTMAX), ("scores", SCORES_TILED), ("values", VALUES_TILED)] {
        m.insert(k.to_string(), source(v));
    }
    m.insert("delta_scan".into(), source(&delta_scan_four()));
    m.insert("int8_input".into(), grouped::quantize());
    m.insert("gemm_fallback".into(), grouped::float_gemm());
    m.insert("gemm".into(), include_str!(concat!(env!("WEBGPU"), "/grouped_q4_tuned.wgsl")).to_string());
    for (k, v) in m { println!("{k}\t{}", v.replace('\\', "\\\\").replace('\n', "\\n")); }
}
'''


def shaders():
    grouped = (SOURCE / 'grouped_q4.rs').read_text()
    # Keep the pure WGSL builders; the GPU check below them needs the full crate.
    grouped = grouped[grouped.index('pub(super) fn quantize'):grouped.index('pub(crate) async fn check')]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / 'grouped.rs').write_text(grouped)
        (directory / 'main.rs').write_text(PROGRAM)
        subprocess.run(['rustc', '--edition', '2021', '--cfg', 'feature="kernel-checks"', '-O', 'main.rs', '-o', 'dump'],
                       cwd=directory, check=True, env={**__import__('os').environ, 'WEBGPU': str(SOURCE)})
        output = subprocess.run([directory / 'dump'], check=True, capture_output=True, text=True).stdout
    result = {}
    for line in output.splitlines():
        name, body = line.split('\t', 1)
        result[name] = re.sub(r'\\(.)', lambda m: '\n' if m[1] == 'n' else m[1], body)
    return result


def main():
    page = PAGE.read_text()
    begin, end = '// BEGIN GENERATED SHADERS (scripts/pplx_flash/build_opportunities.py)\n', '// END GENERATED SHADERS'
    head, rest = page.split(begin)
    _, tail = rest.split(end)
    PAGE.write_text(f'{head}{begin}const S={json.dumps(shaders(), sort_keys=True)};\n{end}{tail}')


if __name__ == '__main__':
    main()
