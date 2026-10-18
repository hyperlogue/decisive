import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def bundle(destination, dp4a=False, suite=False):
    dp4a = dp4a or suite
    paths = ['src/models/pplx_flash/webgpu/shaders.rs',
             'examples/pplx-flash/quantization-probe.js',
             'examples/browser/pplx_flash_quantization.html',
             'scripts/pplx_flash/package_quantization.py']
    if dp4a:
        paths[2] = ('examples/browser/pplx_flash_int8_check.html' if suite
                    else 'examples/browser/pplx_flash_dp4a.html')
        paths.extend(['examples/pplx-flash/dp4a-probe.js','examples/pplx-flash/dp4a-core.js','examples/pplx-flash/dp4a-primitive.js'])
    if suite:
        paths.append('examples/pplx-flash/dp4a-suite.js')
    rust = (ROOT / paths[0]).read_text()
    def extract(name):
        match = re.search(r'\bconst ' + name + r': &str = r#"(.*?)"#;', rust, re.DOTALL)
        if not match:
            raise ValueError(f'Missing shader {name}')
        return match[1]
    source = {'header': extract('HEADER'), 'gemm': extract('GEMM')}
    provenance = {'baseline': ('FP16 reference derived from production GEMM; variant-specific storage and tiling recorded in report' if dp4a else 'Unmodified production BF16 GEMM; quantized variants replace only the weight decoder'),
                  'sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}}
    javascript = (ROOT / paths[1]).read_text()
    if re.search('</script', javascript, re.IGNORECASE):
        raise ValueError('Script cannot be embedded safely')
    html = (ROOT / paths[2]).read_text()
    if suite:
        content = (ROOT / 'examples/pplx-flash/dp4a-suite.js').read_text()
        if re.search('</script', content, re.IGNORECASE) or html.count('/*__SUITE_JS__*/') != 1:
            raise ValueError('Invalid one-click suite marker or script')
        html = html.replace('/*__SUITE_JS__*/', content)
    if dp4a:
        extra = (ROOT / 'examples/pplx-flash/dp4a-probe.js').read_text()
        core = (ROOT / 'examples/pplx-flash/dp4a-core.js').read_text()
        if re.search('</script', core, re.IGNORECASE):
            raise ValueError('Core script cannot be embedded safely')
        if html.count('/*__DP4A_CORE_JS__*/') != 1:
            raise ValueError('Expected one core DP4a marker')
        html = html.replace('/*__DP4A_CORE_JS__*/', core)
        primitive = (ROOT / 'examples/pplx-flash/dp4a-primitive.js').read_text()
        if re.search('</script', primitive, re.IGNORECASE):
            raise ValueError('Primitive script cannot be embedded safely')
        if html.count('/*__DP4A_PRIMITIVE_JS__*/') != 1:
            raise ValueError('Expected one primitive DP4a marker')
        html = html.replace('/*__DP4A_PRIMITIVE_JS__*/', primitive)
        if re.search('</script', extra, re.IGNORECASE):
            raise ValueError('Script cannot be embedded safely')
        if html.count('/*__DP4A_JS__*/') != 1:
            raise ValueError('Expected one DP4a marker')
        html = html.replace('/*__DP4A_JS__*/', extra)
    for marker, content in {
        '__KERNEL_JSON__': json.dumps(source).replace('<', '\\u003c'),
        '__PROVENANCE_JSON__': json.dumps(provenance),
        '/*__PROBE_JS__*/': javascript,
    }.items():
        if html.count(marker) != 1:
            raise ValueError(f'Expected one marker: {marker}')
        html = html.replace(marker, content)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as output:
        output.write(html)
    print(f'Open in Chrome: {destination} ({destination.stat().st_size / 1024:.0f} KiB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?', default=ROOT / 'artifacts/web/pplx-quantization-check.html')
    parser.add_argument('--dp4a', action='store_true', help='Compare W8A8 packed integer dots against FP16')
    parser.add_argument('--suite', action='store_true', help='One-click packed INT8 verdict')
    args = parser.parse_args()
    try:
        bundle(args.output, args.dp4a, args.suite)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Packaging failed: {error}\n')
