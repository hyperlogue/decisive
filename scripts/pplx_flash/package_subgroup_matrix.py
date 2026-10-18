import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def bundle(output):
    paths = ['src/models/pplx_flash/webgpu/shaders.rs',
             'examples/pplx-flash/quantization-probe.js',
             'examples/pplx-flash/dp4a-probe.js',
             'examples/pplx-flash/dp4a-core.js',
             'examples/pplx-flash/subgroup-matrix.js',
             'scripts/pplx_flash/package_subgroup_matrix.py']
    rust = (ROOT / paths[0]).read_text()
    source = {name.lower(): re.search(r'\bconst ' + name + r': &str = r#"(.*?)"#;', rust, re.S)[1]
              for name in ['HEADER', 'GEMM']}
    scripts = '\n'.join((ROOT / p).read_text() for p in paths[1:5])
    if re.search('</script', scripts, re.I):
        raise ValueError('Unsafe inline script')
    provenance = {'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    html = '''<!doctype html><meta charset="utf-8"><title>Subgroup matrix probe</title>
<h1>Subgroup matrix probe</h1><p>Experimental Chrome capability and resident GEMM test. Browser settings are never changed.</p>
<button id="run">Run probe</button><pre id="status">Ready</pre><pre id="result"></pre><script>
'''+scripts+f'\nconst source={json.dumps(source)},provenance={json.dumps(provenance)};'+'''
document.querySelector('#run').onclick=async()=>{
 const button=document.querySelector('#run');button.disabled=true;let hidden=document.hidden;
 const visibility=()=>{hidden ||= document.hidden;};document.addEventListener('visibilitychange',visibility);
 const report=await PplxSubgroupMatrix.run(source,provenance,message=>document.querySelector('#status').textContent=message);
 document.removeEventListener('visibilitychange',visibility);report.page_hidden_during_run=hidden;
 document.querySelector('#status').textContent=report.completed?(report.supported?'Complete':'Feature unavailable'):'Failed';
 document.querySelector('#result').textContent=JSON.stringify(report,null,2);
 const endpoint=new URLSearchParams(location.search).get('report');
 if(endpoint)await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)});
 button.disabled=false;
};if(new URLSearchParams(location.search).get('autorun')==='1')document.querySelector('#run').click();
</script>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as file:
        file.write(html)
    print(f'Packaged {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    bundle(parser.parse_args().output)
