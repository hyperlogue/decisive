#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import urllib.request
p=argparse.ArgumentParser()
p.add_argument('--sources', type=Path, required=True)
p.add_argument('--llama', type=Path, required=True)
a=p.parse_args()
BASE='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
KEV='bf75a6a8848ea6960ff2ed108d9ed44c2941174f'
JEV='b023d1f9c7858fbf01504577a3bfc349ea5c7385'
GGUF='b8356a83beb560cf34cbe6f9a20c2076e1b532d3'
def fetch(repo, revision, remote, local, expected=None):
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        temp=local.with_suffix(local.suffix+'.partial')
        with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{remote}') as source, temp.open('wb') as dest:
            while chunk:=source.read(1024*1024): dest.write(chunk)
        temp.replace(local)
    if expected:
        with local.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected, f'Checksum mismatch: {local.name}'
fetch('jaredpalmer/kev-0.8b', KEV, 'training_config.json', Path('models/kev-0.8b/training_config.json'))
fetch('jaredpalmer/kev-0.8b', KEV, 'tokenizer_config.json', Path('models/kev-0.8b/tokenizer_config.json'))
fetch('chaoliangUNSW/Jev-Style-0.8B-Decision-v3', JEV, 'tokenizer_config.json', Path('models/jev-style-0.8b-v3/tokenizer_config.json'))
for name in ['tokenizer.json','tokenizer_config.json']:
    fetch('Qwen/Qwen3.5-0.8B-Base', BASE, name, Path('models/qwen3.5-0.8b-base')/name)
fetch('chaoliangUNSW/Jev-Style-0.8B-Decision-v3', JEV, 'jev_style_decision.py', a.sources/'jev/jev_style_decision.py')
repo='chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF'
for name in ['jev_score.cpp','build_jev_score.sh','jev_style_decision_gguf.py']:
    fetch(repo,GGUF,name,a.sources/'jev-gguf'/name)
for name in ['readout_config.json','tokenizer/tokenizer.json','Jev-Style-0.8B-Decision-v3-F16.gguf']:
    digest='a33f709e10009c3fe182b51e4945d440288b137a35ad5d481e4ac1f2800b1a27' if name.endswith('.gguf') else None
    fetch(repo,GGUF,name,Path('models/jev-style-0.8b-v3-gguf')/name,digest)
base=Path('models/qwen3.5-0.8b-base').resolve()
assert json.loads((base/'beam-revision.json').read_text())['revision']==BASE
assert json.loads(Path('models/kev-0.8b/beam-revision.json').read_text())['revision']==KEV
out=Path('models/kev-0.8b-f32.gguf')
if not out.exists():
    sys.path[:0]=[str(a.llama),str(a.llama/'gguf-py')]
    import huggingface_hub
    original_download=huggingface_hub.hf_hub_download
    original_snapshot=huggingface_hub.snapshot_download
    def download(repo_id, filename, **kw):
        if repo_id=='Qwen/Qwen3.5-0.8B-Base':
            assert kw.get('revision') in (None,BASE)
            path=base/filename
            assert path.is_file(), filename
            return str(path)
        return original_download(repo_id,filename,**kw)
    def snapshot(repo_id, **kw):
        if repo_id=='Qwen/Qwen3.5-0.8B-Base':
            assert kw.get('revision') in (None,BASE)
            return str(base)
        return original_snapshot(repo_id,**kw)
    huggingface_hub.hf_hub_download=download
    huggingface_hub.snapshot_download=snapshot
    import torch
    torch.set_num_threads(6)
    temp=out.with_suffix('.partial')
    sys.argv=[str(a.llama/'convert_hf_to_gguf.py'),'models/kev-0.8b','--outfile',str(temp),'--outtype','f32']
    runpy.run_path(str(a.llama/'convert_hf_to_gguf.py'),run_name='__main__')
    temp.replace(out)

assert json.loads(Path('models/jev-style-0.8b-v3/beam-revision.json').read_text())['revision']==JEV
jev_out=Path('models/jev-style-0.8b-v3-f32.gguf')
if not jev_out.exists():
    import subprocess
    subprocess.run([sys.executable,str(a.llama/'convert_hf_to_gguf.py'),
                    'models/jev-style-0.8b-v3','--outtype','f32','--outfile',str(jev_out.with_suffix('.partial'))],
                   check=True,env={**__import__('os').environ,'PYTHONPATH':str(a.llama)+':'+str(a.llama/'gguf-py')})
    jev_out.with_suffix('.partial').replace(jev_out)
