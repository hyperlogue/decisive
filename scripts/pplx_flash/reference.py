import sys,json,time,types,argparse,importlib.metadata
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer,Qwen3_5TextRotaryEmbedding,Qwen3_5RMSNorm
parser=argparse.ArgumentParser(description="Text-only FP32 parity oracle using publisher protocol and Transformers layers. Not a tuned serving baseline.")
parser.add_argument('--model',type=Path,default=Path('models/pplx-decider-flash-v1.2'))
parser.add_argument('--source',type=Path,required=True,help='Pinned release source/src/pplx_decider directory')
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--threads',type=int,default=3)
a=parser.parse_args()
root=a.model;src=a.source
pkg=types.ModuleType('pplx_decider');pkg.__path__=[str(src)];sys.modules['pplx_decider']=pkg
from pplx_decider.protocol import Encoder,answer
class Processor:
 tokenizer=AutoTokenizer.from_pretrained(root)
 def apply_chat_template(self,messages,**kw):return self.tokenizer.apply_chat_template(messages,**kw)
 def __call__(self,text,images,return_tensors):return self.tokenizer(text,return_tensors=return_tensors)
request=json.loads(a.input.read_text());codes=json.load(open(root/'decision_config.json'))['codes'];encoder=Encoder(Processor(),codes);encoded=encoder.plan(request,262144)[0]
print('prompt',encoder.text(request),'tokens',encoded.length,flush=True)
output={'request':request,'ids':encoded.input_ids.tolist(),'markers':encoded.markers.tolist(),'keys':list(encoded.keys),'counts':list(encoded.counts),'layers':[], 'reference':{'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'dtype':'float32','attention':'eager, noncausal full attention','threads':a.threads}}
torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
config=Qwen3_5TextConfig(**json.load(open(root/'config.json'))['text_config']);config._attn_implementation='eager'
files=[safe_open(root/f'model-0000{i}-of-00002.safetensors',framework='pt') for i in [1,2]]
lookup={k:f for f in files for k in f.keys()}
def get(k):return lookup[k].get_tensor(k)
with torch.inference_mode():
 x=get('language_model.embed_tokens.weight')[encoded.input_ids].float().unsqueeze(0)
 positions=torch.arange(x.shape[1]).view(1,1,-1).expand(3,1,-1)
 rotary=Qwen3_5TextRotaryEmbedding(config);pe=rotary(x,positions)
 for i in range(32):
  t=time.perf_counter()
  with torch.device('meta'):layer=Qwen3_5DecoderLayer(config,i)
  prefix=f'language_model.layers.{i}.'
  layer.load_state_dict({k[len(prefix):]:get(k).float() for k in lookup if k.startswith(prefix)},assign=True);layer.eval()
  loaded=time.perf_counter()
  x=layer(x,position_embeddings=pe,attention_mask=None,position_ids=positions[0],use_cache=False)
  output['layers'].append({'layer':i,'compute_ms':(time.perf_counter()-loaded)*1000,'load_ms':(loaded-t)*1000,'first':x[0,encoded.markers[0],:8].tolist()})
  print(i,output['layers'][-1],flush=True);del layer
 norm=Qwen3_5RMSNorm(config.hidden_size,eps=config.rms_norm_eps);norm.weight=torch.nn.Parameter(get('language_model.norm.weight').float());x=norm(x)
 with safe_open(root/'readout.safetensors',framework='pt') as f:readout=f.get_tensor('weight').float()
 logits=x[0,encoded.markers]@readout.T
 temperature=json.load(open(root/'decision_config.json'))['temperature']
 output['logits']={k:logits[i,:encoded.counts[i]].tolist() for i,k in enumerate(encoded.keys)}
 output['response']={'answers':{k:answer(request['questions'][k],(logits[i,:encoded.counts[i]]/temperature).softmax(-1).tolist()) for i,k in enumerate(encoded.keys)},'usage':{'input_tokens':encoded.length,'output_tokens':0,'sequences':1}}
a.output.write_text(json.dumps(output,indent=2));print(output['response'],flush=True)
