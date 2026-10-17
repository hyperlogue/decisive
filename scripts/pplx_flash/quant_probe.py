"""Offline weight-quantization sensitivity experiment, not an inference backend.

Reconstructs grouped affine low-bit weights and runs the FP32 publisher oracle.
This measures output sensitivity only, never GPU performance or FP16 arithmetic.
"""
import argparse
import importlib.metadata
import json
import sys
import types
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, Qwen3_5RMSNorm,
)


def reconstruct(weight, bits, group):
    assert weight.ndim == 2 and weight.shape[1] % group == 0
    blocks = weight.reshape(-1, group)
    lo = blocks.amin(-1, keepdim=True)
    scale = ((blocks.amax(-1, keepdim=True) - lo) / ((1 << bits) - 1))
    codes = ((blocks - lo) / scale.clamp_min(1e-30)).round().clamp(0, (1 << bits) - 1)
    return (codes * scale.half().float() + lo.half().float()).reshape(weight.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/pplx-decider-flash-v1.2'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', type=int, default=64, choices=[32, 64, 128])
    parser.add_argument('--threads', type=int, default=3)
    a = parser.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    package = types.ModuleType('pplx_decider')
    package.__path__ = [str(a.source)]
    sys.modules['pplx_decider'] = package
    from pplx_decider.protocol import Encoder, answer

    class Processor:
        tokenizer = AutoTokenizer.from_pretrained(a.model)

        def apply_chat_template(self, messages, **kw):
            return self.tokenizer.apply_chat_template(messages, **kw)

        def __call__(self, text, images, return_tensors):
            return self.tokenizer(text, return_tensors=return_tensors)

    decision = json.loads((a.model / 'decision_config.json').read_text())
    encoder = Encoder(Processor(), decision['codes'])
    requests = [json.loads(p.read_text()) for p in a.input]
    encoded = []
    for request in requests:
        chunks = encoder.plan(request, 262144)
        assert len(chunks) == 1, 'probe requires one sequence per input'
        encoded.append(chunks[0])
    config = Qwen3_5TextConfig(**json.loads((a.model / 'config.json').read_text())['text_config'])
    config._attn_implementation = 'eager'
    files = [safe_open(a.model / f'model-0000{i}-of-00002.safetensors', framework='pt') for i in (1, 2)]
    lookup = {key: f for f in files for key in f.keys()}
    variants = ['bf16', 'q4', 'q3', 'q2', 'q4_mlp_q8_other']
    states = {}
    positions = []
    rotary = Qwen3_5TextRotaryEmbedding(config)
    for item in encoded:
        p = torch.arange(item.length).view(1, 1, -1).expand(3, 1, -1)
        positions.append(p)
    for name in variants:
        states[name] = [lookup['language_model.embed_tokens.weight'].get_tensor(
            'language_model.embed_tokens.weight')[item.input_ids].float().unsqueeze(0) for item in encoded]
    pe = [rotary(x, p) for x, p in zip(states['bf16'], positions)]
    sizes = {name: 0 for name in variants}
    quantized_names = []
    for i in range(config.num_hidden_layers):
        prefix = f'language_model.layers.{i}.'
        original = {k[len(prefix):]: lookup[k].get_tensor(k).float() for k in lookup if k.startswith(prefix)}
        for name in variants:
            with torch.device('meta'):
                layer = Qwen3_5DecoderLayer(config, i)
            state = {}
            for key, weight in original.items():
                bits = 16 if name == 'bf16' else (4 if 'mlp' in key else 8) if name == 'q4_mlp_q8_other' else int(name[-1])
                quantized = bits < 16 and weight.ndim == 2 and key.endswith('.weight')
                if quantized:
                    state[key] = reconstruct(weight, bits, a.group)
                    # One output row at a time, packed into u32 words; FP16 scale and offset.
                    per_word = 32 // bits
                    sizes[name] += weight.shape[0] * ((weight.shape[1] + per_word - 1) // per_word) * 4
                    sizes[name] += weight.numel() // a.group * 4
                    if name == 'q4':
                        quantized_names.append(prefix + key)
                else:
                    state[key] = weight
                    sizes[name] += weight.numel() * 2
            layer.load_state_dict(state, assign=True)
            layer.eval()
            for index, (x, p) in enumerate(zip(states[name], positions)):
                states[name][index] = layer(x, position_embeddings=pe[index], attention_mask=None,
                                            position_ids=p[0], use_cache=False)
            del layer, state
        print(f'layer {i + 1}/{config.num_hidden_layers}', flush=True)
    norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.weight = torch.nn.Parameter(lookup['language_model.norm.weight'].get_tensor('language_model.norm.weight').float())
    with safe_open(a.model / 'readout.safetensors', framework='pt') as file:
        readout = file.get_tensor('weight').float()
    result = {
        'repository': 'perplexity-ai/pplx-decider-flash-v1.2',
        'revision': '6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d',
        'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
        'method': 'affine min-max round-to-nearest, per output row/input group; FP16 scale and offset; FP32 dequantization and computation',
        'group_size': a.group, 'scope': 'all decoder 2D linear weights; mixed variant uses Q4 MLP and Q8 other projections; embeddings, conv, norm, readout unchanged',
        'limitation': 'small synthetic sensitivity probe; not task-quality approval, not WebGPU, no activation quantization, no calibrated AWQ/GPTQ',
        'decoder_storage_bytes': sizes, 'quantized_tensors': quantized_names,
        'cases': [],
    }
    for index, (request, item) in enumerate(zip(requests, encoded)):
        case = {'request': request, 'tokens': item.length, 'variants': {}}
        for name in variants:
            logits = norm(states[name][index])[0, item.markers] @ readout.T
            probabilities = [(logits[j, :item.counts[j]] / decision['temperature']).softmax(-1).tolist() for j in range(len(item.keys))]
            case['variants'][name] = {
                'probabilities': dict(zip(item.keys, probabilities)),
                'answers': {key: answer(request['questions'][key], p) for key, p in zip(item.keys, probabilities)},
            }
        result['cases'].append(case)
    result['comparison'] = {}
    for name in variants[1:]:
        differences = []
        flips = 0
        for case in result['cases']:
            base = case['variants']['bf16']['probabilities']
            other = case['variants'][name]['probabilities']
            for key, probabilities in base.items():
                differences.extend(abs(x - y) for x, y in zip(probabilities, other[key]))
                flips += max(range(len(probabilities)), key=probabilities.__getitem__) != max(range(len(other[key])), key=other[key].__getitem__)
        result['comparison'][name] = {'max_probability_error': max(differences), 'mean_probability_error': sum(differences) / len(differences), 'argmax_changes': flips, 'questions': sum(len(x.keys) for x in encoded)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['comparison'], indent=2), flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
