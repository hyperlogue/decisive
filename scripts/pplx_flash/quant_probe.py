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


def reconstruct_symmetric(weight, bits, group):
    assert weight.ndim == 2 and weight.shape[1] % group == 0
    blocks = weight.reshape(-1, group)
    limit = (1 << (bits - 1)) - 1
    scale = blocks.abs().amax(-1, keepdim=True) / limit
    codes = (blocks / scale.clamp_min(torch.finfo(torch.float32).tiny)).round().clamp(-limit, limit)
    return (codes * scale).reshape(weight.shape)


def precision(variant, key):
    if variant == 'bf16':
        return 16
    if variant == 'q8_mlp_bf16_other':
        return 8 if key.startswith('mlp.') else 16
    if variant == 'q4_mlp_q8_other':
        return 4 if key.startswith('mlp.') else 8
    return int(variant[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/pplx-decider-flash-v1.2'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', type=int, default=64, choices=[32, 64, 128])
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--variants', nargs='+', choices=['bf16', 'q8_mlp_bf16_other', 'q4', 'q3', 'q2', 'q4_mlp_q8_other'],
                        default=['q8_mlp_bf16_other', 'q4', 'q3', 'q2', 'q4_mlp_q8_other'],
                        help='BF16 is always included as the reference')
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
    variants = list(dict.fromkeys(['bf16', *a.variants]))
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
    quantized_names = {name: [] for name in variants}
    for i in range(config.num_hidden_layers):
        prefix = f'language_model.layers.{i}.'
        original = {k[len(prefix):]: lookup[k].get_tensor(k).float() for k in lookup if k.startswith(prefix)}
        for name in variants:
            with torch.device('meta'):
                layer = Qwen3_5DecoderLayer(config, i)
            state = {}
            for key, weight in original.items():
                bits = precision(name, key)
                quantized = bits < 16 and weight.ndim == 2 and key.endswith('.weight')
                if quantized:
                    fn = reconstruct_symmetric if name == 'q8_mlp_bf16_other' else reconstruct
                    state[key] = fn(weight, bits, a.group)

                    per_word = 32 // bits
                    sizes[name] += weight.shape[0] * ((weight.shape[1] + per_word - 1) // per_word) * 4
                    sizes[name] += weight.numel() // a.group * 4
                    quantized_names[name].append(prefix + key)
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
        'method': 'q8_mlp_bf16_other: symmetric max-absolute, FP32 scale, nearest ties-to-even; other quantized variants: affine min-max, FP16 scale and offset; all use FP32 dequantization and computation',
        'group_size': a.group, 'scope': 'q8_mlp_bf16_other changes only MLP 2D weights; q4_mlp_q8_other uses Q4 MLP/Q8 other projections; other variants quantize all decoder 2D linear weights; embeddings, conv, norm, readout unchanged',
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
