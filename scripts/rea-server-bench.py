#!/usr/bin/env python3
"""Reproducible single-slot completion benchmark; no model/config changes.

Run once per build after warmup. Compare tokens as well as throughput.
Each result is appended immediately, so partial runs remain usable.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def request(base, payload):
    req = urllib.request.Request(base + '/completion', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    return result, time.monotonic() - start


def run(base, label, output, repeats=3):
    prompts = {
        'short': 'The following is a detailed explanation of how a computer works:\n',
        'medium': ('A computer processes data using memory, arithmetic, and instructions. ' * 64) + '\nExplain this in detail:\n',
        'long': ('A computer processes data using memory, arithmetic, and instructions. ' * 1024) + '\nExplain this in detail:\n',
    }
    settings = dict(n_predict=512, seed=12345, temperature=0.85, top_k=20,
                    top_p=0.95, min_p=0.0, presence_penalty=1.1, repeat_penalty=1.0,
                    repeat_last_n=64, frequency_penalty=0.0, backend_sampling=False,
                    ignore_eos=True, cache_prompt=False, return_tokens=True)
    with urllib.request.urlopen(base + '/props', timeout=10) as response:
        props = json.load(response)
    identity = {key: props.get(key) for key in ('build_info', 'model_alias', 'model_ftype')}
    identity['context'] = props['default_generation_settings']['n_ctx']
    # Warm every prompt shape, including its prefill graph, before timing it.
    for prompt in prompts.values():
        request(base, dict(settings, prompt=prompt, n_predict=64))
    with Path(output).open('a') as f:
        for repeat in range(repeats):
            for name, prompt in prompts.items():
                result, wall = request(base, dict(settings, prompt=prompt))
                tokens = result['tokens']
                assert len(tokens) == settings['n_predict'], (name, len(tokens))
                row = dict(label=label, repeat=repeat, case=name, wall_s=wall,
                           request_settings=dict(settings, prompt=prompt), server=identity,
                           response_backend_sampling=result.get('generation_settings', {}).get('backend_sampling'),
                           timings=result['timings'], tokens=tokens,
                           content_sha256=hashlib.sha256(result['content'].encode()).hexdigest())
                f.write(json.dumps(row) + '\n')
                f.flush()
                print(label, repeat, name, 'prompt_n=', result['timings']['prompt_n'],
                      'decode_tps=', round(result['timings']['predicted_per_second'], 3), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Explicit server origin, e.g. http://127.0.0.1:8080')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    run(args.base, args.label, args.output, args.repeats)
