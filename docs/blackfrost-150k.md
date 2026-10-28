# Blackfrost Qwen3.8 27B: 150K-context customization tests

## Target and publisher guidance

These tests use the actual **Blackfrost Qwen3.8 27B** weights, not Agents A1
behind an old Hermes display label.

- Model: `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`, `Q4_K_M`.
- Pinned Hugging Face revision: `994bb4e69663ec880a4d9a61604e6debc3a49b9a`.
- Local file: `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf`, 16,810,716,384 bytes.
- Actual SHA-256, checked against the download metadata:
  `5d53637a59cfcd3a4d8354e254ffd44943e5a693da2405a3e228c62962355509`.
- GGUF name: `Qwen3.8 27B ABLITERATED BF16`; runtime architecture identifier
  `qwen35`; 65 blocks including one NextN/MTP block, 262144 architectural context.
  This runtime architecture identifier does not mean the selected file is Agents A1.

The complete [publisher model card at the tested revision](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/blob/994bb4e69663ec880a4d9a61604e6debc3a49b9a/README.md)
was read before testing. It specifies embedded MTP, no separate draft model,
`--jinja`, and a serving example with temperature 1.0, top-p 0.95 and top-k 20.
Its MTP example uses maximum draft length 3; this test compares 1 and 3 instead
of assuming that one is always fastest.

The card's 109.31 t/s DFlash2 result used a 96 GB RTX PRO 6000 Blackwell Server
Edition, an additional BF16 sidecar and a separately pinned implementation. It
is not an RTX 3090 expectation and was not reproduced here. No sidecar, vision
projector or new model download was used for these tests.

## Controlled setup

Hardware: the same RTX 3090 24 GB and i7-13700KF as the Agents A1 experiment.
Runtime: llama.cpp b11887, unchanged original CUDA12.8 backend and executable.

The requested **150000 context tokens were allocated as 150016** in every run.
K and V caches were both Q8_0. The model metadata implies about 9.16 GiB of
attention KV at 150000 tokens with F16, versus about 4.86 GiB with Q8_0 before
additional state, MTP context, compute buffers and overhead. Q8_0 was therefore
used consistently to leave GPU headroom; this experiment does not establish
quality equivalence between Q8_0 and F16 caches.

Common server options:

```bash
llama-server \
  --model Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf \
  --alias Qwen3.8-27B-ABLITERATED \
  --ctx-size 150000 --parallel 1 --gpu-layers all --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 256 --ubatch-size 128 --jinja \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --presence-penalty 1.1 --repeat-penalty 1.0 --offline \
  --host 127.0.0.1 --port 18081
```

Presence penalty 1.1 was held constant to exercise the customized penalty path;
it is not presented as a publisher-recommended Blackfrost setting. Requests also
explicitly set frequency penalty 0, repetition window 64, CPU sampling, seed
12345, cache_prompt=false, ignore_eos=true, return_tokens=true, and 256 generated
tokens. Every prompt shape received a 64-token warmup in each process.

The no-speculation core comparison alternated **baseline, modified, baseline,
modified**, with two repetitions per prompt per process. Each was a Release
core built with GCC 15.2 from the same upstream revision and paired with the
same original compute backend. Process maps verified which core was loaded.
MTP configurations were measured afterward, with two repetitions per prompt.
These feature experiments were not themselves interleaved with new baseline
processes and should be interpreted as exploratory comparisons.

Core hashes:

| Core | SHA-256 |
|---|---|
| Unmodified local baseline | `9020bebe9ac8f3d44e11edddde55c491e2e5ed7572f963d9e5646e2aa2262e96` |
| Modified REA core | `e9eb726028ef76756f987661fd495ddd4beec0e293ef3d16bc28857626ff9299` |

## Results: decode throughput

Median tokens/second. Baseline and modified-core-only have **n=4 per case**;
each MTP setting has **n=2 per case**.

| Prompt tokens | Baseline, no MTP | Modified core, no MTP | Modified + MTP max 1 | Modified + MTP max 3 |
|---:|---:|---:|---:|---:|
| 13 | 36.348 | 36.410 | 51.402 | 48.313 |
| 776 | 36.065 | 36.246 | 50.670 | 56.313 |
| 3080 | 35.770 | 35.947 | 52.131 | 59.603 |

| Prompt tokens | Core-only change vs baseline | MTP max 1 vs baseline | MTP max 3 vs baseline |
|---:|---:|---:|---:|
| 13 | +0.17% | +41.42% | +32.92% |
| 776 | +0.50% | +40.50% | +56.14% |
| 3080 | +0.49% | +45.74% | +66.63% |

**The custom CPU penalty change does not show a meaningful end-to-end speedup
on this dense Blackfrost workload.** Its 0.17-0.50% descriptive median changes
are small relative to ordinary sample variation. Do not transfer the Agents A1
6.1-7.1% result to Blackfrost.

The substantial gains here come from **existing upstream MTP**, not newly written
CUDA kernels. Max 1 wins the shortest case, while max 3 wins the medium and long
cases. The publisher's max-3 example is therefore useful guidance, not a universal
winner. Task quality, latency distribution and real conversation workloads are
separate from these throughput samples.

## Results: complete request wall time

Median seconds, including prompt processing, generation and HTTP overhead:

| Prompt tokens | Baseline | Modified core | Modified + MTP max 1 | Modified + MTP max 3 |
|---:|---:|---:|---:|---:|
| 13 | 7.361 | 7.346 | 5.338 | 5.659 |
| 776 | 7.932 | 7.897 | 5.994 | 5.481 |
| 3080 | 9.931 | 9.908 | 7.871 | 7.266 |

## Correctness and actual speculation

Across all **24 no-MTP measured completions**, the unmodified and modified core
returned exactly the same generated token sequence per prompt. Every measured
request contained 256 generated tokens. The experiment contains **36 measured
requests total**, plus warmups.

The server returned positive drafted and accepted counts for both MTP modes:

| Prompt tokens | MTP max 1 accepted/drafted, both repeats | MTP max 3 accepted/drafted, both repeats |
|---:|---:|---:|
| 13 | 210 / 298 | 268 / 724 |
| 776 | 206 / 304 | 304 / 612 |
| 3080 | 220 / 290 | 320 / 568 |

These counters establish that native drafting executed, not merely that a flag
was configured. The no-MTP exact-token conclusion does not establish MTP sampling
distribution equivalence, task quality, or recurrent rollback correctness over
all possible inputs.

## Testing-profile Hermes client smoke

A real Hermes CLI request used `HERMES_HOME=~/.hermes/profiles/testing`, with
launch-only overrides `--provider custom`, the isolated localhost test endpoint,
and `--model Qwen3.8-27B-ABLITERATED`. It ran against the modified Blackfrost
core with MTP max 1 at the same 150016 allocation; no saved profile model or
endpoint was edited.

The observed stream contained an actual `terminal` tool call:

```text
command: printf BLACKFROST_REA_OK
stdout:  BLACKFROST_REA_OK
exit:    0
```

Hermes then returned:

> Exit code 0. Exact stdout: `BLACKFROST_REA_OK` (no trailing newline; no stderr).

The client exited 0. The retained result reports 15059 input tokens and 144
output tokens across the turn. This is a bounded tool-call/result/final-answer
smoke test, not comprehensive agent or safety validation. The first attempt
failed before inference because the live gateway does not support the CLI's
`--run-budget` forwarding; removing only that flag made the retry execute. No
policy, gateway configuration or model weights were changed to bypass it.

## Limits and scope

- 150K context was **allocated**, not filled. Measured prompts had 13, 776 and
  3080 tokens; no full-150K prefill, retrieval or quality evaluation was done.
- The longer prompts are synthetic repetitive text, not a representative
  document corpus or tool-use workload.
- The code change was already reviewed and tested; this task adds model-specific
  execution evidence rather than another unmeasured engine rewrite.
- Reduced context allocation creates memory headroom; it does not by itself
  establish faster decoding for every active sequence length.
- No claims are made about vision/video, DFlash2, concurrent users or other GPUs.
- Tests used a separate localhost port. Persistent testing-profile model aliases,
  the production service definition, and original model files were not changed.

The test servers were stopped and the pre-existing Agents A1 production service
was restored, with its optimized core and MTP max 1 still active; health was
checked afterward. Blackfrost was tested, not silently made the permanent model.

Raw measurements, including exact request settings, model/core identity,
full server commands, token IDs and timing/acceptance counters:
[blackfrost-150k-results.jsonl](blackfrost-150k-results.jsonl).
