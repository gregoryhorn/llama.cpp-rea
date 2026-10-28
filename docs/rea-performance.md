# REA performance fork: first measured optimization

This is an experimental performance branch of llama.cpp **b11887**, based on
`6050b467c07a95c7b87debd4ddac2f8ba8d9b807`. It is not a new inference engine,
not a replacement CUDA backend, and not a claim that all llama.cpp workloads
are inefficient. The first change removes avoidable CPU sampling overhead.

## Verified code-level result

Hardware: RTX 3090 24 GB; Intel i7-13700KF; Linux x86-64.
Model: `Agents-A1-Uncensored-MTP-APEX-I-Compact.gguf`, reported file type Q4_K
Medium but **mixed tensor precision**, `qwen35moe`, embedding width 2048,
256 experts with 8 active, and one embedded MTP layer. Metadata reports 41 blocks
including the MTP block. Actual tensor counts are Q6_K: 121, F32: 310,
Q4_K: 218, Q3_K: 93, Q8_0: 11. The routed expert matrices include 93 Q3_K,
27 Q4_K and 3 Q8_0 tensors; tuning only Q4_K would miss most expert tensors.
The embedded MTP weight matrices are Q8_0, with F32 normalization/router tensors.

All inference measurements below used:

- The same model file, original b11887 CUDA12.8 backend and server executable.
- Full GPU layer offload, flash attention, f16 K/V cache, one slot.
- Configured context 180000 (reported allocation 180224).
- Seed 12345, temperature 0.85, top-k 20, top-p 0.95, presence penalty 1.1,
  repetition penalty 1.0, min-p 0, and 512 generated tokens per request.
- Prompt cache disabled, EOS ignored, startup excluded. The original campaign
  used one 64-token short-prompt warmup per process; the first medium/long
  prefill shapes were not separately warmed, so their whole-request samples
  can include shape-initialization cost. Decode measurements followed prefill.
- No speculative decoding or backend sampling in the controlled code A/B test.

The first campaign inherited repeat_last_n=64 and frequency_penalty=0 from the
verified server defaults. Following review, the reusable benchmark now sets
both explicitly, records request settings and server identity, requires an
explicit endpoint, and warms every prompt shape. The original raw records were
not rewritten to imply they contained those added metadata fields.

Only `libllama.so` changed between the controlled baseline and optimized runs.
Both were compiled from the same revision with GCC 15.2.0 and the same Release
CMake configuration; the baseline was built before editing the sampler.
The run order was **baseline, optimized, baseline, optimized**, with two
repetitions of each prompt per process: four samples per case per build.
Three additional stock-release samples per case checked the original install.

### Decode throughput (median tokens/second)

| Prompt tokens | Stock release (n=3) | Local baseline (n=4) | Modified core (n=4) | Gain vs local baseline |
|---:|---:|---:|---:|---:|
| 13 | 131.439 | 130.964 | 140.205 | 7.06% |
| 776 | 129.872 | 130.551 | 138.927 | 6.42% |
| 12296 | 124.096 | 124.532 | 132.135 | 6.11% |

### End-to-end request time (median seconds)

| Prompt tokens | Local baseline | Modified core |
|---:|---:|---:|
| 13 | 4.081 | 3.838 |
| 776 | 4.216 | 3.979 |
| 12296 | 7.278 | 7.049 |

The long prompt is deliberately repetitive synthetic text, not a representative
long-document quality evaluation. Decode time and whole-request time are
separate: this CPU sampling change does not accelerate prompt matrix multiplies.
These are descriptive medians from one machine, not confidence intervals or a
cross-hardware performance guarantee.

**All 33 completions produced exactly the same token sequence for their prompt**
across the stock release, local baseline and modified core. Each completion
contained exactly 512 tokens. This is bounded equivalence on these test cases,
not proof of equivalence for every possible model or request.

## Mechanism and implementation

`llama_sampler_penalties_apply` previously called `unordered_map::find` for
every candidate. With this vocabulary, that means 248320 hash lookups per
sampling step against a history of at most 64 tokens. The release disassembly
contains integer bucket-division and hash-chain traversal in that loop.

The new CPU path:

1. Lazily allocates a vocabulary-indexed integer count table.
2. Populates only the entries present in the history map.
3. Scans candidates using a direct count lookup instead of hashing.
4. Clears just the populated entries, leaving scratch independent of semantic
   sampler state.

The original history map and GPU sampler path remain intact. Arbitrarily ordered
candidate arrays, reduced subsets, duplicates and out-of-range IDs retain their
behavior; out-of-range IDs use the original map lookup. Penalty arithmetic and
sampler ordering are unchanged. The cost is one extra integer per vocabulary
entry for each CPU penalty sampler that actually runs.

The controlled latency gate was set at **300 microseconds** before the code
change. The unmodified local core failed at **718.114 microseconds** median;
the modified core passed at **113.769 microseconds**. These isolated sampler
measurements must not be confused with full-model speedup.

## Correctness tests

- Existing upstream `test-sampling`: passed before and after the change.
- `rea-penalties-bench`: exact floating-point comparisons against the original
  penalty formula for full ordered and shuffled vocabularies, subsets, eviction,
  reset, clone, state copy, repeated application, duplicates and out-of-range IDs.
- CTest registers correctness only; the optional timing ceiling is not enforced
  in generic CI because hardware and contention differ.
- Seeded inference token IDs were compared, not just decoded text or answer quality.

Run:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BACKEND_DL=ON -DGGML_CPU=OFF -DGGML_CUDA=OFF \
  -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_UI=OFF
cmake --build build --target llama test-sampling rea-penalties-bench -j 4
ctest --test-dir build -R '^(test-sampling|rea-penalties-bench)$' --output-on-failure
build/bin/rea-penalties-bench
# Optional, only on a controlled comparable machine:
build/bin/rea-penalties-bench 300
```

This core-only configuration deliberately builds no compute backend. For local
inference it was paired with the **matching b11887** prebuilt CPU/CUDA backends
via the dynamic backend loader. It does not provide GPU inference by itself.
There was no CUDA compiler on this host, and an attempted local CPU backend
build hit an upstream const-conversion error in `ggml-cpu/arch/x86/repack.cpp`.
Neither issue was concealed with permissive compiler flags or unrelated patches.
A normal full CUDA source build remains possible when its toolchain is installed;
it was not exercised here.

To repeat an HTTP benchmark against an already running compatible server:

```bash
python scripts/rea-server-bench.py --base http://SERVER:8080 \
  --label baseline --output baseline.jsonl --repeats 3
# Restart with the candidate core, retaining identical model/settings, then:
python scripts/rea-server-bench.py --base http://SERVER:8080 \
  --label optimized --output optimized.jsonl --repeats 3
```

Use a fresh output file for a new experiment. Each line contains timings, wall
time, generated token IDs and a content digest. Do not run other inference
requests or competing CPU benchmarks during the measurement window.

## REA evidence and limits

[Focused shipped-binary evidence](rea-performance-evidence.json) records the
release library SHA-256 and REA symbol observation. REA identified the actual
release function at `0x35dee0`; direct GNU objdump disassembly then established
the bucket-division instructions at `0x35e01b` and `0x35e04c`.
Source inspection alone did not establish runtime cost: the isolated benchmark
and alternating full-model runs supplied that evidence.

The complete three-record REA export was retained locally for verification.
The public evidence file is explicitly an excerpt, not a mutated full Evidence
record. No decompiler analysis was needed for this source-available change.

Kernel profiling through `perf` was blocked by `perf_event_paranoid=4`; no
privilege settings were changed. CUDA kernels were reused, not rebuilt or
profiled with Nsight. Kernel-level bottlenecks, other GPUs, multi-user throughput,
full 180k-token workloads and production task-quality effects remain unmeasured.

## Additional measured configuration experiments

These reuse **upstream features**, not additional custom-code optimizations.
Each candidate used the modified CPU core, the same context/model/settings and
2 samples per prompt after warmup. They were run after the alternating A/B
experiment, so the comparisons below are exploratory rather than another
interleaved controlled test.

| Configuration | 13-token prompt | 776-token prompt | 12296-token prompt |
|---|---:|---:|---:|
| Modified core, backend sampling requested | 142.132 t/s | 140.686 t/s | 134.372 t/s |
| Modified core, MTP draft max 1 | 170.327 t/s | 168.513 t/s | 160.118 t/s |
| Modified core, MTP draft max 3 | 175.570 t/s | 156.783 t/s | 151.075 t/s |

The backend-sampling requests matched the baseline tokens on all six samples,
but their whole-request medians (3.859, 4.014, 7.073 seconds) did not beat the
modified CPU-only sampler's corresponding medians. This small decode difference
is not sufficient evidence to change the default sampler placement.

MTP draft max 1 was the more balanced experimental setting: approximately
28.6–30.1% higher decode throughput than the local non-MTP baseline on these
cases, but it uses an existing upstream MTP implementation. Max 3 was faster on
the short case and slower than max 1 on the other two. The first max-1 sample
accepted 221/290 drafted tokens on the short case, 219/292 on medium and 218/292
on long; the server returned these counters directly.

**All twelve stochastic MTP completions differed from the non-MTP token
sequence.** That is not by itself evidence of incorrect text or a changed
sampling distribution: different verification batches can change numerical
results and random-number consumption. It does mean that the exact-token
validation of the CPU patch cannot be extended to MTP. No task-quality or
sampling-distribution evaluation was completed. MTP is therefore documented as
an opt-in experiment, not silently enabled in the default deployment.

For this model the existing optional flags are:

```bash
--spec-type draft-mtp --spec-draft-n-max 1
```

Do not add `--model-draft` with the same large model file: this version can create
its MTP context against the target model's shared weights. The full 180k-token
workload was not tested even though the configured allocation was retained.

Raw records: [controlled A/B and stock release](rea-server-benchmark.jsonl),
[configuration candidates](rea-candidate-benchmark.jsonl).

## Local deployment and rollback

The tested core is installed separately at
`~/.local/opt/llama.cpp-rea-b11887/lib/libllama.so.0.7.0`. The existing
`qwen-local.service` loads it through
`~/.config/systemd/user/qwen-local.service.d/rea-performance.conf`.
The original server executable, CUDA/CPU backend libraries, model, sampling
parameters, tools and context configuration were left intact.

After restarting the service, `/proc/<pid>/maps` confirmed both the custom
core and the original CUDA backend were actually mapped. Health returned
`{"status":"ok"}`; `/props` still reported context allocation 180224,
speculation `none`, backend sampling `false`, and the same model alias.
Three additional deployed requests matched the controlled baseline's token IDs
exactly. Their decode rates were 142.479, 139.559 and 134.550 t/s for the short,
medium and long cases respectively. These are single deployment smoke samples,
not replacements for the controlled medians above.

The real browser UI was rendered, its Send control was exercised, and the
model's `REA_RUNTIME_OK` reply was visually verified. The code change did not
require a UI replacement.

Rollback removes only this drop-in, not the original runtime:

```bash
rm ~/.config/systemd/user/qwen-local.service.d/rea-performance.conf
systemctl --user daemon-reload
systemctl --user restart qwen-local.service
```

Do not remove the original runtime libraries while this deployment uses them.
The drop-in must be revisited before upgrading the underlying llama.cpp version;
this locally built core is bound to the tested b11887 library ABI.

## Remaining code-level candidates — hypotheses, not completed improvements

1. **SM86 MoE dispatch/launch tuning.** Ampere uses the generic MMVQ parameter
   table in `ggml/src/ggml-cuda/mmvq.cu:106–143`. Benchmark actual Q3_K/Q4_K
   expert shapes and verification widths before adding an RTX 3090-specific
   policy. Existing multi-token MMVQ and expert fusion are not new work.
2. **Chunked CUDA Gated DeltaNet prefill.** The pinned CUDA implementation has
   an explicit chunked-prefill TODO at `gated_delta_net.cu:192`. A change here
   requires output AND final recurrent-state tests, including partial chunks
   and multiple ubatches. Flash attention is not a substitute for optimizing
   the recurrent layers.
3. **GPU-resident MTP hidden-state exchange.** The pinned MTP driver stores
   pending/verification hidden rows in host vectors. Device-backed exchange
   could reduce synchronization, but must preserve rejection/rollback state.
   No synchronization profile was acquired, so its cost remains unknown.
4. **Large-vocabulary small-k GPU selection.** Compare any new fused selector
   against the already present radix/CUB implementations before calling it an
   improvement. Preserve penalty/grammar ordering and test ties/infinities.

Flash attention and graph reuse were already active, with zero reported graph
reallocations during the controlled tests, so turning them on again is not an
optimization. Kernel tuning requires a CUDA compiler and a separate correctness
and performance campaign; this branch does not pretend to have completed it.

Pinned implementation references:

- [Original CPU penalty loop](https://github.com/ggml-org/llama.cpp/blob/6050b467c07a95c7b87debd4ddac2f8ba8d9b807/src/llama-sampler.cpp#L2957-L2987)
- [Existing MTP implementation](https://github.com/ggml-org/llama.cpp/blob/6050b467c07a95c7b87debd4ddac2f8ba8d9b807/common/speculative.cpp#L1489-L1601)
- [Shared-model MTP context initialization](https://github.com/ggml-org/llama.cpp/blob/6050b467c07a95c7b87debd4ddac2f8ba8d9b807/common/speculative.cpp#L2704-L2761)
