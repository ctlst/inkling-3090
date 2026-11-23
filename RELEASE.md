# Inkling 2 Mini on RTX 3090 · v0.1.0-rc.1

A source-based Docker release for one 24 GB RTX 3090. Build the pinned vLLM image with the included SM86 patch and download the official Inkling 2 Mini checkpoint at the revision in the README. The build checks the resulting model code by SHA-256. The checkpoint manifest covers the target, drafter, tokenizer and serving configuration. [Build and run](README.md#build-and-run)

## Validated profile

The engine serves one active sequence with 196,608 total tokens, FP8 target KV, all routed experts resident in VRAM, and seven greedy DSpark proposals. The expert weights retain their released NVFP4/MXFP8 formats. Input embeddings and vision weights use host memory. The recipe leaves power and fan settings under the operator’s control.

| Workload | Single RTX 3090 result |
|---|---|
| Short prose, chat, chart reading, tool call | 120, 146, 140 and 180 decode tok/s respectively, medians of three trials per prompt at 220 W |
| Tests-first coding | 233 decode tok/s, 17 external checks and 22 generated self-tests passed at 220 W |
| Five-turn agent replay | 130.84 s with cold prefixes, 43.73 s with shared prefixes at 220 W |
| Cold 195,015-token retrieval | Correct B, 105.36 s to first token, 1,851 prompt tok/s and 95.2 decode tok/s at 220 W |
| Two-hour app-building session | 171.9 active decode tok/s and 150.1 wall-clock output tok/s, including reasoning and retries, mostly at 350 W |

The candidate reached health 69.3 s after a fresh container start with new runtime caches and the model already on disk. Arithmetic, vision, structured tools, FP8 encoding/decoding and 48 routed batches passed. Each short task uses one prompt. Coding and cached-image speeds describe those specific workloads. [Cold boot and replay](benchmarks/cold-boot-3090.json) · [Task checks and outputs](benchmarks/workloads-3090.json) · [Two-hour run](benchmarks/omp-3090.json)

## Upstream accuracy check

On 24 fixed public MMLU-Pro questions, the 3090 scored 17/24, upstream target-only vLLM on B200 scored 17/24, and upstream vLLM with DSpark on B200 scored 18/24. The 3090 agreed with those references on 24 and 22 final choices respectively. All three answered the same 195,015-token public LongBench v2 question correctly. The initial target-only B200 attempt changed three choices on repeat. This spot check uses six subjects, one long document and distinct serving configurations. [Protocol and reproducible public checks](benchmarks/accuracy.md)

This release includes the source recipe, original-weight verification, and measured results. Build the image locally. Validation covers Linux x86-64 on a single RTX 3090. [License](LICENSE) · [Notices](NOTICE)
