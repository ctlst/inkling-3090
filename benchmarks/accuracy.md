# Paired accuracy spot check

The test uses four public MMLU-Pro questions from each of six subjects. Within each subject, the harness sorts the test split by SHA-256 of `inkling-419/{question_id}` and selects the first four. Selection happened before either run. Each server receives the same single user message, with the question, lettered options and a request for one letter. Requests use temperature 0, seed 419, low reasoning effort and a 2,048-token completion budget. Grading uses the final message only. A missing or unparseable answer counts as wrong. This zero-shot sample uses a different protocol and sample size from a model-card benchmark. [Public dataset and revision](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be)

## Reproduce

From the repository root with Python and uv, run the commands below for each endpoint. The preparation step verifies the downloaded test split by SHA-256. The scorer uses the Python standard library and writes progress after every response.

```bash
mkdir -p .verify/accuracy
cd .verify/accuracy
curl -fL -o test.parquet https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet
uv run --with pyarrow python ../../benchmarks/accuracy_sample.py prepare test.parquet fixtures.json
python3 ../../benchmarks/accuracy_sample.py run http://127.0.0.1:8000 fixtures.json results.json
```

Both engines use the official weights at `ed40e7da625f594fc9c4990f5f66be24873c22d7`. The live RTX 3090 uses the tuned SM86 serving path, FP8 target KV, and seven greedy draft proposals. The B200 reference uses the unmodified upstream image pinned in the Dockerfile. Both B200 profiles request eager execution and leave KV at its upstream default (auto with BF16 model dtype). The serial run serves one sequence; the batched run serves up to 24. Both use a 196,608 context ceiling. The multiple-choice prompts are short.

The serial upstream B200 run used the stock image with the pinned checkpoint mounted at `/weights`, the bundled draft checkpoint, and probabilistic DSpark proposals. The matched user requests still set temperature to zero.

```bash
vllm serve /weights --served-model-name inkling-2-mini \
  --max-model-len 196608 --max-num-seqs 1 --enforce-eager \
  --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

For the batched reference, remove the speculative configuration, set `--max-num-seqs 24`, and add `--max-num-batched-tokens 2048 --kv-cache-memory-bytes 12884901888 --language-model-only --disable-flashinfer-autotune`. Pass `--concurrency 24` to the scorer to send one independent HTTP request per item with up to 24 concurrent clients. These settings preserve the prompt, template, decoding budget and grading across systems. Device, KV precision, kernels, batching and speculation can all affect numerical results.

## Reproduce the long-context check

The LongBench v2 user-guide item has 195,015 prompt tokens with the pinned tokenizer and template. Run the following from the repo root after starting the service. The runner checks the prompt hash, token count, final choice and stop reason.

```bash
mkdir -p .verify/longbench
curl -fL -o .verify/longbench/data.json https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json
curl -fL -o .verify/longbench/0shot.txt https://raw.githubusercontent.com/THUDM/LongBench/2e00731f/prompts/0shot.txt
python3 benchmarks/longbench_guide.py .verify/longbench/data.json .verify/longbench/0shot.txt http://127.0.0.1:8000
```

## Results

All three configurations returned a parseable final answer and a normal stop on every question.

| Configuration | Correct | Same choice as the 3090 |
|---|---:|---:|
| Tuned RTX 3090, FP8 target KV, greedy DSpark 7 | 17/24 | 24/24 |
| Upstream B200, default KV, target only | 17/24 | 24/24 |
| Upstream B200, default KV, probabilistic DSpark 7 | 18/24 | 22/24 |

On the serial upstream DSpark run, history question 4884 was correct where the other two configurations were wrong. The other disagreement, history question 4858, was wrong on all three. All three answered the intact 195,015-token LongBench v2 user-guide question correctly (B), with the same prompt hash and input token count. These results cover one run per configuration on 24 questions and one long document. [Item-level results](accuracy-3090-vs-upstream.json)
