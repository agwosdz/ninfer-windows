# Real-world serving benchmarks

`bench/real-world/` measures the complete public serving route — `ninfer-serve` behind its
OpenAI-compatible API — against real, deterministic workloads, judged by the wall-clock metrics a
user sees: TTFT, decode throughput, aggregate throughput under concurrency, and output validity.

Companion to `bench/` (in-process `ninfer_bench`) and `eval/` (EvalScope accuracy). Mirrors the
`qwen3.8-27b-local-inference-benchmark` candidate methodology: 5 measured rounds, medians,
validity rules, concurrent aggregate.

## Concepts
- **Target**: served model config (artifact, alias, kv dtype, spec, context, port), launched and
  stopped by the harness.
- **Prompt set**: JSONL {id, prompt, prompt_type, budget_tokens}.
- **Validity**: stream complete, finish_reason expected, completion tokens == budget, no transport
  error; a bad round is retried once.

## Metrics (medians over valid results)
- TTFT s, decode tok/s, e2e tok/s, aggregate tok/s (C), batch wall s.

## Usage
```powershell
Copy-Item bench/real-world/benchmark.json.example bench/real-world/benchmark.json
python bench/real-world/src/harness.py bench/real-world/benchmark.json --dry-run
```

## Config shape
```json
{ "runs":5, "rounds":5, "concurrency":4, "max_tokens":512, "warmup_requests":1,
  "temperature":0, "prompt_token_tolerance":16, "request_timeout_seconds":300,
  "prompts":"prompts/workloads.jsonl",
  "serve":{ "binary":"build/apps/ninfer-serve" },
  "targets":[ { "id":"qwen3.6-27b-int8", "artifact":"models/qwen3_6_27b.ninfer",
                "alias":"qwen3.6-27b", "kv_dtype":"int8", "spec":"none",
                "max_context":16384, "port":8080 } ] }
```
KV dtype names: `bf16|int8|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8`.
