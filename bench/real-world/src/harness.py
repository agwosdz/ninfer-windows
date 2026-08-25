#!/usr/bin/env python3
"""Real-world serving benchmark harness for ninfer-serve."""
from __future__ import annotations
import argparse, json, statistics, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None

def load_json(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)

def _openai_chat_request(base_url, model, prompt, max_tokens, temperature, timeout_seconds):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))

class Backend:
    def open(self, target, config): raise NotImplementedError
    def close(self): raise NotImplementedError
    def ready(self): raise NotImplementedError
    def request(self, prompt, max_tokens, temperature, request_index=0): raise NotImplementedError

class MockBackend(Backend):
    def __init__(self, tokens_per_second=100.0, ttft=0.02):
        self._tps = tokens_per_second; self._ttft = ttft
    def open(self, target, config): pass
    def close(self): pass
    def ready(self): return True
    def request(self, prompt, max_tokens, temperature, request_index=0):
        start = time.monotonic(); time.sleep(self._ttft)
        time.sleep((max_tokens + (request_index % 3)) / self._tps)
        elapsed = time.monotonic() - start
        return {"ttft_s": self._ttft, "elapsed_s": elapsed,
                "prompt_tokens": len(prompt) // 2, "completion_tokens": max_tokens,
                "finish_reason": "length", "cached_tokens": 0, "valid": True}

def compute_metrics(reqs, wall_seconds=None):
    valid = [r for r in reqs if r.get("valid")]
    ttft = median([r.get("ttft_s") for r in valid])
    e2e = median([r.get("elapsed_s") for r in valid])
    decode = median([(r["completion_tokens"] - 1) / (r["elapsed_s"] - r["ttft_s"])
                     for r in valid if r.get("ttft_s") and r.get("elapsed_s")
                     and r["elapsed_s"] > r["ttft_s"]])
    aggregate = None
    if valid:
        total = sum(r["completion_tokens"] for r in valid)
        aggregate = total / wall_seconds if wall_seconds else (total / max(r["elapsed_s"] for r in valid))
    return {"ttft_s": ttft, "decode_tok_s": decode, "e2e_tok_s": e2e,
            "aggregate_tok_s": aggregate, "valid_count": len(valid), "request_count": len(reqs)}

def run_single(backend, prompt_req, config):
    max_tokens = int(config.get("max_tokens", 512)); temperature = int(config.get("temperature", 0))
    runs = int(config.get("runs", 5)); warmup = int(config.get("warmup_requests", 1))
    for _ in range(warmup): backend.request(prompt_req["prompt"], 8, temperature)
    return [backend.request(prompt_req["prompt"], max_tokens, temperature, i) for i in range(runs)]

def run_concurrent(backend, prompt_req, config):
    max_tokens = int(config.get("max_tokens", 512)); temperature = int(config.get("temperature", 0))
    rounds = int(config.get("rounds", 5)); concurrency = int(config.get("concurrency", 4))
    retries = int(config.get("round_retries", 1))
    all_results = []; round_walls = []
    for _ in range(rounds):
        for attempt in range(retries + 1):
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(backend.request, prompt_req["prompt"], max_tokens, temperature, i)
                           for i in range(concurrency)]
                reqs = [f.result() for f in futures]
            wall = time.monotonic() - start
            if all(r.get("valid") for r in reqs):
                round_walls.append(wall); all_results.extend(reqs); break
            if attempt == retries: all_results.extend(reqs)
    return all_results, round_walls

def format_summary(target_id, mode, metrics):
    ttft = f"{metrics['ttft_s']:.4f}s" if metrics['ttft_s'] is not None else "n/a"
    decode = f"{metrics['decode_tok_s']:.2f}" if metrics['decode_tok_s'] is not None else "n/a"
    e2e = f"{metrics['e2e_tok_s']:.2f}" if metrics['e2e_tok_s'] is not None else "n/a"
    agg = f"{metrics['aggregate_tok_s']:.2f}" if metrics['aggregate_tok_s'] is not None else "n/a"
    return (f"{target_id:24s} {mode:12s} ttft={ttft:>8s} decode={decode:>8s} e2e={e2e:>8s} agg={agg:>8s} valid={metrics['valid_count']}/{metrics['request_count']}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="ninfer-serve real-world bench")
    parser.add_argument("config")
    parser.add_argument("--dry-run", action="store_true", help="mock backend; no server")
    parser.add_argument("--limit", type=int, default=None, help="first N prompts")
    parser.add_argument("--modes", default="single,concurrent", help="single,concurrent")
    args = parser.parse_args(argv)
    config = load_json(args.config)
    base_dir = Path(args.config).resolve().parent
    prompts_path = Path(config.get("prompts", "prompts/workloads.jsonl"))
    if not prompts_path.is_absolute(): prompts_path = base_dir / prompts_path
    prompts = []
    with prompts_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line: prompts.append(json.loads(line))
    if args.limit: prompts = prompts[:args.limit]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.dry_run:
        backend = MockBackend()
        targets = [{"id": "mock", "artifact": "models/mock.ninfer", "alias": "mock",
                    "kv_dtype": "int8", "spec": "none", "port": 0}]
    else:
        serve = config.get("serve", {})
        backend = ServeBackendStub()
        targets = config.get("targets", [])
        if not targets: raise SystemExit("no targets in config")
    try:
        for target in targets:
            backend.open(target, config)
            print(f"target {target['id']}: backend ready")
            for prompt in prompts:
                print(f"  prompt {prompt['id']} ({prompt.get('prompt_type','?')})")
                if "single" in modes:
                    m = compute_metrics(run_single(backend, prompt, config), None)
                    print("    " + format_summary(target["id"], "single", m))
                if "concurrent" in modes:
                    results, walls = run_concurrent(backend, prompt, config)
                    m = compute_metrics(results, statistics.median(walls) if walls else None)
                    print("    " + format_summary(target["id"], "concurrent", m))
    finally:
        backend.close()
    return 0

class ServeBackendStub:
    def open(self, target, config): raise SystemExit("serve backend not implemented in dry-run")
    def close(self): pass
    def ready(self): return False
    def request(self, prompt, max_tokens, temperature, request_index=0): return {"valid": False, "error": "serve stub"}

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
