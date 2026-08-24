#!/usr/bin/env python3
"""A/B: same graded items through kv-dtype variants with greedy mtp0.

Launches ninfer-serve per kv-dtype sequentially, asks the same few items,
prints completions so drift is visible, and grades them with the same
strategies run_serve_graded uses.
"""
import http.client
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "tools/bench")
from run_serve_corpus import RunningServer, SPECULATIVE_MODES, SEEDS
from run_serve_graded import load_graded_items, grade_answer, DEFAULT_SEED_DATASET
from run_graded_suite import serve_resolve, load_identity, to_target_key

ARTIFACT = Path("models/qwen3_8_27b_dflash2.ninfer").resolve()
KV_VARIANTS = ["bf16", "int8", "rk4v4-e8"]
MAX_TOKENS = 128


def command(serve, port, kv, log_path):
    return [
        str(serve), str(ARTIFACT),
        "--host", "127.0.0.1", "--port", str(port),
        "--model-id", "qwen3.8-27b",
        "--max-context", "8192",
        "--prefill-chunk", "1024",
        "--log-stats-interval-ms", "0",
        "--device", "0",
        "--request-log-jsonl", str(log_path),
        "--kv-dtype", kv,
        "--no-prefix-reuse",
        "--greedy",
    ]


def post(conn, payload):
    body = json.dumps(payload).encode()
    conn.request("POST", "/v1/chat/completions", body=body, headers={
        "Content-Type": "application/json", "Content-Length": str(len(body))})
    resp = conn.getresponse()
    return json.loads(resp.read().decode())


def main() -> int:
    serve = Path("bin/ninfer-serve.exe").resolve()
    items = load_graded_items([DEFAULT_SEED_DATASET], 6)
    out = Path("profiles/graded/ab-kv") 
    out.mkdir(parents=True, exist_ok=True)
    port = 8300
    for kv in KV_VARIANTS:
        log = out / f"ab_{kv}.jsonl"
        with RunningServer(command(serve, port, kv, log), "127.0.0.1", port, log) as srv:
            srv.wait_until_ready()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            conn.request("POST", "/v1/chat/completions", body=json.dumps({
                "model": "qwen3.8-27b",
                "messages": [{"role": "user", "content": "Reply with exactly one word: ready"}],
                "max_completion_tokens": 8, "stream": False, "enable_thinking": False,
            }).encode(), headers={"Content-Type": "application/json"})
            conn.getresponse().read()
            print(f"\n=== kv={kv} ===")
            n = 0; correct = 0
            for item in items:
                payload = {"model": "qwen3.8-27b",
                           "messages": [{"role": "user", "content": item.question}],
                           "max_completion_tokens": MAX_TOKENS, "stream": False,
                           "enable_thinking": False}
                resp = post(conn, payload)
                content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
                verdict = grade_answer(item.answer, content, item.match)
                correct += verdict["correct"]
                n += 1
                print(f"  [{item.item_id}] {'OK ' if verdict['correct'] else 'BAD'} exp={item.answer!r} got={content[:60]!r}")
            print(f"  => {correct}/{n} correct")
            conn.close()
        port += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())