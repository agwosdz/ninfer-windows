#!/usr/bin/env python3
"""Determinism check: same rk4v4-e8 request twice; compare completions."""
import http.client
import json
import sys
from pathlib import Path

sys.path.insert(0, "tools/bench")
from run_serve_corpus import RunningServer

ARTIFACT = Path("models/qwen3_8_27b_dflash2.ninfer").resolve()


def command(serve, port, kv, log):
    return [str(serve), str(ARTIFACT), "--host", "127.0.0.1", "--port", str(port),
            "--model-id", "qwen3.8-27b", "--max-context", "8192", "--prefill-chunk", "1024",
            "--log-stats-interval-ms", "0", "--device", "0",
            "--request-log-jsonl", str(log), "--kv-dtype", kv, "--no-prefix-reuse", "--greedy"]


def ask(conn, q):
    body = json.dumps({"model": "qwen3.8-27b",
                       "messages": [{"role": "user", "content": q}],
                       "max_completion_tokens": 64, "stream": False,
                       "enable_thinking": False}).encode()
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
    return json.loads(conn.getresponse().read().decode())


def main():
    out = Path("profiles/graded/ab-kv")
    out.mkdir(parents=True, exist_ok=True)
    q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    serve = Path("bin/ninfer-serve.exe").resolve()
    for kv in ("int8", "rk4v4-e8"):
        port = 8400
        log = out / f"det_{kv}.jsonl"
        with RunningServer(command(serve, port, kv, log), "127.0.0.1", port, log) as srv:
            srv.wait_until_ready()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            r1 = ask(conn, q)
            r2 = ask(conn, q)
            c1 = (r1.get("choices") or [{}])[0].get("message", {}).get("content", "")
            c2 = (r2.get("choices") or [{}])[0].get("message", {}).get("content", "")
            conn.close()
            print(f"=== {kv} ===")
            print("run1:", repr(c1[:120]))
            print("run2:", repr(c2[:120]))
            print("identical:", c1 == c2)
        port += 1


if __name__ == "__main__":
    raise SystemExit(main())