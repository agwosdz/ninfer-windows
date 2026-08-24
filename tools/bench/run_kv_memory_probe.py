#!/usr/bin/env python3
"""Memory probe: weights + KV footprint per artifact, KV dtype, and context.

Each engine start reports its startup-frozen memory plan in the server_start
event (weights capacity, KV payload bytes, runtime reservation, headroom,
sequence/workspace arenas, CUDA-graph allowance). This tool sweeps
`--max-context` values per artifact x KV dtype x optional speculative mode and
emits a flat CSV plus a console table, so the base-model KV cost and its
growth to full context are directly measurable.

Usage:

    # Base model memory (mtp0/no spec) for one artifact across contexts.
    python tools/bench/run_kv_memory_probe.py \
        --artifact base=models/qwen3_8_27b.ninfer \
        --max-context 2048,8192,32768,98304 \
        --kv-dtype bf16,int8,rk4v4-e8 \
        --output profiles/memory/qwen38

    # Include the DFlash2 drafter's own KV (labels the extra footprint).
    python tools/bench/run_kv_memory_probe.py \
        --artifact dflash2=models/qwen3_8_27b_dflash2.ninfer \
        --family dflash7 \
        --max-context 2048,8192,32768 \
        --kv-dtype int8 \
        --output profiles/memory/qwen38-dflash2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

try:  # repository context
    from tools.bench import run_serve_corpus as corpus
    from tools.bench.run_serve_graded import GradedSpec  # noqa: F401 (reuse)
    from tools.bench.run_graded_suite import (
        KV_DTYPES,
        REPO_ROOT,
        discover_artifacts,
        load_identity,
        serve_resolve,
        to_target_key,
    )
except ImportError:  # direct script execution
    from run_serve_corpus import (  # type: ignore[no-redef]
        CampaignError,
        RunningServer,
        SPECULATIVE_MODES,
        TARGET_MODEL_IDS,
    )
    from run_ninfer_bench_matrix import KV_DTYPES  # type: ignore[no-redef]
    from run_graded_suite import (  # type: ignore[no-redef]
        REPO_ROOT,
        discover_artifacts,
        load_identity,
        serve_resolve,
        to_target_key,
    )

    import run_serve_corpus as corpus  # type: ignore[no-redef]

DEFAULT_PORT = 8093


def parse_kv(text: str) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for piece in text.split(","):
        dtype = piece.strip().lower()
        if dtype not in KV_DTYPES:
            raise SystemExit(f"unsupported --kv-dtype {piece!r}; expected {','.join(KV_DTYPES)}")
        seen[dtype] = None
    if not seen:
        raise SystemExit("--kv-dtype selected nothing")
    return tuple(seen)


def parse_contexts(texts: Sequence[str]) -> tuple[int, ...]:
    values: list[int] = []
    for text in texts:
        for piece in text.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                value = int(piece)
            except ValueError:
                raise SystemExit(f"invalid --max-context value {piece!r}") from None
            if value < 1024:
                raise SystemExit("--max-context must be at least 1024")
            values.append(value)
    unique: list[int] = list(dict.fromkeys(values))
    if not unique:
        raise SystemExit("--max-context selected nothing")
    return tuple(unique)


def server_command(
    serve: Path,
    artifact: Path,
    model_id: str,
    port: int,
    max_context: int,
    kv_dtype: str,
    family: str,
    log_path: Path,
    device: int,
) -> list[str]:
    command = [
        str(serve),
        str(artifact),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model-id", model_id,
        "--max-context", str(max_context),
        "--prefill-chunk", "1024",
        "--log-stats-interval-ms", "0",
        "--device", str(device),
        "--request-log-jsonl", str(log_path),
        "--kv-dtype", kv_dtype,
        "--no-prefix-reuse",
        "--greedy",
    ]
    backend, tokens = corpus.SPECULATIVE_MODES.get(family, ("none", 0))
    if backend != "none":
        command += ["--spec", backend, "--draft-tokens", str(tokens), "--lm-head-draft"]
        if backend == "dflash":
            command += ["--no-cuda-graph"]
    return command


def probe_one(
    serve: Path,
    artifact: Path,
    model_id: str,
    port: int,
    max_context: int,
    kv_dtype: str,
    family: str,
    log_path: Path,
    device: int,
) -> dict:
    command = server_command(serve, artifact, model_id, port, max_context, kv_dtype, family,
                             log_path, device)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with corpus.RunningServer(command, "127.0.0.1", port, log_path) as server:
            start_event = server.wait_until_ready()
    except corpus.CampaignError as exc:
        return {"failed": str(exc)}
    load_seconds = time.monotonic() - started

    engine = start_event.get("engine", {})
    artifact_info = start_event.get("artifact", {})
    memory = start_event.get("memory", {})
    weights = memory.get("weights", {})
    sequence = memory.get("sequence", {})
    workspace = memory.get("workspace", {})
    request_transient = memory.get("request_transient", {})
    return {
        "artifact": artifact.name,
        "family": family,
        "kv_dtype": kv_dtype,
        "max_context": max_context,
        "model_id": model_id,
        "weights_id": artifact_info.get("weights_id"),
        "load_seconds": round(load_seconds, 3),
        "weights_capacity_bytes": weights.get("capacity_bytes"),
        "kv_payload_bytes": memory.get("kv_payload_bytes"),
        "runtime_reservation_bytes": memory.get("runtime_reservation_bytes"),
        "sequence_capacity_bytes": sequence.get("capacity_bytes"),
        "workspace_capacity_bytes": workspace.get("capacity_bytes"),
        "request_transient_capacity_bytes": request_transient.get("capacity_bytes"),
        "cuda_graph_allowance_bytes": memory.get("cuda_graph_allowance_bytes"),
        "available_after_weights_bytes": memory.get("available_after_weights_bytes"),
        "available_after_startup_bytes": memory.get("available_after_startup_bytes"),
        "total_device_memory_bytes": start_event.get("environment", {}).get(
            "total_device_memory_bytes"
        ),
        "kv_capacity": engine.get("kv_capacity"),
        "max_ctx_cfg": max_context,
        "gpu_name": start_event.get("environment", {}).get("gpu_name"),
    }


def human_gib(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value / (1024 ** 3):.2f} GiB"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--artifact-dir", action="append", type=Path, default=[])
    parser.add_argument("--family", default="mtp0",
                        choices=tuple(corpus.SPECULATIVE_MODES))
    parser.add_argument("--kv-dtype", default=",".join(KV_DTYPES))
    parser.add_argument("--max-context", action="append", required=True,
                        help="comma list of context lengths to sweep (repeatable)")
    parser.add_argument("--serve", type=Path, default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    serve = serve_resolve(args.serve)
    if args.dry_run:
        print(f"serve: {serve}")
        print(f"families: {args.family}")
        print(f"kv: {','.join(parse_kv(args.kv_dtype))}")
        print(f"contexts: {','.join(str(c) for c in parse_contexts(args.max_context))}")
        return 0

    if serve is None:
        raise SystemExit("ninfer-serve not found; pass --serve")

    artifacts: list[dict] = []
    if args.artifact:
        for value in args.artifact:
            name, sep, raw = value.partition("=")
            if not sep:
                raise SystemExit(f"invalid --artifact {value!r}; expected NAME=PATH")
            path = Path(raw).expanduser().resolve()
            if not path.is_file():
                raise SystemExit(f"--artifact {name!r}: file not found: {path}")
            identity = load_identity(path)
            artifacts.append({"name": name, "path": path,
                              "model_id": identity["model_id"], "weights_id": identity["weights_id"]})
    else:
        for entry in discover_artifacts(args.models_dir, args.artifact_dir):
            artifacts.append({"name": entry["name"], "path": entry["path"],
                              "model_id": entry["model_id"], "weights_id": entry["weights_id"]})

    kv_dtypes = parse_kv(args.kv_dtype)
    contexts = parse_contexts(args.max_context)
    rows: list[dict] = []

    print(f"probing {len(artifacts)} artifact(s) x {args.family} x {len(kv_dtypes)} kv "
          f"x {len(contexts)} contexts = {len(artifacts) * len(kv_dtypes) * len(contexts)} points",
          flush=True)

    port = args.port
    for entry in artifacts:
        for kv_dtype in kv_dtypes:
            for max_context in contexts:
                log_path = args.output / "logs" / (
                    f"probe_{entry['name']}_{args.family}_{kv_dtype}_{max_context}.jsonl"
                )
                row = probe_one(serve, entry["path"], entry["model_id"], port, max_context,
                                kv_dtype, args.family, log_path, args.device)
                row["artifact"] = entry["name"]
                rows.append(row)
                print(f"  {entry['name']}/{args.family}/{kv_dtype}/ctx={max_context}: "
                      f"weights={human_gib(row.get('weights_capacity_bytes'))} "
                      f"kv={human_gib(row.get('kv_payload_bytes'))} "
                      f"after_weights={human_gib(row.get('available_after_weights_bytes'))} "
                      f"after_startup={human_gib(row.get('available_after_startup_bytes'))}",
                      flush=True)
                port += 1

    args.output.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "artifact", "model_id", "weights_id", "family", "kv_dtype", "max_context",
        "backend_bytes", "kv_capacity", "speculative_backend", "speculative_draft_window",
        "proposal_head", "decode_path",
        "weights_capacity_bytes", "kv_payload_bytes", "runtime_reservation_bytes",
        "sequence_capacity_bytes", "workspace_capacity_bytes",
        "request_transient_capacity_bytes", "cuda_graph_allowance_bytes",
        "available_after_weights_bytes", "available_after_startup_bytes",
        "total_device_memory_bytes", "gpu_name", "failed",
    ]
    with (args.output / "memory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.output / 'memory.csv'} ({len(rows)} points)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except corpus.CampaignError as error:
        print(f"memory probe failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error