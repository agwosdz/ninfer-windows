#!/usr/bin/env python3
"""Graded real-text serving campaign across artifact, draft-mode, and KV axes.

This is the accuracy gate that the token-id engine matrix cannot provide: it
drives a real `ninfer-serve` process with short gradeable questions, scores each
completion against its expected answer (exact match), and reports accuracy next
to throughput for every selected configuration. Greedy sampling is the default
so an answer change between two KV storages or draft modes is a deterministic
signal, not sampling noise.

Datasets are vendored JSONL files (`bench/fixtures/graded/*.jsonl`) so campaigns
run offline and byte-stable; one JSON object per line:

    {"id": "...", "category": "...", "match": "auto|boxed|number|exact",
     "answer": "72", "question": "..."}

`match` selects the scoring strategy: `number` compares canonicalized numbers
(last number in the completion), `boxed` compares \\boxed{} content, `exact`
compares normalized whole answers, `auto` tries boxed, then number, then exact.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import http.client
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

try:  # repository/pytest context
    from tools.bench.run_ninfer_bench_matrix import KV_DTYPES
    from tools.bench import run_serve_corpus as corpus
except ImportError:  # direct script execution
    import run_ninfer_bench_matrix as _matrix  # type: ignore[no-redef]

    from run_ninfer_bench_matrix import KV_DTYPES  # type: ignore[no-redef]
    import run_serve_corpus as corpus  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_DATASET = REPO_ROOT / "bench/fixtures/graded/reasoning_seed.jsonl"
DEFAULT_MAX_CONTEXT = 32768
DEFAULT_MAX_TOKENS = 512

# Report names must match what the server logs in server_start events
# (src/serve/request_log.cpp).
KV_REPORT_NAMES = {
    "bf16": "bf16",
    "int8": "int8-group64",
    "rk8v4": "rk8v4",
    "rk4v4": "rk4v4",
    "rk4v4-e8": "rk4v4-e8",
    "rk2v4-e8": "rk2v4-e8",
}
MATCH_STRATEGIES = ("auto", "boxed", "number", "exact")


class GradedCampaignError(RuntimeError):
    pass


# --- answer grading -----------------------------------------------------------


def canonical_number(text: str) -> str | None:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return None
    value = match.group(0).replace(",", "").rstrip(".")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value or "0"


def extract_last_number(text: str) -> str | None:
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return canonical_number(matches[-1])


def extract_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth = 1
    index = start + len(marker)
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    if depth:
        return None
    content = text[start + len(marker) : index - 1].strip()
    return content or None


def normalize_exact(text: str) -> str:
    cleaned = re.sub(r"\s+", "", text.strip().lower())
    return cleaned.strip(".,;:!")


def grade_answer(expected: str, completion: str, strategy: str) -> dict[str, Any]:
    """Score one completion; returns correctness plus what was extracted."""
    expected_number = canonical_number(expected)
    extracted: str | None = None

    def boxed_matches() -> bool:
        nonlocal extracted
        boxed = extract_boxed(completion)
        if boxed is None:
            return False
        extracted = boxed
        if normalize_exact(boxed) == normalize_exact(expected):
            return True
        return expected_number is not None and canonical_number(boxed) == expected_number

    def number_matches() -> bool:
        nonlocal extracted
        last = extract_last_number(completion)
        if last is None or expected_number is None:
            return False
        extracted = last
        return last == expected_number

    def exact_matches() -> bool:
        nonlocal extracted
        extracted = completion.strip()
        return normalize_exact(completion) == normalize_exact(expected)

    if strategy == "auto":
        correct = boxed_matches() or number_matches() or exact_matches()
    elif strategy == "boxed":
        correct = boxed_matches()
    elif strategy == "number":
        correct = number_matches()
    elif strategy == "exact":
        correct = exact_matches()
    else:
        raise GradedCampaignError(f"unknown match strategy {strategy!r}")
    return {"correct": correct, "extracted": extracted}


# --- dataset loading ----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GradedItem:
    item_id: str
    category: str | None
    question: str
    answer: str
    match: str


def load_graded_items(paths: Sequence[Path], max_items: int | None) -> list[GradedItem]:
    items: list[GradedItem] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise GradedCampaignError(f"dataset file not found: {path}")
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GradedCampaignError(f"{path}:{line_number} is not valid JSONL: {exc}") from exc
            missing = [field for field in ("id", "question", "answer") if not row.get(field)]
            if missing:
                raise GradedCampaignError(f"{path}:{line_number} missing fields: {missing}")
            item_id = str(row["id"])
            if item_id in seen:
                continue
            seen.add(item_id)
            items.append(
                GradedItem(
                    item_id=item_id,
                    category=row.get("category"),
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    match=str(row.get("match", "auto")),
                )
            )
    for item in items:
        if item.match not in MATCH_STRATEGIES:
            raise GradedCampaignError(
                f"item {item.item_id!r} has unknown match strategy {item.match!r}; "
                f"expected {MATCH_STRATEGIES}"
            )
    if max_items is not None:
        items = items[:max_items]
    if not items:
        raise GradedCampaignError("no graded items were loaded")
    return items


# --- campaign planning --------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GradedSpec:
    target: str
    model_id: str
    artifact: Path
    mode_name: str
    speculative_backend: str
    draft_tokens: int
    kv_dtype: str
    sampling_mode: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.target, self.mode_name, self.kv_dtype, self.sampling_mode)


def plan_configs(
    artifacts: Sequence[tuple[str, Path]],
    modes: Sequence[str],
    kv_dtypes: Sequence[str],
    sampling_mode: str,
) -> list[GradedSpec]:
    specs: list[GradedSpec] = []
    for target, artifact in artifacts:
        for mode_name in modes:
            backend, draft_tokens = corpus.SPECULATIVE_MODES[mode_name]
            for kv_dtype in kv_dtypes:
                specs.append(
                    GradedSpec(
                        target=target,
                        model_id=corpus.TARGET_MODEL_IDS[target],
                        artifact=artifact,
                        mode_name=mode_name,
                        speculative_backend=backend,
                        draft_tokens=draft_tokens,
                        kv_dtype=kv_dtype,
                        sampling_mode=sampling_mode,
                    )
                )
    return specs


def graded_request_payload(spec: GradedSpec, item: GradedItem, seed: int, max_tokens: int) -> dict[str, Any]:
    return {
        "model": spec.model_id,
        "messages": [{"role": "user", "content": item.question}],
        "max_completion_tokens": max_tokens,
        "seed": seed,
        "stream": False,
        "enable_thinking": False,
    }


WARMUP_PAYLOAD = {
    "messages": [{"role": "user", "content": "Reply with exactly one word: ready"}],
}


def graded_server_command(
    serve: Path,
    spec: GradedSpec,
    log_path: Path,
    port: int,
    device: int,
    max_context: int,
) -> list[str]:
    command = [
        str(serve),
        str(spec.artifact),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        spec.model_id,
        "--max-context",
        str(max_context),
        "--prefill-chunk",
        "1024",
        "--log-stats-interval-ms",
        "0",
        "--device",
        str(device),
        "--request-log-jsonl",
        str(log_path),
        "--kv-dtype",
        spec.kv_dtype,
        "--no-prefix-reuse",
    ]
    if spec.speculative_backend != "none":
        command.extend(
            [
                "--spec",
                spec.speculative_backend,
                "--draft-tokens",
                str(spec.draft_tokens),
                "--lm-head-draft",
            ]
        )
    if spec.sampling_mode == "greedy":
        command.append("--greedy")
    else:
        # Same explicit stochastic profile as the published serving campaigns.
        command.extend(
            [
                "--temperature",
                "0.6",
                "--top-p",
                "0.95",
                "--top-k",
                "20",
                "--min-p",
                "0",
                "--presence-penalty",
                "1.0",
                "--frequency-penalty",
                "0",
            ]
        )
    return command


def validate_graded_server_start(
    event: dict[str, Any], spec: GradedSpec, device: int, max_context: int
) -> tuple[str, str]:
    corpus.require_server_log_identity(event, "server_start")
    engine = event.get("engine", {})
    expected_engine = {
        "device": device,
        "max_context": max_context,
        "kv_capacity": max_context,
        "prefill_chunk": 1024,
        "kv_cache": KV_REPORT_NAMES[spec.kv_dtype],
        "cuda_graph": True,
        "prefix_reuse": False,
        "speculative_backend": spec.speculative_backend,
        "speculative_draft_window": spec.draft_tokens,
        "proposal_head": "optimized" if spec.draft_tokens else "full",
    }
    if engine != expected_engine:
        raise GradedCampaignError(f"server_start Engine configuration mismatch: {engine!r}")
    weights_id = event.get("artifact", {}).get("weights_id")
    if not isinstance(weights_id, str) or not weights_id:
        raise GradedCampaignError("server_start has no canonical artifact weights_id")
    server_instance_id = event.get("server_instance_id")
    if not isinstance(server_instance_id, str) or not server_instance_id:
        raise GradedCampaignError("server_start has no server_instance_id")
    return server_instance_id, weights_id


def completion_text(response: dict[str, Any], item_id: str) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise GradedCampaignError(f"completion for {item_id!r} has no choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise GradedCampaignError(f"completion for {item_id!r} has no message content")
    return content


def run_graded_config(
    serve: Path,
    spec: GradedSpec,
    items: Sequence[GradedItem],
    output_dir: Path,
    *,
    port: int,
    device: int,
    max_context: int,
    max_tokens: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one configuration; returns (item records, failures)."""
    log_path = output_dir / f"serve_{spec.target}_{spec.mode_name}_{spec.kv_dtype}.jsonl"
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    command = graded_server_command(serve, spec, log_path, port, device, max_context)
    try:
        with corpus.RunningServer(command, "127.0.0.1", port, log_path) as server:
            start_event = server.wait_until_ready()
            instance_id, weights_id = validate_graded_server_start(
                start_event, spec, device, max_context
            )
            connection = http.client.HTTPConnection(
                "127.0.0.1", port, timeout=corpus.REQUEST_TIMEOUT_SECONDS
            )
            try:
                warmup = dict(WARMUP_PAYLOAD)
                warmup.update(
                    {
                        "model": spec.model_id,
                        "max_completion_tokens": 16,
                        "seed": seed,
                        "stream": False,
                        "enable_thinking": False,
                    }
                )
                corpus.post_json(connection, warmup)
                for item in items:
                    payload = graded_request_payload(spec, item, seed, max_tokens)
                    started = time.perf_counter()
                    response = corpus.post_json(connection, payload)
                    wall_seconds = time.perf_counter() - started
                    usage = response.get("usage", {}) or {}
                    completion = completion_text(response, item.item_id)
                    verdict = grade_answer(item.answer, completion, item.match)
                    records.append(
                        {
                            "target": spec.target,
                            "weights_id": weights_id,
                            "mode": spec.mode_name,
                            "kv_dtype": spec.kv_dtype,
                            "sampling": spec.sampling_mode,
                            "item_id": item.item_id,
                            "category": item.category,
                            "correct": verdict["correct"],
                            "extracted": verdict["extracted"],
                            "expected": item.answer,
                            "completion_tokens": usage.get("completion_tokens"),
                            "wall_seconds": round(wall_seconds, 6),
                        }
                    )
            finally:
                connection.close()
    except corpus.CampaignError as exc:
        failures.append({"config": spec.key, "error": str(exc)})
    except GradedCampaignError as exc:
        failures.append({"config": spec.key, "error": str(exc)})
    return records, failures


def aggregate_config(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for record in records if record["correct"])
    wall = sum(float(record["wall_seconds"]) for record in records)
    tokens = sum(int(record["completion_tokens"] or 0) for record in records)
    return {
        "items": total,
        "accuracy": round(correct / total, 4) if total else None,
        "wall_seconds_mean": round(wall / total, 4) if total else None,
        "output_tok_s": round(tokens / wall, 2) if wall > 0 else None,
    }


# --- output -------------------------------------------------------------------


def write_outputs(
    output_dir: Path,
    records: Sequence[dict[str, Any]],
    aggregates: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
) -> None:
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    fieldnames = [
        "target",
        "weights_id",
        "mode",
        "kv_dtype",
        "sampling",
        "items",
        "accuracy",
        "wall_seconds_mean",
        "output_tok_s",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregates)
    headers = ["target", "mode", "kv", "sampling", "items", "acc%", "wall s/item", "tok/s"]
    rows = [
        [
            row["target"],
            row["mode"],
            row["kv_dtype"],
            row["sampling"],
            str(row["items"]),
            "-" if row["accuracy"] is None else f"{row['accuracy'] * 100:.1f}",
            "-" if row["wall_seconds_mean"] is None else f"{row['wall_seconds_mean']:.3f}",
            "-" if row["output_tok_s"] is None else f"{row['output_tok_s']:.1f}",
        ]
        for row in aggregates
    ]
    table = ["| " + " | ".join(headers) + " | ", "| " + " | ".join("---" for _ in headers) + " | "]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    report = "# Graded serving campaign\n\n```\n" + "\n".join(table) + "\n```\n"
    if failures:
        report += "\n## Failures\n\n```json\n" + json.dumps(failures, indent=2) + "\n```\n"
    (output_dir / "summary.md").write_text(report, encoding="utf-8")


# --- entry point --------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=Path, default=REPO_ROOT / "build/apps/ninfer-serve")
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="TARGET=PATH",
        help="artifact for a registered target; repeat to sweep multiple targets",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=tuple(corpus.SPECULATIVE_MODES),
        help="speculative mode; repeatable (default: mtp0 and mtp3)",
    )
    parser.add_argument(
        "--kv-dtype",
        default=",".join(("bf16", "int8")),
        help=f"comma list among {','.join(KV_DTYPES)} (default: bf16,int8)",
    )
    parser.add_argument(
        "--sampling",
        choices=("greedy", "stochastic"),
        default="greedy",
        help="greedy makes answer deltas deterministic (default)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        type=Path,
        metavar="JSONL",
        help="graded dataset file; repeatable (default: bench/fixtures/graded/reasoning_seed.jsonl)",
    )
    parser.add_argument("--max-items", type=int, default=None, help="cap on distinct graded items")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    parser.add_argument("--seed", type=int, default=corpus.SEEDS[0])
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="campaign output directory")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without serving")
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.max_context < 1024:
        parser.error("--max-tokens must be positive and --max-context at least 1024")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = corpus.parse_artifacts(args.artifact)
    modes = tuple(args.mode or ("mtp0", "mtp3"))
    kv_dtypes = _parse_kv_subset(args.kv_dtype)
    dataset_paths = list(args.dataset) or [DEFAULT_SEED_DATASET]
    items = load_graded_items(dataset_paths, args.max_items)
    specs = plan_configs(artifacts, modes, kv_dtypes, args.sampling)

    print(f"configs : {len(specs)} ({len(artifacts)} artifacts x {len(modes)} modes x "
          f"{len(kv_dtypes)} kv x [{args.sampling}])")
    print(f"items   : {len(items)} from {len(dataset_paths)} dataset file(s)")
    if args.sampling != "greedy":
        print("warning: stochastic sampling makes single-sample accuracy noisy; "
              "use greedy for compression A/B decisions")

    if args.dry_run:
        sample = graded_request_payload(specs[0], items[0], args.seed, args.max_tokens)
        print("sample payload:", json.dumps(sample, ensure_ascii=False))
        for spec in specs:
            command = graded_server_command(
                args.serve.expanduser().resolve(),
                spec,
                args.output / f"serve_{spec.target}_{spec.mode_name}_{spec.kv_dtype}.jsonl",
                args.port,
                args.device,
                args.max_context,
            )
            print("serve:", " ".join(str(part) for part in command[:6]), "...")
        return 0

    serve = args.serve.expanduser().resolve()
    if not serve.is_file():
        raise SystemExit(f"ninfer-serve executable not found: {serve}")
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        label = f"{spec.target}/{spec.mode_name}/{spec.kv_dtype}/{spec.sampling}"
        print(f"[{index}/{len(specs)}] {label} ...", flush=True)
        records, failures = run_graded_config(
            serve,
            spec,
            items,
            output_dir,
            port=args.port,
            device=args.device,
            max_context=args.max_context,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        all_records.extend(records)
        all_failures.extend(failures)
        summary = {"target": spec.target, "weights_id": None, "mode": spec.mode_name,
                   "kv_dtype": spec.kv_dtype, "sampling": spec.sampling_mode}
        summary.update(aggregate_config(records))
        summary["weights_id"] = records[0]["weights_id"] if records else None
        aggregates.append(summary)
        accuracy = summary["accuracy"]
        print(f"    accuracy={'' if accuracy is None else format(accuracy * 100, '.1f') + '%'} "
              f"items={summary['items']} tok/s={summary['output_tok_s']}")

    write_outputs(output_dir, all_records, aggregates, all_failures)
    print(f"results : {output_dir / 'results.jsonl'}")
    print(f"summary : {output_dir / 'summary.csv'}")
    return 1 if all_failures else 0


def _parse_kv_subset(text: str) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for piece in text.split(","):
        dtype = piece.strip().lower()
        if dtype not in KV_DTYPES:
            allowed = ", ".join(KV_DTYPES)
            raise SystemExit(f"unsupported --kv-dtype {piece!r}; expected {allowed}")
        seen[dtype] = None
    if not seen:
        raise SystemExit("--kv-dtype selected nothing")
    return tuple(seen)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except corpus.CampaignError as error:
        print(f"graded campaign failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
