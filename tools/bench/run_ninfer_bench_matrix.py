#!/usr/bin/env python3
"""Run the native NInfer product performance matrix.

The matrix sweeps three experiment axes on top of layered test shapes instead of
a fully factorial campaign:

* Draft models: speculative modes named like the serving campaigns
  (mtp0, mtp1..mtp5, dflash7). Core presets compare mtp0 baseline, the mtp3
  primary path, and the dflash7 DFlash round; mtp0..mtp5 is additionally swept
  on representative context-decode cases for MTP window tuning.
* Variants: repeatable ``--variant NAME=PATH`` artifacts (for example the
  registered qwen3.8-27b groupwise-int, nvfp4, and groupwise-int-dflash2
  profiles). Every case runs once per selected variant.
* KV-cache compression: ``--kv-dtype`` sweeps the registered storages
  (bf16, int8, rk8v4, rk4v4, rk4v4-e8, rk2v4-e8) for every case.

Run ``--pick`` for an interactive one-shot campaign: it lists discovered
.ninfer artifacts (repo root, out/, plus repeatable ``--artifact-dir``
directories), speculative draft modes, KV compressions, and benchmark suites,
then executes the full cross product of the selections through the normal
matrix path.

Raw ninfer_bench reports stay under profiles/bench. This script writes a
descriptive manifest, exact commands, per-case logs, raw JSON reports, and a flat
summary CSV/JSON that is easy to compare across configurations.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = REPO_ROOT / "build/bench/ninfer_bench"
DEFAULT_WEIGHTS = REPO_ROOT / "out/qwen3_6_27b.ninfer"
DEFAULT_CORPUS = REPO_ROOT / "bench/fixtures/bench_corpus.ids"

PREFILL_LENGTHS_CORE = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
PREFILL_LENGTHS_FULL_EXTRA = (32768, 65536)
PREFILL_CHUNKS = (128, 256, 512, 1024, 2048, 4096)
PURE_DECODE_GENS = (16, 64, 128, 512, 2048)
CONTEXT_CORE = ((512, 512), (2048, 512), (8192, 512))
CONTEXT_FULL_EXTRA = ((32768, 256), (65536, 128))

# Speculative modes use the run_serve_corpus.py vocabulary so engine-level and
# serving-level results join on one set of names.
DRAFT_MODES: dict[str, tuple[str, int]] = {
    "mtp0": ("none", 0),
    "mtp1": ("mtp", 1),
    "mtp2": ("mtp", 2),
    "mtp3": ("mtp", 3),
    "mtp4": ("mtp", 4),
    "mtp5": ("mtp", 5),
    "dflash7": ("dflash", 7),
}
PRIMARY_DRAFTS = ("mtp0", "mtp3", "dflash7")
MTP_SWEEP_DRAFTS = ("mtp0", "mtp1", "mtp2", "mtp3", "mtp4", "mtp5")

# KV storages use the product CLI/serve spellings (apps/cli/options.cpp). The
# rk* entries are the rotated compressed-KV line: rotated int8 keys with int4
# values, packed int4 keys+values, and their E8-lattice / E8-root variants.
KV_DTYPES = ("bf16", "int8", "rk8v4", "rk4v4", "rk4v4-e8", "rk2v4-e8")
KV_DESCRIPTIONS = {
    "bf16": "BF16 KV storage (baseline)",
    "int8": "INT8 group-64",
    "rk8v4": "rotated INT8 keys + INT4 values group-64",
    "rk4v4": "rotated packed INT4 keys + INT4 values group-64",
    "rk4v4-e8": "INT4 K/V with E8-lattice codes",
    "rk2v4-e8": "INT4 K/V with E8-root codes",
}

# Near-capacity stress prompts are pinned corpus offsets per speculative mode so
# repeated campaigns measure identical request shapes.
TAIL_PROMPT_OFFSETS: dict[tuple[str, int], int] = {
    ("mtp", 3): 8174,
    ("mtp", 5): 8170,
    ("dflash", 7): 8162,
}
REPORT_SCHEMA_VERSION = 12
REPORT_ARTIFACT_TYPE = "ninfer_bench_report"
REPORT_TOOL = "ninfer_bench"


@dataclasses.dataclass(frozen=True)
class BenchCase:
    suite: str
    name: str
    args: tuple[str, ...]
    repetitions: int
    warmup: int
    notes: str = ""


def csv_list(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def pair_list(values: Iterable[tuple[int, int]]) -> str:
    return ";".join(f"{p},{g}" for p, g in values)


def draft_spec(mode: str) -> tuple[str, int]:
    return DRAFT_MODES[mode]


def draft_tag(mode: str) -> str:
    backend, tokens = draft_spec(mode)
    if backend == "none":
        return "k0"
    prefix = "df" if backend == "dflash" else "k"
    return f"{prefix}{tokens}"


def uses_cuda_graph(mode: str) -> bool:
    # The qwen3.8-27B DFlash2 decode round runs eager by target contract; keep
    # requested options and reported decode_path truthful instead of letting the
    # Engine silently downgrade a graph request.
    return draft_spec(mode)[0] != "dflash"


def draft_args(mode: str) -> tuple[str, ...]:
    """Bench flags for one speculative mode; mirrors the serve campaign flags."""
    backend, tokens = draft_spec(mode)
    if backend == "none":
        return ()
    args = ("--spec", backend, "--draft-tokens", str(tokens), "--lm-head-draft")
    if not uses_cuda_graph(mode):
        args += ("--no-cuda-graph",)
    return args


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")


def count_corpus_tokens(path: Path) -> int:
    if not path.is_file():
        raise SystemExit(f"corpus file not found: {path}")
    return len(path.read_text(encoding="utf-8").split())


def add_repetition_args(
    base_args: list[str], case: BenchCase, repetitions_override: int | None,
    warmup_override: int | None,
) -> list[str]:
    repetitions = repetitions_override if repetitions_override is not None else case.repetitions
    warmup = warmup_override if warmup_override is not None else case.warmup
    return [*base_args, "-r", str(repetitions), "--warmup", str(warmup)]


def graph_suffix(mode: str, graph: bool) -> str:
    if not uses_cuda_graph(mode):
        return "eager"
    return "graph" if graph else "eager"


def build_cases(preset: str, drafts: Sequence[str]) -> list[BenchCase]:
    if preset == "smoke":
        return [
            BenchCase(
                "prefill_length",
                f"prefill_p128_{draft_tag('mtp0')}",
                ("-p", "128", *draft_args("mtp0")),
                1,
                0,
            ),
            BenchCase(
                "pure_decode",
                f"tg8_{draft_tag('mtp3')}_graph",
                ("-n", "8", *draft_args("mtp3")),
                1,
                0,
            ),
            BenchCase(
                "context_decode",
                f"ctx_p128_g8_{draft_tag('dflash7')}_eager",
                ("-pg", "128,8", "--max-ctx", "256", *draft_args("dflash7")),
                1,
                0,
            ),
        ]

    include_full = preset == "full"
    prefill_lengths = PREFILL_LENGTHS_CORE + (PREFILL_LENGTHS_FULL_EXTRA if include_full else ())
    context_pairs = CONTEXT_CORE + (CONTEXT_FULL_EXTRA if include_full else ())
    sweep_pairs = ((2048, 512),) + (((32768, 256),) if include_full else ())

    cases: list[BenchCase] = []

    for mode in drafts:
        cases.append(
            BenchCase(
                "prefill_length",
                f"prefill_lengths_{draft_tag(mode)}",
                ("-p", csv_list(prefill_lengths), *draft_args(mode)),
                3,
                1,
                "prefill length curve",
            )
        )

    for mode in drafts:
        for chunk in PREFILL_CHUNKS:
            cases.append(
                BenchCase(
                    "prefill_chunk",
                    f"prefill_p8192_chunk{chunk}_{draft_tag(mode)}",
                    (
                        "-p",
                        "8192",
                        "--prefill-chunk",
                        str(chunk),
                        *draft_args(mode),
                    ),
                    3,
                    1,
                    "workspace and chunk-size sensitivity",
                )
            )

    for mode in drafts:
        for graph in ((True, False) if uses_cuda_graph(mode) else (False,)):
            suffix = graph_suffix(mode, graph)
            args = ["-n", csv_list(PURE_DECODE_GENS), *draft_args(mode)]
            if not graph:
                args.append("--no-cuda-graph")
            cases.append(
                BenchCase(
                    "pure_decode",
                    f"tg_lengths_{draft_tag(mode)}_{suffix}",
                    tuple(args),
                    5,
                    1,
                    "pure decode throughput; tg seeds only one token",
                )
            )

    for mode in drafts:
        for graph in ((True, False) if uses_cuda_graph(mode) else (False,)):
            suffix = graph_suffix(mode, graph)
            args = ["-pg", pair_list(context_pairs), *draft_args(mode)]
            if not graph and uses_cuda_graph(mode):
                args.append("--no-cuda-graph")
            cases.append(
                BenchCase(
                    "context_decode",
                    f"context_decode_{draft_tag(mode)}_{suffix}",
                    tuple(args),
                    3,
                    1,
                    "decode at real context offsets",
                )
            )

    for mode in MTP_SWEEP_DRAFTS:
        cases.append(
            BenchCase(
                "mtp_sweep",
                f"mtp_sweep_{draft_tag(mode)}_graph",
                ("-pg", pair_list(sweep_pairs), *draft_args(mode)),
                3,
                1,
                "primary MTP draft-window sweep",
            )
        )

    for mode in drafts:
        spec = draft_spec(mode)
        prompt_offset = TAIL_PROMPT_OFFSETS.get(spec)
        if prompt_offset is None:
            continue
        for graph in ((True, False) if uses_cuda_graph(mode) else (False,)):
            suffix = graph_suffix(mode, graph)
            args = [
                "-pg",
                f"{prompt_offset},12",
                "--max-ctx",
                "8192",
                *draft_args(mode),
            ]
            if not graph and uses_cuda_graph(mode):
                args.append("--no-cuda-graph")
            cases.append(
                BenchCase(
                    "tail_stress",
                    f"tail_{draft_tag(mode)}_{suffix}",
                    tuple(args),
                    3,
                    1,
                    "near-capacity fallback stress",
                )
            )

    return cases


def filtered_cases(cases: list[BenchCase], suites: Sequence[str], limit: int | None) -> list[BenchCase]:
    selected = cases
    if suites:
        allowed = set(suites)
        selected = [case for case in selected if case.suite in allowed]
    if limit is not None:
        selected = selected[:limit]
    return selected


def max_prompt_in_cases(cases: Sequence[BenchCase]) -> int:
    max_prompt = 0
    for case in cases:
        args = list(case.args)
        for flag in ("-p", "--n-prompt"):
            if flag in args:
                raw = args[args.index(flag) + 1]
                max_prompt = max(max_prompt, *(int(piece) for piece in raw.split(",")))
        for flag in ("-pg", "--prompt-gen"):
            if flag in args:
                raw = args[args.index(flag) + 1]
                for pair in raw.split(";"):
                    p, _ = pair.split(",", 1)
                    max_prompt = max(max_prompt, int(p))
    return max_prompt


def parse_variant(value: str, seen: dict[str, Path]) -> None:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise SystemExit(f"invalid --variant value {value!r}; expected NAME=PATH")
    if name in seen:
        raise SystemExit(f"duplicate variant name: {name}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"artifact not found for variant {name!r}: {path}")
    seen[name] = path


def parse_variants(values: Sequence[str], default_weights: Path) -> tuple[tuple[str, Path], ...]:
    seen: dict[str, Path] = {}
    for value in values:
        parse_variant(value, seen)
    if not seen:
        if not default_weights.is_file():
            raise SystemExit(f"weights file not found: {default_weights}")
        return (("default", default_weights),)
    return tuple(seen.items())


def parse_kv_dtypes(text: str | None) -> tuple[str, ...]:
    if not text:
        return ("bf16",)
    seen: dict[str, None] = {}
    for piece in text.split(","):
        dtype = piece.strip().lower()
        if dtype not in KV_DTYPES:
            allowed = ", ".join(KV_DTYPES)
            raise SystemExit(f"unsupported --kv-dtype {piece!r}; expected {allowed}")
        seen[dtype] = None
    return tuple(seen)


def parse_drafts(text: str | None) -> tuple[str, ...]:
    if not text:
        return PRIMARY_DRAFTS
    seen: dict[str, None] = {}
    for piece in text.split(","):
        mode = piece.strip().lower()
        if mode not in DRAFT_MODES:
            allowed = ", ".join(DRAFT_MODES)
            raise SystemExit(f"unknown draft mode {piece!r}; expected {allowed}")
        seen[mode] = None
    return tuple(seen)


# --- interactive picker (--pick) --------------------------------------------


def split_pick_args(argv: Sequence[str]) -> tuple[list[Path], list[str]]:
    """Extract wizard-only flags; everything else passes through to main()."""
    dirs: list[Path] = []
    rest: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--artifact-dir":
            if index + 1 >= len(argv):
                raise SystemExit("--artifact-dir needs a path")
            dirs.append(Path(argv[index + 1]).expanduser().resolve())
            index += 2
        else:
            rest.append(argv[index])
            index += 1
    return dirs, rest


def describe_artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        size_gib = resolved.stat().st_size / (1024**3)
    except OSError:
        size_gib = 0.0
    model_id = ""
    weights_id = ""
    conversion_path = Path(str(resolved) + ".conversion.json")
    try:
        report = json.loads(conversion_path.read_text(encoding="utf-8"))
        identity = report.get("identity", {})
        model_id = str(identity.get("model_id", ""))
        weights_id = str(identity.get("weights_id", ""))
    except (OSError, json.JSONDecodeError):
        pass
    identity_text = f"{model_id or '?'}/{weights_id or '?'}"
    return {
        "path": resolved,
        "name": resolved.name,
        "model_id": model_id,
        "weights_id": weights_id,
        "label": f"{resolved.name}  [{identity_text}]  {size_gib:.1f} GiB",
    }


def discover_artifacts(extra_dirs: Sequence[Path] = ()) -> list[dict[str, Any]]:
    roots = [REPO_ROOT, REPO_ROOT / "out", *extra_dirs]
    seen: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.ninfer")):
            if path.is_file():
                described = describe_artifact(path)
                seen.setdefault(str(described["path"]), described)
    return [seen[key] for key in sorted(seen)]


def sanitize_variant_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", name.strip().lower()).strip("_")
    return cleaned or "artifact"


def assign_variant_names(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    used: set[str] = set()
    named = []
    for entry in entries:
        base = sanitize_variant_name(Path(entry["name"]).stem)
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        named.append({**entry, "variant": name})
    return named


def parse_index_selection(text: str, size: int, *, default: Sequence[int] | None = None) -> list[int]:
    text = text.strip()
    if not text:
        if default is None:
            raise SystemExit("a selection is required")
        return sorted(set(default))
    if text.lower() in {"a", "all"}:
        return list(range(size))
    picked: set[int] = set()
    for token in text.replace(",", " ").split():
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if not match:
            raise SystemExit(f"invalid selection token {token!r}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if start < 1 or end < start or end > size:
            raise SystemExit(f"selection {token!r} outside 1..{size}")
        picked.update(range(start - 1, end))
    return sorted(picked)


def _prompt_selection(title: str, labels: Sequence[str], *, default: Sequence[int]) -> list[int]:
    print(title)
    for number, label in enumerate(labels, start=1):
        print(f"  [{number}] {label}")
    hint = f"comma/range/a=all/Enter=default({len(default)}) "
    while True:
        try:
            raw = input(hint + ": ")
        except EOFError:
            raise SystemExit("--pick needs interactive input; use explicit flags instead") from None
        try:
            return parse_index_selection(raw, len(labels), default=default)
        except SystemExit as error:
            print(f"  {error}")


def assemble_pick_argv(
    preset: str,
    suites: Sequence[str],
    drafts: Sequence[str],
    kv_dtypes: Sequence[str],
    variants: Sequence[dict[str, Any]],
) -> list[str]:
    argv = ["--preset", preset]
    for suite in suites:
        argv += ["--suite", suite]
    argv += ["--drafts", ",".join(drafts), "--kv-dtype", ",".join(kv_dtypes)]
    for variant in variants:
        argv += ["--variant", f"{variant['variant']}={variant['path']}"]
    return argv


def plan_with_picker(extra_dirs: Sequence[Path] = ()) -> list[str]:
    entries = assign_variant_names(discover_artifacts(extra_dirs))
    if not entries:
        raise SystemExit(
            "no .ninfer artifacts found in repo root, out/, or --artifact-dir directories"
        )

    print(f"Found {len(entries)} artifact(s):")
    picked = _prompt_selection(
        "\nArtifacts to benchmark:",
        [entry["label"] for entry in entries],
        default=range(len(entries)),
    )
    variants = [entries[index] for index in picked]

    draft_labels = [
        f"{mode:<9} {draft_spec(mode)[1]} - "
        + ("no speculation (baseline)" if draft_spec(mode)[0] == "none" else
           "DFlash block=8 round (needs the groupwise-int-dflash2 profile)"
           if draft_spec(mode)[0] == "dflash" else
           f"MTP draft window {draft_spec(mode)[1]}")
        for mode in DRAFT_MODES
    ]
    draft_default = [list(DRAFT_MODES).index(mode) for mode in PRIMARY_DRAFTS]
    draft_picked = _prompt_selection("\nSpeculative draft modes:", draft_labels, default=draft_default)
    drafts = tuple(list(DRAFT_MODES)[index] for index in draft_picked)

    kv_picked = _prompt_selection(
        "\nKV-cache compression:",
        [f"{dtype:<10} {KV_DESCRIPTIONS[dtype]}" for dtype in KV_DTYPES],
        default=range(len(KV_DTYPES)),
    )
    kv_selected = tuple(KV_DTYPES[index] for index in kv_picked)

    preset_picked = _prompt_selection(
        "\nBenchmark preset:",
        [
            "smoke   - three quick shapes, sanity check",
            "core    - published-length curves (default)",
            "full    - core plus 32k/64k context lengths",
        ],
        default=[1],
    )[0]
    presets = ("smoke", "core", "full")
    preset = presets[preset_picked]

    all_cases = build_cases(preset, drafts)
    suite_counts: dict[str, int] = {}
    for case in all_cases:
        suite_counts[case.suite] = suite_counts.get(case.suite, 0) + 1
    suites = tuple(suite_counts)
    suite_picked = _prompt_selection(
        "\nSuites to run:", [f"{s:<15} {suite_counts[s]} shape(s)" for s in suites],
        default=range(len(suites)),
    )
    selected_suites = tuple(suites[index] for index in suite_picked)

    selected_cases = filtered_cases(all_cases, list(selected_suites), None)
    points = len(selected_cases) * len(kv_selected) * len(variants)

    if any(draft_spec(mode)[0] == "dflash" for mode in drafts):
        if not any("dflash" in entry["weights_id"] for entry in variants):
            print(
                "\nwarning: dflash modes were selected but no selected artifact carries a "
                "dflash weights profile; those combinations will fail at Engine startup and be "
                "recorded in failures.json"
            )

    print("\nCampaign plan:")
    print(f"  artifacts : {', '.join(entry['name'] for entry in variants)}")
    print(f"  drafts    : {', '.join(drafts)}")
    print(f"  kv        : {', '.join(kv_selected)}")
    print(f"  preset    : {preset}")
    print(f"  suites    : {', '.join(selected_suites)}")
    print(f"  points    : {points} ({len(selected_cases)} shapes x {len(kv_selected)} kv x "
          f"{len(variants)} variants)")

    try:
        confirm = input("\nStart the campaign? [Y/n]: ").strip().lower()
    except EOFError:
        raise SystemExit("--pick needs interactive input; use explicit flags instead") from None
    if confirm not in {"", "y", "yes"}:
        raise SystemExit("campaign cancelled")

    return assemble_pick_argv(preset, selected_suites, drafts, kv_selected, variants)


def load_bench_report(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("benchmark report root must be an object")
    identity = (
        report.get("schema_version"),
        report.get("artifact_type"),
        report.get("tool"),
    )
    expected = (REPORT_SCHEMA_VERSION, REPORT_ARTIFACT_TYPE, REPORT_TOOL)
    if identity != expected:
        raise ValueError(
            "unsupported benchmark report identity: "
            f"schema_version={identity[0]!r}, artifact_type={identity[1]!r}, "
            f"tool={identity[2]!r}; expected {expected!r}"
        )
    return report


def report_rows(
    report_path: Path,
    case: BenchCase,
    axes: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    report = load_bench_report(report_path)
    config = report.get("config", {})
    load = report.get("load", {})
    memory = report.get("memory", {})
    weights_memory = memory.get("weights", {})
    sequence_memory = memory.get("sequence", {})
    workspace_memory = memory.get("workspace", {})
    request_transient_memory = memory.get("request_transient", {})
    rows = []
    for test in report.get("tests", []):
        speculative = test.get("speculative", {})
        row = {
            "suite": case.suite,
            "case": case.name,
            "report": str(report_path),
            "label": test.get("label"),
            "kind": test.get("kind"),
            "n_prompt": test.get("n_prompt"),
            "n_gen": test.get("n_gen"),
            "requested_output_tokens": test.get("requested_output_tokens"),
            "target": load.get("target"),
            "weights_id": load.get("weights_id"),
            "artifact_path": report.get("artifact", {}).get("path"),
            "max_context": config.get("max_context"),
            "kv_capacity": memory.get("kv_capacity"),
            "prefill_chunk": config.get("prefill_chunk"),
            "kv_cache": config.get("kv_cache"),
            "speculative_backend": config.get("speculative_backend"),
            "draft_tokens": config.get("draft_tokens"),
            "proposal_head": config.get("proposal_head"),
            "decode_path": config.get("decode_path"),
            "decode_graph_primed": config.get("decode_graph_prime", {}).get("primed"),
            "decode_graph_prime_output_tokens": config.get("decode_graph_prime", {}).get(
                "output_tokens"
            ),
            "repetitions": config.get("repetitions"),
            "warmup": config.get("warmup"),
            "load_seconds": load.get("load_seconds"),
            "upload_seconds": load.get("upload_seconds"),
            "artifact_bytes_read": load.get("artifact_bytes_read"),
            "host_to_device_bytes": load.get("host_to_device_bytes"),
            "peak_staging_bytes": load.get("peak_staging_bytes"),
            "kv_payload_bytes": memory.get("kv_payload_bytes"),
            "weights_capacity_bytes": weights_memory.get("capacity_bytes"),
            "sequence_capacity_bytes": sequence_memory.get("capacity_bytes"),
            "workspace_capacity_bytes": workspace_memory.get("capacity_bytes"),
            "request_transient_capacity_bytes": request_transient_memory.get("capacity_bytes"),
            "cuda_graph_allowance_bytes": memory.get("cuda_graph_allowance_bytes"),
            "workspace_peak_bytes": test.get("workspace_peak_bytes"),
            "workspace_allocator_peak_bytes": test.get("workspace_allocator_peak_bytes"),
            "prefill_tok_s_mean": test.get("prefill_tok_s_mean"),
            "prefill_tok_s_stddev": test.get("prefill_tok_s_stddev"),
            "decode_output_tok_s_mean": test.get("decode_output_tok_s_mean"),
            "decode_output_tok_s_stddev": test.get("decode_output_tok_s_stddev"),
            "decode_engine_tok_s_mean": test.get("decode_engine_tok_s_mean"),
            "decode_engine_tok_s_stddev": test.get("decode_engine_tok_s_stddev"),
            "prepare_seconds_mean": test.get("prepare_seconds_mean"),
            "prefill_seconds_mean": test.get("prefill_seconds_mean"),
            "decode_seconds_mean": test.get("decode_seconds_mean"),
            "total_seconds_mean": test.get("total_seconds_mean"),
            "spec_acceptance_rate": speculative.get("acceptance_rate"),
            "spec_acceptance_length": speculative.get("acceptance_length"),
            "spec_rounds": speculative.get("rounds"),
            "spec_drafted_tokens": speculative.get("drafted_tokens"),
            "spec_accepted_tokens": speculative.get("accepted_tokens"),
            "spec_fallback_steps": speculative.get("fallback_steps"),
            "spec_accepted_per_position": json.dumps(
                speculative.get("accepted_per_position", []), separators=(",", ":")
            ),
            "gpu_name": report.get("environment", {}).get("gpu_name"),
        }
        if axes:
            row.update(axes)
        rows.append(row)
    return rows


def write_summary(rows: Sequence[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        (out_dir / "summary.csv").write_text("", encoding="utf-8")
        (out_dir / "summary.json").write_text("[]\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def run_command(command: Sequence[str], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return process.returncode


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    cases: Sequence[BenchCase],
    commands: Sequence[dict[str, Any]],
    variants: Sequence[tuple[str, Path]],
    kv_dtypes: Sequence[str],
    drafts: Sequence[str],
) -> None:
    manifest = {
        "artifact_type": "ninfer_bench_matrix_run",
        "schema_version": 4,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "preset": args.preset,
        "draft_modes": list(drafts),
        "variants": [{"name": name, "artifact": str(path)} for name, path in variants],
        "kv_dtypes": list(kv_dtypes),
        "repo_root": str(REPO_ROOT),
        "bench": str(args.bench),
        "corpus": str(args.corpus),
        "corpus_tokens": count_corpus_tokens(args.corpus),
        "dry_run": args.dry_run,
        "resume": args.resume,
        "case_count": len(cases),
        "commands": list(commands),
        "notes": [
            "Primary speculative comparison: mtp0 baseline, mtp3 primary MTP path, dflash7 DFlash.",
            "mtp0..mtp5 is swept by the mtp_sweep suite for MTP window decisions.",
            "DFlash cases run eager: the qwen3.8-27B DFlash2 decode round does not capture CUDA graphs.",
            "Use variant x kv-dtype x draft columns in summary.csv for paired configuration deltas.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "core", "full"), default="core")
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="single-variant .ninfer artifact used when --variant is not given",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named .ninfer artifact variant; repeat to sweep multiple variants",
    )
    parser.add_argument(
        "--kv-dtype",
        default="bf16",
        help="comma list among bf16,int8 swept over every case (default: bf16)",
    )
    parser.add_argument(
        "--drafts",
        default=None,
        help=(
            "comma list of speculative modes "
            f"({','.join(DRAFT_MODES)}); default: {','.join(PRIMARY_DRAFTS)}"
        ),
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--suite", action="append", default=[], help="suite to run; repeatable")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N selected cases")
    parser.add_argument("--repetitions", type=int, default=None, help="override all case repetitions")
    parser.add_argument("--warmup", type=int, default=None, help="override all case warmup repetitions")
    parser.add_argument("--dry-run", action="store_true", help="write commands but do not execute")
    parser.add_argument("--resume", action="store_true", help="skip cases with an existing valid JSON report")
    parser.add_argument(
        "--no-build", action="store_true", help="do not build build/bench/ninfer_bench"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.repetitions is not None and args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.warmup is not None and args.warmup < 0:
        raise SystemExit("--warmup must be nonnegative")

    args.bench = args.bench.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    args.corpus = args.corpus.expanduser().resolve()

    variants = parse_variants(args.variant, args.weights)
    kv_dtypes = parse_kv_dtypes(args.kv_dtype)
    drafts = parse_drafts(args.drafts)
    corpus_tokens = count_corpus_tokens(args.corpus)

    all_cases = build_cases(args.preset, drafts)
    cases = filtered_cases(all_cases, args.suite, args.limit)
    if not cases:
        raise SystemExit("selected matrix is empty")
    max_prompt = max_prompt_in_cases(cases)
    if max_prompt > corpus_tokens:
        raise SystemExit(
            f"selected matrix needs prompt length {max_prompt}, but corpus has {corpus_tokens} tokens"
        )

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = REPO_ROOT / "profiles/bench" / f"ninfer-{args.preset}-{utc_stamp()}"
    out_dir = out_dir.expanduser().resolve()
    json_dir = out_dir / "json"
    log_dir = out_dir / "logs"
    json_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    command_records: list[dict[str, Any]] = []
    commands_sh: list[str] = []
    for case in cases:
        for kv_dtype in kv_dtypes:
            for variant_name, variant_artifact in variants:
                report_path = json_dir / case.suite / f"{case.name}_{kv_dtype}_{variant_name}.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                base_args = [
                    str(args.bench),
                    "--weights",
                    str(variant_artifact),
                    "--corpus",
                    str(args.corpus),
                    "--device",
                    str(args.device),
                    "--kv-dtype",
                    kv_dtype,
                    *case.args,
                    "--output",
                    "json",
                    "--output-file",
                    str(report_path),
                ]
                command = add_repetition_args(base_args, case, args.repetitions, args.warmup)
                command_records.append(
                    {
                        "suite": case.suite,
                        "case": case.name,
                        "variant": variant_name,
                        "artifact": str(variant_artifact),
                        "kv_dtype": kv_dtype,
                        "report": str(report_path),
                        "notes": case.notes,
                        "command": command,
                    }
                )
                commands_sh.append(shell_join(command))
    commands_text = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(commands_sh) + "\n"
    (out_dir / "commands.sh").write_text(commands_text, encoding="utf-8")
    write_manifest(out_dir, args, cases, command_records, variants, kv_dtypes, drafts)

    if args.dry_run:
        print(f"wrote dry-run matrix to {out_dir}")
        print(f"cases: {len(command_records)} ({len(cases)} shapes x {len(kv_dtypes)} kv x {len(variants)} variants)")
        return 0

    failures: list[dict[str, Any]] = []
    if not args.no_build:
        build_stdout = log_dir / "build.stdout.txt"
        build_stderr = log_dir / "build.stderr.txt"
        rc = run_command(
            ["cmake", "--build", "build", "-j", "--target", "ninfer_bench"],
            build_stdout,
            build_stderr,
        )
        if rc != 0:
            failures.append(
                {
                    "case": "build",
                    "returncode": rc,
                    "stdout": str(build_stdout),
                    "stderr": str(build_stderr),
                }
            )
            (out_dir / "failures.json").write_text(
                json.dumps(failures, indent=2) + "\n", encoding="utf-8"
            )
            print(f"build failed; see {build_stderr}", file=sys.stderr)
            return 1

    if not args.bench.is_file():
        raise SystemExit(f"bench binary not found after build: {args.bench}")

    for index, record in enumerate(command_records, start=1):
        axis_stride = len(kv_dtypes) * len(variants)
        case = cases[(index - 1) // axis_stride]
        report_path = Path(record["report"])
        if args.resume and report_path.is_file():
            try:
                load_bench_report(report_path)
                print(f"[{index}/{len(command_records)}] skip {record['report']} (existing report)")
                continue
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass

        stdout_path = log_dir / f"{Path(record['report']).stem}.stdout.txt"
        stderr_path = log_dir / f"{Path(record['report']).stem}.stderr.txt"
        print(f"[{index}/{len(command_records)}] run {case.suite}/{Path(record['report']).stem}")
        rc = run_command(record["command"], stdout_path, stderr_path)
        if rc != 0:
            failures.append(
                {
                    "suite": record["suite"],
                    "case": record["case"],
                    "variant": record["variant"],
                    "kv_dtype": record["kv_dtype"],
                    "returncode": rc,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "command": record["command"],
                }
            )
            print(f"  failed with rc={rc}; see {stderr_path}", file=sys.stderr)
            continue
        if not report_path.is_file():
            failures.append(
                {
                    "suite": record["suite"],
                    "case": record["case"],
                    "variant": record["variant"],
                    "kv_dtype": record["kv_dtype"],
                    "returncode": rc,
                    "error": "report file was not created",
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "command": record["command"],
                }
            )

    rows: list[dict[str, Any]] = []
    for record in command_records:
        report_path = Path(record["report"])
        if not report_path.is_file():
            continue
        try:
            rows.extend(
                report_rows(
                    report_path,
                    # Suite/name bookkeeping comes from the record itself because
                    # expanded report names carry the kv/variant suffixes.
                    BenchCase(record["suite"], Path(record["report"]).stem, (), 1, 0),
                    axes={
                        "variant": record["variant"],
                        "kv_dtype": record["kv_dtype"],
                    },
                )
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "suite": record["suite"],
                    "case": record["case"],
                    "variant": record["variant"],
                    "kv_dtype": record["kv_dtype"],
                    "report": str(report_path),
                    "error": f"failed to parse report: {exc}",
                }
            )

    write_summary(rows, out_dir)
    if failures:
        (out_dir / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
        print(f"completed with {len(failures)} failure(s); see {out_dir / 'failures.json'}")
        return 1

    print(f"completed {len(command_records)} benchmark points")
    print(f"summary: {out_dir / 'summary.csv'}")
    return 0


def run_cli(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--pick" in raw:
        raw = [item for item in raw if item != "--pick"]
        artifact_dirs, rest = split_pick_args(raw)
        return main([*plan_with_picker(artifact_dirs), *rest])
    return main(raw)


if __name__ == "__main__":
    raise SystemExit(run_cli())
