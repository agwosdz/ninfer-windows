#!/usr/bin/env python3
"""Graded campaign suite: pick artifacts, drafters, and KV storages interactively.

A thin, capability-aware front end over `run_serve_graded.py`. It keeps the
real-text grading campaign intact (exact-match accuracy + throughput per
configuration) and adds:

  * artifact discovery in `models/` (plus optional `--artifact-dir`s); each
    `.ninfer` is paired with its sibling `.conversion.json` report to recover
    the clamped identity (model id, weights id) that drives capability;
  * a drafter picker that only offers families each artifact can actually run:
      - MTP modes  (mtp3, mtp5)  on every registered target,
      - DFlash      (dflash7)     only for the qwen3.6-35b-a3b profile and for
        qwen3.8-27b artifacts whose weights id carries the dflash2 profile;
  * a KV-cache compression sweep over every registered storage.

Usage:

    # Interactive wizard (default when --artifact/--family are absent).
    python3 tools/bench/run_graded_suite.py

    # Repeatable direct campaign (every wizard choice has a flag).
    python3 tools/bench/run_graded_suite.py \
        --artifact dflash2=models/qwen3_8_27b_dflash2.ninfer \
        --family mtp0 --family dflash7 \
        --kv-dtype bf16,int8 --max-items 64 \
        --output profiles/graded/my-run

    # Print the capability matrix and planned configurations without serving.
    python3 tools/bench/run_graded_suite.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

try:  # repository/pytest context
    from tools.bench import run_serve_corpus as corpus
    from tools.bench import run_serve_graded as graded
    from tools.bench.run_ninfer_bench_matrix import KV_DTYPES, parse_index_selection
except ImportError:  # direct script execution (python3 tools/bench/run_graded_suite.py)
    from run_serve_corpus import (  # type: ignore[no-redef]
        CampaignError,
        SPECULATIVE_MODES,
        TARGET_MODEL_IDS,
    )
    from run_serve_graded import (  # type: ignore[no-redef]
        DEFAULT_MAX_CONTEXT,
        DEFAULT_MAX_TOKENS,
        DEFAULT_SEED_DATASET,
        GradedSpec,
        aggregate_config,
        load_graded_items,
        run_graded_config,
        write_outputs,
    )
    from run_ninfer_bench_matrix import KV_DTYPES, parse_index_selection  # type: ignore[no-redef]

    import run_serve_corpus as corpus  # type: ignore[no-redef]
    import run_serve_graded as graded  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_GRADED_DIR = REPO_ROOT / "bench" / "fixtures" / "graded"
DEFAULT_PORT = 8090

# Drafter families in wizard order. mtp0 (no speculation) is the baseline
# every artifact supports; the rest are capability-filtered per artifact.
DRAFTER_FAMILIES = ("mtp0", "mtp3", "mtp5", "dflash7")
FAMILY_LABELS = {
    "mtp0": "no speculation (baseline)",
    "mtp3": "MTP, 3 draft tokens (optimized proposal head)",
    "mtp5": "MTP, 5 draft tokens",
    "dflash7": "DFlash / DFlash2, 7 draft tokens (text-only; eager)",
}


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")


def serve_resolve(explicit: Path) -> Path | None:
    """Locate ninfer-serve: explicit, then build/apps, then bin/, then any
    build-*/apps tree by newest mtime, then the build-windows app dir."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(REPO_ROOT / "build/apps/ninfer-serve.exe")
    candidates.append(REPO_ROOT / "build/apps/ninfer-serve")
    candidates.append(REPO_ROOT / "bin/ninfer-serve.exe")
    candidates.append(REPO_ROOT / "bin/ninfer-serve")
    for build_dir in sorted(REPO_ROOT.glob("build-*/apps"), key=lambda p: p.stat().st_mtime,
                            reverse=True):
        candidates.append(build_dir / "ninfer-serve.exe")
        candidates.append(build_dir / "ninfer-serve")
    candidates.append(REPO_ROOT / "build-edge-ninja/apps/ninfer-serve.exe")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def utc_now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def load_identity(path: Path) -> dict[str, str]:
    """Recover {model_id, weights_id} for capability mapping.

    Prefers the sibling .conversion.json report; falls back to the filename,
    which stably encodes the target family and dflash/dflash2 marker in this
    repository (e.g. qwen3_6_35b_a3b.ninfer, qwen3_8_27b_dflash2.ninfer).
    """
    report_path = Path(str(path) + ".conversion.json")
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
        identity = report.get("identity") or {}
        model_id = str(identity.get("model_id", ""))
        weights_id = str(identity.get("weights_id", ""))
        if model_id:
            return {"model_id": model_id, "weights_id": weights_id}

    name = path.name
    # Filename-derived identity: strip the .ninfer suffix; the stem encodes the
    # corpus target key (e.g. qwen3_6_35b_a3b) with optional variant markers.
    stem = name[: -len(".ninfer")] if name.endswith(".ninfer") else name
    model_id = ""
    for key, registered in corpus.TARGET_MODEL_IDS.items():
        if stem.startswith(key):
            model_id = registered
            break
    weights_id = "groupwise-int"
    if "dflash2" in stem:
        weights_id = "groupwise-int-dflash2"
    elif "nvfp4" in stem:
        weights_id = "nvfp4"
    return {"model_id": model_id, "weights_id": weights_id}


def to_target_key(model_id: str) -> str:
    """Map an artifact identity model_id onto the corpus target key."""
    for key, registered in corpus.TARGET_MODEL_IDS.items():
        if model_id == registered:
            return key
    return model_id.replace("-", "_") if model_id else "unknown"


def discover_artifacts(models_dir: Path, extra_dirs: Sequence[Path] = ()) -> list[dict]:
    """Enumerate .ninfer artifacts (deduped) with their clamped identity."""
    roots = [models_dir, REPO_ROOT / "out", *extra_dirs]
    entries: dict[str, dict] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.ninfer")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            identity = load_identity(resolved)
            size_gib = resolved.stat().st_size / (1024**3)
            label = f"{resolved.name}  [{identity['model_id'] or '?'}/"
            label += f"{identity['weights_id'] or '?'}]  {size_gib:.1f} GiB"
            entries.setdefault(
                str(resolved),
                {
                    "path": resolved,
                    "name": resolved.name,
                    "model_id": identity["model_id"],
                    "weights_id": identity["weights_id"],
                    "label": label,
                },
            )
    return [entries[key] for key in sorted(entries)]


def drafter_families_for(model_id: str, weights_id: str) -> tuple[str, ...]:
    """Families an artifact of this identity can run."""
    modes = {"mtp0", "mtp3", "mtp5"}
    if model_id == corpus.TARGET_MODEL_IDS["qwen3_6_35b_a3b"]:
        modes.add("dflash7")
    if model_id == corpus.TARGET_MODEL_IDS["qwen3_8_27b"]:
        if "dflash" in weights_id.lower():
            modes.add("dflash7")
    return tuple(mode for mode in DRAFTER_FAMILIES if mode in modes)


def usable_families(entry: dict) -> tuple[str, ...]:
    return drafter_families_for(entry["model_id"], entry["weights_id"])


def make_spec(entry: dict, family: str, kv_dtype: str, sampling: str) -> graded.GradedSpec:
    """Build the GradedSpec for one configuration from a discovered artifact."""
    target_key = to_target_key(entry["model_id"])
    backend, tokens = corpus.SPECULATIVE_MODES.get(family, ("none", 0))
    return graded.GradedSpec(
        target=target_key,
        model_id=corpus.TARGET_MODEL_IDS.get(target_key, entry["model_id"]),
        artifact=entry["path"],
        mode_name=family,
        speculative_backend=backend,
        draft_tokens=tokens,
        kv_dtype=kv_dtype,
        sampling_mode=sampling,
    )


def build_plan(
    artifacts: Sequence[dict],
    families: Sequence[str],
    kv_dtypes: Sequence[str],
    sampling: str,
) -> list[dict]:
    """Per-config records (entry, family, kv, spec) where the family is legal."""
    plan: list[dict] = []
    for entry in artifacts:
        usable = set(usable_families(entry))
        for family in families:
            if family not in usable:
                continue
            for kv_dtype in kv_dtypes:
                plan.append(
                    {
                        "entry": entry,
                        "family": family,
                        "kv": kv_dtype,
                        "spec": make_spec(entry, family, kv_dtype, sampling),
                    }
                )
    return plan


def print_capability(artifacts: Sequence[dict]) -> None:
    print("capability matrix:")
    for entry in artifacts:
        usable = usable_families(entry)
        print(f"  {entry['name']:52s} families={','.join(usable)}")


# --- wizard ----------------------------------------------------------------


def _prompt_selection(title: str, labels: Sequence[str], *, default: Sequence[int]) -> list[int]:
    print(title)
    for number, label in enumerate(labels, start=1):
        print(f"  [{number}] {label}")
    hint = f"comma/range/a=all/Enter=default({','.join(str(i + 1) for i in default)}) "
    while True:
        try:
            raw = input(hint + ": ")
        except EOFError:
            print("\nno interactive input; use explicit flags (see --help)")
            raise SystemExit(2) from None
        try:
            return parse_index_selection(raw, len(labels), default=default)
        except SystemExit as error:
            print(f"  {error}")


def run_wizard(artifacts: Sequence[dict]) -> tuple[list[dict], list[str], tuple[str, ...]]:
    if not artifacts:
        raise SystemExit("no artifacts found (add --models-dir or --artifact-dir)")

    picked_artifacts = _prompt_selection(
        "Artifacts to benchmark:",
        [entry["label"] for entry in artifacts],
        default=range(len(artifacts)),
    )
    selected = [artifacts[index] for index in picked_artifacts]

    usable = set().union(*(usable_families(entry) for entry in selected))
    order = [family for family in DRAFTER_FAMILIES if family in usable]
    labels = [f"{family:<8} {FAMILY_LABELS[family]}" for family in order]

    family_default = [order.index(family) for family in ("mtp0", "mtp3") if family in order]
    if "dflash7" in order:
        family_default.append(order.index("dflash7"))
    picked = _prompt_selection("Drafter families:", labels, default=sorted(family_default))
    selected_families = [order[index] for index in picked]

    kv_labels = [f"{dtype:<10} {granularity(dtype)}" for dtype in KV_DTYPES]
    kv_picked = _prompt_selection(
        "KV-cache compressions:", kv_labels, default=range(len(KV_DTYPES))
    )
    kv_selected = tuple(KV_DTYPES[index] for index in kv_picked)

    return selected_families, kv_selected, selected


def granularity(dtype: str) -> str:
    descriptions = {
        "bf16": "BF16 KV storage (baseline)",
        "int8": "INT8 group-64",
        "rk8v4": "rotated INT8 keys + INT4 values",
        "rk4v4": "rotated packed INT4 K + V",
        "rk4v4-e8": "INT4 K/V with E8-lattice codes",
        "rk2v4-e8": "INT4 K/V with E8-root codes",
    }
    return descriptions.get(dtype, "")


# --- CLI -------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--artifact-dir", action="append", type=Path, default=[])
    parser.add_argument("--artifact", action="append", default=[],
                        metavar="NAME=PATH", help="explicit artifact (repeatable)")
    parser.add_argument("--family", action="append", default=[],
                        choices=DRAFTER_FAMILIES,
                        help="drafter family (repeatable); omit for the wizard")
    parser.add_argument("--kv-dtype", default=",".join(KV_DTYPES),
                        help="comma list among " + ",".join(KV_DTYPES))
    parser.add_argument("--sampling", choices=("greedy", "stochastic"), default="greedy")
    parser.add_argument("--dataset", action="append", type=Path, default=[])
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=graded.DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-context", type=int, default=graded.DEFAULT_MAX_CONTEXT)
    parser.add_argument("--seed", type=int, default=graded.corpus.SEEDS[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--serve", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Explicit artifacts first (--artifact NAME=PATH).
    artifacts: list[dict] = []
    for value in args.artifact:
        name, separator, raw = value.partition("=")
        if not separator:
            raise SystemExit(f"invalid --artifact {value!r}; expected NAME=PATH")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"--artifact {name!r}: file not found: {path}")
        identity = load_identity(path)
        artifacts.append({
            "path": path,
            "name": name,
            "model_id": identity["model_id"],
            "weights_id": identity["weights_id"],
            "label": f"{path.name} [{name} / {identity['weights_id'] or '?'}]",
        })
    discovered = discover_artifacts(args.models_dir, args.artifact_dir)
    known = {str(entry["path"]) for entry in artifacts}
    artifacts.extend(entry for entry in discovered if str(entry["path"]) not in known)

    use_wizard = not args.family
    if args.dry_run and use_wizard:
        # Inspection without interaction: show the full cross-product plan.
        use_wizard = False
        families_list = list(DRAFTER_FAMILIES)
    else:
        families_list = []
    kv_list: list[str] = list(parse_kv(args.kv_dtype))
    artifacts_for_plan = artifacts
    if use_wizard:
        wizard_families, wizard_kv, wizard_artifacts = run_wizard(artifacts)
        families_list = wizard_families
        kv_list = list(wizard_kv)
        artifacts_for_plan = wizard_artifacts
        use_wizard = False
    elif not families_list:
        families_list = list(args.family)

    dataset_paths = list(args.dataset) or [graded.DEFAULT_SEED_DATASET]
    items = graded.load_graded_items(dataset_paths, args.max_items)
    plan = build_plan(artifacts_for_plan, families_list, kv_list, args.sampling)

    print(f"artifacts : {len(artifacts_for_plan)}")
    print(f"families  : {', '.join(families_list)}")
    print(f"kv        : {', '.join(kv_list)}")
    print(f"items     : {len(items)} from {len(dataset_paths)} file(s)")
    print_capability(artifacts_for_plan)
    print(f"configs   : {len(plan)}")

    if args.dry_run:
        print("dry-run: not serving")
        return 0

    serve = serve_resolve(args.serve)
    if serve is None:
        raise SystemExit("ninfer-serve not found; pass --serve <path to ninfer-serve.exe>")
    output_dir = (args.output or REPO_ROOT / "profiles/bench" / f"graded-{utc_stamp()}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    all_failures: list[dict] = []
    aggregates: list[dict] = []
    suite_rows: list[dict] = []
    index = 0
    for config in plan:
        index += 1
        entry, family, kv = config["entry"], config["family"], config["kv"]
        spec = config["spec"]
        label = f"{entry['name']}/{family}/{kv}/{args.sampling}"
        print(f"[{index}/{len(plan)}] {label} ...", flush=True)
        records, failures = graded.run_graded_config(
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
        aggregate = graded.aggregate_config(records)
        # Base summary row: matches run_serve_graded.write_outputs fieldnames.
        summary_row = {
            "target": spec.target,
            "weights_id": records[0]["weights_id"] if records else None,
            "mode": spec.mode_name,
            "kv_dtype": spec.kv_dtype,
            "sampling": spec.sampling_mode,
        }
        summary_row.update(aggregate)
        aggregates.append(summary_row)
        # Suite-side detail view keeps the artifact/family axes write_outputs
        # does not carry.
        suite_rows.append(
            {
                "artifact": entry["name"],
                "family": family,
                "kv_dtype": kv,
                "sampling": args.sampling,
                **aggregate,
            }
        )
        all_records.extend(records)
        all_failures.extend(failures)
        acc = "" if aggregate["accuracy"] is None else format(aggregate["accuracy"] * 100, ".1f")
        print(f"    accuracy={acc}% items={aggregate['items']} tok/s={aggregate['output_tok_s']}")

    graded.write_outputs(output_dir, all_records, aggregates, all_failures)
    # Supplementary artifact x family view the graded writer does not carry.
    if suite_rows:
        import csv

        suite_fields = ["artifact", "family", "kv_dtype", "sampling", "items", "accuracy",
                        "wall_seconds_mean", "output_tok_s"]
        with (output_dir / "summary.suite.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=suite_fields)
            writer.writeheader()
            writer.writerows(suite_rows)
    print(f"results : {output_dir / 'results.jsonl'}")
    print(f"summary : {output_dir / 'summary.csv'}")
    if suite_rows:
        print(f"details : {output_dir / 'summary.suite.csv'}")
    return 1 if all_failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except corpus.CampaignError as error:
        print(f"suite failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error