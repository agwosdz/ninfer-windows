"""Unit tests for the graded-suite capability front end.

Keeps the pure planning logic (identity fallback, drafter-family restrictions,
configuration filtering) locked down without requiring a CUDA device, a serve
binary, a model artifact, or the interactive wizard.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.bench.run_graded_suite import (
    DRAFTER_FAMILIES,
    build_plan,
    discover_artifacts,
    drafter_families_for,
    load_identity,
    make_spec,
    to_target_key,
)


def _identity(model_id: str = "", weights_id: str = ""):
    return {"model_id": model_id, "weights_id": weights_id}


def test_identity_prefers_conversion_report(tmp_path: Path) -> None:
    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"\0")
    (tmp_path / "model.ninfer.conversion.json").write_text(
        json.dumps({"identity": {"model_id": "qwen3.8-27b", "weights_id": "groupwise-int-dflash2"}}),
        encoding="utf-8",
    )
    assert load_identity(artifact) == {
        "model_id": "qwen3.8-27b",
        "weights_id": "groupwise-int-dflash2",
    }


def test_identity_filename_fallback(tmp_path: Path) -> None:
    cases = {
        "qwen3_6_27b.ninfer": ("qwen3.6-27b", "groupwise-int"),
        "qwen3_6_27b_nvfp4.ninfer": ("qwen3.6-27b", "nvfp4"),
        "qwen3_6_35b_a3b.ninfer": ("qwen3.6-35b-a3b", "groupwise-int"),
        "qwen3_8_27b_dflash2.ninfer": ("qwen3.8-27b", "groupwise-int-dflash2"),
    }
    for name, (model_id, weights_id) in cases.items():
        artifact = tmp_path / name
        artifact.write_bytes(b"x")
        assert load_identity(artifact) == _identity(model_id, weights_id), name


def test_drafter_families_per_identity() -> None:
    # MTP everywhere, dflash only on 35B and dflash2 profiles.
    assert drafter_families_for("qwen3.6-27b", "groupwise-int") == (
        "mtp0", "mtp3", "mtp5"
    )
    assert drafter_families_for("qwen3.6-35b-a3b", "groupwise-int") == (
        "mtp0", "mtp3", "mtp5", "dflash7"
    )
    assert drafter_families_for("qwen3.8-27b", "groupwise-int") == (
        "mtp0", "mtp3", "mtp5"
    )
    assert drafter_families_for("qwen3.8-27b", "groupwise-int-dflash2") == (
        "mtp0", "mtp3", "mtp5", "dflash7"
    )


def test_plan_filters_illegal_families_per_artifact(tmp_path: Path) -> None:
    plain = tmp_path / "qwen3_8_27b.ninfer"
    plain.write_bytes(b"x")
    dflash2 = tmp_path / "qwen3_8_27b_dflash2.ninfer"
    dflash2.write_bytes(b"x")

    all_artifacts = discover_artifacts(tmp_path)
    artifacts = [
        entry for entry in all_artifacts
        if str(entry["path"]).startswith(str(tmp_path.resolve()))
    ]
    assert len(artifacts) == 2, f"found {len(artifacts)}: {[e['name'] for e in artifacts]}"

    plan = build_plan(artifacts, ["mtp0", "dflash7"], ["bf16", "int8"], "greedy")
    # 2 artifacts x mtp0 x 2 kv = 4; only the dflash2 artifact gets dflash7 x 2 = 2
    assert len(plan) == 6, f"plan has {len(plan)}: {[(c['entry']['name'], c['family'], c['kv']) for c in plan]}"
    family_by_name: dict[str, set[str]] = {}
    for config in plan:
        family_by_name.setdefault(config["entry"]["name"], set()).add(config["family"])
    assert "dflash7" not in family_by_name["qwen3_8_27b.ninfer"]
    assert "dflash7" in family_by_name["qwen3_8_27b_dflash2.ninfer"]


def test_target_key_best_effort() -> None:
    assert to_target_key("qwen3.8-27b") == "qwen3_8_27b"
    assert to_target_key("") == "unknown"