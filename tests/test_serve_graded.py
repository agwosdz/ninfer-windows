from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bench.run_serve_graded import (
    DEFAULT_SEED_DATASET,
    GradedSpec,
    aggregate_config,
    canonical_number,
    extract_boxed,
    extract_last_number,
    grade_answer,
    graded_request_payload,
    graded_server_command,
    load_graded_items,
    plan_configs,
)


def test_canonical_number_strips_separators_and_trailing_zeros() -> None:
    assert canonical_number("1,234.500") == "1234.5"
    assert canonical_number("$72.") == "72"
    assert canonical_number("-3") == "-3"
    assert canonical_number("no digits") is None


def test_extract_last_number_and_boxed_pick_the_final_candidate() -> None:
    assert extract_last_number("First I get 24, then 48, total 72 clips.") == "72"
    assert extract_last_number("nothing here") is None
    text = r"wrong \boxed{99} then corrected \boxed{55}"
    assert extract_boxed(text) == "55"


def test_grade_answer_strategies() -> None:
    assert grade_answer("72", "So the total is \\boxed{72} clips.", "auto")["correct"] is True
    assert grade_answer("72", "She sold 72 clips in total.", "number")["correct"] is True
    assert (
        grade_answer("72", "She sold 72 clips, i.e. 7,2e1? no. 72!", "number")["extracted"]
        == "72"
    )
    assert grade_answer("Au", "Au", "exact")["correct"] is True
    assert grade_answer("Au", " au. ", "exact")["correct"] is True
    assert grade_answer("Au", "The symbol is Au.", "exact")["correct"] is False
    assert grade_answer("40", "f(4) = 48 - 8 = \\boxed{40}", "boxed")["correct"] is True
    # Wrong answer must not pass even when a number exists.
    verdict = grade_answer("120", "She has read 180 pages, so 60 remain.", "number")
    assert verdict["correct"] is False and verdict["extracted"] == "60"


def test_seed_dataset_items_self_grade(tmp_path: Path) -> None:
    """Every vendored item must be gradeable from an ideal completion."""
    items = load_graded_items([DEFAULT_SEED_DATASET], None)
    assert len(items) >= 16
    for item in items:
        if item.match == "number":
            completion = f"Step by step we find {item.answer}."
        elif item.match == "boxed":
            completion = rf"The answer is \boxed{{{item.answer}}}."
        else:
            completion = item.answer
        assert grade_answer(item.answer, completion, item.match)["correct"], item.item_id


def test_load_graded_items_dedupes_caps_and_validates(tmp_path: Path) -> None:
    good = tmp_path / "a.jsonl"
    good.write_text(
        "\n".join(
            [
                json.dumps({"id": "x1", "question": "q1", "answer": "1", "match": "number"}),
                json.dumps({"id": "x2", "question": "q2", "answer": "2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    duplicate = tmp_path / "b.jsonl"
    duplicate.write_text(
        json.dumps({"id": "x1", "question": "other", "answer": "9"}) + "\n", encoding="utf-8"
    )
    items = load_graded_items([good, duplicate], None)
    assert [item.item_id for item in items] == ["x1", "x2"]
    assert items[1].match == "auto"

    capped = load_graded_items([good], 1)
    assert len(capped) == 1

    broken = tmp_path / "broken.jsonl"
    broken.write_text(json.dumps({"id": "", "question": "q", "answer": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="missing fields"):
        load_graded_items([broken], None)

    bad_match = tmp_path / "badmatch.jsonl"
    bad_match.write_text(
        json.dumps({"id": "m", "question": "q", "answer": "a", "match": "regex"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="unknown match strategy"):
        load_graded_items([bad_match], None)

    with pytest.raises(Exception, match="not found"):
        load_graded_items([tmp_path / "missing.jsonl"], None)


def _spec(mode: str = "mtp0", kv: str = "bf16") -> GradedSpec:
    return GradedSpec(
        target="qwen3_8_27b",
        model_id="qwen3.8-27b",
        artifact=Path("out/qwen3_8_27b.ninfer"),
        mode_name=mode,
        speculative_backend={"mtp0": "none", "mtp3": "mtp", "dflash7": "dflash"}[mode],
        draft_tokens={"mtp0": 0, "mtp3": 3, "dflash7": 7}[mode],
        kv_dtype=kv,
        sampling_mode="greedy",
    )


def test_plan_configs_builds_the_full_cross_product_in_declaration_order() -> None:
    artifacts = [("qwen3_8_27b", Path("out/a.ninfer")), ("qwen3_6_27b", Path("out/b.ninfer"))]
    specs = plan_configs(artifacts, ("dflash7", "mtp0"), ("rk4v4-e8", "bf16"), "greedy")
    keys = [spec.key for spec in specs]
    assert len(keys) == len(set(keys)) == 8
    assert specs[0].key == ("qwen3_8_27b", "dflash7", "rk4v4-e8", "greedy")
    assert specs[-1].key == ("qwen3_6_27b", "mtp0", "bf16", "greedy")


def test_request_payload_is_single_turn_greedy_friendly() -> None:
    spec = _spec()
    from tools.bench.run_serve_graded import GradedItem

    item = GradedItem("i1", "cat", "What is 2+2?", "4", "number")
    payload = graded_request_payload(spec, item, seed=7, max_tokens=256)
    assert payload["messages"][0]["content"] == "What is 2+2?"
    assert payload["max_completion_tokens"] == 256
    assert payload["enable_thinking"] is False
    assert payload["seed"] == 7


def test_server_command_encodes_kv_dtype_and_sampling(tmp_path: Path) -> None:
    command = graded_server_command(
        Path("build/apps/ninfer-serve"),
        _spec("mtp3", "rk4v4-e8"),
        tmp_path / "log.jsonl",
        port=8091,
        device=1,
        max_context=32768,
    )
    joined = " ".join(str(part) for part in command)
    assert "--kv-dtype rk4v4-e8" in joined
    assert "--spec mtp --draft-tokens 3 --lm-head-draft" in joined
    assert "--greedy" in joined
    assert "--max-context 32768" in joined

    baseline = " ".join(
        str(part)
        for part in graded_server_command(
            Path("build/apps/ninfer-serve"),
            _spec("mtp0", "bf16"),
            tmp_path / "log2.jsonl",
            port=8091,
            device=1,
            max_context=32768,
        )
    )
    assert "--spec" not in baseline and "--greedy" in baseline


def test_aggregate_config_reports_accuracy_and_throughput() -> None:
    records = [
        {"correct": True, "wall_seconds": 1.0, "completion_tokens": 50},
        {"correct": False, "wall_seconds": 3.0, "completion_tokens": 100},
    ]
    summary = aggregate_config(records)
    assert summary["items"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["wall_seconds_mean"] == 2.0
    assert summary["output_tok_s"] == 37.5
