from __future__ import annotations

import json

from tools.bench.run_ninfer_bench_matrix import (
    DRAFT_MODES,
    KV_DTYPES,
    REPORT_SCHEMA_VERSION,
    BenchCase,
    assemble_pick_argv,
    assign_variant_names,
    build_cases,
    describe_artifact,
    discover_artifacts,
    draft_args,
    draft_tag,
    filtered_cases,
    parse_drafts,
    parse_index_selection,
    parse_kv_dtypes,
    parse_variants,
    report_rows,
    sanitize_variant_name,
    split_pick_args,
)


def _sample_report() -> dict:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "ninfer_bench_report",
        "tool": "ninfer_bench",
        "artifact": {"path": "model.ninfer"},
        "environment": {"gpu_name": "RTX 5090"},
        "load": {
            "target": "qwen3_8_27b",
            "weights_id": "groupwise-int-dflash2",
            "load_seconds": 2.5,
            "upload_seconds": 2.0,
            "artifact_bytes_read": 17_500_000_000,
            "host_to_device_bytes": 17_400_000_000,
            "peak_staging_bytes": 134_217_728,
        },
        "memory": {
            "kv_capacity": 8192,
            "kv_payload_bytes": 123_456,
            "weights": {"capacity_bytes": 17_400_000_000},
            "sequence": {"capacity_bytes": 2_000_000_000},
            "workspace": {"capacity_bytes": 100_000_000},
            "request_transient": {"capacity_bytes": 50_000_000},
            "cuda_graph_allowance_bytes": 150_000_000,
        },
        "config": {
            "max_context": 4096,
            "prefill_chunk": 1024,
            "kv_cache": "int8-group64",
            "speculative_backend": "dflash",
            "draft_tokens": 7,
            "proposal_head": "optimized",
            "decode_path": "dflash_eager",
            "decode_graph_prime": {"primed": False, "output_tokens": 15},
            "repetitions": 2,
            "warmup": 1,
        },
        "tests": [
            {
                "label": "tg3",
                "kind": "tg",
                "n_prompt": 0,
                "n_gen": 3,
                "requested_output_tokens": 4,
                "workspace_peak_bytes": 1_048_576,
                "workspace_allocator_peak_bytes": 524_288,
                "decode_output_tok_s_mean": 4.5,
                "decode_engine_tok_s_mean": 7.5,
                "total_seconds_mean": 0.875,
                "speculative": {
                    "acceptance_rate": 1.0,
                    "acceptance_length": 5.0,
                    "rounds": 1,
                    "drafted_tokens": 5,
                    "accepted_tokens": 5,
                    "fallback_steps": 3,
                    "accepted_per_position": [1, 1, 1, 1, 1],
                },
            }
        ],
    }


def test_schema_v12_report_is_flattened_with_axes(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_sample_report()), encoding="utf-8")

    rows = report_rows(
        report_path,
        BenchCase("pure_decode", "tg3_df7_int8_q38d2", (), repetitions=2, warmup=1),
        axes={"variant": "q38d2", "kv_dtype": "int8"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert (row["suite"], row["case"], row["label"], row["kind"]) == (
        "pure_decode",
        "tg3_df7_int8_q38d2",
        "tg3",
        "tg",
    )
    assert (row["variant"], row["kv_dtype"]) == ("q38d2", "int8")
    assert (row["target"], row["weights_id"], row["artifact_path"], row["gpu_name"]) == (
        "qwen3_8_27b",
        "groupwise-int-dflash2",
        "model.ninfer",
        "RTX 5090",
    )
    assert (row["speculative_backend"], row["draft_tokens"]) == ("dflash", 7)
    assert (row["decode_path"], row["decode_graph_primed"]) == (
        "dflash_eager",
        False,
    )
    assert (row["kv_cache"], row["kv_capacity"]) == ("int8-group64", 8192)
    assert row["decode_output_tok_s_mean"] == 4.5
    assert row["decode_engine_tok_s_mean"] == 7.5
    assert row["spec_fallback_steps"] == 3
    assert row["spec_accepted_per_position"] == "[1,1,1,1,1]"


def test_draft_args_mirror_the_serving_campaign_flags() -> None:
    assert draft_args("mtp0") == ()
    assert draft_args("mtp3") == (
        "--spec",
        "mtp",
        "--draft-tokens",
        "3",
        "--lm-head-draft",
    )
    # DFlash runs the qwen3.8 v2 decode round eager by target contract.
    assert draft_args("dflash7") == (
        "--spec",
        "dflash",
        "--draft-tokens",
        "7",
        "--lm-head-draft",
        "--no-cuda-graph",
    )


def test_draft_tags_are_unique_and_stable() -> None:
    tags = {draft_tag(mode) for mode in DRAFT_MODES}
    assert len(tags) == len(DRAFT_MODES)
    assert draft_tag("mtp0") == "k0"
    assert draft_tag("mtp3") == "k3"
    assert draft_tag("dflash7") == "df7"


def test_core_cases_embed_their_speculative_mode(tmp_path) -> None:
    cases = build_cases("core", ("mtp0", "mtp3", "dflash7"))
    names = [case.name for case in cases]

    assert f"prefill_lengths_{draft_tag('dflash7')}" in names
    assert f"tg_lengths_{draft_tag('mtp3')}_graph" in names
    # DFlash shapes exist only in the eager graph variant.
    assert f"tg_lengths_{draft_tag('dflash7')}_eager" in names
    assert not any(
        name.startswith("tg_lengths_") and name.endswith("_graph") and "df7" in name
        for name in names
    )
    # Every MTP window appears in the fixed sweep suite.
    for k in range(6):
        assert f"mtp_sweep_k{k}_graph" in names
    assert len({(case.suite, case.name) for case in cases}) == len(cases)

    dflash_case = next(case for case in cases if case.name.endswith("df7_eager")
                       and case.suite == "context_decode")
    assert "--no-cuda-graph" in dflash_case.args


def test_axis_parsers_validate_and_dedupe(tmp_path) -> None:
    artifact = tmp_path / "a.ninfer"
    artifact.write_bytes(b"")

    variants = parse_variants((f"gi={artifact}", f"fp4={artifact}"), tmp_path / "missing.ninfer")
    assert [name for name, _ in variants] == ["gi", "fp4"]
    fallback = parse_variants((), artifact)
    assert fallback == (("default", artifact),)

    try:
        parse_variants(("broken",), artifact)
    except SystemExit as exc:
        assert "NAME=PATH" in str(exc)
    else:
        raise AssertionError("malformed variant accepted")

    assert parse_kv_dtypes("bf16, int8,bf16") == ("bf16", "int8")
    try:
        parse_kv_dtypes("fp8")
    except SystemExit as exc:
        assert "fp8" in str(exc)
    else:
        raise AssertionError("unsupported kv dtype accepted")

    assert parse_drafts(None) == ("mtp0", "mtp3", "dflash7")
    assert parse_drafts("mtp5,dflash7") == ("mtp5", "dflash7")
    try:
        parse_drafts("dflash9")
    except SystemExit as exc:
        assert "dflash9" in str(exc)
    else:
        raise AssertionError("unknown draft mode accepted")


def test_filtered_cases_keeps_suite_selection_order() -> None:
    cases = build_cases("smoke", ("mtp0", "mtp3", "dflash7"))
    assert [case.name for case in cases] == [
        "prefill_p128_k0",
        "tg8_k3_graph",
        "ctx_p128_g8_df7_eager",
    ]
    limited = filtered_cases(cases, ("pure_decode",), 5)
    assert [case.name for case in limited] == ["tg8_k3_graph"]


def test_kv_dtype_axis_covers_every_registered_storage() -> None:
    assert KV_DTYPES == ("bf16", "int8", "rk8v4", "rk4v4", "rk4v4-e8", "rk2v4-e8")
    assert parse_kv_dtypes(",".join(KV_DTYPES)) == KV_DTYPES


def test_parse_index_selection_supports_ranges_defaults_and_all() -> None:
    assert parse_index_selection("", 4, default=[0, 2]) == [0, 2]
    assert parse_index_selection("all", 3) == [0, 1, 2]
    assert parse_index_selection("a", 3) == [0, 1, 2]
    assert parse_index_selection("2-4", 5) == [1, 2, 3]
    assert parse_index_selection("1, 3-4,1", 5) == [0, 2, 3]

    try:
        parse_index_selection("x", 3)
    except SystemExit as exc:
        assert "invalid selection token" in str(exc)
    else:
        raise AssertionError("garbage token accepted")

    try:
        parse_index_selection("9", 3)
    except SystemExit as exc:
        assert "outside 1..3" in str(exc)
    else:
        raise AssertionError("out-of-range token accepted")


def test_artifact_discovery_labels_identity_from_conversion_report(tmp_path) -> None:
    labeled = tmp_path / "qwen3_8_27b.ninfer"
    labeled.write_bytes(b"artifact")
    (tmp_path / "qwen3_8_27b.ninfer.conversion.json").write_text(
        json.dumps({"identity": {"model_id": "qwen3.8-27b", "weights_id": "groupwise-int"}}),
        encoding="utf-8",
    )
    plain = tmp_path / "plain.ninfer"
    plain.write_bytes(b"artifact")

    entries = discover_artifacts([tmp_path])
    by_name = {entry["name"]: entry for entry in entries}
    assert set(by_name) >= {"qwen3_8_27b.ninfer", "plain.ninfer"}
    assert by_name["qwen3_8_27b.ninfer"]["weights_id"] == "groupwise-int"
    assert "groupwise-int" in by_name["qwen3_8_27b.ninfer"]["label"]
    assert by_name["plain.ninfer"]["model_id"] == ""

    named = assign_variant_names(
        [by_name["qwen3_8_27b.ninfer"], by_name["plain.ninfer"], by_name["plain.ninfer"]]
    )
    assert [entry["variant"] for entry in named] == ["qwen3_8_27b", "plain", "plain_2"]
    assert sanitize_variant_name("Qwen3.8 (27B) dflash2!") == "qwen3.8_27b_dflash2"


def test_split_pick_args_extracts_wizard_only_flags() -> None:
    dirs, rest = split_pick_args(
        ["--dry-run", "--artifact-dir", "C:/a", "--no-build", "--artifact-dir", "C:/b"]
    )
    assert len(dirs) == 2 and rest == ["--dry-run", "--no-build"]


def test_assemble_pick_argv_builds_full_cross_product_arguments() -> None:
    variants = [{"variant": "gi", "path": "out/gi.ninfer"}, {"variant": "fp4", "path": "out/fp4.ninfer"}]
    argv = assemble_pick_argv(
        "core",
        ("pure_decode", "context_decode"),
        ("mtp0", "dflash7"),
        ("bf16", "rk4v4-e8"),
        variants,
    )
    assert argv[:2] == ["--preset", "core"]
    assert argv.count("--suite") == 2
    assert "--drafts" in argv and argv[argv.index("--drafts") + 1] == "mtp0,dflash7"
    assert argv[argv.index("--kv-dtype") + 1] == "bf16,rk4v4-e8"
    assert argv.count("--variant") == 2
    assert f"gi={variants[0]['path']}" in argv
