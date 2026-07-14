"""`matrix.__main__`, hermetic: CLI arg parsing, `MatrixRunConfig` assembly, git_sha/run_id
resolution, and case loading -- every piece testable with no live call. `main()`'s own end-to-end
live sweep (real TEI/OpenAI/Anthropic/Ollama + a real Postgres) is the operator/burst step
(`task matrix:live`), never run here; this file only proves the wiring around it: that `main()`
reports a missing required env var as a clean, worded failure (exit code 2) rather than a raw
traceback, BEFORE attempting any live call.
"""
from __future__ import annotations

import json

import pytest

from dataset_tools.manifest import DATASET_VERSION
from rag_tools.chunker import chunker_hash

from matrix.cases import MatrixCase
from matrix.live_driver import MissingEnvVarError
from matrix.__main__ import (
    build_arg_parser,
    build_matrix_run_config,
    default_run_id,
    load_cases_for_run,
    main,
    resolve_git_sha,
)


# ---- argument parsing -----------------------------------------------------------------------------


def test_build_arg_parser_applies_documented_defaults():
    args = build_arg_parser().parse_args([])
    assert args.corpus_version == "corpus-0.1.1"
    assert args.dataset_version == DATASET_VERSION
    assert args.k_retrieval == 5
    assert args.seed == 20260721
    assert args.n_top_configs == 2
    assert args.reranker_depths == "20,50,100"
    assert args.embedders == "bge-m3,openai"
    assert args.generators == "anthropic,openai,ollama"
    assert args.variant_generator == "ollama"
    assert args.avg_output_tokens == 300
    assert args.limit_cases is None
    assert args.pool_size is None


def test_build_arg_parser_honors_explicit_overrides():
    args = build_arg_parser().parse_args([
        "--corpus-version", "corpus-9.9.9",
        "--k-retrieval", "3",
        "--seed", "7",
        "--reranker-depths", "10,40",
        "--embedders", "bge-m3",
        "--generators", "ollama",
        "--variant-generator", "none",
        "--limit-cases", "2",
    ])
    assert args.corpus_version == "corpus-9.9.9"
    assert args.k_retrieval == 3
    assert args.seed == 7
    assert args.reranker_depths == "10,40"
    assert args.embedders == "bge-m3"
    assert args.generators == "ollama"
    assert args.variant_generator == "none"
    assert args.limit_cases == 2


def test_build_arg_parser_rejects_an_unknown_variant_generator_choice():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--variant-generator", "voyage"])


# ---- git_sha / run_id resolution -------------------------------------------------------------------


def test_resolve_git_sha_returns_the_explicit_value_when_given():
    assert resolve_git_sha("abc1234") == "abc1234"


def test_resolve_git_sha_falls_back_to_the_env_var_when_no_explicit_value(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "deadbeef")
    assert resolve_git_sha(None) == "deadbeef"


def test_resolve_git_sha_falls_back_to_a_real_git_rev_parse_when_neither_is_given(monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    sha = resolve_git_sha(None)
    assert len(sha) >= 7
    assert all(c in "0123456789abcdef" for c in sha)


def test_default_run_id_is_a_non_empty_string_prefixed_for_this_lane():
    run_id = default_run_id()
    assert run_id.startswith("matrix-live-")
    assert len(run_id) > len("matrix-live-")


# ---- MatrixRunConfig assembly ----------------------------------------------------------------------


def test_build_matrix_run_config_assembles_every_field_from_args():
    args = build_arg_parser().parse_args([
        "--run-id", "test-run-1", "--reranker-depths", "20,50",
        "--k-retrieval", "5", "--seed", "42", "--n-top-configs", "1",
    ])
    config = build_matrix_run_config(args, git_sha="a" * 40)
    assert config.run_id == "test-run-1"
    assert config.git_sha == "a" * 40
    assert config.reranker_depths == (20, 50)
    assert config.k_retrieval == 5
    assert config.seed == 42
    assert config.n_top_configs == 1


def test_build_matrix_run_config_defaults_run_id_when_not_given():
    args = build_arg_parser().parse_args([])
    config = build_matrix_run_config(args, git_sha="a" * 40)
    assert config.run_id  # some non-empty default run id


def test_build_matrix_run_config_defaults_chunker_config_hash_to_the_real_chunker_hash():
    args = build_arg_parser().parse_args([])
    config = build_matrix_run_config(args, git_sha="a" * 40)
    assert config.chunker_config_hash == chunker_hash()


def test_build_matrix_run_config_honors_an_explicit_chunker_config_hash():
    args = build_arg_parser().parse_args(["--chunker-config-hash", "custom-hash-value"])
    config = build_matrix_run_config(args, git_sha="a" * 40)
    assert config.chunker_config_hash == "custom-hash-value"


# ---- case loading (a real, committed, deterministic file read) -------------------------------------


def test_load_cases_for_run_reads_the_real_committed_seed_set():
    args = build_arg_parser().parse_args([])
    cases = load_cases_for_run(args)
    assert len(cases) > 0
    assert all(isinstance(c, MatrixCase) for c in cases)


def test_load_cases_for_run_honors_limit_cases():
    args = build_arg_parser().parse_args(["--limit-cases", "3"])
    cases = load_cases_for_run(args)
    assert len(cases) == 3


def test_load_cases_for_run_reads_a_custom_cases_file(tmp_path):
    custom = tmp_path / "cases.jsonl"
    row = {
        "case_id": "c1", "turns": [{"user": "how much is plan a"}],
        "expected_doc_ids": ["d1"], "hop_count": 1,
    }
    custom.write_text(json.dumps(row) + "\n")
    args = build_arg_parser().parse_args(["--cases", str(custom)])
    cases = load_cases_for_run(args)
    assert [c.case_id for c in cases] == ["c1"]


# ---- main(): the worded failure fires before any live call, never a raw traceback ------------------


def test_main_reports_a_missing_required_env_var_cleanly_and_exits_nonzero(monkeypatch, tmp_path, capsys):
    for var in ("ATLAS_PG_DSN", "ATLAS_TEI_EMBED_URL", "ATLAS_TEI_RERANK_URL", "ATLAS_OPENAI_INDEX_DIR"):
        monkeypatch.delenv(var, raising=False)
    exit_code = main([
        "--limit-cases", "1",
        "--generators", "ollama",
        "--variant-generator", "none",
        "--output-dir", str(tmp_path / "out"),
        "--cassette-dir", str(tmp_path / "cassettes"),
        "--cache-dir", str(tmp_path / "cache"),
        "--run-id", "hermetic-missing-env-test",
        "--git-sha", "a" * 40,
    ])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "ATLAS_PG_DSN" in captured.err or "ATLAS_PG_DSN" in captured.out


def test_main_never_raises_missingenvvarerror_uncaught(monkeypatch, tmp_path):
    """`main()` is the CLI boundary: a `MissingEnvVarError` must be CAUGHT and reported (a clean exit
    code), never let escape as a raw traceback -- the same "never a silent/ugly failure at the
    boundary" discipline this driver's own worded-failure contract already promises."""
    for var in ("ATLAS_PG_DSN", "ATLAS_TEI_EMBED_URL", "ATLAS_TEI_RERANK_URL", "ATLAS_OPENAI_INDEX_DIR"):
        monkeypatch.delenv(var, raising=False)
    try:
        exit_code = main([
            "--limit-cases", "1", "--generators", "ollama", "--variant-generator", "none",
            "--output-dir", str(tmp_path / "out"), "--cassette-dir", str(tmp_path / "cassettes"),
            "--cache-dir", str(tmp_path / "cache"), "--run-id", "hermetic-test-2", "--git-sha", "a" * 40,
        ])
    except MissingEnvVarError:
        pytest.fail("main() must catch MissingEnvVarError at the CLI boundary, never let it escape")
    assert exit_code != 0


def test_main_rejects_an_unknown_embedders_value_before_any_live_call(tmp_path):
    exit_code = main([
        "--embedders", "voyage", "--limit-cases", "1", "--run-id", "hermetic-test-3",
        "--git-sha", "a" * 40, "--output-dir", str(tmp_path / "out"),
    ])
    assert exit_code == 2


# ---- main(): threads the variant stage's reconciled spend gate into run_matrix ----------------------


def test_main_threads_the_variant_stages_updated_spend_gate_into_run_matrix(monkeypatch, tmp_path):
    """The Important review finding (cross stage spend gate reconciliation): `build_variants_config`
    hands `main()` back an UPDATED gate with the variant stage's admitted estimate already recorded
    into it; `main()` must pass THAT gate, not the pristine one it built at the top of the function,
    into `run_matrix`'s own `spend_gate` argument -- otherwise stage 3's own admission inside
    `run_matrix` would independently recheck the SAME starting budget the variant stage was ALSO
    just admitted against (`matrix.live_driver.build_variants_config`'s own docstring). Every real
    live-construction call site is stubbed out here: this test is only about the wiring between
    `build_variants_config`'s return value and the `run_matrix` call, never a real component build."""
    import matrix.__main__ as matrix_main
    from matrix.spend_gate import SpendGate

    pristine_seen = {}
    updated_gate = SpendGate(spent={"anthropic": 999.0})

    def fake_build_variants_config(*, gate, **kwargs):
        pristine_seen["gate"] = gate
        return object(), updated_gate, None

    captured = {}

    def fake_run_matrix(*args, **kwargs):
        captured["spend_gate"] = kwargs["spend_gate"]
        return {"dropped_cells": []}

    monkeypatch.setattr(matrix_main, "_build_embedders", lambda *a, **k: [])
    monkeypatch.setattr(matrix_main, "build_reranker_components", lambda *a, **k: ())
    monkeypatch.setattr(matrix_main, "_build_generators", lambda *a, **k: [])
    monkeypatch.setattr(matrix_main, "_build_judges", lambda *a, **k: [])
    monkeypatch.setattr(matrix_main, "build_variants_config", fake_build_variants_config)
    monkeypatch.setattr(matrix_main, "run_matrix", fake_run_matrix)

    exit_code = matrix_main.main([
        "--limit-cases", "1", "--variant-generator", "anthropic",
        "--output-dir", str(tmp_path / "out"), "--cassette-dir", str(tmp_path / "cassettes"),
        "--cache-dir", str(tmp_path / "cache"), "--run-id", "hermetic-reconcile-test",
        "--git-sha", "a" * 40,
    ])

    assert exit_code == 0
    assert pristine_seen["gate"].spent_usd("anthropic") == 0.0  # build_variants_config got the pristine gate
    assert captured["spend_gate"] is updated_gate  # run_matrix got the UPDATED gate back, never the pristine one
