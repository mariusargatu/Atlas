"""`matrix.runner`, hermetic, end to end: the FULL staged runner (embedders -> rerankers ->
generators) against seeded REPLAY fixtures, producing a deterministic manifest plus HELM style per
query result files. Every explicit hermetic test bullet SP9 task 4's own spec names lives here:
a deterministic manifest + per query files; the content hash cache skipping recompute on a rerun;
`panel_vote` invoked in stage 3; the BM25 + exact_scan baseline rows present; two runs producing a
byte identical manifest.
"""
from __future__ import annotations

import json

import jsonschema
import pytest
from contract_tools import loader
from langchain_core.messages import HumanMessage

from replay.cassette_store import seed_cassette
from replay.gateway import GatewayChatModel

from judge.rubric import RUBRIC_GROUNDEDNESS, prompt as judge_prompt

from atlas.adapters.cassette_reranker import CassetteReranker
from atlas.ports.knowledge import Chunk

from matrix.cache import MatrixCache
from matrix.cases import MatrixCase
from matrix.embedders import BASELINE_COMPONENT_IDS, BM25_COMPONENT_ID, EXACT_SCAN_COMPONENT_ID, EmbedderComponent
from matrix.generators import GeneratorComponent, build_generate_prompt
from matrix.rerankers import RerankerComponent
from matrix.runner import MatrixRunConfig, run_matrix

_CASE_A = MatrixCase("case-a", "how much is plan a", frozenset({"d1"}), ({"fact_id": "a:price", "value": "10"},))
_CASE_B = MatrixCase("case-b", "how much is plan b", frozenset({"d2"}), ({"fact_id": "b:price", "value": "20"},))
_CASES = (_CASE_A, _CASE_B)

_D1 = Chunk(chunk_id="d1", doc_id="d1", text="plan a costs 10")
_D2 = Chunk(chunk_id="d2", doc_id="d2", text="plan b costs 20")
_D3 = Chunk(chunk_id="d3", doc_id="d3", text="extra filler text")
_D4 = Chunk(chunk_id="d4", doc_id="d4", text="more filler text")

_BGE_M3_ID = "bge-m3-local"
_OPENAI_ID = "openai-text-embedding-3-small"
_BGE_M3_MODEL = {"id": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"}
_OPENAI_MODEL = {"id": "text-embedding-3-small", "revision": "text-embedding-3-small"}

_RERANKER_ID = "bge-reranker-v2-m3"

_GEN_A_ID = "claude-sonnet-5-test"
_GEN_B_ID = "gpt-test"
_JUDGE_IDS = ("judge-claude", "judge-gpt", "judge-mixtral")

_ANSWER_A = "Plan a costs 10."
_ANSWER_B = "Plan b costs 20."


class _CountingSearch:
    def __init__(self, table: dict[str, list[Chunk]]) -> None:
        self._table = table
        self.calls = 0

    def __call__(self, case: MatrixCase) -> list[Chunk]:
        self.calls += 1
        return self._table[case.case_id]


def _perfect_table() -> dict[str, list[Chunk]]:
    return {"case-a": [_D1, _D3, _D2, _D4], "case-b": [_D2, _D3, _D1, _D4]}


def _bad_table() -> dict[str, list[Chunk]]:
    return {"case-a": [_D3, _D1, _D2, _D4], "case-b": [_D3, _D2, _D1, _D4]}


def _embedders() -> tuple[list[EmbedderComponent], dict[str, _CountingSearch]]:
    searches = {
        _BGE_M3_ID: _CountingSearch(_perfect_table()),
        _OPENAI_ID: _CountingSearch(_bad_table()),
        BM25_COMPONENT_ID: _CountingSearch(_perfect_table()),
        EXACT_SCAN_COMPONENT_ID: _CountingSearch(_perfect_table()),
    }
    components = [
        EmbedderComponent(_BGE_M3_ID, searches[_BGE_M3_ID], embedding_model=_BGE_M3_MODEL),
        EmbedderComponent(_OPENAI_ID, searches[_OPENAI_ID], embedding_model=_OPENAI_MODEL),
        EmbedderComponent(BM25_COMPONENT_ID, searches[BM25_COMPONENT_ID], embedding_model=None, is_baseline=True),
        EmbedderComponent(
            EXACT_SCAN_COMPONENT_ID, searches[EXACT_SCAN_COMPONENT_ID], embedding_model=_BGE_M3_MODEL, is_baseline=True
        ),
    ]
    return components, searches


def _rerankers() -> list[RerankerComponent]:
    return [RerankerComponent(_RERANKER_ID, CassetteReranker({}))]  # empty table: identity, no reorder


def _generators(cassette_dir) -> list[GeneratorComponent]:
    gw_a = GatewayChatModel(model_id=_GEN_A_ID, cassette_dir=cassette_dir, mode="replay")
    gw_b = GatewayChatModel(model_id=_GEN_B_ID, cassette_dir=cassette_dir, mode="replay")
    return [
        GeneratorComponent(_GEN_A_ID, {"provider": "anthropic", "model_id": _GEN_A_ID, "revision": _GEN_A_ID}, gw_a),
        GeneratorComponent(_GEN_B_ID, {"provider": "openai", "model_id": _GEN_B_ID, "revision": _GEN_B_ID}, gw_b),
    ]


def _judges(cassette_dir) -> list[GatewayChatModel]:
    return [GatewayChatModel(model_id=jid, cassette_dir=cassette_dir, mode="replay") for jid in _JUDGE_IDS]


def _seed_all_cassettes(cassette_dir) -> None:
    # Every (case, generator) generate call: the top-1 candidate under the WINNING (bge-m3 family)
    # config is the SAME chunk regardless of which of the two selected top configs produced it (the
    # reranker axis here is an identity no-op), so exactly ONE cassette entry per (case, generator)
    # covers both the primary and the off diagonal secondary cell.
    for gen_id in (_GEN_A_ID, _GEN_B_ID):
        seed_cassette(
            cassette_dir, [HumanMessage(build_generate_prompt(_CASE_A.query, [_D1]))],
            {"content": _ANSWER_A, "tool_calls": []}, gen_id,
        )
        seed_cassette(
            cassette_dir, [HumanMessage(build_generate_prompt(_CASE_B.query, [_D2]))],
            {"content": _ANSWER_B, "tool_calls": []}, gen_id,
        )
    # Judge panel: case-a unanimous PASS, case-b split PASS/PASS/FAIL (a real disagreement signal).
    verdicts_a = {"judge-claude": "PASS", "judge-gpt": "PASS", "judge-mixtral": "PASS"}
    verdicts_b = {"judge-claude": "PASS", "judge-gpt": "PASS", "judge-mixtral": "FAIL"}
    for jid, verdict in verdicts_a.items():
        seed_cassette(
            cassette_dir, judge_prompt(RUBRIC_GROUNDEDNESS, _CASE_A.query, _ANSWER_A, _D1.text),
            {"content": verdict, "tool_calls": []}, jid,
        )
    for jid, verdict in verdicts_b.items():
        seed_cassette(
            cassette_dir, judge_prompt(RUBRIC_GROUNDEDNESS, _CASE_B.query, _ANSWER_B, _D2.text),
            {"content": verdict, "tool_calls": []}, jid,
        )


def _config() -> MatrixRunConfig:
    return MatrixRunConfig(
        run_id="run-sp9-task4-test",
        git_sha="a" * 40,
        corpus_version="corpus-test-0.0.1",
        dataset_version="dataset-test-0.0.1",
        chunker_config_hash="chk-test",
        k_retrieval=1,
        seed=20260721,
        n_top_configs=2,
        reranker_depths=(20,),
    )


def _run(tmp_path, *, run_name: str, cache_dir, variants=None):
    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir(exist_ok=True)
    _seed_all_cassettes(cassette_dir)
    output_dir = tmp_path / run_name
    embedders, searches = _embedders()
    manifest = run_matrix(
        _CASES,
        embedders=embedders,
        rerankers=_rerankers(),
        generators=_generators(cassette_dir),
        judges=_judges(cassette_dir),
        judge_ids=_JUDGE_IDS,
        cache=MatrixCache(cache_dir),
        config=_config(),
        output_dir=output_dir,
        variants=variants,
    )
    return manifest, output_dir, searches


# ---- the full staged runner produces a manifest + per query files ------------------------------


def test_full_run_writes_a_manifest_and_per_query_files(tmp_path):
    manifest, output_dir, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk == manifest

    per_query_dir = output_dir / "per_query"
    assert (per_query_dir / "case-a.json").exists()
    assert (per_query_dir / "case-b.json").exists()
    case_a = json.loads((per_query_dir / "case-a.json").read_text())
    assert set(case_a) == {"embedders", "rerankers", "generators"}
    assert _BGE_M3_ID in case_a["embedders"]
    assert case_a["embedders"][_BGE_M3_ID]["recall_at_k"] == 1.0


def test_stages_present_embedders_rerankers_generators():
    pass  # covered structurally by the manifest shape assertions below; kept as a marker only


def test_manifest_has_all_three_stages(tmp_path):
    manifest, _, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    assert set(manifest["stages"]) == {"embedders", "rerankers", "generators"}
    assert len(manifest["stages"]["embedders"]) == 4
    assert len(manifest["stages"]["rerankers"]) == 4  # 4 embedders x 1 reranker x 1 depth
    assert len(manifest["stages"]["generators"]) >= 2  # primary x 2 generators, + the off diagonal cell


# ---- SP9 final review (finding I1): the variant comparison stage, wired into the manifest ---------


class _VariantRetriever:
    """Query string keyed canned responses, the same convention `test_matrix_variants.py`'s own
    fixtures use: enough to prove the WIRING (the manifest gains a real `variant_comparison` key
    when, and only when, a caller supplies `variants`); the variant-vs-variant BEHAVIOUR itself is
    proven in `test_matrix_variants.py`, not re-derived here."""

    def __init__(self, table: dict[str, list[Chunk]]) -> None:
        self._table = table

    def search_chunks(self, query: str, k: int, config) -> list[Chunk]:
        return self._table.get(query, [])[:k]


def test_variant_comparison_is_none_by_default_never_computed_when_variants_is_omitted(tmp_path):
    """Backward compatible: every caller before this stage existed (no `variants` argument at all)
    sees the SAME manifest shape as before, plus one new, always present key set to `None`."""
    manifest, _, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    assert manifest["variant_comparison"] is None


def test_variant_comparison_is_populated_with_all_three_variants_when_supplied(tmp_path):
    from atlas.adapters.inmemory_graph import InMemoryGraph
    from atlas.orchestration.agentic_rag import build_generate_prompt as build_variant_prompt
    from matrix.variants import VariantsConfig

    variant_cassette_dir = tmp_path / "variant-cassettes"
    variant_cassette_dir.mkdir()
    seed_cassette(
        variant_cassette_dir, [HumanMessage(build_variant_prompt(_CASE_A.query, [_D1], corrective=False))],
        {"content": _ANSWER_A, "tool_calls": []},
    )
    seed_cassette(
        variant_cassette_dir, [HumanMessage(build_variant_prompt(_CASE_B.query, [_D2], corrective=False))],
        {"content": _ANSWER_B, "tool_calls": []},
    )
    variants = VariantsConfig(
        retriever=_VariantRetriever({_CASE_A.query: [_D1], _CASE_B.query: [_D2]}),
        reranker=CassetteReranker({}),
        graph=InMemoryGraph((), ()),
        gateway=GatewayChatModel(model_id="claude-test", cassette_dir=variant_cassette_dir, mode="replay"),
    )
    manifest, _, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache", variants=variants)
    rows = manifest["variant_comparison"]
    assert [row["variant_id"] for row in rows] == ["agentic", "graph", "naive"]  # sorted, real rows
    assert all(row["n"] == 2 for row in rows)  # both cases scored under every variant


# ---- BM25 + exact_scan baseline rows present -----------------------------------------------------


def test_bm25_and_exact_scan_baseline_rows_are_present(tmp_path):
    manifest, _, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    embedder_ids = {row["component_id"] for row in manifest["stages"]["embedders"]}
    assert BASELINE_COMPONENT_IDS <= embedder_ids
    by_id = {row["component_id"]: row for row in manifest["stages"]["embedders"]}
    assert by_id[BM25_COMPONENT_ID]["is_baseline"] is True
    assert by_id[BM25_COMPONENT_ID]["lineage"]["embedding_model"]["id"] == "not-applicable"
    assert by_id[EXACT_SCAN_COMPONENT_ID]["is_baseline"] is True
    assert by_id[EXACT_SCAN_COMPONENT_ID]["lineage"]["embedding_model"] == _BGE_M3_MODEL


# ---- panel_vote invoked in stage 3 ---------------------------------------------------------------


def test_panel_vote_ran_in_stage_3_disagreement_and_labels_present(tmp_path):
    manifest, output_dir, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    assert any(row["disagreement_rate"] > 0.0 for row in manifest["stages"]["generators"])

    case_b = json.loads((output_dir / "per_query" / "case-b.json").read_text())
    panel_entries = list(case_b["generators"].values())
    assert panel_entries  # at least one (config, generator) cell ran for case-b
    assert any(e["panel_disagreed"] is True for e in panel_entries)  # the seeded PASS/PASS/FAIL split
    assert all(sorted(e["panel_votes"]) in ([0, 1, 1], [1, 1, 1]) for e in panel_entries)


# ---- quality/stats + quality/gate wiring ----------------------------------------------------------


def test_stage1_stats_are_holm_corrected_paired_deltas_with_95pct_ci(tmp_path):
    manifest, _, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    assert manifest["stage1_stats"]  # 4 embedders -> C(4,2) = 6 pairwise deltas
    for delta in manifest["stage1_stats"]:
        assert {"a", "b", "diff", "ci_lo", "ci_hi", "p_value", "p_value_holm"} <= set(delta)
        assert delta["p_value_holm"] >= delta["p_value"]


def test_stage3_stats_compare_the_generator_axis_under_the_primary_config(tmp_path):
    manifest, _, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    assert len(manifest["stage3_stats"]) == 1  # 2 generators -> exactly one pairwise comparison


def test_the_manifest_carries_no_always_pass_promotion_gate(tmp_path):
    """The gate this replaced ran `gate_on_lower_bound` with threshold=0.0 and variance_budget=1.0
    over nDCG in [0, 1], so PASS was the only reachable verdict; nothing read it. Stage 3 entry is
    decided by `select_top_configs`, which this asserts is what the manifest actually reports."""
    manifest, _, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    assert "promotion_gate" not in manifest
    assert manifest["top_configs"]  # the real selection mechanism, non empty


def test_off_diagonal_check_is_present_recorded_not_asserted(tmp_path):
    manifest, _, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    off = manifest["off_diagonal"]
    assert off is not None
    assert isinstance(off["retrieval_ranking_holds"], bool)
    assert off["primary_config_id"] != off["secondary_config_id"]


# ---- every stage row's lineage validates against the real D26 contract schema --------------------


def test_every_stage_rows_lineage_validates_against_the_manifest_contract(tmp_path):
    manifest, _, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    schema = loader.load_schema("manifest")
    for stage_rows in manifest["stages"].values():
        for row in stage_rows:
            jsonschema.validate(row["lineage"], schema)


# ---- content hash cache skips recompute on a rerun (full run granularity) ------------------------


def test_content_hash_cache_skips_recompute_on_a_full_run_rerun(tmp_path):
    cache_dir = tmp_path / "cache"
    _, _, searches1 = _run(tmp_path, run_name="run1", cache_dir=cache_dir)
    calls_after_first = {cid: s.calls for cid, s in searches1.items()}
    assert all(n > 0 for n in calls_after_first.values())

    _, _, searches2 = _run(tmp_path, run_name="run2", cache_dir=cache_dir)  # same cache dir: a rerun
    calls_after_second = {cid: s.calls for cid, s in searches2.items()}
    # searches2 are FRESH counting objects (never called yet before run_matrix); if the cache truly
    # skipped recompute, none of THIS run's own search callables were ever invoked.
    assert all(n == 0 for n in calls_after_second.values())


# ---- determinism: two runs produce a byte identical manifest -------------------------------------


def test_two_runs_produce_a_byte_identical_manifest(tmp_path):
    cache_dir = tmp_path / "cache"
    _, output_dir_1, _ = _run(tmp_path, run_name="run1", cache_dir=cache_dir)
    _, output_dir_2, _ = _run(tmp_path, run_name="run2", cache_dir=cache_dir)
    bytes_1 = (output_dir_1 / "manifest.json").read_bytes()
    bytes_2 = (output_dir_2 / "manifest.json").read_bytes()
    assert bytes_1 == bytes_2


def test_two_runs_with_independent_fresh_caches_still_produce_a_byte_identical_manifest(tmp_path):
    """Determinism does not depend on cache reuse at all: two runs with their OWN, independent
    (never shared) cache directories must still agree byte for byte, since every score is a pure
    function of (cases, components, config, seed)."""
    _, output_dir_1, _ = _run(tmp_path, run_name="fresh1", cache_dir=tmp_path / "cache-fresh-1")
    _, output_dir_2, _ = _run(tmp_path, run_name="fresh2", cache_dir=tmp_path / "cache-fresh-2")
    assert (output_dir_1 / "manifest.json").read_bytes() == (output_dir_2 / "manifest.json").read_bytes()


def test_run_matrix_raises_when_no_retrieval_config_exists(tmp_path):
    with pytest.raises(ValueError, match="at least one retrieval config"):
        run_matrix(
            _CASES, embedders=[], rerankers=[], generators=[], judges=[], judge_ids=(),
            cache=MatrixCache(tmp_path / "cache"), config=_config(), output_dir=tmp_path / "empty-run",
        )


# ---- SP9 task 5: the spend gate wired into the manifest's own dropped_cells list ------------------


def test_dropped_cells_is_present_and_empty_when_no_spend_gate_is_passed(tmp_path):
    """Backward compatible: every existing caller (no spend_gate argument at all) sees the SAME
    manifest shape as before this task, plus one new, always present, empty key."""
    manifest, _, _ = _run(tmp_path, run_name="run1", cache_dir=tmp_path / "cache")
    assert manifest["dropped_cells"] == []


def test_a_cell_over_budget_is_skipped_and_logged_never_silently(tmp_path):
    from matrix.spend_gate import SpendGate

    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir(exist_ok=True)
    _seed_all_cassettes(cassette_dir)
    gw_a = GatewayChatModel(model_id=_GEN_A_ID, cassette_dir=cassette_dir, mode="replay")
    gw_b = GatewayChatModel(model_id=_GEN_B_ID, cassette_dir=cassette_dir, mode="replay")
    generators = [
        GeneratorComponent(
            _GEN_A_ID, {"provider": "anthropic", "model_id": _GEN_A_ID, "revision": _GEN_A_ID}, gw_a,
            estimated_usd=1.0,
        ),
        # openai's remaining budget (the default $20 ceiling) cannot cover this cell's own estimate.
        GeneratorComponent(
            _GEN_B_ID, {"provider": "openai", "model_id": _GEN_B_ID, "revision": _GEN_B_ID}, gw_b,
            estimated_usd=100.0,
        ),
    ]
    embedders, _ = _embedders()
    manifest = run_matrix(
        _CASES, embedders=embedders, rerankers=_rerankers(), generators=generators,
        judges=_judges(cassette_dir), judge_ids=_JUDGE_IDS, cache=MatrixCache(tmp_path / "cache"),
        config=_config(), output_dir=tmp_path / "gated-run", spend_gate=SpendGate(),
    )
    assert len(manifest["dropped_cells"]) == 1
    dropped = manifest["dropped_cells"][0]
    assert dropped["provider"] == "openai"
    assert _GEN_B_ID in dropped["component_id"]
    assert "openai" in dropped["reason"]

    generator_ids = {row["generator_component_id"] for row in manifest["stages"]["generators"]}
    assert _GEN_A_ID in generator_ids  # anthropic's cell still ran (within its own ceiling)
    assert _GEN_B_ID not in generator_ids  # openai's cell never ran at all


def test_an_always_runs_provider_is_never_dropped_even_with_a_huge_estimate(tmp_path):
    from matrix.spend_gate import SpendGate

    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir(exist_ok=True)
    _seed_all_cassettes(cassette_dir)
    gw_a = GatewayChatModel(model_id=_GEN_A_ID, cassette_dir=cassette_dir, mode="replay")
    generators = [
        GeneratorComponent(
            _GEN_A_ID, {"provider": "ollama", "model_id": _GEN_A_ID, "revision": _GEN_A_ID}, gw_a,
            estimated_usd=1_000_000.0,
        ),
    ]
    embedders, _ = _embedders()
    manifest = run_matrix(
        _CASES, embedders=embedders, rerankers=_rerankers(), generators=generators,
        judges=_judges(cassette_dir), judge_ids=_JUDGE_IDS, cache=MatrixCache(tmp_path / "cache"),
        config=_config(), output_dir=tmp_path / "ollama-run", spend_gate=SpendGate(),
    )
    assert manifest["dropped_cells"] == []
    assert manifest["stages"]["generators"]  # the cell ran


def test_paid_generator_cells_left_at_the_default_zero_estimate_are_dropped_not_silently_admitted(tmp_path):
    """The live-money safety net, exercised at the `run_matrix` level: `_generators()` builds both
    cells with `GeneratorComponent`'s own default `estimated_usd=0.0` (a live driver bug this repo
    must never trust silently), against a fresh `SpendGate` with FULL headroom on both providers.
    Neither cell may be admitted as "free" -- both are dropped and logged, never silently run."""
    from matrix.spend_gate import SpendGate

    cassette_dir = tmp_path / "cassettes"
    cassette_dir.mkdir(exist_ok=True)
    _seed_all_cassettes(cassette_dir)
    embedders, _ = _embedders()
    manifest = run_matrix(
        _CASES, embedders=embedders, rerankers=_rerankers(), generators=_generators(cassette_dir),
        judges=_judges(cassette_dir), judge_ids=_JUDGE_IDS, cache=MatrixCache(tmp_path / "cache"),
        config=_config(), output_dir=tmp_path / "zero-estimate-run", spend_gate=SpendGate(),
    )
    assert len(manifest["dropped_cells"]) == 2
    dropped_providers = {d["provider"] for d in manifest["dropped_cells"]}
    assert dropped_providers == {"anthropic", "openai"}
    assert manifest["stages"]["generators"] == []  # neither cell ran
