"""`matrix.generators`, hermetic (stage 3): the generator axis over the top retrieval config(s),
scored by reference based correctness (primary) and `judge.panel.panel_vote` (secondary) -- panel_vote
invoked FOR REAL here, its first caller anywhere in this repo, not a stub. Every model call (the
generator's own answer, every judge in the panel) is a real REPLAY round trip through
`replay.gateway.GatewayChatModel` against seeded cassettes, keyless and networkless.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from replay.cassette_store import seed_cassette
from replay.gateway import GatewayChatModel

from judge.rubric import RUBRIC_GROUNDEDNESS, prompt as judge_prompt

from atlas.ports.knowledge import Chunk

from matrix.cases import MatrixCase
from matrix.chunks import serialize_chunk
from matrix.generators import (
    GenerationCell,
    GeneratorComponent,
    build_generate_prompt,
    off_diagonal_validation,
    run_generation_cell,
)
from matrix.select import RetrievalConfigResult

_CASE_A = MatrixCase(
    "case-a", "how much is the plan", frozenset(), ({"fact_id": "x:price", "value": "10"},),
)
_CASE_B = MatrixCase(
    "case-b", "how much is the other plan", frozenset(), ({"fact_id": "y:price", "value": "99"},),
)
_CASES = (_CASE_A, _CASE_B)

_CHUNK_A = Chunk(chunk_id="ca", doc_id="ca", text="the plan costs 10")
_CHUNK_B = Chunk(chunk_id="cb", doc_id="cb", text="the plan costs 20")

_ANSWER_A = "The plan costs 10."
_ANSWER_B = "The plan costs 20."

_GENERATOR_MODEL_ID = "claude-sonnet-5-test"
_JUDGE_IDS = ("judge-claude", "judge-gpt", "judge-mixtral")


def _config() -> RetrievalConfigResult:
    return RetrievalConfigResult(
        "emb::bge-reranker-v2-m3@20",
        {"case-a": (serialize_chunk(_CHUNK_A),), "case-b": (serialize_chunk(_CHUNK_B),)},
        ndcg_point=0.9,
    )


def _generator(cassette_dir) -> GeneratorComponent:
    gw = GatewayChatModel(model_id=_GENERATOR_MODEL_ID, cassette_dir=cassette_dir, mode="replay")
    snapshot = {"provider": "anthropic", "model_id": _GENERATOR_MODEL_ID, "revision": _GENERATOR_MODEL_ID}
    return GeneratorComponent("claude-sonnet-5", snapshot, gw)


def _judges(cassette_dir) -> list[GatewayChatModel]:
    return [GatewayChatModel(model_id=jid, cassette_dir=cassette_dir, mode="replay") for jid in _JUDGE_IDS]


def _seed_generation(cassette_dir) -> None:
    seed_cassette(
        cassette_dir, [HumanMessage(build_generate_prompt(_CASE_A.query, [_CHUNK_A]))],
        {"content": _ANSWER_A, "tool_calls": []}, _GENERATOR_MODEL_ID,
    )
    seed_cassette(
        cassette_dir, [HumanMessage(build_generate_prompt(_CASE_B.query, [_CHUNK_B]))],
        {"content": _ANSWER_B, "tool_calls": []}, _GENERATOR_MODEL_ID,
    )


def _seed_judges_split_on_case_a_unanimous_on_case_b(cassette_dir) -> None:
    """case-a: judges split PASS/PASS/FAIL (panel disagrees, majority PASS). case-b: judges agree
    PASS unanimously (no disagreement)."""
    verdicts_a = {"judge-claude": "PASS", "judge-gpt": "PASS", "judge-mixtral": "FAIL"}
    verdicts_b = {"judge-claude": "PASS", "judge-gpt": "PASS", "judge-mixtral": "PASS"}
    context_a = _CHUNK_A.text
    context_b = _CHUNK_B.text
    for jid, verdict in verdicts_a.items():
        seed_cassette(
            cassette_dir, judge_prompt(RUBRIC_GROUNDEDNESS, _CASE_A.query, _ANSWER_A, context_a),
            {"content": verdict, "tool_calls": []}, jid,
        )
    for jid, verdict in verdicts_b.items():
        seed_cassette(
            cassette_dir, judge_prompt(RUBRIC_GROUNDEDNESS, _CASE_B.query, _ANSWER_B, context_b),
            {"content": verdict, "tool_calls": []}, jid,
        )


def test_run_generation_cell_produces_one_answer_and_correctness_per_case(tmp_path):
    _seed_generation(tmp_path)
    _seed_judges_split_on_case_a_unanimous_on_case_b(tmp_path)
    cell = run_generation_cell(_config(), _generator(tmp_path), _CASES, judges=_judges(tmp_path))
    assert isinstance(cell, GenerationCell)
    assert cell.per_case["case-a"]["answer"] == _ANSWER_A
    assert cell.per_case["case-a"]["correctness"] == 1.0  # "10" IS in the answer text
    assert cell.per_case["case-b"]["correctness"] == 0.0  # "99" is NOT in "The plan costs 20."


def test_panel_vote_is_really_invoked_majority_and_disagreement_flag_are_correct(tmp_path):
    """The direct proof panel_vote ran for real: case-a's PASS/PASS/FAIL becomes majority label 1
    WITH disagreed=True; case-b's unanimous PASS becomes label 1 with disagreed=False. Neither
    value is achievable by any single judge_label call alone, only by panel_vote's own aggregation."""
    _seed_generation(tmp_path)
    _seed_judges_split_on_case_a_unanimous_on_case_b(tmp_path)
    cell = run_generation_cell(_config(), _generator(tmp_path), _CASES, judges=_judges(tmp_path))
    assert cell.per_case["case-a"]["panel_label"] == 1
    assert cell.per_case["case-a"]["panel_disagreed"] is True
    assert sorted(cell.per_case["case-a"]["panel_votes"]) == [0, 1, 1]
    assert cell.per_case["case-b"]["panel_label"] == 1
    assert cell.per_case["case-b"]["panel_disagreed"] is False
    assert cell.per_case["case-b"]["panel_votes"] == [1, 1, 1]


def test_ties_fail_closed_through_the_real_panel_vote_call(tmp_path):
    """An even panel split (2 judges) ties: `judge.panel.panel_vote` fails closed to label 0. Proven
    through this module's own real call, not asserted directly against `panel_vote` in isolation."""
    seed_cassette(
        tmp_path, [HumanMessage(build_generate_prompt(_CASE_A.query, [_CHUNK_A]))],
        {"content": _ANSWER_A, "tool_calls": []}, _GENERATOR_MODEL_ID,
    )
    two_judge_ids = ("judge-claude", "judge-gpt")
    seed_cassette(
        tmp_path, judge_prompt(RUBRIC_GROUNDEDNESS, _CASE_A.query, _ANSWER_A, _CHUNK_A.text),
        {"content": "PASS", "tool_calls": []}, "judge-claude",
    )
    seed_cassette(
        tmp_path, judge_prompt(RUBRIC_GROUNDEDNESS, _CASE_A.query, _ANSWER_A, _CHUNK_A.text),
        {"content": "FAIL", "tool_calls": []}, "judge-gpt",
    )
    judges = [GatewayChatModel(model_id=jid, cassette_dir=tmp_path, mode="replay") for jid in two_judge_ids]
    config = RetrievalConfigResult(
        "emb::bge-reranker-v2-m3@20", {"case-a": (serialize_chunk(_CHUNK_A),)}, ndcg_point=0.9,
    )
    cell = run_generation_cell(config, _generator(tmp_path), [_CASE_A], judges=judges)
    assert cell.per_case["case-a"]["panel_label"] == 0
    assert cell.per_case["case-a"]["panel_disagreed"] is True


def test_off_diagonal_validation_records_whether_the_retrieval_ranking_held():
    primary = GenerationCell("primary-config", "gen", {"c1": {"correctness": 1.0}, "c2": {"correctness": 1.0}})
    secondary_worse = GenerationCell("secondary-config", "gen", {"c1": {"correctness": 0.0}, "c2": {"correctness": 0.5}})
    check = off_diagonal_validation(primary, secondary_worse, shared_generator_id="gen")
    assert check.primary_config_id == "primary-config"
    assert check.secondary_config_id == "secondary-config"
    assert check.primary_mean_correctness == 1.0
    assert check.secondary_mean_correctness == 0.25
    assert check.retrieval_ranking_holds is True  # primary (the nDCG-ranked best) really did score higher


def test_generation_cell_prompt_hash_is_a_real_digest_over_the_cells_own_prompts(tmp_path):
    from determinism.canonical import digest

    from matrix.generators import build_generate_prompt

    _seed_generation(tmp_path)
    _seed_judges_split_on_case_a_unanimous_on_case_b(tmp_path)
    cell = run_generation_cell(_config(), _generator(tmp_path), _CASES, judges=_judges(tmp_path))
    expected = digest(
        {
            "case-a": build_generate_prompt(_CASE_A.query, [_CHUNK_A]),
            "case-b": build_generate_prompt(_CASE_B.query, [_CHUNK_B]),
        }
    )
    assert cell.prompt_hash == expected
    assert cell.prompt_hash != ""


def test_off_diagonal_validation_is_recorded_even_when_the_ranking_does_not_hold():
    """The check is DATA, never an assertion: a config the retrieval stage ranked worse can still
    score higher under a given generator, and that must be recorded honestly, not hidden."""
    primary = GenerationCell("primary-config", "gen", {"c1": {"correctness": 0.0}})
    secondary_better = GenerationCell("secondary-config", "gen", {"c1": {"correctness": 1.0}})
    check = off_diagonal_validation(primary, secondary_better, shared_generator_id="gen")
    assert check.retrieval_ranking_holds is False
