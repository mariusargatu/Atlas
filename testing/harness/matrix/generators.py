"""Stage 3: the top 1 to 2 retrieval configs (`matrix.select`) times the generator axis
`{Claude, GPT, qwen2.5:7b}`, scored by `quality.agent_metrics.answer_correctness_rate` (primary,
reference based) and `judge.panel.panel_vote` (secondary, D15's headline jury -- ITS FIRST REAL
CALLER anywhere in this repo; `judge/panel.py`'s own module docstring names "SP9's benchmark matrix
runner" as exactly this caller) -- plus the ONE off diagonal validation cell (research 14): does the
best retrieval config under one generator stay best under a second, checked (recorded), never
asserted.

Every model call in this stage -- the generator's own answer AND every judge in the panel -- goes
through `replay.gateway.GatewayChatModel`; the hermetic lane pins every one of them to REPLAY mode
against seeded cassette fixtures (keyless, zero egress). A live caller (deferred to the batched live
capture session, spend gated behind SP9 task 5) swaps the SAME gateways to RECORD mode; nothing in
this module's own contract changes either way.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from determinism.canonical import digest

from judge.llm_judge import judge_label
from judge.panel import PanelVote, panel_vote
from judge.rubric import RUBRIC_GROUNDEDNESS, Rubric

from quality.agent_metrics import answer_correctness_rate

from matrix.cases import MatrixCase
from matrix.chunks import deserialize_chunk
from matrix.select import RetrievalConfigResult

_GENERATE_INSTRUCTION = "Answer using ONLY the passages below; do not add facts the passages do not contain."


@dataclass(frozen=True)
class GeneratorComponent:
    """`model_snapshot` is the manifest contract's own shape (`{"provider", "model_id", "revision"}`);
    `gateway` is any `BaseChatModel` (a `replay.gateway.GatewayChatModel` in every hermetic test and
    every live caller alike -- this module never imports the gateway module itself, only the
    `BaseChatModel` interface, so a caller's own choice of mode is invisible here).

    `estimated_usd` (SP9 task 5): the pre call dollar estimate `matrix.runner.run_matrix`'s own
    spend gate checks BEFORE this cell runs at all (a real per token cost is only known AFTER a
    live call returns `usage_metadata`; this is the caller's own upfront budget estimate, never a
    measured actual). Defaults to `0.0`, so every existing hermetic caller (REPLAY mode, no live
    spend, no gate passed to `run_matrix` at all) is completely unaffected by this field's mere
    presence."""

    component_id: str
    model_snapshot: dict
    gateway: BaseChatModel
    estimated_usd: float = 0.0


@dataclass(frozen=True)
class GenerationCell:
    config_id: str
    generator_component_id: str
    per_case: dict[str, dict]  # case_id -> {"answer", "correctness", "panel_label", "panel_disagreed", "panel_votes"}
    #: A content addressed digest over every prompt this cell actually built (one lineage row per
    #: CELL, not per case, per D26's own "avoid a denormalized repeat per row" spirit -- a per query
    #: file traces back to its cell by `config_id`/`generator_component_id`, never carrying the
    #: cell's own lineage a second time). Honest and traceable: the SAME digest function the cassette
    #: key already uses, over the prompts this cell's own `generate` calls were keyed on.
    prompt_hash: str = ""


def build_generate_prompt(query: str, chunks: Sequence) -> str:
    """The one prompt building function this stage's `generate` and every hermetic cassette seeder
    share, so a test can seed a REPLAY cassette keyed on the EXACT string the runner will build."""
    context = "\n".join(f"- {c.text}" for c in chunks)
    return f"{_GENERATE_INSTRUCTION}\n\nQuestion: {query}\n\nPassages:\n{context}"


def _generate_answer(gateway: BaseChatModel, query: str, chunks: Sequence) -> str:
    result = gateway.invoke([HumanMessage(build_generate_prompt(query, chunks))])
    content = result.content
    return content if isinstance(content, str) else str(content)


def _panel_dict(panel: PanelVote) -> dict:
    return {"panel_label": panel.label, "panel_disagreed": panel.disagreed, "panel_votes": list(panel.votes)}


def run_generation_cell(
    config: RetrievalConfigResult,
    generator: GeneratorComponent,
    cases: Sequence[MatrixCase],
    *,
    judges: Sequence[BaseChatModel],
    rubric: Rubric = RUBRIC_GROUNDEDNESS,
) -> GenerationCell:
    """One (retrieval config, generator) cell: every case gets ONE generate call plus ONE panel of
    `len(judges)` judge calls, `judge.panel.panel_vote` aggregating the panel's labels -- the first
    real invocation of that function anywhere in this repo."""
    per_case: dict[str, dict] = {}
    prompts: dict[str, str] = {}
    for case in cases:
        chunks = [deserialize_chunk(d) for d in config.candidates.get(case.case_id, ())]
        prompts[case.case_id] = build_generate_prompt(case.query, chunks)
        answer = _generate_answer(generator.gateway, case.query, chunks)
        correctness = answer_correctness_rate(case.expected_facts, answer)
        context = "\n".join(c.text for c in chunks)
        labels = [judge_label(judge, rubric, case.query, answer, context) for judge in judges]
        panel = panel_vote(labels)
        per_case[case.case_id] = {"answer": answer, "correctness": correctness, **_panel_dict(panel)}
    prompt_hash = digest({case_id: prompts[case_id] for case_id in sorted(prompts)})
    return GenerationCell(
        config_id=config.config_id, generator_component_id=generator.component_id,
        per_case=per_case, prompt_hash=prompt_hash,
    )


@dataclass(frozen=True)
class OffDiagonalCheck:
    """Research 14's own recommendation, checked here rather than asserted: does the retrieval
    stage's own ranking with the best result first (`matrix.select`'s nDCG order) predict the generation stage's
    ranking too, for at least one generator shared by both configs. `retrieval_ranking_holds` is
    DATA (recorded in the run manifest), never a test assertion -- a tiny fixture's honest small n
    interval can easily disagree by noise alone, and failing a hermetic test on that would punish
    the measurement for being honest."""

    primary_config_id: str
    secondary_config_id: str
    shared_generator_id: str
    primary_mean_correctness: float
    secondary_mean_correctness: float
    retrieval_ranking_holds: bool


def off_diagonal_validation(
    primary_cell: GenerationCell, secondary_cell: GenerationCell, *, shared_generator_id: str
) -> OffDiagonalCheck:
    primary_scores = [c["correctness"] for c in primary_cell.per_case.values()]
    secondary_scores = [c["correctness"] for c in secondary_cell.per_case.values()]
    primary_mean = sum(primary_scores) / len(primary_scores) if primary_scores else 0.0
    secondary_mean = sum(secondary_scores) / len(secondary_scores) if secondary_scores else 0.0
    return OffDiagonalCheck(
        primary_config_id=primary_cell.config_id,
        secondary_config_id=secondary_cell.config_id,
        shared_generator_id=shared_generator_id,
        primary_mean_correctness=primary_mean,
        secondary_mean_correctness=secondary_mean,
        retrieval_ranking_holds=primary_mean >= secondary_mean,
    )


__all__ = [
    "GenerationCell",
    "GeneratorComponent",
    "OffDiagonalCheck",
    "build_generate_prompt",
    "off_diagonal_validation",
    "run_generation_cell",
]
