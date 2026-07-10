"""Variant comparison stage (SP9 final review, finding I1): naive vs agentic vs graph RAG, measured
over the SAME `MatrixCase` sequence with the SAME existing metrics
(`quality.agent_metrics.answer_correctness_rate`, `quality.ir_metrics.recall_at_k`/`ndcg_at_k`),
never a fourth metric definition. This closes the review's own finding: `orchestration/agentic_rag.py`
and `orchestration/graph_rag.py` are real, tamper-proven subgraphs, but nothing in
`testing/harness/matrix/` ever invoked either one -- the matrix's earlier three stages sweep
embedders, rerankers, and generators, never the variant axis SP9 is named for. This module is that
caller.

All three variants share the query-in/chunks-plus-answer-out contract (D6), over the SAME injected
`retriever`/`reranker`/`graph`/`gateway`, so any difference between the three rows is genuinely the
variant's own mechanism (CRAG rewrite, graph traversal), never a different corpus or a different
model:

  naive    -- the fixed pipeline every earlier matrix stage already sweeps in spirit: retrieve
              (widened, `rerank_enabled=False`) -> rerank -> truncate -> ONE generate call, no CRAG
              grading, no rewrite, no graph traversal. Built here as a plain async function, never a
              LangGraph `StateGraph` (there is no conditional edge to model), sharing
              `agentic_rag.build_generate_prompt` so its prompt text is byte-identical to the other
              two variants whenever they too take the `corrective=False` path -- the "one shared
              prompt shape, never re-derived" discipline `graph_rag.py`'s own module docstring
              already holds.
  agentic  -- `orchestration.agentic_rag.build_agentic_rag_graph`, invoked with the case's own
              `query_entity_ids` (`matrix.cases`'s real registry-derived supplier) as CRAG grading's
              ground truth, so `grade_documents` is exercised for real rather than the vacuous
              pass-through an empty set forces.
  graph    -- `orchestration.graph_rag.build_graph_rag_graph`, invoked over the SAME retriever and a
              caller-supplied `KnowledgeGraph`.

Fully hermetic by construction: `retriever`/`reranker`/`graph`/`gateway` are all injected
(deterministic fixtures in every hermetic test, a real adapter and a REPLAY/RECORD
`GatewayChatModel` in a live matrix run, with no change to this module's own contract) -- the SAME
"seeded fixture now, live swap deferred" shape `matrix.embedders`/`matrix.rerankers` already use.
`run_matrix` (`matrix/runner.py`) is this stage's own caller, wired the same optional,
backward-compatible way SP9 task 5's `spend_gate` was: a caller that passes no `VariantsConfig` sees
`manifest["variant_comparison"]` stay `None`, exactly the prior manifest shape.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from quality import ir_metrics
from quality.agent_metrics import answer_correctness_rate

from atlas.domain.retrieval import RetrievalConfig
from atlas.orchestration.agentic_rag import build_agentic_rag_graph, build_generate_prompt
from atlas.orchestration.graph_rag import build_graph_rag_graph
from atlas.ports.knowledge import Chunk, Retriever
from atlas.ports.knowledge_graph import KnowledgeGraph
from atlas.ports.reranker import Reranker

from matrix.cases import MatrixCase

NAIVE = "naive"
AGENTIC = "agentic"
GRAPH = "graph"
#: Every id this stage ever produces a row for, in a stable (never dict/set-derived) order.
VARIANT_IDS: tuple[str, ...] = (AGENTIC, GRAPH, NAIVE)

# Retrieval widths, matching `agentic_rag`/`graph_rag`'s own `_K_FUSED`/`_K_FINAL` (the naive
# variant's fixed pipeline uses the identical widths so a "wider pool, then truncate" comparison is
# apples to apples across all three, never a narrower or wider naive baseline than its siblings get).
_K_FUSED = 20
_K_FINAL = 3


@dataclass(frozen=True)
class VariantCaseResult:
    """One variant's one-case result: `doc_ids` is the final (already truncated) chunk order, the
    same shape `quality.ir_metrics` scores; `answer` is the generated text
    `quality.agent_metrics.answer_correctness_rate` dereferences `case.expected_facts` against."""

    case_id: str
    doc_ids: tuple[str, ...]
    answer: str
    correctness: float
    recall_at_k: float
    ndcg_at_k: float

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "doc_ids": list(self.doc_ids),
            "answer": self.answer,
            "correctness": self.correctness,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class VariantResult:
    """One variant's row across every case: `per_case` keyed by `case_id` (never a list -- a
    caller joining against a specific case does not have to scan), the three means the manifest row
    reports, guarded to 0.0 on an empty case set (the same convention `quality.agent_metrics`'s own
    module docstring documents for every metric here, applied uniformly)."""

    variant_id: str
    per_case: dict[str, VariantCaseResult] = field(default_factory=dict)

    @property
    def mean_correctness(self) -> float:
        return _mean([r.correctness for r in self.per_case.values()])

    @property
    def mean_recall_at_k(self) -> float:
        return _mean([r.recall_at_k for r in self.per_case.values()])

    @property
    def mean_ndcg_at_k(self) -> float:
        return _mean([r.ndcg_at_k for r in self.per_case.values()])

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "n": len(self.per_case),
            "mean_correctness": self.mean_correctness,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_ndcg_at_k": self.mean_ndcg_at_k,
        }


def _score(case: MatrixCase, chunks: Sequence[Chunk], answer: str, k: int) -> VariantCaseResult:
    doc_ids = tuple(c.doc_id for c in chunks)
    return VariantCaseResult(
        case_id=case.case_id,
        doc_ids=doc_ids,
        answer=answer,
        correctness=answer_correctness_rate(case.expected_facts, answer),
        recall_at_k=ir_metrics.recall_at_k(doc_ids, case.relevant_doc_ids, k),
        ndcg_at_k=ir_metrics.ndcg_at_k(doc_ids, case.relevant_doc_ids, k),
    )


async def _agenerate_text(gateway: BaseChatModel, prompt: str) -> str:
    result = await gateway._agenerate([HumanMessage(prompt)])
    message = result.generations[0].message
    return message.content if isinstance(message.content, str) else str(message.content)


async def _run_naive(
    cases: Sequence[MatrixCase], *, retriever: Retriever, reranker: Reranker, gateway: BaseChatModel, k: int,
) -> VariantResult:
    per_case: dict[str, VariantCaseResult] = {}
    for case in cases:
        candidates = retriever.search_chunks(
            case.query, k=_K_FUSED, config=RetrievalConfig(rerank_enabled=False, k_fused=_K_FUSED)
        )
        chunks = reranker.rerank(case.query, candidates)[:_K_FINAL]
        prompt = build_generate_prompt(case.query, chunks, corrective=False)
        answer = await _agenerate_text(gateway, prompt)
        per_case[case.case_id] = _score(case, chunks, answer, k)
    return VariantResult(variant_id=NAIVE, per_case=per_case)


async def _run_agentic(
    cases: Sequence[MatrixCase], *, retriever: Retriever, reranker: Reranker, gateway: BaseChatModel, k: int,
) -> VariantResult:
    graph_app = build_agentic_rag_graph(gateway, retriever=retriever, reranker=reranker)
    per_case: dict[str, VariantCaseResult] = {}
    for case in cases:
        out = await graph_app.ainvoke(
            {"query": case.query, "query_entity_ids": tuple(sorted(case.query_entity_ids))}
        )
        per_case[case.case_id] = _score(case, out["chunks"], out["answer"], k)
    return VariantResult(variant_id=AGENTIC, per_case=per_case)


async def _run_graph(
    cases: Sequence[MatrixCase],
    *,
    retriever: Retriever,
    reranker: Reranker,
    graph: KnowledgeGraph,
    gateway: BaseChatModel,
    k: int,
) -> VariantResult:
    graph_app = build_graph_rag_graph(gateway, retriever=retriever, reranker=reranker, graph=graph)
    per_case: dict[str, VariantCaseResult] = {}
    for case in cases:
        out = await graph_app.ainvoke({"query": case.query})
        per_case[case.case_id] = _score(case, out["chunks"], out["answer"], k)
    return VariantResult(variant_id=GRAPH, per_case=per_case)


async def run_variant_comparison(
    cases: Sequence[MatrixCase],
    *,
    retriever: Retriever,
    reranker: Reranker,
    graph: KnowledgeGraph,
    gateway: BaseChatModel,
    k: int = _K_FINAL,
) -> dict[str, VariantResult]:
    """Run naive, agentic, and graph over the SAME `cases`, sharing the SAME
    retriever/reranker/graph/gateway fixtures. Returns one `VariantResult` per `VARIANT_IDS` entry,
    keyed by variant id. Async because both `agentic`/`graph` are compiled LangGraph subgraphs
    (`.ainvoke`); `matrix.runner.run_matrix` (a sync function) bridges this with `asyncio.run` only
    when a caller actually supplies a `VariantsConfig`."""
    naive = await _run_naive(cases, retriever=retriever, reranker=reranker, gateway=gateway, k=k)
    agentic = await _run_agentic(cases, retriever=retriever, reranker=reranker, gateway=gateway, k=k)
    graph_result = await _run_graph(
        cases, retriever=retriever, reranker=reranker, graph=graph, gateway=gateway, k=k
    )
    return {NAIVE: naive, AGENTIC: agentic, GRAPH: graph_result}


@dataclass(frozen=True)
class VariantsConfig:
    """Every fixture `run_variant_comparison` needs, bundled so `run_matrix` takes exactly one new
    optional parameter. Omitted (`None`, the default for every caller before this stage existed)
    skips the stage entirely: `manifest["variant_comparison"]` stays `None`, the same
    "absent config, absent effect" shape SP9 task 5's `spend_gate=None` already established."""

    retriever: Retriever
    reranker: Reranker
    graph: KnowledgeGraph
    gateway: BaseChatModel
    k: int = _K_FINAL


def variant_comparison_rows(results: dict[str, VariantResult]) -> list[dict]:
    """The manifest's own `variant_comparison` value: one row per `VARIANT_IDS` entry, sorted by
    `variant_id` so two runs over the same inputs agree byte for byte (the same determinism
    discipline `matrix.runner`'s own `dropped_cells` list already holds to)."""
    return [results[vid].to_dict() for vid in sorted(results)]


__all__ = [
    "AGENTIC",
    "GRAPH",
    "NAIVE",
    "VARIANT_IDS",
    "VariantCaseResult",
    "VariantResult",
    "VariantsConfig",
    "run_variant_comparison",
    "variant_comparison_rows",
]
