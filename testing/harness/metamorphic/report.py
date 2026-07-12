"""The three deterministic, judge free invariants (D32) and the report that runs them over a
frozen family: no judge, no live model call, no wall clock. `run_family` is the one function that
actually calls the retrieval stack (`InMemoryRetriever.search_chunks`, real, deterministic keyword
overlap); everything downstream of that call is pure arithmetic over the returned id lists,
`quality.ir_metrics` and `quality.agent_metrics` unmodified, no third copy of either living here.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.retrieval import RetrievalConfig
from atlas.ports.knowledge import Chunk

from quality import agent_metrics, ir_metrics

from metamorphic.families import ALL_FAMILIES, STUB_CORPUS, STUB_K, MetamorphicFamily

__all__ = [
    "FamilyResult",
    "MetamorphicReport",
    "id_based_retrieval_agreement",
    "rank_overlap_floor_holds",
    "registry_answer_equivalence_holds",
    "run_all_families",
    "run_family",
]


# ---- invariant 1: ID based retrieval agreement -----------------------------------------------------


def id_based_retrieval_agreement(
    retrieved_by_member: Sequence[Sequence[str]], ground_truth_doc_id: str, k: int
) -> bool:
    """True iff EVERY member's top-k retrieval contains the registry's ground truth chunk,
    regardless of how the question was worded. `hit_rate_at_k` is exactly this per member (1.0 iff
    the chunk id is present); this just requires it of every member, not merely on average."""
    return all(
        ir_metrics.hit_rate_at_k(retrieved, {ground_truth_doc_id}, k) == 1.0
        for retrieved in retrieved_by_member
    )


# ---- invariant 2: rank overlap floor ---------------------------------------------------------------


def rank_overlap_floor_holds(retrieved_by_member: Sequence[Sequence[str]], k: int, floor: float) -> bool:
    """True iff every PAIR of members' retrieved id sets overlaps at or above `floor`
    (`quality.ir_metrics.rank_overlap_at_k`). A family of one member trivially holds (no pair to
    check); every family this module ships has at least three."""
    pairs = itertools.combinations(retrieved_by_member, 2)
    return all(ir_metrics.rank_overlap_at_k(a, b, k) >= floor for a, b in pairs)


def _min_rank_overlap(retrieved_by_member: Sequence[Sequence[str]], k: int) -> float:
    pairs = itertools.combinations(retrieved_by_member, 2)
    overlaps = [ir_metrics.rank_overlap_at_k(a, b, k) for a, b in pairs]
    return min(overlaps) if overlaps else 1.0


# ---- invariant 3: registry derived answer equivalence ----------------------------------------------


def registry_answer_equivalence_holds(expected_facts: Sequence[dict[str, object]], answer: str) -> bool:
    """True iff `answer` fully grounds `expected_facts` (`quality.agent_metrics.
    answer_correctness_rate` == 1.0). Every family here shares ONE frozen answer across all its
    members BY DESIGN: the metamorphic relation is that the correct answer never depends on how the
    question was worded, so checking it once per family already checks it for every member."""
    return agent_metrics.answer_correctness_rate(expected_facts, answer) == 1.0


# ---- the report -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyResult:
    family_id: str
    kind: str
    contradiction_id: str
    retrieved_by_member: tuple[tuple[str, ...], ...]
    id_agreement: bool
    min_rank_overlap: float
    rank_overlap_floor: float
    rank_overlap_holds: bool
    answer_equivalence: bool

    @property
    def holds(self) -> bool:
        return self.id_agreement and self.rank_overlap_holds and self.answer_equivalence

    def render(self) -> str:
        mark = "PASS" if self.holds else "FAIL"
        lines = [
            f"[{mark}] {self.family_id}  (kind={self.kind}, seed={self.contradiction_id})",
            f"       id based retrieval agreement: {self.id_agreement}",
            f"       rank overlap: min={self.min_rank_overlap:.2f} floor={self.rank_overlap_floor:.2f}"
            f" holds={self.rank_overlap_holds}",
            f"       registry derived answer equivalence: {self.answer_equivalence}",
        ]
        for retrieved in self.retrieved_by_member:
            lines.append(f"         retrieved: {list(retrieved)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class MetamorphicReport:
    results: tuple[FamilyResult, ...]

    @property
    def all_hold(self) -> bool:
        return all(result.holds for result in self.results)

    def render(self) -> str:
        lines = [
            "# Metamorphic report (registry seeded, D32 judge free invariants)",
            f"families: {len(self.results)}  all_hold={self.all_hold}",
            "",
        ]
        for result in self.results:
            lines.append(result.render())
            lines.append("")
        return "\n".join(lines)


def run_family(
    family: MetamorphicFamily, *, corpus: Sequence[Chunk] = STUB_CORPUS, k: int = STUB_K
) -> FamilyResult:
    """Runs the REAL retrieval stack (`InMemoryRetriever.search_chunks`, deterministic keyword
    overlap, no wall clock, no randomness) once per member, then checks all three invariants over
    the results. `corpus`/`k` default to this module's own frozen stub fixture, so a bare
    `run_family(PARAPHRASE_FAMILY)` reproduces the hermetic report; a caller wanting the live lane
    (pinned pgvector index, `metamorphic.__main__`'s own report) passes a real `PgvectorRetriever`
    driven corpus/k instead -- `run_family` itself never imports or constructs one, keeping this
    module retriever agnostic."""
    retriever = InMemoryRetriever(list(corpus))
    config = RetrievalConfig()
    retrieved_by_member = tuple(
        tuple(chunk.doc_id for chunk in retriever.search_chunks(member.question, k=k, config=config))
        for member in family.members
    )
    return FamilyResult(
        family_id=family.family_id,
        kind=family.kind,
        contradiction_id=family.contradiction_id,
        retrieved_by_member=retrieved_by_member,
        id_agreement=id_based_retrieval_agreement(retrieved_by_member, family.ground_truth_doc_id, k),
        min_rank_overlap=_min_rank_overlap(retrieved_by_member, k),
        rank_overlap_floor=family.rank_overlap_floor,
        rank_overlap_holds=rank_overlap_floor_holds(retrieved_by_member, k, family.rank_overlap_floor),
        answer_equivalence=registry_answer_equivalence_holds(family.expected_facts, family.answer),
    )


def run_all_families(families: Sequence[MetamorphicFamily] = ALL_FAMILIES) -> MetamorphicReport:
    """The deterministic report `task metamorphic` always runs: every committed family, replayed
    against the frozen stub fixture. Defaults to `families.ALL_FAMILIES`; a caller may pass a
    subset (a hermetic test isolating one family) or, in the live lane, the same families run
    through a real retriever instead."""
    return MetamorphicReport(results=tuple(run_family(family) for family in families))
