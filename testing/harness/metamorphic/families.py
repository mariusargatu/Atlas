"""The frozen metamorphic families (see the package docstring for the full picture): literal,
committed data, never recomputed at import time. "Frozen" is the operative word twice over here,
mirroring `evals.datasets.metamorphic_golden`'s own precedent: the QUESTION TEXT is frozen (a human
curated snapshot, "generate, freeze, replay"), and the STUB retrieval corpus these questions run
against is frozen too (real content copied verbatim from the committed corpus render, not read
from disk at import time, so this module has zero filesystem dependency and zero import time I/O).

`testing/tests/test_metamorphic.py::test_family_ground_truth_matches_the_registry_contradiction`
is the machine check that this frozen data has not silently drifted from the real registry: it
loads `corpus/registry/core.yaml` and asserts `WINNING_FACT_ID`/`WINNING_FACT_VALUE` below dereference
to the SAME `conflict-daniel-contract` winning fact `judge.provisional.manufactured_cases` already
consumes. The two real chunk ids (`WINNING_DOC_ID`/`LOSING_DOC_ID`) are NOT derivable from the
registry alone (they come from the separate render/chunk pipeline); those are pinned against the
identical constants `dataset_tools/seed_cases.jsonl`'s own `gen-fact-contract_term-daniel-2025-
contract_months` and `gen-fact-plan-fiber-100-contract_months` cases already commit, so there is
exactly one place in this whole repo that has ever had to know these two hashes by heart.
"""
from __future__ import annotations

from dataclasses import dataclass

from atlas.ports.knowledge import Chunk

# ---- the registry ground truth this whole module is seeded from (conflict-daniel-contract) -------

WINNING_FACT_ID = "contract_term-daniel-2025:contract_months"
WINNING_FACT_VALUE = 12
EXPECTED_FACTS: tuple[dict[str, object], ...] = ({"fact_id": WINNING_FACT_ID, "value": WINNING_FACT_VALUE},)

# The two real chunk ids from the committed corpus render (corpus/rendered/corpus-0.1.1), pinned to
# the SAME values `dataset_tools/seed_cases.jsonl` already commits for these two facts:
#   - WINNING_DOC_ID: gen-fact-contract_term-daniel-2025-contract_months's expected_doc_ids[0]
#   - LOSING_DOC_ID:  gen-fact-plan-fiber-100-contract_months's expected_doc_ids[0]
WINNING_DOC_ID = "2514487e4633b47b"  # Daniel's individual, 12 month contract term (the truth)
LOSING_DOC_ID = "e36752f7a1c6c439"  # plan-fiber-100's generic "No contract" marketing page (the trap)

# Verbatim from corpus/rendered/corpus-0.1.1/docs/doc-contract_terms-contract_term-daniel-2025.txt.
_WINNING_TEXT = (
    "# Contract Terms: Daniel\n\n"
    "Reference: Daniel's 2025 Contract Term\n\n"
    "This document confirms the contract terms on record for Daniel.\n\n"
    "Contract length: 12 months\n"
    "Effective vintage: 2025\n\n"
    "## Early termination\n\n"
    "If this contract is cancelled early, the Early Termination Fee of\n"
    "150.00 applies.\n\n"
    "These terms supersede any pricing or contract information shown on current\n"
    "plan pages.\n"
)

# Verbatim from corpus/rendered/corpus-0.1.1/docs/doc-plan_page-plan-fiber-100.txt.
_LOSING_TEXT = (
    "## Contract terms\n\n"
    "No contract. Cancel any time.\n\n"
    "## Whats included\n\n"
    "Installation and equipment fees for your region are listed in the\n"
    "regional fee schedule.\n\n"
    "# Fiber 100\n\n"
    "We are pleased to offer the Fiber 100 broadband plan, providing\n"
    "100 Mbps download and 100 Mbps upload for\n"
    "29.99 per month.\n"
)

# The stub retrieval fixture every family below is perturbed against: the retrieval STACK
# (`InMemoryRetriever.search_chunks`, deterministic keyword overlap) is genuinely exercised over
# real content carrying real ids, never a hand frozen "as if retrieved" list. This is what makes
# the invariant checks a real test of retrieval under perturbation rather than a test of hand
# authored fixture data.
STUB_CORPUS: list[Chunk] = [
    Chunk(
        chunk_id=WINNING_DOC_ID, doc_id=WINNING_DOC_ID, doc_type="contract_terms",
        text=_WINNING_TEXT, entity_ids=("contract_term-daniel-2025",),
    ),
    Chunk(
        chunk_id=LOSING_DOC_ID, doc_id=LOSING_DOC_ID, doc_type="plan_page",
        text=_LOSING_TEXT, entity_ids=("plan-fiber-100",),
    ),
]

# The stub corpus above holds exactly the two chunks the conflict names, so k=2 asks for the whole
# stub universe -- a full agreement floor of 1.0 is achievable and meaningful (unlike a k larger
# than the corpus, which would cap every family below 1.0 by construction, penalising a family for
# a corpus that is simply too small to fill it, the same short list convention `precision_at_k`
# already documents). The live lane (`metamorphic.__main__`) uses the real deployed
# `knowledge_server.DEPLOYED_K` against the real, much larger index instead.
STUB_K = 2

# One frozen, correctly grounded answer: the SAME text every family member's question should
# resolve to, regardless of how the question was worded -- the literal meaning of "answer
# equivalence" here. Deliberately mentions the fact value (12) as a literal substring so
# `quality.agent_metrics.answer_correctness_rate` grounds it against `EXPECTED_FACTS`.
WINNING_ANSWER = (
    "According to Daniel's 2025 Contract Term (contract_term-daniel-2025), you're on a 12 month "
    "contract that runs through 2025, with a 150.00 early termination fee if you cancel before it "
    "ends."
)

# A plausible but WRONG answer (the same shape of false claim the pre rewrite family's toy corpus
# render guard used to catch): grounded in the LOSING chunk's own text instead of the winning one,
# so it never mentions "12" and fails `answer_correctness_rate` -- the "has teeth" fixture
# `test_metamorphic.py` uses to prove the equivalence check is not vacuously true.
DRIFTED_ANSWER = (
    "Good news, your plan has no contract and no minimum term, so you're free to cancel any "
    "time with no fee."
)


@dataclass(frozen=True)
class MetamorphicMember:
    """One member of a family: a single worded question. No per member answer or retrieval field:
    both are computed fresh by `report.run_family` (retrieval, real, over `STUB_CORPUS`) or shared
    by the whole family (`MetamorphicFamily.answer`, since the metamorphic relation IS that every
    member resolves to the same answer)."""

    question: str


@dataclass(frozen=True)
class MetamorphicFamily:
    """A frozen family: a versioned identity (`family_id`, bumped whenever a human curates a new
    member in, mirroring `JudgeContract`'s own versioned identity discipline elsewhere in SP8), the
    registry contradiction it is seeded from, its members, the shared ground truth every member is
    checked against, and the family's OWN rank overlap floor (a paraphrase or a typo may honestly
    lose a distractor also retrieved alongside it; pure formatting noise should not lose anything
    at all, so the floor is a property of the family's kind, never a single global constant)."""

    family_id: str
    kind: str  # "paraphrase" | "typo_noise" | "query_perturbation"
    contradiction_id: str
    members: tuple[MetamorphicMember, ...]
    ground_truth_doc_id: str
    expected_facts: tuple[dict[str, object], ...]
    answer: str
    rank_overlap_floor: float


# ---- 1. paraphrase: natural rewordings of "is my plan contract free" -----------------------------
#
# The first three questions are the SAME wording already curated as real seed cases in
# `dataset_tools/seed_cases.jsonl` (case_ids seed-flagship-daniel-contract-free,
# seed-grounded-daniel-contract-2, seed-grounded-daniel-contract-3), copied here as a stable
# snapshot rather than read at import time, per this module's own "frozen means frozen" docstring
# rule; the last two are new, equally natural paraphrasings of the same underlying question.
PARAPHRASE_FAMILY = MetamorphicFamily(
    family_id="paraphrase-conflict-daniel-contract-v1",
    kind="paraphrase",
    contradiction_id="conflict-daniel-contract",
    members=(
        MetamorphicMember("Is my plan contract free?"),
        MetamorphicMember("I signed up a while back, do I still have a contract on my line?"),
        MetamorphicMember("Can I cancel without paying anything, or am I locked into a term?"),
        MetamorphicMember("Am I tied into a contract for my plan?"),
        MetamorphicMember("Do I have a minimum term commitment on my plan?"),
    ),
    ground_truth_doc_id=WINNING_DOC_ID,
    expected_facts=EXPECTED_FACTS,
    answer=WINNING_ANSWER,
    rank_overlap_floor=0.5,
)

# ---- 2. typo/noise: character level typos of the same question -----------------------------------

TYPO_NOISE_FAMILY = MetamorphicFamily(
    family_id="typo-noise-conflict-daniel-contract-v1",
    kind="typo_noise",
    contradiction_id="conflict-daniel-contract",
    members=(
        MetamorphicMember("Is my paln contract free?"),
        MetamorphicMember("Is my plan contrct free?"),
        MetamorphicMember("Iz my plan contract free?"),
    ),
    ground_truth_doc_id=WINNING_DOC_ID,
    expected_facts=EXPECTED_FACTS,
    answer=WINNING_ANSWER,
    rank_overlap_floor=0.5,
)

# ---- 3. query perturbation: pure surface noise, zero semantic content -----------------------------
#
# Casing, doubled whitespace, leading/trailing padding, an extra question mark: none of these
# change a single meaningful token, so this family's floor is the strongest of the three, exact
# agreement (1.0), not merely a floor.
QUERY_PERTURBATION_FAMILY = MetamorphicFamily(
    family_id="query-perturbation-conflict-daniel-contract-v1",
    kind="query_perturbation",
    contradiction_id="conflict-daniel-contract",
    members=(
        MetamorphicMember("Is my plan contract free?"),
        MetamorphicMember("Is my plan  contract free?"),
        MetamorphicMember("IS MY PLAN CONTRACT FREE?"),
        MetamorphicMember("  Is my plan contract free?  "),
        MetamorphicMember("Is my plan contract free??"),
    ),
    ground_truth_doc_id=WINNING_DOC_ID,
    expected_facts=EXPECTED_FACTS,
    answer=WINNING_ANSWER,
    rank_overlap_floor=1.0,
)

ALL_FAMILIES: tuple[MetamorphicFamily, ...] = (
    PARAPHRASE_FAMILY, TYPO_NOISE_FAMILY, QUERY_PERTURBATION_FAMILY,
)

__all__ = [
    "ALL_FAMILIES",
    "DRIFTED_ANSWER",
    "EXPECTED_FACTS",
    "LOSING_DOC_ID",
    "PARAPHRASE_FAMILY",
    "QUERY_PERTURBATION_FAMILY",
    "STUB_CORPUS",
    "STUB_K",
    "TYPO_NOISE_FAMILY",
    "WINNING_ANSWER",
    "WINNING_DOC_ID",
    "WINNING_FACT_ID",
    "WINNING_FACT_VALUE",
    "MetamorphicFamily",
    "MetamorphicMember",
]
