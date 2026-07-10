"""Matrix case loading: the retrieval slice of the 76 case seed set (`dataset_tools.seed_cases.jsonl`)
turned into the small, typed `MatrixCase` shape every stage of the staged runner reads.

Reuses `dataset_tools.manifest.case_slice`'s own established slice buckets rather than re deriving a
second definition: `RETRIEVAL_SLICES` is the exact same three slices (`factoid_one_hop`,
`factoid_two_hop`, `grounded_not_true`) `test_sp7_retrieval_metrics_live.py`'s own
`_retrieval_relevant_cases` filters to (every case whose `expected_doc_ids` is nonempty on the
committed seed set; `hallucination_bait` is answerable: false by construction, nothing to retrieve,
and `other` carries no case level `expected_doc_ids` either). No retriever, no network, anywhere in
this module: it is a pure transform from JSONL to a dataclass.

`query_entity_ids` (the real supplier T1/T2's own reports name): every `expected_facts` entry's
`fact_id` is already `dataset_tools.generator`'s own `{entity.id}:{field}` shape, the registry's own
identity, so `quality.agent_metrics.expected_entity_ids` (the SAME extraction SP7's citation grading
already uses) reads it back with no new data and no rederivation. Every case in `RETRIEVAL_SLICES`
carries at least one `expected_facts` entry, so this is never vacuously empty on the committed set --
`agentic_rag.py`'s `grade_documents` (and the variant comparison stage, `matrix.variants`) get a real
CRAG grading target instead of the vacuous pass through an empty set forces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dataset_tools import manifest as dataset_manifest

from quality.agent_metrics import expected_entity_ids

RETRIEVAL_SLICES = frozenset({"factoid_one_hop", "factoid_two_hop", "grounded_not_true"})


@dataclass(frozen=True)
class MatrixCase:
    """One case's query plus its own declared ground truth, nothing computed: `relevant_doc_ids`
    feeds `quality.ir_metrics` (stage 1/2), `expected_facts` feeds
    `quality.agent_metrics.answer_correctness_rate` (stage 3), `query_entity_ids` is the registry's
    own entity ids named by `expected_facts` (see the module docstring), CRAG grading's real
    ground truth."""

    case_id: str
    query: str
    relevant_doc_ids: frozenset[str]
    expected_facts: tuple[dict, ...] = field(default_factory=tuple)
    query_entity_ids: frozenset[str] = frozenset()


def _to_matrix_case(case: dict) -> MatrixCase:
    facts = tuple(case.get("expected_facts") or ())
    return MatrixCase(
        case_id=case["case_id"],
        query=case["turns"][0]["user"],
        relevant_doc_ids=frozenset(case.get("expected_doc_ids") or ()),
        expected_facts=facts,
        query_entity_ids=expected_entity_ids(facts),
    )


def load_matrix_cases(path: Path, *, slices: frozenset[str] = RETRIEVAL_SLICES) -> tuple[MatrixCase, ...]:
    """Load `path` (a JSONL file shaped like the seed set) and keep only cases whose `case_slice` is in
    `slices` (the retrieval relevant slice by default). Order preserved from the file, never
    resorted through a set or dict, so a caller that wants a stable case sequence in file order gets
    one."""
    cases = dataset_manifest.load_cases_from_jsonl(Path(path))
    return tuple(_to_matrix_case(c) for c in cases if dataset_manifest.case_slice(c) in slices)


__all__ = ["MatrixCase", "RETRIEVAL_SLICES", "load_matrix_cases"]
