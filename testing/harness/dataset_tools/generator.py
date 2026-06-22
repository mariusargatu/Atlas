"""Mechanical registry to case generator (SP7 Task 1): zero LLM. Walks the hand authored core
registry (`corpus/registry/core.yaml`, HLD D4, loaded via `corpus_tools.registry`) into dataset
contract cases (`contracts/dataset/schema.json`, v0.1.0) whose ground truth is derived BY
CONSTRUCTION, never authored:

  - one hop and two hop factoid cases: `Registry.entities` (fact_id `entity_id:field`, value
    straight off `Entity.fields`) and `Registry.edges` (chains one entity's fact to a second,
    connected entity's fact), walked in registry file order.
  - `grounded_not_true` adversarial cases: `Registry.contradictions`, one per contradiction,
    `expected_facts` holding only the WINNING fact (the resolved, correct answer); `question_hint`
    seeds the case's own phrasing.
  - answerable false hallucination bait cases: the never rendered entities (`render: false`
    in the registry). Nothing grounds them anywhere in the corpus, so `answerable` is False and
    `expected_doc_ids` is empty by construction, not by a separate hand written rule.

Only `corpus/registry/core.yaml` is loaded, not `generated_variants.yaml`: the digest and the
registry's own docstring name core.yaml as the hand authored narrative root ("documents, golden
answers, and the knowledge graph all derive from these entities... nothing here is generated");
`generated_variants.yaml`'s region variant entities are rendering expansion artifacts, not
independently authored facts, and are deliberately excluded from ground truth generation.

`expected_doc_ids` holds retrieval unit ids (`rag_tools.chunker.ChunkRecord.chunk_id`), not raw
`doc_id` strings: retrieval happens at chunk granularity, so a case's ground truth has to be
directly comparable against what retrieval actually returns (D14's ID based context precision and
recall). `dataset_tools.provenance_join` computes these ids by joining a fact's placement span
(`provenance/<doc_id>.json`) against `rag_tools.chunker`'s own span overlap rule, mirroring that
rule exactly rather than inventing a third variant. `doc_version` (the manifest's per doc content
hash) is folded into that chunk id computation by `provenance_join`, so it never needs a field of
its own on the dataset contract.

`expected_tool_calls` is deliberately left absent by this generator, on every case it emits.
It names MCP tool calls tied to agent RUNTIME behavior (`account.get_contract`, `catalog.get_plan`,
...), not a registry fact: nothing in the registry, the corpus, or the provenance sidecars can
derive it mechanically. Hand authoring fills it in later (SP7 Task 6, the seed set); a generated
case simply omits the field (the dataset schema does not require it), never emits an empty array or
null placeholder for it.

`split` defaults to "dev" for every generated case. Stratified, seeded split assignment across
dev/test is a separate, later concern (SP7 Task 4's dataset manifest), not this walker's job: a
generated case is a candidate, not yet a placed member of the seed set (SP7 Task 6 curates that,
"included via Task 1's generator output, curated, not blindly dumped").

`intent` is "troubleshooting" on every generated case: none of these are write/action requests, and
`atlas.domain.binding.classify_intent`'s own deterministic heuristic (ACTION cues vs everything
else) would classify plain lookup questions the same way, so a future intent confusion matrix
(SP7 Task 3) compares against the intent the runtime would actually produce for this phrasing.

No reference free faithfulness, judge, or rubric anything lives here or anywhere in SP7: that is
SP8's one calibrated judge (D15), out of scope by the 04/05 grader boundary this repo's CLAUDE.md
names. This generator never reads or emits `atlas.judge.*`.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import jsonschema
from contract_tools import loader
from corpus_tools.registry import Entity, Registry, load_registry

from dataset_tools import provenance_join
from dataset_tools.provenance_join import CorpusIndex

CORE_REGISTRY = Path("corpus/registry/core.yaml")
DEFAULT_CORPUS_DIR = provenance_join.DEFAULT_CORPUS_DIR

_INTENT = "troubleshooting"
_ORIGIN = "synthetic"
_CANDIDATE_SOURCE = "registry_render"
_DEFAULT_SPLIT = "dev"


def validate_case(case: dict, schema: dict | None = None) -> None:
    """Validate one case dict against the dataset contract. Accepts `origin: promoted` exactly
    like any other origin value: promotion is a later, human gated activation (SP8's judge/HITL
    loop consumes it), never a schema level restriction (nothing here produces one; the schema and
    this validator both accept one all the same)."""
    if schema is None:
        schema = loader.load_schema("dataset")
    jsonschema.validate(case, schema)


def _base_case(
    case_id: str,
    *,
    hop_count: int | None,
    doc_type: str | None,
    adversarial_class: str | None,
    answerable: bool,
    expected_doc_ids: tuple[str, ...],
    expected_facts: list[dict] | None,
    user: str,
) -> dict:
    case: dict = {
        "case_id": case_id,
        "split": _DEFAULT_SPLIT,
        "origin": _ORIGIN,
        "candidate_source": _CANDIDATE_SOURCE,
        "source_trace_id": None,
        "intent": _INTENT,
        "adversarial_class": adversarial_class,
        "failure_class": None,
        "answerable": answerable,
        "expected_doc_ids": list(expected_doc_ids),
        "refusal_class": None,
        "persona": None,
        "turns": [{"user": user}],
        "end_state": None,
    }
    if hop_count is not None:
        case["hop_count"] = hop_count
    if doc_type is not None:
        case["doc_type"] = doc_type
    if expected_facts is not None:
        case["expected_facts"] = expected_facts
    return case


def _one_hop_cases(reg: Registry, index: CorpusIndex) -> tuple[dict, ...]:
    cases: list[dict] = []
    for entity in reg.entities:
        for field in entity.fields:
            fact_ref = f"{entity.id}:{field}"
            doc_ids = provenance_join.docs_for_fact(index, fact_ref)
            if not doc_ids:
                continue  # never rendered anywhere: not a mechanically answerable factoid case
            cases.append(
                _base_case(
                    f"gen-fact-{entity.id}-{field}",
                    hop_count=1,
                    doc_type=provenance_join.doc_type_for_fact(index, fact_ref),
                    adversarial_class=None,
                    answerable=True,
                    expected_doc_ids=provenance_join.chunk_ids_for_fact(index, fact_ref),
                    expected_facts=[{"fact_id": fact_ref, "value": entity.fields[field]}],
                    user=f"What is the {field} of {entity.id}?",
                )
            )
    return tuple(cases)


def _first_placed_field(entity: Entity, index: CorpusIndex) -> str | None:
    """The first field, in the entity's own declared order, that is placed somewhere in the
    corpus: the same "mechanically answerable" test `_one_hop_cases` applies per field, used here
    to pick one representative fact per edge endpoint."""
    for field in entity.fields:
        if provenance_join.docs_for_fact(index, f"{entity.id}:{field}"):
            return field
    return None


def _two_hop_cases(reg: Registry, index: CorpusIndex) -> tuple[dict, ...]:
    cases: list[dict] = []
    for edge in reg.edges:
        src, dst = reg.entity(edge.src), reg.entity(edge.dst)
        src_field, dst_field = _first_placed_field(src, index), _first_placed_field(dst, index)
        if src_field is None or dst_field is None:
            continue  # neither endpoint mechanically groundable (never true on the committed registry)

        src_ref, dst_ref = f"{edge.src}:{src_field}", f"{edge.dst}:{dst_field}"
        doc_ids = sorted(
            set(provenance_join.chunk_ids_for_fact(index, src_ref))
            | set(provenance_join.chunk_ids_for_fact(index, dst_ref))
        )
        relation_text = edge.relation.replace("_", " ")
        cases.append(
            _base_case(
                f"gen-edge-{edge.relation}-{edge.src}-{edge.dst}",
                hop_count=2,
                doc_type=provenance_join.doc_type_for_fact(index, src_ref),
                adversarial_class=None,
                answerable=True,
                expected_doc_ids=tuple(doc_ids),
                expected_facts=[
                    {"fact_id": src_ref, "value": src.fields[src_field]},
                    {"fact_id": dst_ref, "value": dst.fields[dst_field]},
                ],
                user=f"Given {edge.src} is {relation_text} {edge.dst}, what is {edge.dst}'s {dst_field}?",
            )
        )
    return tuple(cases)


def _phrase_question(hint: str, fallback: str) -> str:
    text = hint.strip() or fallback
    text = text[0].upper() + text[1:]
    if not text.endswith(("?", ".", "!")):
        text += "?"
    return text


def _contradiction_cases(reg: Registry, index: CorpusIndex) -> tuple[dict, ...]:
    cases: list[dict] = []
    for c in reg.contradictions:
        entity_id, _, field = c.winning_fact.partition(":")
        value = reg.entity(entity_id).fields[field]
        cases.append(
            _base_case(
                f"gen-contradiction-{c.id}",
                hop_count=c.hops,
                doc_type=provenance_join.doc_type_for_fact(index, c.winning_fact),
                adversarial_class="grounded_not_true",
                answerable=True,
                expected_doc_ids=provenance_join.chunk_ids_for_fact(index, c.winning_fact),
                expected_facts=[{"fact_id": c.winning_fact, "value": value}],
                user=_phrase_question(c.question_hint, c.id),
            )
        )
    return tuple(cases)


def _bait_cases(reg: Registry) -> tuple[dict, ...]:
    cases: list[dict] = []
    for entity in reg.entities:
        if entity.render:
            continue
        field = next(iter(entity.fields), "name")
        cases.append(
            _base_case(
                f"gen-bait-{entity.id}",
                hop_count=None,
                doc_type=None,
                adversarial_class="hallucination_bait",
                answerable=False,
                expected_doc_ids=(),
                expected_facts=None,
                user=f"What is the {field} of {entity.id}?",
            )
        )
    return tuple(cases)


def generate_cases(reg: Registry | None = None, corpus_dir: Path = DEFAULT_CORPUS_DIR) -> tuple[dict, ...]:
    """Walk the registry (file order) into every mechanically derivable case, self validating each
    one against the dataset contract before returning. Deterministic: two calls with the same
    registry and corpus produce byte identical `to_jsonl` output (no set/dict reordering anywhere
    in the walk, per D16)."""
    if reg is None:
        reg = load_registry([CORE_REGISTRY])
    index = provenance_join.load_corpus_index(corpus_dir)

    cases = (
        *_one_hop_cases(reg, index),
        *_two_hop_cases(reg, index),
        *_contradiction_cases(reg, index),
        *_bait_cases(reg),
    )

    schema = loader.load_schema("dataset")
    for case in cases:
        validate_case(case, schema)
    return cases


def to_jsonl(cases: Sequence[dict]) -> str:
    """One case object per line, compact and key sorted so two runs over the same input are byte
    identical regardless of dict construction order. Git native (D16): plain text, line diffable."""
    lines = [json.dumps(case, sort_keys=True, separators=(",", ":")) for case in cases]
    return "".join(f"{line}\n" for line in lines)


def write_jsonl(cases: Sequence[dict], path: Path) -> None:
    path.write_text(to_jsonl(cases))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="dataset_tools.generator")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_DIR), help="rendered corpus_version directory")
    parser.add_argument("--out", required=True, help="output JSONL path")
    args = parser.parse_args(argv)

    cases = generate_cases(corpus_dir=Path(args.corpus))
    write_jsonl(cases, Path(args.out))
    print(f"wrote {len(cases)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
