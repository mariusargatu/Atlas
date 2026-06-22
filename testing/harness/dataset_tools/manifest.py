"""dataset_manifest.json, deterministic seeded stratified splits, and the two direction
contamination lint (SP7 Task 4). Modeled on `corpus_tools.build`'s own manifest shape (a semver,
an upstream version pin, per artifact counts, a content addressed list, stage then only write on a
clean gate), the same way that module is the committed convention for a versioned build artifact.

This module never edits `dataset_tools.generator`; it only consumes it: `generate_cases()` (or a
JSONL file already written by `generator.write_jsonl`) is the input, cases in, a manifest plus a
finalized case list out. `generator.py`'s own docstring names the reason `split` is "dev" on every
case it emits as a provisional placeholder: "SP7 Task 4 owns final stratified dev/test assignment".
This module is that Task 4 assignment: `assign_splits` computes a fresh, deterministic, seeded,
stratified split for every case_id and OVERWRITES the placeholder, never trusts it.

Two independent contamination checks, per D16, "both directions":

  1. `lint_verbatim_leakage` (hard fail): a case's own phrasing (`turns[*].user`) must never
     appear verbatim inside a rendered corpus doc UNLESS that doc is one the case's own
     `expected_doc_ids` already declares. A case legitimately grounded on a doc is allowed to
     echo its wording; an undeclared verbatim match means a doc's answer leaked into the
     question, making retrieval trivially winnable by string match rather than real retrieval.
     Mirrors `corpus_tools.build`'s own stage then only write discipline: `build_manifest` raises
     `ContaminationLintError` before anything is written, never a partial or tainted manifest.
  2. `_fact_overlap` (declared, never gated): the registry (21 entities, 2 contradictions) is
     small enough that dev/test splits necessarily share coverage of some underlying facts (a
     one hop case for `entity:field` and a two hop edge case referencing that same fact can land
     in different splits by chance of the seeded assignment). This is measured and reported in
     the manifest's `fact_overlap` field, always present even when its count is zero: the small
     registry honesty risk this repo's planning digest names twice by design, declared, never
     silent, never gated (D16 does not ask this direction to fail the build, unlike direction 1).

No reference free faithfulness, judge, or rubric anything lives here: SP8's boundary (04/05 grader
boundary, this repo's CLAUDE.md), out of scope for SP7 entirely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dataset_tools import generator, provenance_join
from rag_tools import chunker

DEFAULT_CORPUS_DIR = provenance_join.DEFAULT_CORPUS_DIR
# Matches corpus_tools.expand.DEFAULT_SEED's own convention (a fixed, committed integer, not
# recomputed from the wall clock): split assignment must be reproducible run over run.
DEFAULT_SEED = 20260718
DEFAULT_TEST_FRACTION = 0.2
# TASK 6 REQUIREMENT, do not miss this: on Task 1's current provisional generator output (88
# mechanically generated cases), the two case grounded_not_true and two case hallucination_bait
# strata each round to zero test coverage at this fraction (round(2 * 0.2) equals 0), so today's
# test split carries no adversarial (contradiction or bait) case at all; see
# test_stratified_split_counts_pinned_against_committed_generated_set in test_dataset_manifest.py
# for the pinned proof. This is acceptable only because Task 4 ships the split machinery over a
# provisional set, not the final dataset (the HLD's own sizing target for the frozen test split
# is 250+ cases within a 300 to 500 case full sweep). When Task 6 replaces this set with its hand
# curated seed set, it MUST ensure every case_slice value, including both adversarial classes,
# has non zero coverage in BOTH the dev and test splits. A test split with zero adversarial
# coverage cannot catch a regression in contradiction or hallucination bait handling, which
# defeats the point of holding out a test split at all.
# The golden dataset's own semver (distinct from contracts/dataset/schema.json's
# x-contract-version, which versions the CASE shape, not the case SET's content). Task 4 ships
# the manifest and split machinery over Task 1's provisional generator output; Task 6 owns the
# hand curated seed set this version will actually describe once committed.
DATASET_VERSION = "0.1.0"

# Below this length a substring match is far more likely to be an ordinary common phrase than a
# real phrasing leak (the thing direction 1 of the lint exists to catch, per D16: "prevents answer
# leakage into phrasing"). Chosen well under the shortest real case question in the committed set.
MIN_LINT_TEXT_LEN = 15


class ContaminationLintError(Exception):
    """Raised by `build_manifest` when `lint_verbatim_leakage` finds an undeclared verbatim
    match. Mirrors `corpus_tools.build`'s own discipline: a failing lint must never let a
    manifest, partial or otherwise, get written."""

    def __init__(self, violations: tuple[dict, ...]) -> None:
        self.violations = violations
        super().__init__(f"{len(violations)} contamination lint violation(s): {violations}")


# ---- case classification and split assignment -----------------------------------------------------


# The full set of case classes Task 1's generator currently produces. A case landing outside
# this set buckets as "other" (see case_slice's own docstring): a future case shape, not yet
# classified, never one of today's committed classes. build_manifest's per split slice counts
# declare every one of these four explicitly, even at zero, matching fact_overlap's own
# declared, never silent principle.
CASE_SLICES: tuple[str, ...] = (
    "factoid_one_hop",
    "factoid_two_hop",
    "grounded_not_true",
    "hallucination_bait",
)


def case_slice(case: dict) -> str:
    """The per slice bucket a case falls into for stratification and manifest reporting. Checks
    `adversarial_class` before `hop_count`: a contradiction case carries a real `hop_count` too
    (1 or 2), but it must bucket as `grounded_not_true`, never get folded into the plain factoid
    slices. Falls back to "other" for a case shape none of Task 1's three classes produce (a
    future case class, e.g. Task 5's fairness or multi turn cases): never raises on an unknown
    shape (the "code first" directive: new origins and shapes are late binding, not blocked), but
    every case in the CURRENTLY committed generated set is asserted elsewhere to never land here.
    """
    adversarial = case.get("adversarial_class")
    if adversarial == "grounded_not_true":
        return "grounded_not_true"
    if adversarial == "hallucination_bait":
        return "hallucination_bait"
    hop_count = case.get("hop_count")
    if hop_count == 1:
        return "factoid_one_hop"
    if hop_count == 2:
        return "factoid_two_hop"
    return "other"


def assign_splits(
    cases: Sequence[dict],
    *,
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> dict[str, str]:
    """case_id -> "dev"/"test", deterministic and seeded, stratified by (case_slice, intent).
    Ignores any `split` value already on the case (the generator's own placeholder, per its
    docstring): this is the one function that decides the real assignment.

    Every source of nondeterminism is closed: strata are visited in `sorted()` order (never a
    bare dict iteration), the case_ids within a stratum are sorted before sampling, and the RNG
    is seeded per stratum with a string key (`f"{seed}:{slice}:{intent}"`), the same
    `random.Random(seed_key)` pattern `corpus_tools.render` already uses for reproducible
    variation per entity. `random.Random` on a string seed hashes via sha512 internally, not the
    process's `hash()`, which depends on PYTHONHASHSEED, so this is stable across processes and
    machines.
    """
    strata: dict[tuple[str, str], list[str]] = {}
    for case in cases:
        key = (case_slice(case), case["intent"])
        strata.setdefault(key, []).append(case["case_id"])

    assignment: dict[str, str] = {}
    for key in sorted(strata):
        case_ids = sorted(strata[key])
        n_test = round(len(case_ids) * test_fraction)
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        test_ids = set(rng.sample(case_ids, k=n_test)) if n_test else set()
        for case_id in case_ids:
            assignment[case_id] = "test" if case_id in test_ids else "dev"
    return assignment


# ---- content addressing ----------------------------------------------------------------------------


def _canonical_case_json(case: dict) -> str:
    return json.dumps(case, sort_keys=True, separators=(",", ":"))


def _case_content_hash(case: dict) -> str:
    return hashlib.sha256(_canonical_case_json(case).encode()).hexdigest()


def _aggregate_content_hash(case_entries: dict[str, dict]) -> str:
    """sha256 over the canonical sorted case id and content list (D16 phrasing): one line per
    case, `case_id:content_hash`, sorted, then hashed as a single blob. Mirrors
    `corpus_tools.build`'s own `content_hash` (sha256 over sorted per doc hashes)."""
    lines = sorted(f"{case_id}:{entry['content_hash']}" for case_id, entry in case_entries.items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ---- corpus context: one pass building both the chunk_id -> doc_id map and doc texts --------------


@dataclass(frozen=True)
class _CorpusContext:
    corpus_version: str
    chunk_to_doc: dict[str, str]
    doc_texts: dict[str, str]


def _load_corpus_context(corpus_dir: Path) -> _CorpusContext:
    index = provenance_join.load_corpus_index(corpus_dir)
    chunk_to_doc: dict[str, str] = {}
    doc_texts: dict[str, str] = {}
    for doc_id in sorted(index.doc_versions):
        text = (corpus_dir / "docs" / f"{doc_id}.txt").read_text()
        doc_texts[doc_id] = text
        chunks = chunker.chunk_document(
            doc_id=doc_id,
            doc_type=index.doc_types[doc_id],
            text=text,
            doc_version=index.doc_versions[doc_id],
            corpus_version=index.corpus_version,
            placements=index.placements_by_doc[doc_id],
        )
        for chunk in chunks:
            chunk_to_doc[chunk.chunk_id] = doc_id
    return _CorpusContext(corpus_version=index.corpus_version, chunk_to_doc=chunk_to_doc, doc_texts=doc_texts)


# ---- contamination lint direction 1: verbatim case text leaking an undeclared doc ------------------


def _lint_verbatim_leakage(cases: Sequence[dict], context: _CorpusContext) -> tuple[dict, ...]:
    violations: list[dict] = []
    for case in cases:
        declared_docs = {
            context.chunk_to_doc[chunk_id]
            for chunk_id in (case.get("expected_doc_ids") or ())
            if chunk_id in context.chunk_to_doc
        }
        for turn in case.get("turns") or ():
            text = turn.get("user", "")
            if len(text) < MIN_LINT_TEXT_LEN:
                continue
            for doc_id in sorted(context.doc_texts):
                if doc_id in declared_docs:
                    continue
                if text in context.doc_texts[doc_id]:
                    violations.append({"case_id": case["case_id"], "doc_id": doc_id, "text": text})
    return tuple(violations)


def lint_verbatim_leakage(
    cases: Sequence[dict], corpus_dir: Path = DEFAULT_CORPUS_DIR
) -> tuple[dict, ...]:
    """Direction 1 of the contamination lint (D16): every `(case_id, doc_id, text)` violation
    found, empty when clean. See the module docstring for the rule."""
    context = _load_corpus_context(corpus_dir)
    return _lint_verbatim_leakage(cases, context)


# ---- contamination lint direction 2: dev/test fact coverage overlap, declared never silent ---------


def _fact_overlap(cases: Sequence[dict]) -> dict:
    fact_splits: dict[str, set[str]] = {}
    for case in cases:
        for fact in case.get("expected_facts") or ():
            fact_splits.setdefault(fact["fact_id"], set()).add(case["split"])
    overlapping = sorted(fact_id for fact_id, splits in fact_splits.items() if len(splits) > 1)
    return {
        "declared": True,
        "count": len(overlapping),
        "fact_ids": overlapping,
        "note": (
            "The registry is small enough that dev and test splits can share coverage of the "
            "same underlying fact (a one hop case for a fact and a two hop edge case referencing "
            "that same fact can land in different splits). Measured and declared here, never "
            "silently hidden; this direction is reported, not gated."
        ),
    }


# ---- JSONL round trip: consuming the generator's own output -----------------------------------------


def load_cases_from_jsonl(path: Path) -> tuple[dict, ...]:
    lines = Path(path).read_text().splitlines()
    return tuple(json.loads(line) for line in lines if line.strip())


def write_cases_jsonl(cases: Sequence[dict], path: Path) -> None:
    # Reuses generator.write_jsonl (Task 1's own byte reproducible writer) rather than a second
    # serialization variant: same compact, key sorted, newline terminated JSONL shape.
    generator.write_jsonl(cases, path)


def write_manifest(built: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(built, sort_keys=True, indent=2) + "\n")


# ---- the manifest build itself -----------------------------------------------------------------------


def build_manifest(
    cases: Sequence[dict],
    *,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    dataset_version: str = DATASET_VERSION,
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple[dict, tuple[dict, ...]]:
    """Build `(dataset_manifest_dict, final_cases)` from an input case sequence. `final_cases`
    carries the real, computed split (never the generator's "dev" placeholder). Raises
    `ValueError` on a duplicate case_id (fail fast, never a silently clobbered content hash entry)
    and `ContaminationLintError` on any direction 1 lint violation, both before anything about a
    manifest is constructed: nothing partial or tainted is ever returned."""
    case_ids = [case["case_id"] for case in cases]
    seen: set[str] = set()
    duplicates: list[str] = []
    for case_id in case_ids:
        if case_id in seen:
            duplicates.append(case_id)
        seen.add(case_id)
    if duplicates:
        raise ValueError(f"duplicate case_id(s) in input: {sorted(set(duplicates))}")

    context = _load_corpus_context(corpus_dir)
    assignment = assign_splits(cases, seed=seed, test_fraction=test_fraction)
    final_cases = tuple({**case, "split": assignment[case["case_id"]]} for case in cases)

    violations = _lint_verbatim_leakage(final_cases, context)
    if violations:
        raise ContaminationLintError(violations)

    overlap = _fact_overlap(final_cases)

    per_split: dict[str, dict] = {}
    for split in ("dev", "test"):
        split_cases = [c for c in final_cases if c["split"] == split]
        # Every known case class starts at 0, so an empty stratum (e.g. an adversarial class that
        # rounds to zero test coverage, see the CASE_SLICES comment above) is still declared
        # explicitly rather than silently absent from the map.
        slices: dict[str, int] = dict.fromkeys(CASE_SLICES, 0)
        for c in split_cases:
            s = case_slice(c)
            slices[s] = slices.get(s, 0) + 1
        per_split[split] = {"count": len(split_cases), "slices": dict(sorted(slices.items()))}

    case_entries = {c["case_id"]: {"split": c["split"], "content_hash": _case_content_hash(c)} for c in final_cases}
    content_hash = _aggregate_content_hash(case_entries)

    built = {
        "dataset_version": dataset_version,
        "corpus_version": context.corpus_version,
        "seed": seed,
        "test_fraction": test_fraction,
        "case_count": len(final_cases),
        "splits": per_split,
        "cases": dict(sorted(case_entries.items())),
        "content_hash": content_hash,
        "fact_overlap": overlap,
        "contamination_lint": {"status": "pass", "checked_cases": len(final_cases)},
    }
    return built, final_cases


# ---- CLI: what `task dataset:build` runs -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dataset_tools.manifest")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_DIR), help="rendered corpus_version directory")
    parser.add_argument(
        "--cases",
        default=None,
        help="input cases JSONL (a generator.write_jsonl output); defaults to generator.generate_cases()",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="split assignment seed")
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--cases-out", required=True, help="output JSONL, cases with the assigned split")
    parser.add_argument("--manifest-out", required=True, help="output dataset_manifest.json")
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus)
    cases = load_cases_from_jsonl(Path(args.cases)) if args.cases else generator.generate_cases(corpus_dir=corpus_dir)

    try:
        built, final_cases = build_manifest(
            cases,
            corpus_dir=corpus_dir,
            dataset_version=args.dataset_version,
            seed=args.seed,
            test_fraction=args.test_fraction,
        )
    except ContaminationLintError as exc:
        print(f"error: {exc}")
        return 1

    write_cases_jsonl(final_cases, Path(args.cases_out))
    write_manifest(built, Path(args.manifest_out))
    print(f"wrote {len(final_cases)} cases to {args.cases_out} and the manifest to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
