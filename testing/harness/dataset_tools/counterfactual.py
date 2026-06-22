"""Fairness counterfactual pairs (SP7 Task 5, D33): persona paired cases identical in registry
ground truth, varying only the persona attached to the case. Reuses Task 3's
`quality.agent_metrics.counterfactual_equivalent` as the ONE equivalence check (the asserted
`expected_facts` set plus `refusal_class` equal across a pair), never a second, softer, or fuzzy
measure of "the same answer for two different customers": that function is imported, never
reimplemented, per the plan's own instruction.

The persona table below is hand authored: a fixed, ordered tuple, never inferred from a real
customer record or a production trace (D33's own rule, "pairs authored from personas, no
demographic inference on real signals"). Every persona case is built from ONE shared base case
(typically one of Task 1's registry derived cases, `dataset_tools.generator.generate_cases()`):
every field is copied unchanged except `case_id` (must stay unique per case) and `persona`, so
equivalence holds by construction, not by luck.

D33's digest paragraph names three possible varying axes: the customer name, dialect or register
(the contract's `persona.style`), or region. The dataset contract's persona block (v0.1.0,
`contracts/dataset/schema.json`, the `gc-0002` shape this module follows) declares only two
properties, `name` and `style` (`additionalProperties: false`); it carries no `region` field, and
the plan's own Task 5 text ties the varying axes back to "the dataset contract's persona block
(gc-0002 shape)", the two field shape. Adding a third schema property is a shared, cross task
contract change, out of this module's own footprint, so the authored table below writes name and
style onto a generated case; region is carried on `PERSONAS` for authoring intent only (so it is
not silently dropped) and is never serialized onto a case's `persona` field.

No reference free faithfulness, judge, or rubric anything lives here: SP8's boundary (the 04/05
grader boundary this repo's CLAUDE.md names), out of scope for SP7 entirely.
"""
from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence

import jsonschema
from contract_tools import loader

from quality.agent_metrics import counterfactual_equivalent

# Hand authored, ordered, deterministic: never generated, never inferred from a real customer
# record or a production trace (D33). `region` is carried for authoring intent only; the module
# docstring above explains why it is not written onto a generated case's `persona` field.
PERSONAS: tuple[dict, ...] = (
    {"name": "sarah", "style": "direct", "region": "north"},
    {"name": "chen", "style": "formal", "region": "south"},
    {"name": "amara", "style": "casual", "region": "north"},
    {"name": "yusuf", "style": "direct", "region": "south"},
)


def _persona_field(persona: Mapping[str, str]) -> dict:
    # only the two fields the dataset contract's persona block declares (additionalProperties:
    # false); `region`, if present on the authored table, stays off the serialized case.
    return {"name": persona["name"], "style": persona["style"]}


def generate_cohort(
    base_case: Mapping[str, object], *, personas: Sequence[Mapping[str, str]] = PERSONAS
) -> tuple[dict, ...]:
    """One case per persona, every field identical to `base_case` except `case_id` (suffixed with
    the persona name, so it stays unique within the cohort) and `persona`. Deterministic:
    `personas`' own order, never a set or dict walk, so two calls over the same inputs return byte
    identical output."""
    cohort = []
    for persona in personas:
        case = {
            **base_case,
            "case_id": f"{base_case['case_id']}-persona-{persona['name']}",
            "persona": _persona_field(persona),
        }
        cohort.append(case)
    return tuple(cohort)


def cohort_pairs(
    cohort: Sequence[Mapping[str, object]]
) -> tuple[tuple[Mapping[str, object], Mapping[str, object]], ...]:
    """Every distinct pair within one persona cohort. `itertools.combinations` walks the cohort in
    its own given order (the personas' authored order), so the pair list is deterministic and
    never a set walk."""
    return tuple(itertools.combinations(cohort, 2))


def generate_cohorts(
    base_cases: Sequence[Mapping[str, object]], *, personas: Sequence[Mapping[str, str]] = PERSONAS
) -> tuple[tuple[dict, ...], ...]:
    """One cohort per base case, walked in `base_cases`' own order. A caller typically supplies
    Task 1's `generator.generate_cases()` output, itself registry file order; this function never
    reorders its input."""
    return tuple(generate_cohort(case, personas=personas) for case in base_cases)


def flatten_cohorts(cohorts: Sequence[Sequence[Mapping[str, object]]]) -> tuple[dict, ...]:
    """Every generated persona case, base case order then persona order, one flat tuple: the shape
    `validate_cases`/JSONL writers want, mirroring `generator.generate_cases`'s own flat tuple
    return."""
    return tuple(dict(case) for cohort in cohorts for case in cohort)


def check_pair_equivalence(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]]
) -> tuple[dict, ...]:
    """Every pair that FAILS Task 3's exact equivalence check (`expected_facts` set plus
    `refusal_class` both equal): empty when every pair is equivalent, the steady state a freshly
    generated cohort always reaches by construction (`generate_cohort` copies every field but two).
    This exists to catch drift AFTER the fact, e.g. a hand edited persona case landing in a later
    curated seed set with a diverging `expected_facts` entry, not to prove the generator itself."""
    divergent = []
    for case_a, case_b in pairs:
        if not counterfactual_equivalent(case_a, case_b):
            divergent.append({"case_id_a": case_a["case_id"], "case_id_b": case_b["case_id"]})
    return tuple(divergent)


def validate_cases(cases: Sequence[Mapping[str, object]], schema: dict | None = None) -> None:
    """Validate every generated persona case against the dataset contract, the same schema Task
    1's generator self validates against."""
    if schema is None:
        schema = loader.load_schema("dataset")
    for case in cases:
        jsonschema.validate(case, schema)


def to_jsonl(cases: Sequence[Mapping[str, object]]) -> str:
    """One case object per line, compact and key sorted, matching `generator.to_jsonl`'s own byte
    reproducible convention exactly (a shared reader/writer never needs to tell the two files'
    output apart)."""
    lines = [json.dumps(case, sort_keys=True, separators=(",", ":")) for case in cases]
    return "".join(f"{line}\n" for line in lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from dataset_tools import generator

    parser = argparse.ArgumentParser(prog="dataset_tools.counterfactual")
    parser.add_argument(
        "--cases", default=None, help="input cases JSONL; defaults to generator.generate_cases()"
    )
    parser.add_argument("--out", required=True, help="output JSONL path, one persona case per line")
    args = parser.parse_args(argv)

    base_cases = (
        _load_cases_from_jsonl(Path(args.cases)) if args.cases else generator.generate_cases()
    )
    cohorts = generate_cohorts(base_cases)
    cases = flatten_cohorts(cohorts)
    validate_cases(cases)
    Path(args.out).write_text(to_jsonl(cases))
    print(f"wrote {len(cases)} persona cases ({len(cohorts)} cohorts) to {args.out}")
    return 0


def _load_cases_from_jsonl(path) -> tuple[dict, ...]:
    lines = path.read_text().splitlines()
    return tuple(json.loads(line) for line in lines if line.strip())


if __name__ == "__main__":
    import sys

    sys.exit(main())
