"""Every declared field is exercised by at least one golden example (HLD D25)."""

from __future__ import annotations

import pytest
from contract_tools import diff, loader


def _instance_paths(value, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}{key}"
            paths.add(path)
            paths |= _instance_paths(child, prefix=f"{path}/")
    elif isinstance(value, list):
        for child in value:
            paths |= _instance_paths(child, prefix=prefix)
    return paths


@pytest.mark.parametrize("family", ["dataset", "manifest"])
def test_every_declared_property_appears_in_some_example(family: str) -> None:
    schema_paths = set(diff.flatten_properties(loader.load_schema(family)))
    example_paths: set[str] = set()
    for example in loader.load_examples(family).values():
        example_paths |= _instance_paths(example)
    uncovered = schema_paths - example_paths
    assert not uncovered, f"{family}: fields never exercised by an example: {sorted(uncovered)}"


def test_every_trace_structural_field_appears_in_an_example() -> None:
    """The generic "every property appears in an example" rule above, narrowed to the trace
    family's STRUCTURAL fields only (`name`/`span_kind`/`events`/...) -- unchanged from the pre
    freeze rule. `atlas.*` attribute properties are checked separately, by
    `test_every_trace_attribute_is_emitted_or_narrowed` below: post freeze, a real single captured
    span (what the two curated golden examples now honestly are) can never simultaneously carry
    every one of the 30 reserved attributes the way the old aspirational composite pretended to,
    since a real trace is many DIFFERENT span shapes (turn/guard/stage/...), not one."""
    schema_paths = set(diff.flatten_properties(loader.load_schema("trace")))
    attribute_paths = {p for p in schema_paths if p.startswith("attributes/")}
    structural_paths = schema_paths - attribute_paths
    example_paths: set[str] = set()
    for example in loader.load_examples("trace").values():
        example_paths |= _instance_paths(example)
    uncovered = structural_paths - example_paths
    assert not uncovered, f"trace: structural fields never exercised by an example: {sorted(uncovered)}"


def test_every_trace_attribute_is_emitted_or_narrowed() -> None:
    """SP6 task 7 (the v1.0.0 freeze): for `atlas.*` attribute properties specifically, "documented"
    means governed by `contract_tools.freeze_check`'s own dual evidence walk (a hermetic translation
    run AND the full committed live capture archive, `contracts/trace/freeze_evidence.json` -- 26
    real exported spans across 4 real scenarios, a far more exhaustive proof than 2 curated example
    files could ever be) or `ADR-029`'s narrowed set.
    This is the SAME check `test_freeze_check.py::test_the_real_committed_freeze_is_clean` runs from
    the freeze gate's own side; pinned here too so the contract coverage suite does not silently
    regress to "nothing checks trace attribute coverage at all" if that file is ever skipped."""
    from contract_tools import freeze_check

    statuses = freeze_check.evaluate(
        hermetic_emitted=freeze_check.hermetic_emitted_attributes(),
        live_emitted=freeze_check.live_emitted_attributes(),
        narrowed=freeze_check.load_narrowed(),
    )
    failing = [s.attribute for s in statuses if not s.ok]
    assert not failing, f"trace: attributes with no real evidence and not narrowed: {sorted(failing)}"


def test_every_sse_event_type_appears_in_some_sequence() -> None:
    declared = set(loader.load_schema("sse")["$defs"])
    seen = {event["event"] for seq in loader.load_examples("sse").values() for event in seq}
    assert declared == seen, f"sse events never exercised: {sorted(declared - seen)}"


def test_every_sse_field_appears_in_some_example() -> None:
    schema_paths = set(diff.flatten_properties(loader.load_schema("sse")))
    example_paths: set[str] = set()
    for sequence in loader.load_examples("sse").values():
        for event in sequence:
            example_paths |= _instance_paths(event, prefix=f"$defs/{event['event']}/")
    uncovered = schema_paths - example_paths
    assert not uncovered, f"sse: fields never exercised by an example: {sorted(uncovered)}"


def test_embedded_contract_tuples_match_the_loader() -> None:
    versions = loader.contract_versions()
    manifest = loader.load_examples("manifest")["benchmark_run"]
    assert manifest["contract_versions"] == versions
    for example in loader.load_examples("trace").values():
        attrs = example["attributes"]
        assert attrs["atlas.contract.trace_version"] == versions["trace"]
        assert attrs["atlas.contract.dataset_version"] == versions["dataset"]
        assert attrs["atlas.contract.manifest_version"] == versions["manifest"]
        assert attrs["atlas.contract.sse_version"] == versions["sse"]
