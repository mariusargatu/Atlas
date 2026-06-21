"""D11 style hermetic byte diff of the committed golden span inventory
(`contracts/trace/span_inventory.json`) against a fresh run of `contract_tools.trace_inventory`'s
own scenarios (SP6 task 2). Mirrors `test_mcp_snapshots.py`'s exact discipline for MCP tool schemas:
this is what makes an UNREVIEWED drift in `atlas_graph.py`'s tracer.open() vocabulary (a renamed
span, an added kwarg) fail loud instead of silently reaching `trace_translation.py`'s fail closed
gate for the first time in production.
"""
from __future__ import annotations

from contract_tools.trace_inventory import INVENTORY_PATH, render_inventory


def test_committed_inventory_is_byte_identical_to_a_fresh_run():
    fresh = render_inventory()
    assert INVENTORY_PATH.is_file(), f"no committed span inventory at {INVENTORY_PATH}"
    committed = INVENTORY_PATH.read_text()
    assert committed == fresh, (
        f"the trace span inventory drifted from the committed file ({INVENTORY_PATH}). Regenerate "
        "via `uv run python -m contract_tools.trace_inventory --write` and review the diff before "
        "committing, if the change is intentional."
    )


def test_inventory_rendering_is_deterministic_across_repeated_calls():
    # a regression here (e.g. dict/set iteration order leaking into the dump) would make the byte
    # diff test above flaky rather than a reliable gate; pin the determinism claim directly.
    assert render_inventory() == render_inventory()


def test_every_entry_has_the_three_expected_fields():
    import json

    entries = json.loads(INVENTORY_PATH.read_text())
    assert entries, "the committed inventory must not be empty"
    for entry in entries:
        assert set(entry) == {"name", "kind", "attrs"}
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["kind"], str) and entry["kind"]
        assert isinstance(entry["attrs"], list)


def test_tool_kind_entries_are_only_ever_recorded_under_the_wildcard_name():
    import json

    from contract_tools.trace_inventory import TOOL_WILDCARD

    entries = json.loads(INVENTORY_PATH.read_text())
    tool_entries = [e for e in entries if e["kind"] == "tool"]
    assert tool_entries, "expected at least one kind=tool entry"
    assert all(e["name"] == TOOL_WILDCARD for e in tool_entries)


def test_the_four_read_loop_stage_names_are_all_present():
    import json

    entries = json.loads(INVENTORY_PATH.read_text())
    stage_names = {e["name"] for e in entries if e["kind"] == "stage"}
    assert stage_names == {"embed", "retrieve", "rerank", "assemble"}
