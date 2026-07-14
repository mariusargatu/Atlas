"""`testing/harness/contract_tools/freeze_check.py`, hermetic (SP6 task 7): the trace contract's
v1.0.0 freeze gate. Every test here is offline -- no docker, no network, no live process -- either
by stubbing evidence directly (`evaluate`'s own pure, injectable shape) or by reading files already
committed to this repo (the span inventory, the narrowed companion, the live capture artifact).
"""
from __future__ import annotations

import json

import pytest
import yaml
from contract_tools import freeze_check, loader


# ---- evaluate(): pure, injectable -- the three scenarios the plan names by hand ------------------


def test_green_path_every_reserved_attribute_emitted_or_narrowed():
    reserved = ("a", "b", "c")
    statuses = freeze_check.evaluate(
        reserved=reserved,
        hermetic_emitted=frozenset({"a", "b"}),
        live_emitted=frozenset({"a", "b"}),
        narrowed={"c": freeze_check.NarrowedEntry(owner="SP9", reason="not built yet")},
    )
    assert [s.status for s in statuses] == ["emitted", "emitted", "narrowed"]
    assert all(s.ok for s in statuses)


def test_missing_attribute_path_stops_the_freeze():
    reserved = ("a", "b")
    statuses = freeze_check.evaluate(
        reserved=reserved,
        hermetic_emitted=frozenset({"a"}),  # "b" has no evidence anywhere, and is not narrowed
        live_emitted=frozenset({"a"}),
        narrowed={},
    )
    by_attr = {s.attribute: s for s in statuses}
    assert by_attr["a"].status == "emitted"
    assert by_attr["b"].status == "missing_both"
    assert not by_attr["b"].ok
    assert any(not s.ok for s in statuses)


def test_narrowed_attribute_path_is_exempt_even_with_zero_evidence():
    reserved = ("a",)
    statuses = freeze_check.evaluate(
        reserved=reserved,
        hermetic_emitted=frozenset(),
        live_emitted=frozenset(),
        narrowed={"a": freeze_check.NarrowedEntry(owner="SP7", reason="no producer yet")},
    )
    assert statuses[0].status == "narrowed"
    assert statuses[0].ok


# ---- evaluate(): the two partial evidence shapes, distinguished --------------------------------


def test_hermetic_only_evidence_is_missing_live_not_emitted():
    statuses = freeze_check.evaluate(
        reserved=("a",), hermetic_emitted=frozenset({"a"}), live_emitted=frozenset(), narrowed={},
    )
    assert statuses[0].status == "missing_live"
    assert not statuses[0].ok


def test_live_only_evidence_is_missing_hermetic_not_emitted():
    statuses = freeze_check.evaluate(
        reserved=("a",), hermetic_emitted=frozenset(), live_emitted=frozenset({"a"}), narrowed={},
    )
    assert statuses[0].status == "missing_hermetic"
    assert not statuses[0].ok


def test_attribute_status_rejects_an_unknown_status():
    with pytest.raises(ValueError, match="status"):
        freeze_check.AttributeStatus("x", "bogus")


# ---- load_narrowed(): the real committed companion, and its error paths -------------------------


def test_the_real_narrowed_yaml_parses_and_names_exactly_the_three_gap_attributes():
    """SP8 task 1 closed four of the original ten (the judge trio plus the subject pseudonym);
    SP9 task 5 closed the usage accounting trio (ADR-029's own Amendments). Three remain, all RAG
    observability."""
    narrowed = freeze_check.load_narrowed()
    assert set(narrowed) == {
        "atlas.retrieval.doc_ids",
        "atlas.rerank.scores_pre",
        "atlas.rerank.scores_post",
    }
    for entry in narrowed.values():
        assert entry.owner
        assert entry.reason


def test_load_narrowed_rejects_a_missing_file(tmp_path):
    with pytest.raises(freeze_check.NarrowedFormatError, match="no such file"):
        freeze_check.load_narrowed(tmp_path / "does-not-exist.yaml")


def test_load_narrowed_rejects_malformed_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("narrowed: [this is not: valid: yaml: at all")
    with pytest.raises(freeze_check.NarrowedFormatError, match="invalid YAML"):
        freeze_check.load_narrowed(path)


def test_load_narrowed_rejects_a_missing_top_level_key(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("something_else: []\n")
    with pytest.raises(freeze_check.NarrowedFormatError, match="narrowed"):
        freeze_check.load_narrowed(path)


def test_load_narrowed_rejects_an_entry_missing_a_required_field(tmp_path):
    path = tmp_path / "incomplete.yaml"
    path.write_text(yaml.safe_dump({"narrowed": [{"attribute": "atlas.judge.id", "owner": "x"}]}))
    with pytest.raises(freeze_check.NarrowedFormatError, match="attribute.*owner.*reason"):
        freeze_check.load_narrowed(path)


def test_load_narrowed_rejects_an_attribute_the_trace_contract_does_not_reserve(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump({
        "narrowed": [{"attribute": "atlas.jugde.id", "owner": "x", "reason": "typo'd name"}]
    }))
    with pytest.raises(freeze_check.NarrowedFormatError, match="not a RESERVED_TRACE_ATTRIBUTES"):
        freeze_check.load_narrowed(path)


# ---- hermetic_emitted_attributes(): the real translation code against the real inventory --------


def test_hermetic_emitted_attributes_covers_exactly_the_twenty_seven_real_emitters():
    """19 at the v1.0.0 freeze, plus atlas.turn.seq (1.1.0), plus the four judge/pseudonym
    attributes SP8 task 1 closed at 1.2.0, plus the usage accounting trio SP9 task 5 closed at
    1.3.0."""
    emitted = freeze_check.hermetic_emitted_attributes()
    reserved = set(loader.RESERVED_TRACE_ATTRIBUTES)
    covered = reserved & emitted
    assert covered == reserved - set(freeze_check.load_narrowed())


def test_hermetic_emitted_attributes_accepts_an_injected_entries_override():
    entries = [{"name": "turn", "kind": "turn", "attrs": ["input", "intent", "customer_id"]}]
    emitted = freeze_check.hermetic_emitted_attributes(entries)
    assert "input" in emitted
    assert "atlas.contract.trace_version" in emitted  # REQUIRED_SPAN_ATTRIBUTES, always merged in
    assert "atlas.variant" in emitted  # BUILD_ATTRIBUTES, always merged in
    assert "atlas.config.hash" in emitted  # settings sourced, named explicitly


def test_hermetic_emitted_attributes_skips_stage_entries_but_still_counts_their_duration_attr():
    entries = [{"name": "embed", "kind": "stage", "attrs": []}]
    emitted = freeze_check.hermetic_emitted_attributes(entries)
    assert "atlas.stage.embed_ms" in emitted


# ---- live_emitted_attributes(): parsing the committed OTLP shaped evidence artifact --------------


def test_live_emitted_attributes_walks_resource_scope_spans_to_the_attribute_keys(tmp_path):
    path = tmp_path / "evidence.json"
    export = {
        "resourceSpans": [{
            "scopeSpans": [{
                "spans": [
                    {"name": "turn", "attributes": [
                        {"key": "atlas.variant", "value": {"stringValue": "graph"}},
                        {"key": "atlas.privacy.synthetic", "value": {"boolValue": True}},
                    ]},
                    {"name": "embed", "attributes": [
                        {"key": "atlas.stage.embed_ms", "value": {"doubleValue": 12.5}},
                    ]},
                ]
            }]
        }]
    }
    path.write_text(json.dumps({"_capture": "test fixture", "exports": [export]}))
    keys = freeze_check.live_emitted_attributes(path)
    assert keys == {"atlas.variant", "atlas.privacy.synthetic", "atlas.stage.embed_ms"}


def test_live_emitted_attributes_unions_across_multiple_exports(tmp_path):
    path = tmp_path / "evidence.json"
    export_a = {"resourceSpans": [{"scopeSpans": [{"spans": [
        {"name": "turn", "attributes": [{"key": "atlas.variant", "value": {}}]},
    ]}]}]}
    export_b = {"resourceSpans": [{"scopeSpans": [{"spans": [
        {"name": "refusal", "attributes": [{"key": "atlas.degradation.mode", "value": {}}]},
    ]}]}]}
    path.write_text(json.dumps({"_capture": "test fixture", "exports": [export_a, export_b]}))
    keys = freeze_check.live_emitted_attributes(path)
    assert keys == {"atlas.variant", "atlas.degradation.mode"}


def test_live_emitted_attributes_handles_an_export_with_no_spans_gracefully(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"_capture": "test fixture", "exports": []}))
    assert freeze_check.live_emitted_attributes(path) == frozenset()


# ---- main(): CLI wiring, real sources -----------------------------------------------------------


def test_the_real_committed_freeze_is_clean(capsys):
    """The actual gate: `contracts/trace/freeze_evidence.json` (the committed live capture) and
    `contracts/trace/freeze_narrowed.yaml` (the committed narrowing) together must leave every one
    of the 30 real RESERVED_TRACE_ATTRIBUTES either emitted (both evidence sources) or narrowed (29
    at the v1.0.0 freeze, plus atlas.turn.seq, trace 1.1.0, I1 fix, SP6 final review). This is the
    test that keeps the freeze honest forever: if a future edit ever removes an emitter, or the
    narrowed file drifts from what is really emitted, this goes red."""
    code = freeze_check.main([])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "freeze clean" in out


def test_main_reports_a_missing_narrowed_file_as_an_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(freeze_check, "NARROWED_PATH", tmp_path / "does-not-exist.yaml")
    code = freeze_check.main([])
    assert code == 2
    assert "error" in capsys.readouterr().out
