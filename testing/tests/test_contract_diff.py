"""The evolution rules as a table: each row is a rule from contracts/README.md."""

from __future__ import annotations

import copy

import pytest
from contract_tools import diff, loader

BASE = {
    "type": "object",
    "required": ["a"],
    "properties": {
        "a": {"type": "string"},
        "b": {"type": "integer"},
        "mode": {"enum": ["x", "y"]},
        "nested": {
            "type": "object",
            "required": ["inner"],
            "properties": {"inner": {"type": "string"}},
        },
        "items_field": {"type": "array", "items": {"type": "object", "properties": {"leaf": {"type": "number"}}}},
    },
}


def _with(**overrides) -> dict:
    schema = {**BASE, "properties": {**BASE["properties"]}}
    schema.update({k: v for k, v in overrides.items() if k != "properties"})
    if "properties" in overrides:
        schema["properties"].update(overrides["properties"])
    return schema


def _without_property(name: str) -> dict:
    schema = {**BASE, "properties": {k: v for k, v in BASE["properties"].items() if k != name}}
    return schema


@pytest.mark.parametrize(
    ("new_schema", "level", "reason_fragment"),
    [
        (_without_property("b"), "major", "removed property: b"),
        (_with(properties={"b": {"type": "string"}}), "major", "retyped property: b"),
        (_with(required=["a", "b"]), "major", "new required"),
        (_with(properties={"mode": {"enum": ["x"]}}), "major", "narrowed enum"),
        (_with(properties={"c": {"type": "boolean"}}), "minor", "added optional"),
        (_with(properties={"mode": {"enum": ["x", "y", "z"]}}), "minor", "widened enum"),
        (_with(title="cosmetic"), "patch", "metadata"),
        (BASE, "patch", "no change"),
    ],
)
def test_rule_table(new_schema: dict, level: str, reason_fragment: str) -> None:
    report = diff.classify_change(BASE, new_schema)
    assert report.level == level
    assert any(reason_fragment in r for r in report.reasons)


def test_nested_removal_is_major() -> None:
    new = {**BASE, "properties": {**BASE["properties"], "nested": {"type": "object", "required": ["inner"], "properties": {}}}}
    assert diff.classify_change(BASE, new).level == "major"


def test_nested_new_required_is_major() -> None:
    new = {**BASE, "properties": {**BASE["properties"], "nested": {"type": "object", "required": ["inner", "extra"], "properties": {"inner": {"type": "string"}, "extra": {"type": "string"}}}}}
    assert diff.classify_change(BASE, new).level == "major"


def test_array_item_retype_is_major() -> None:
    new = {**BASE, "properties": {**BASE["properties"], "items_field": {"type": "array", "items": {"type": "object", "properties": {"leaf": {"type": "string"}}}}}}
    assert diff.classify_change(BASE, new).level == "major"


@pytest.mark.parametrize(
    ("old_v", "new_v", "level", "ok"),
    [
        ("1.2.0", "2.0.0", "major", True),
        ("1.2.0", "1.3.0", "major", False),
        ("0.1.0", "0.2.0", "major", True),
        ("0.1.0", "0.1.1", "major", False),
        ("1.2.0", "1.3.0", "minor", True),
        ("1.2.0", "1.2.1", "minor", False),
        ("1.2.0", "1.2.1", "patch", True),
        ("1.2.0", "1.2.0", "patch", False),
    ],
)
def test_required_bump(old_v: str, new_v: str, level: str, ok: bool) -> None:
    report = diff.ChangeReport(level=level, reasons=("test",))
    got_ok, _ = diff.required_bump(old_v, new_v, report)
    assert got_ok is ok


def test_pre_freeze_relaxation_stops_applying_once_major_is_1() -> None:
    """SP1 carry, closed by SP6 task 7 (the v1 freeze): the digest's own words, "while MAJOR is 0,
    breaking requires at least a MINOR bump," is a PRE FREEZE relaxation only. Once a family's
    `x-contract-version` crosses to MAJOR >= 1, `required_bump` must fall through to the ordinary
    SchemaVer floor (a breaking change needs a full MAJOR bump, MINOR is never enough), never the
    relaxed 0.x rule. `required_bump`'s own `if old_v[0] == 0:` guard (present since the engine's
    first commit, 4df2a35) is exactly what keeps the two regimes separate; this pins the transition
    explicitly (not just via the pre existing 1.2.0 cases) so a future edit cannot silently widen
    the relaxed branch's reach past major 0 again."""
    report = diff.ChangeReport(level="major", reasons=("removed property: x",))
    # AT the freeze boundary itself (old major == 0): the relaxed rule still governs, exactly as it
    # always has -- reaching 1.0.0 from a 0.x breaking change needs only "at least a MINOR bump".
    ok, why = diff.required_bump("0.9.0", "1.0.0", report)
    assert ok, why
    # PAST the freeze (old major == 1): a breaking change bumped only to 1.1.0 (a MINOR bump, what
    # the pre freeze rule would have accepted) must now be REJECTED.
    ok, why = diff.required_bump("1.0.0", "1.1.0", report)
    assert not ok
    assert "MAJOR" in why
    # The correctly bumped equivalent (a real MAJOR increment) is accepted.
    ok, why = diff.required_bump("1.0.0", "2.0.0", report)
    assert ok, why
    # A second major cycle floors the same way (never retriggers the 0.x relaxation).
    ok, why = diff.required_bump("2.3.0", "2.4.0", report)
    assert not ok
    ok, why = diff.required_bump("2.3.0", "3.0.0", report)
    assert ok, why


def test_change_report_rejects_an_unknown_level() -> None:
    with pytest.raises(ValueError, match="level"):
        diff.ChangeReport(level="bogus", reasons=("x",))


def test_flatten_uses_slash_separator_because_attribute_names_contain_dots() -> None:
    flat = diff.flatten_properties(BASE)
    assert "nested/inner" in flat
    assert "items_field/leaf" in flat


def _sse_schema() -> dict:
    return loader.load_schema("sse")


def test_deleting_a_def_from_the_sse_schema_is_major() -> None:
    old = _sse_schema()
    new = copy.deepcopy(old)
    del new["$defs"]["error"]
    new["oneOf"] = [ref for ref in new["oneOf"] if ref["$ref"] != "#/$defs/error"]
    report = diff.classify_change(old, new)
    assert report.level == "major"
    assert any(r.startswith("removed property: $defs/error/") for r in report.reasons)


def test_narrowing_an_enum_inside_a_def_is_major() -> None:
    old = _sse_schema()
    new = copy.deepcopy(old)
    new["$defs"]["message_end"]["properties"]["finish_reason"]["enum"] = [
        "complete",
        "refusal",
        "truncated",
    ]
    report = diff.classify_change(old, new)
    assert report.level == "major"
    assert any("narrowed enum on $defs/message_end/finish_reason" in r for r in report.reasons)


def test_adding_an_optional_property_inside_a_def_is_minor() -> None:
    old = _sse_schema()
    new = copy.deepcopy(old)
    new["$defs"]["citation"]["properties"]["confidence"] = {"type": "number"}
    report = diff.classify_change(old, new)
    assert report.level == "minor"
    assert any("added optional property: $defs/citation/confidence" in r for r in report.reasons)


def test_union_type_list_order_is_not_a_retype() -> None:
    old = {"type": "object", "properties": {"x": {"type": ["string", "null"]}}}
    new = {"type": "object", "properties": {"x": {"type": ["null", "string"]}}}
    report = diff.classify_change(old, new)
    assert report.level == "patch"
    assert not any("retyped" in r for r in report.reasons)


def _manifest_schema() -> dict:
    return loader.load_schema("manifest")


def test_pattern_tightening_is_an_unmodeled_minor_change() -> None:
    old = _manifest_schema()
    new = copy.deepcopy(old)
    new["properties"]["git_sha"]["pattern"] = "^[0-9a-f]{40}$"
    report = diff.classify_change(old, new)
    assert report.level == "minor"
    assert any(
        "unmodeled change on git_sha" in r and "pattern" in r for r in report.reasons
    )


def test_title_only_change_on_a_real_schema_is_still_patch() -> None:
    old = _manifest_schema()
    new = {**old, "title": "a different title entirely"}
    report = diff.classify_change(old, new)
    assert report.level == "patch"
    assert any("metadata" in r for r in report.reasons)


def _write(tmp_path, name: str, schema: dict) -> str:
    import json

    p = tmp_path / name
    p.write_text(json.dumps(schema))
    return str(p)


def test_cli_accepts_a_legal_minor_bump(tmp_path, capsys) -> None:
    old = {**BASE, "x-contract-version": "0.1.0"}
    new = _with(properties={"c": {"type": "boolean"}})
    new["x-contract-version"] = "0.2.0"
    code = diff.main([_write(tmp_path, "old.json", old), _write(tmp_path, "new.json", new)])
    assert code == 0
    assert "minor" in capsys.readouterr().out


def test_cli_rejects_a_breaking_change_without_a_bump(tmp_path, capsys) -> None:
    old = {**BASE, "x-contract-version": "0.1.0"}
    new = _without_property("b")
    new["x-contract-version"] = "0.1.1"
    code = diff.main([_write(tmp_path, "old.json", old), _write(tmp_path, "new.json", new)])
    assert code == 1
    out = capsys.readouterr().out
    assert "removed property: b" in out


def test_cli_rejects_a_malformed_version_without_a_traceback(tmp_path, capsys) -> None:
    old = {**BASE, "x-contract-version": "v1-beta"}
    new = {**BASE, "x-contract-version": "0.1.0"}
    code = diff.main([_write(tmp_path, "old.json", old), _write(tmp_path, "new.json", new)])
    assert code == 2
    out = capsys.readouterr().out
    assert "error: invalid version" in out


def test_cli_missing_new_file_exits_2_with_cannot_read(tmp_path, capsys) -> None:
    old_path = _write(tmp_path, "old.json", BASE)
    missing_path = str(tmp_path / "does-not-exist.json")
    code = diff.main([old_path, missing_path])
    assert code == 2
    out = capsys.readouterr().out
    assert "cannot read" in out


def test_family_schema_paths_derives_from_loader() -> None:
    assert diff.family_schema_paths() == [
        "contracts/trace/schema.json",
        "contracts/dataset/schema.json",
        "contracts/manifest/schema.json",
        "contracts/sse/schema.json",
    ]


def test_cli_git_ref_failure_exits_2_with_clear_message(capsys) -> None:
    code = diff.main(["--git-ref", "no-such-ref-xyz", "contracts/trace/schema.json"])
    assert code == 2
    out = capsys.readouterr().out
    assert "cannot read" in out


def test_the_real_trace_schema_bump_1_2_0_to_1_3_0_is_a_validated_minor() -> None:
    """SP9 task 5: the cost trio's real emitter closes 1.2.0 to 1.3.0 (ADR-029's own Amendment) --
    additive, since the three properties were already declared, reserved, since v0.1.0; only their
    emitter status moves. The diff engine cannot detect THAT REASON on its own from schema text
    alone (ADR-029's own words: the version bump is this repository's own signal, not something a
    schema diff can infer) -- what it CAN and must validate is that the real committed bump is at
    least the SchemaVer floor a change with the same content ever requires, never a downgrade or a
    version left unchanged."""
    new = loader.load_schema("trace")
    assert new["x-contract-version"] == "1.3.0"
    old = {**new, "x-contract-version": "1.2.0"}  # the only real diff: the version field itself
    report = diff.classify_change(old, new)
    assert report.level == "patch"
    ok, why = diff.required_bump("1.2.0", "1.3.0", report)
    assert ok, why


def test_property_level_description_edit_is_patch() -> None:
    new = {**BASE, "properties": {**BASE["properties"], "a": {"type": "string", "description": "the a field"}}}
    report = diff.classify_change(BASE, new)
    assert report.level == "patch"


def test_dropping_an_enum_constraint_is_minor_widening() -> None:
    new = {**BASE, "properties": {**BASE["properties"], "mode": {"type": "string"}}}
    report = diff.classify_change(BASE, new)
    assert report.level == "minor"
    assert any("widened" in r or "unmodeled" in r for r in report.reasons)


def test_removing_a_required_entry_is_minor_widening() -> None:
    new = {**BASE, "required": []}
    report = diff.classify_change(BASE, new)
    assert report.level == "minor"
    assert any("no longer required" in r for r in report.reasons)


def test_adding_a_type_where_none_existed_is_major() -> None:
    old = {**BASE, "properties": {**BASE["properties"], "open_field": {}}}
    new = {**old, "properties": {**old["properties"], "open_field": {"type": "string"}}}
    report = diff.classify_change(old, new)
    assert report.level == "major"
    assert any("narrowed" in r and "type constraint added" in r for r in report.reasons)


def test_dropping_a_type_constraint_is_minor_widening() -> None:
    old = {**BASE, "properties": {**BASE["properties"], "open_field": {"type": "string"}}}
    new = {**old, "properties": {**old["properties"], "open_field": {}}}
    report = diff.classify_change(old, new)
    assert report.level == "minor"
    assert any("widened" in r and "type constraint dropped" in r for r in report.reasons)


def test_enum_drop_with_explicit_type_stays_minor() -> None:
    """Regression pin for the SP2 Task 1 fix: an enum only field gaining an explicit type
    while its enum is dropped must stay minor, not trip the new type added major rule."""
    new = {**BASE, "properties": {**BASE["properties"], "mode": {"type": "string"}}}
    report = diff.classify_change(BASE, new)
    assert report.level == "minor"
    assert any("enum constraint dropped" in r for r in report.reasons)
    assert not any("type constraint added" in r for r in report.reasons)


def test_adding_an_enum_where_none_existed_is_major() -> None:
    old = {**BASE, "properties": {**BASE["properties"], "open_field": {}}}
    new = {**old, "properties": {**old["properties"], "open_field": {"enum": ["x"]}}}
    report = diff.classify_change(old, new)
    assert report.level == "major"
    assert any("narrowed" in r and "enum constraint added" in r for r in report.reasons)


def test_open_value_field_tightening_is_caught() -> None:
    """Pins the reviewer's exact reproduction: contracts/dataset/schema.json's
    expected_facts[].value is deliberately open ({}); tightening it to a concrete type
    must be reported as major, not silently swallowed as a metadata only patch."""
    old = loader.load_schema("dataset")
    new = copy.deepcopy(old)
    new["properties"]["expected_facts"]["items"]["properties"]["value"] = {"type": "string"}
    report = diff.classify_change(old, new)
    assert report.level == "major"
