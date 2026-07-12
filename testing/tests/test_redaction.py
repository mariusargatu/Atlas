"""D11 style hermetic byte diff of the generated OTel collector redaction config (SP6 task 3),
mirroring `test_span_inventory.py`/`test_mcp_snapshots.py`'s exact discipline: an UNREVIEWED change
to what the trace pipeline actually emits (a new `RESERVED_TRACE_ATTRIBUTES` name, a new gen_ai.*
key `trace_translation.py`'s table starts producing) must fail this gate loud, not silently ship an
allowlist that no longer matches what a real span carries.

Two independent things this file proves, not one:
  1. the generator's drift gate: `contract_tools.redaction.render_collector_config()` byte matches
     the committed `infra/observability/otel-collector.yaml` (the same "regenerate and assert
     equality" mechanism as span_inventory.json/mcp_snapshots).
  2. the allowlist is genuinely GROUNDED in the Task 2 translation table, not a frozen copy that
     happens to agree today: `test_emitted_gen_ai_attributes_reflects_live_changes_...` proves a
     monkeypatched translation table changes what this module derives, and
     `test_todays_emitted_gen_ai_attributes_match_the_reviewed_pin` proves today's real derivation
     matches a hand reviewed, committed expectation -- so a silent change to `_translate_attr`'s gen_ai
     rule would fail THIS test even before anyone remembers to regenerate the collector config.
"""
from __future__ import annotations

import json

from contract_tools.loader import RESERVED_TRACE_ATTRIBUTES
from contract_tools.redaction import (
    COLLECTOR_CONFIG_PATH,
    OPENINFERENCE_SPAN_KIND_ATTRIBUTE,
    SPAN_INVENTORY_PATH,
    allowed_attributes,
    emitted_gen_ai_attributes,
    redaction_policy_version,
    render_collector_config,
)

# Hand reviewed, committed pin (module docstring's point 2 above): today's real derivation must
# equal exactly this set. A change to trace_translation.py's gen_ai mapping rules that is NOT
# accompanied by a reviewed update to this constant fails loud here, independent of whether
# infra/observability/otel-collector.yaml was also regenerated.
_REVIEWED_GEN_AI_ATTRIBUTES = frozenset({"gen_ai.request.model"})


def test_committed_collector_config_is_byte_identical_to_a_fresh_render():
    fresh = render_collector_config()
    assert COLLECTOR_CONFIG_PATH.is_file(), f"no committed collector config at {COLLECTOR_CONFIG_PATH}"
    committed = COLLECTOR_CONFIG_PATH.read_text()
    assert committed == fresh, (
        f"the generated OTel collector config drifted from the committed file "
        f"({COLLECTOR_CONFIG_PATH}). Regenerate via `uv run python -m contract_tools.redaction "
        "--write` and review the diff before committing, if the change is intentional."
    )


def test_collector_config_rendering_is_deterministic_across_repeated_calls():
    assert render_collector_config() == render_collector_config()


def test_todays_emitted_gen_ai_attributes_match_the_reviewed_pin():
    assert emitted_gen_ai_attributes() == _REVIEWED_GEN_AI_ATTRIBUTES


def test_allowed_attributes_is_reserved_names_union_emitted_gen_ai_names_union_openinference_span_kind():
    expected = set(RESERVED_TRACE_ATTRIBUTES) | _REVIEWED_GEN_AI_ATTRIBUTES | {OPENINFERENCE_SPAN_KIND_ATTRIBUTE}
    assert set(allowed_attributes()) == expected


def test_openinference_span_kind_is_always_allowed():
    """Found live (this task's own operator verification): D13's own "Base" framing names
    openinference.span.kind as a peer category to atlas.* and gen_ai.*, not an extension of either.
    otel_tracer.py stamps it on EVERY exported span (the OTel wire encoding of the contract's own
    top level span_kind field); omitting it from the allowlist strips exactly what Phoenix's own UI
    uses to categorize a span, confirmed by a real turn through the compose observability profile
    showing it in the collector's own redaction.redacted.keys diagnostic before this constant existed."""
    assert OPENINFERENCE_SPAN_KIND_ATTRIBUTE in allowed_attributes()


def test_allowed_attributes_has_no_duplicates_and_is_sorted():
    attrs = allowed_attributes()
    assert len(attrs) == len(set(attrs))
    assert attrs == tuple(sorted(attrs))


def test_every_reserved_trace_attribute_is_in_the_allowlist():
    # RESERVED_TRACE_ATTRIBUTES are ALL approved ahead of time in v1 (HLD D13), whether or not each one has a
    # real emitter yet (Task 2's review found only 10 of 29 do today) -- the allowlist must cover
    # every reserved name unconditionally, not just the currently emitted subset.
    missing = set(RESERVED_TRACE_ATTRIBUTES) - set(allowed_attributes())
    assert not missing, f"reserved trace attributes missing from the redaction allowlist: {missing}"


def test_committed_collector_config_lists_every_allowed_attribute_as_an_allowed_key():
    text = COLLECTOR_CONFIG_PATH.read_text()
    for attr in allowed_attributes():
        assert f"- {attr}" in text, f"{attr!r} missing from the committed collector config's allowed_keys"


def test_committed_collector_config_never_names_an_informal_passthrough_key():
    # The informal diagnostic keys with no namespace that trace_translation.py passes through unchanged
    # (reason, tool, args, result, ...) must NEVER appear in the allowlist: they carry potentially
    # real turn content (a user's raw query text, a tool's raw args/result) and must be redacted
    # away by the collector, not preserved by an accidental allowlist entry.
    informal_keys = ("reason", "tool", "tools", "args", "result", "proposal", "input", "output")
    allowed = set(allowed_attributes())
    for key in informal_keys:
        assert key not in allowed


def test_langsmith_and_langwatch_appear_only_as_commented_configuration():
    """D13: LangSmith/LangWatch prove pluggability as commented collector YAML, never a live export
    target. Every line naming either must be a YAML comment (a prose note ABOUT the commented block,
    like this test's own docstring, is fine too -- what matters is that parsing the file as YAML,
    which drops comments entirely, never surfaces either name anywhere active)."""
    import yaml

    text = COLLECTOR_CONFIG_PATH.read_text()
    lines_naming_either = [
        line for line in text.splitlines() if "langsmith" in line.lower() or "langwatch" in line.lower()
    ]
    assert lines_naming_either, "expected at least one LangSmith/LangWatch mention (commented, per D13)"
    for line in lines_naming_either:
        assert line.lstrip().startswith("#"), f"LangSmith/LangWatch must stay commented: {line!r}"

    doc = yaml.safe_load(text)  # YAML parsing drops every comment; nothing active should remain
    rendered_active = json.dumps(doc).lower()
    assert "langsmith" not in rendered_active
    assert "langwatch" not in rendered_active


def test_collector_config_wires_phoenix_as_a_real_exporter():
    text = COLLECTOR_CONFIG_PATH.read_text()
    assert "otlp/phoenix" in text
    pipeline_start = text.index("service:")
    assert "otlp/phoenix" in text[pipeline_start:]


def test_collector_config_archives_raw_otlp_to_a_file_exporter():
    # D13: "raw OTLP archived." A file exporter in the same (already redacted, see the module
    # docstring's own note on pipeline ordering) pipeline as Phoenix.
    text = COLLECTOR_CONFIG_PATH.read_text()
    assert "file/raw_archive" in text
    pipeline_start = text.index("service:")
    assert "file/raw_archive" in text[pipeline_start:]


def test_collector_config_is_valid_yaml():
    import yaml

    doc = yaml.safe_load(COLLECTOR_CONFIG_PATH.read_text())
    assert doc["processors"]["redaction"]["allow_all_keys"] is False
    assert set(doc["processors"]["redaction"]["allowed_keys"]) == set(allowed_attributes())
    assert doc["service"]["pipelines"]["traces"]["exporters"] == ["otlp/phoenix", "file/raw_archive"]
    assert doc["service"]["pipelines"]["traces"]["processors"] == ["redaction"]


def test_emitted_gen_ai_attributes_reflects_live_changes_in_the_translation_table_not_a_frozen_copy(monkeypatch):
    """Adversarial proof: `emitted_gen_ai_attributes` is really grounded in
    `trace_translation.translate_attributes`'s CURRENT behaviour, not a hardcoded literal that
    happens to agree with it today. If this ever silently drifted, `test_todays_emitted_gen_ai_
    attributes_match_the_reviewed_pin` would also start failing the moment `trace_translation.py`'s
    real mapping changes -- this test proves the mechanism that makes that true."""
    import contract_tools.redaction as redaction_module

    original = redaction_module.trace_translation.translate_attributes

    def _patched(name, kind, attrs, **kwargs):
        result = dict(original(name, kind, attrs, **kwargs))
        if name == "agent" and kind == "llm" and "model" in attrs:
            result["gen_ai.trial.new_signal"] = "x"
        return result

    monkeypatch.setattr(redaction_module.trace_translation, "translate_attributes", _patched)
    keys = redaction_module.emitted_gen_ai_attributes()
    assert "gen_ai.trial.new_signal" in keys
    assert "gen_ai.trial.new_signal" not in _REVIEWED_GEN_AI_ATTRIBUTES


def test_emitted_gen_ai_attributes_accepts_an_explicit_entries_override():
    # entries=... is test only, the same style trace_translation.translate_attributes's own
    # inventory= override already uses; production callers always read the committed file.
    from contract_tools.redaction import emitted_gen_ai_attributes as fn

    real_entries = json.loads(SPAN_INVENTORY_PATH.read_text())
    assert fn(entries=real_entries) == fn()


def test_span_inventory_path_matches_the_task_2_trace_inventory_module():
    from contract_tools.trace_inventory import INVENTORY_PATH

    assert SPAN_INVENTORY_PATH == INVENTORY_PATH


def test_reserved_trace_attributes_constant_has_30_names():
    # sanity: this test file's whole premise depends on RESERVED_TRACE_ATTRIBUTES actually being the
    # 30 name list the loader promises (29 at the v1.0.0 freeze, plus atlas.turn.seq added in the
    # trace 1.1.0 MINOR bump, I1 fix, SP6 final review) -- a regression here would make every
    # assertion above vacuous.
    assert len(RESERVED_TRACE_ATTRIBUTES) == 30


# ---- redaction_policy_version(): content addressed, SP6 task 7 review fix round 1, Important 1 ----


def test_redaction_policy_version_is_a_16_character_hex_digest():
    value = redaction_policy_version()
    assert len(value) == 16
    assert all(c in "0123456789abcdef" for c in value)


def test_redaction_policy_version_is_deterministic_across_repeated_calls():
    assert redaction_policy_version() == redaction_policy_version()


def test_redaction_policy_version_matches_a_manual_sha256_over_the_canonical_sorted_allowlist():
    import hashlib

    canonical = "\n".join(allowed_attributes())
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    assert redaction_policy_version() == expected


def test_redaction_policy_version_changes_when_an_allowlist_key_is_added(monkeypatch):
    """Tamper proof (SP6 task 7 review fix round 1, Important 1): the old `atlas.contract.
    trace_version` alias would NOT have changed here, because the trace schema's own declared
    property set is untouched by this monkeypatch -- that gap is exactly the versioning trap this
    fix round closes. The content addressed version MUST change the moment the allowlist itself
    gains a key, proving the value is really grounded in `allowed_attributes()`'s current output,
    not a frozen or borrowed identity."""
    import contract_tools.redaction as redaction_module

    baseline = redaction_module.redaction_policy_version()
    original = redaction_module.allowed_attributes
    monkeypatch.setattr(
        redaction_module, "allowed_attributes", lambda: tuple(sorted({*original(), "gen_ai.trial.new_key"}))
    )
    assert redaction_module.redaction_policy_version() != baseline


def test_redaction_policy_version_changes_when_an_allowlist_key_is_removed(monkeypatch):
    """Tamper proof, the mirror case: removing a key must ALSO change the version, not only adding
    one -- a version that only reacted to additions would still miss a real narrowing of the
    allowlist."""
    import contract_tools.redaction as redaction_module

    baseline = redaction_module.redaction_policy_version()
    original = redaction_module.allowed_attributes()
    monkeypatch.setattr(redaction_module, "allowed_attributes", lambda: original[1:])
    assert redaction_module.redaction_policy_version() != baseline


def test_redaction_policy_version_is_embedded_as_a_comment_in_the_generated_collector_config():
    text = render_collector_config()
    assert f"redaction_policy_version: {redaction_policy_version()}" in text


def test_redaction_policy_version_is_not_a_copy_of_any_contract_familys_version():
    # Regression guard for the exact bug this fix round closes: the old value was a literal copy of
    # RESERVED_TRACE_ATTRIBUTES's owning schema's own x-contract-version ("0.1.0"/"1.0.0" today,
    # whichever family). A 16 character lowercase hex digest can never collide with a SchemaVer
    # string in this repo's actual data, so this also doubles as a shape check.
    from contract_tools.loader import contract_versions

    assert redaction_policy_version() not in contract_versions().values()
