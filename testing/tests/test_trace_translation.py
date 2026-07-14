"""`backend/atlas/adapters/trace_translation.py`, hermetic (SP6 task 2): the pure attribute
translation table between `atlas_graph.py`'s informal span vocabulary and the frozen contract's
dotted `atlas.*`/`gen_ai.*` names, fail closed, plus the shape it produces validates against
`contracts/trace/schema.json`.
"""
from __future__ import annotations

import jsonschema
import pytest
from contract_tools import loader

from atlas.adapters.trace_translation import (
    BUILD_ATTRIBUTES,
    REQUIRED_SPAN_ATTRIBUTES,
    STAGE_DURATION_ATTRIBUTE,
    UninventoriedSpanError,
    UnmappedAttributeError,
    span_kind_for,
    translate_attributes,
    translate_span,
)


@pytest.fixture(scope="module")
def schema() -> dict:
    return loader.load_schema("trace")


# ---- the specific renames the plan names by hand -------------------------------------------------


def test_guard_ok_true_becomes_allow():
    out = translate_attributes("pre_render_guard", "guard", {"ok": True, "reason": ""})
    assert out["atlas.guard.decision"] == "allow"


def test_guard_ok_false_becomes_block():
    out = translate_attributes("pre_render_guard", "guard", {"ok": False, "reason": "held"})
    assert out["atlas.guard.decision"] == "block"


def test_no_observed_guard_span_ever_produces_step_up():
    """SP6 plan: step_up is resolved from the observed inventory, never guessed. No guard call site
    in atlas_graph.py carries any signal beyond a bool `ok`, so `atlas.guard.decision` never emits
    "step_up" -- this is the executable proof for that documented (module docstring) decision."""
    for entry in _committed_inventory_entries():
        if entry["kind"] != "guard":
            continue
        attrs = {k: (True if k == "ok" else "x") for k in entry["attrs"]}
        out = translate_attributes(entry["name"], "guard", attrs)
        assert out.get("atlas.guard.decision") in {"allow", "block"}


def test_llm_model_key_becomes_gen_ai_request_model():
    out = translate_attributes("agent", "llm", {"model": "claude-test"})
    assert out["gen_ai.request.model"] == "claude-test"


def test_degradation_mode_key_renames_directly_no_value_transform():
    out = translate_attributes("refusal", "node", {"degradation_mode": "drop_rerank"})
    assert out["atlas.degradation.mode"] == "drop_rerank"


def test_degradation_mode_on_the_fallback_llm_span_also_renames():
    out = translate_attributes("agent", "llm", {"model": "fallback", "degradation_mode": "provider_fallback"})
    assert out["atlas.degradation.mode"] == "provider_fallback"
    assert out["gen_ai.request.model"] == "fallback"


# ---- SP8 task 1: the subject pseudonym and the judge's own span -----------------------------------


def test_customer_id_on_a_turn_span_becomes_a_hmac_pseudonym_never_the_raw_id():
    out = translate_attributes("turn", "turn", {"input": "q", "intent": "account", "customer_id": "cust_1"})
    assert "customer_id" not in out  # the raw id never survives translation
    pseudonym = out["atlas.subject.pseudonym"]
    assert pseudonym != "cust_1"
    assert len(pseudonym) == 16
    assert all(c in "0123456789abcdef" for c in pseudonym)


def test_the_pseudonym_is_stable_for_the_same_customer_id():
    first = translate_attributes("turn", "turn", {"input": "a", "intent": "account", "customer_id": "cust_1"})
    second = translate_attributes("turn", "turn", {"input": "b", "intent": "policy_question", "customer_id": "cust_1"})
    assert first["atlas.subject.pseudonym"] == second["atlas.subject.pseudonym"]


def test_the_pseudonym_differs_for_a_different_customer_id():
    a = translate_attributes("turn", "turn", {"input": "q", "intent": "account", "customer_id": "cust_1"})
    b = translate_attributes("turn", "turn", {"input": "q", "intent": "account", "customer_id": "cust_2"})
    assert a["atlas.subject.pseudonym"] != b["atlas.subject.pseudonym"]


def test_judge_span_keys_translate_to_the_dotted_judge_attributes():
    out = translate_attributes(
        "judge_verdict", "judge",
        {"judge_id": "fingerprint-abc", "rubric_version": "groundedness-v1", "verdict": "grounded"},
    )
    assert out == {
        "atlas.judge.id": "fingerprint-abc",
        "atlas.judge.rubric_version": "groundedness-v1",
        "atlas.judge.verdict": "grounded",
    }


def test_judge_span_carries_an_ungrounded_verdict_too():
    out = translate_attributes(
        "judge_verdict", "judge",
        {"judge_id": "x", "rubric_version": "groundedness-v1", "verdict": "ungrounded"},
    )
    assert out["atlas.judge.verdict"] == "ungrounded"


# ---- SP9 task 5: atlas.cost.* (the usage accounting trio, ADR-029's amendment) --------------------

_COST_INVENTORY = frozenset({
    ("generator_cost", "llm", frozenset({"model", "input_tokens", "output_tokens", "usd"})),
})


def test_cost_keys_on_an_llm_span_translate_to_the_dotted_cost_attributes():
    out = translate_attributes(
        "generator_cost", "llm",
        {"model": "anthropic:claude-sonnet-5", "input_tokens": 120, "output_tokens": 45, "usd": 0.00081},
        inventory=_COST_INVENTORY,
    )
    assert out == {
        "gen_ai.request.model": "anthropic:claude-sonnet-5",
        "atlas.cost.input_tokens": 120,
        "atlas.cost.output_tokens": 45,
        "atlas.cost.usd": 0.00081,
    }


def test_cost_keys_are_scoped_to_kind_llm_only():
    """The cost trio is an `llm` kind rule, not a blanket rename: the same key on a different kind
    must still fail closed (never accidentally pass through or alias)."""
    with pytest.raises(UnmappedAttributeError):
        translate_attributes(
            "budget_guard", "guard", {"usd": 1.0},
            inventory=frozenset({("budget_guard", "guard", frozenset({"usd"}))}),
        )


# ---- passthrough keys: no reserved contract counterpart, kept verbatim ---------------------------


def test_turn_span_input_and_intent_pass_through_unchanged():
    # customer_id is required on a real "turn" shape (SP8 task 1) but is not itself a passthrough
    # key -- see test_customer_id_on_a_turn_span_becomes_a_hmac_pseudonym_never_the_raw_id below.
    out = translate_attributes("turn", "turn", {"input": "q", "intent": "account", "customer_id": "cust_1"})
    assert out["input"] == "q"
    assert out["intent"] == "account"


def test_cache_node_keys_pass_through_unchanged():
    out = translate_attributes("cache", "node", {"hit": True, "generic": False})
    assert out == {"hit": True, "generic": False}


def test_execute_action_success_keys_pass_through_unchanged():
    out = translate_attributes("execute_action", "node", {"applied": True, "reference": "REF-1"})
    assert out == {"applied": True, "reference": "REF-1"}


def test_render_output_key_passes_through_unchanged():
    out = translate_attributes("render", "node", {"output": "the final answer"})
    assert out == {"output": "the final answer"}


def test_bind_guard_reason_free_keys_pass_through_unchanged():
    out = translate_attributes("bind_guard", "guard", {"ok": False, "intent": "policy_question", "tools": ["x"]})
    assert out["intent"] == "policy_question"
    assert out["tools"] == ["x"]
    assert out["atlas.guard.decision"] == "block"


def test_agent_failure_retryable_key_passes_through_unchanged():
    out = translate_attributes("agent_failure", "guard", {"ok": False, "reason": "boom", "retryable": True})
    assert out["retryable"] is True
    assert out["atlas.guard.decision"] == "block"


def test_tool_args_and_result_pass_through():
    out = translate_attributes("*", "tool", {"args": {"a": 1}, "result": "ok"})
    assert out == {"args": {"a": 1}, "result": "ok"}


def test_tool_args_and_proposal_pass_through():
    out = translate_attributes("*", "tool", {"args": {"a": 1}, "proposal": "PLAN-1"})
    assert out == {"args": {"a": 1}, "proposal": "PLAN-1"}


# ---- fail closed: uninventoried tuple -------------------------------------------------------------


def test_an_uninventoried_shape_raises():
    with pytest.raises(UninventoriedSpanError):
        translate_attributes("bind_guard", "guard", {"ok": True, "brand_new_kwarg": 1})


def test_a_known_span_name_with_an_extra_never_seen_key_still_raises():
    with pytest.raises(UninventoriedSpanError):
        translate_attributes("turn", "turn", {"input": "q", "intent": "account", "extra": "surprise"})


def test_fail_closed_uses_the_caller_supplied_inventory_override_when_given():
    empty = frozenset()
    with pytest.raises(UninventoriedSpanError):
        translate_attributes("turn", "turn", {"input": "q", "intent": "account"}, inventory=empty)
    allowed = frozenset({("turn", "turn", frozenset({"input", "intent"}))})
    out = translate_attributes("turn", "turn", {"input": "q", "intent": "account"}, inventory=allowed)
    assert out["input"] == "q"


# ---- fail closed: unmapped key on an otherwise inventoried shape ----------------------------------


def test_an_unmapped_key_on_an_inventoried_shape_raises():
    allowed = frozenset({("turn", "turn", frozenset({"totally_unmapped_key"}))})
    with pytest.raises(UnmappedAttributeError):
        translate_attributes("turn", "turn", {"totally_unmapped_key": "x"}, inventory=allowed)


# ---- tool spans: wildcarded name, per the module's own documented decision -----------------------


def test_tool_kind_is_checked_under_the_wildcard_name_not_the_real_tool_name():
    # any two DIFFERENT real tool names sharing the SAME attribute shape both translate fine,
    # because the inventory check normalizes kind="tool" spans to "*" regardless of name.
    for name in ("get_account_summary", "search_knowledge", "list_plans"):
        out = translate_attributes(name, "tool", {"args": {}, "result": "x"})
        assert out == {"args": {}, "result": "x"}


# ---- span_kind mapping -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kind", "expected"),
    [
        ("turn", "turn", "CHAIN"),
        ("render", "node", "CHAIN"),
        ("agent", "llm", "LLM"),
        ("search_knowledge", "tool", "TOOL"),
        ("pre_render_guard", "guard", "GUARDRAIL"),
        ("judge_verdict", "judge", "EVALUATOR"),
        ("embed", "stage", "RETRIEVER"),
        ("retrieve", "stage", "RETRIEVER"),
        ("rerank", "stage", "RERANKER"),
        ("assemble", "stage", "CHAIN"),
        ("ttft", "stage", "CHAIN"),
    ],
)
def test_span_kind_for_maps_every_kind_to_a_schema_enum_member(name, kind, expected, schema):
    mapped = span_kind_for(name, kind)
    assert mapped == expected
    assert mapped in schema["properties"]["span_kind"]["enum"]


# ---- translate_span: the full record validates against the schema --------------------------------


def test_translate_span_output_validates_against_the_trace_schema(schema):
    record = translate_span("turn", "turn", {"input": "hi", "intent": "account", "customer_id": "cust_1"})
    jsonschema.validate(record, schema)
    assert record["attributes"]["atlas.contract.trace_version"] == loader.contract_versions()["trace"]
    assert record["attributes"]["atlas.privacy.synthetic"] is True


def test_translate_span_of_a_guard_verdict_validates_too(schema):
    record = translate_span("pre_render_guard", "guard", {"ok": False, "reason": "held"})
    jsonschema.validate(record, schema)
    assert record["attributes"]["atlas.guard.decision"] == "block"


def test_every_committed_inventory_entry_translates_and_validates(schema):
    """Every REAL shape atlas_graph.py actually produces (the committed golden inventory) round
    trips through translate_span and validates -- the direct proof that the translation table
    covers every observed call site, not just the handful this file spot checks by hand."""
    for entry in _committed_inventory_entries():
        if entry["kind"] == "stage":
            continue  # stage spans are OtelTracer's own separate mechanism, not translate_span's
        name = "search_knowledge" if entry["name"] == "*" else entry["name"]
        attrs = {k: _sample_value(k) for k in entry["attrs"]}
        record = translate_span(name, entry["kind"], attrs)
        jsonschema.validate(record, schema)


# ---- required attributes stamped by every span -----------------------------------------------------


def test_required_span_attributes_are_exactly_the_schemas_own_required_pair(schema):
    assert set(REQUIRED_SPAN_ATTRIBUTES) == set(schema["properties"]["attributes"]["required"])


# ---- SP6 task 7 (the v1 freeze): BUILD_ATTRIBUTES, build wide constants on every span -------------


def test_build_attributes_are_exactly_the_nine_cheaply_emittable_names_minus_the_settings_sourced_two():
    """The freeze's emitter checklist added 9 new emitters; 7 are build wide constants this module
    computes itself (`BUILD_ATTRIBUTES`), the other 2 (`atlas.corpus.version`/`atlas.index.build_id`)
    are settings sourced and stamped by `otel_tracer.py` directly (this module owns no
    `AtlasSettings` dependency, see its own docstring)."""
    assert set(BUILD_ATTRIBUTES) == {
        "atlas.semconv.version",
        "atlas.variant",
        "atlas.contract.dataset_version",
        "atlas.contract.manifest_version",
        "atlas.contract.sse_version",
        "atlas.privacy.content_captured",
        "atlas.privacy.redaction_policy_version",
    }


def test_build_attributes_are_all_reserved(schema):
    from contract_tools.loader import RESERVED_TRACE_ATTRIBUTES

    for attr in BUILD_ATTRIBUTES:
        assert attr in RESERVED_TRACE_ATTRIBUTES


def test_build_attributes_appear_on_every_translated_span_regardless_of_kind():
    for name, kind, attrs in (
        ("turn", "turn", {"input": "hi", "intent": "account", "customer_id": "cust_1"}),
        ("pre_render_guard", "guard", {"ok": True, "reason": ""}),
        ("agent", "llm", {"model": "claude-test"}),
    ):
        record = translate_span(name, kind, attrs)
        for key, value in BUILD_ATTRIBUTES.items():
            assert record["attributes"][key] == value


def test_atlas_variant_is_graph_this_repos_own_fixed_point():
    assert BUILD_ATTRIBUTES["atlas.variant"] == "graph"


def test_atlas_semconv_version_matches_the_installed_package_pin():
    from importlib.metadata import version

    assert BUILD_ATTRIBUTES["atlas.semconv.version"] == version("opentelemetry-semantic-conventions")


def test_atlas_privacy_content_captured_is_false_matching_the_redaction_allowlist():
    """`contract_tools.redaction.allowed_attributes()` never allowlists a free text passthrough key
    (`input`/`output`/`args`/`result`/`reason`/...), so a REDACTED, exported span never carries
    literal content regardless of which span this lands on -- see `BUILD_ATTRIBUTES`'s own comment."""
    from contract_tools.redaction import allowed_attributes

    allowed = allowed_attributes()
    for key in ("input", "output", "args", "result", "reason", "tool", "proposal"):
        assert key not in allowed
    assert BUILD_ATTRIBUTES["atlas.privacy.content_captured"] is False


def test_atlas_privacy_redaction_policy_version_is_content_addressed_not_a_copy_of_trace_version():
    """SP6 task 7 review fix round 1, Important 1 (the versioning trap): a REGRESSION pin against
    the old bug, not just a value check. `atlas.privacy.redaction_policy_version` used to be a
    literal copy of `atlas.contract.trace_version`, so the two agreeing was never proof the value
    tracked the real redaction allowlist -- it only proved the alias existed. Asserting inequality
    here (both happen to be strings today, "1.0.0" vs a 16 character hex digest) is the actual
    guard against silently reintroducing that alias."""
    assert (
        BUILD_ATTRIBUTES["atlas.privacy.redaction_policy_version"]
        != REQUIRED_SPAN_ATTRIBUTES["atlas.contract.trace_version"]
    )
    value = BUILD_ATTRIBUTES["atlas.privacy.redaction_policy_version"]
    assert len(value) == 16
    assert all(c in "0123456789abcdef" for c in value)


def test_atlas_privacy_redaction_policy_version_matches_the_redaction_generators_own_function():
    """The two independent computations (this module's own `_redaction_policy_version`, and
    `contract_tools.redaction.redaction_policy_version`, which this module can never import --
    `test_import_lint.py`'s own boundary, and it would also be a circular import) must agree without
    ever sharing code, the same cross check discipline
    `test_atlas_contract_dataset_manifest_sse_versions_match_the_loader` below already established
    for the other three contract family versions."""
    from contract_tools.redaction import redaction_policy_version

    assert BUILD_ATTRIBUTES["atlas.privacy.redaction_policy_version"] == redaction_policy_version()


def test_reserved_trace_attributes_from_schema_matches_the_loaders_own_list():
    """This module reads the trace schema's OWN declared attribute property names directly (never a
    `contract_tools.loader` import, the same backend/harness boundary the module docstring's
    redaction policy version comment explains) so it can independently rederive the redaction
    allowlist's reserved category. Pinned here as a set equality against
    `contract_tools.loader.RESERVED_TRACE_ATTRIBUTES`, the SAME 30 names, so the two can never
    silently diverge without a test noticing."""
    from atlas.adapters import trace_translation

    assert set(trace_translation._reserved_trace_attributes_from_schema()) == set(
        loader.RESERVED_TRACE_ATTRIBUTES
    )


def test_atlas_contract_dataset_manifest_sse_versions_match_the_loader(schema):
    versions = loader.contract_versions()
    assert BUILD_ATTRIBUTES["atlas.contract.dataset_version"] == versions["dataset"]
    assert BUILD_ATTRIBUTES["atlas.contract.manifest_version"] == versions["manifest"]
    assert BUILD_ATTRIBUTES["atlas.contract.sse_version"] == versions["sse"]


# ---- degraded_turn.json's own shape: no atlas.stage.rerank_ms on a drop_rerank turn ----------------


def test_golden_examples_are_real_single_spans_from_the_sp6_task_7_live_capture():
    """SP6 task 7 (the v1.0.0 freeze): both golden examples were REGENERATED from a real live
    capture (`contracts/trace/freeze_evidence.json`), replacing the pre freeze aspirational
    composites (one fictional span carrying all 29 attributes at once, including all five
    `atlas.stage.*ms` durations together -- something no real span has ever done: each stage
    duration lands on its OWN separate OTel span, `otel_tracer.py`'s own `close(seq)` stamps exactly
    ONE `atlas.stage.*ms` attribute per span). `chat_turn.json` is a real "turn" span (the richest
    real span this system produces: every `BUILD_ATTRIBUTES` constant plus the three settings
    sourced identity fields); `degraded_turn.json` is a real "refusal" node span (the one real,
    reliably reproducible carrier of a non "none" `atlas.degradation.mode` value -- see
    `ADR-029`'s own "Consequences" section)."""
    examples = loader.load_examples("trace")
    chat_turn, degraded = examples["chat_turn"], examples["degraded_turn"]
    assert chat_turn["name"] == "turn"
    assert degraded["name"] == "refusal"
    assert degraded["attributes"]["atlas.degradation.mode"] == "refusal"
    # No stage duration attribute EVER appears on a "turn" or "refusal" kind span, real or golden:
    # they are a task defined, name keyed mechanism carried by their OWN dedicated stage spans only.
    for record in (chat_turn, degraded):
        for attr in STAGE_DURATION_ATTRIBUTE.values():
            assert attr not in record["attributes"]


def test_stage_duration_attribute_names_are_all_reserved(schema):
    from contract_tools.loader import RESERVED_TRACE_ATTRIBUTES

    for attr in STAGE_DURATION_ATTRIBUTE.values():
        assert attr in RESERVED_TRACE_ATTRIBUTES


# ---- helpers ---------------------------------------------------------------------------------------


def _committed_inventory_entries() -> list[dict]:
    import json

    from atlas.adapters.trace_translation import _INVENTORY_PATH  # test only reach into the module

    return json.loads(_INVENTORY_PATH.read_text())


def _sample_value(key: str):
    if key == "ok":
        return True
    if key == "proposal":
        return "PLAN-1"
    if key == "tools":
        return ["change_plan"]
    if key == "applied":
        return True
    if key == "args":
        return {}
    if key == "result":
        return "ok"
    if key == "degradation_mode":
        return "none"  # a valid contract enum member, never a placeholder value
    if key == "verdict":
        return "grounded"  # a valid contract enum member (SP8 task 1), never a placeholder value
    if key == "customer_id":
        return "cust_1"
    if key in ("input_tokens", "output_tokens"):
        return 120  # atlas.cost.*_tokens is schema typed "integer" (SP9 task 5)
    if key == "usd":
        return 0.00081  # atlas.cost.usd is schema typed "number"
    return "x"
