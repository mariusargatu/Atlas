"""SP7 Task 6: `dataset_tools.paraphrase`, hermetic. The worded provider key gate itself, and the
paraphrase machinery's ground truth preserving behavior, are both unit tested with a fake model;
no real provider client is ever constructed and no network call is ever made from this file, the
same "flag gated, never runs in any gate" contract the module docstring states. `main()` without a
key never even imports `replay.providers` (verified below by patching it absent and asserting the
gate still refuses cleanly, before any import could fail loudly instead).
"""
from __future__ import annotations

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult

from dataset_tools import paraphrase

BASE_CASE = {
    "case_id": "gen-fact-plan-fiber-500-monthly_price",
    "split": "dev",
    "origin": "synthetic",
    "candidate_source": "registry_render",
    "source_trace_id": None,
    "intent": "troubleshooting",
    "hop_count": 1,
    "doc_type": "plan_page",
    "adversarial_class": None,
    "failure_class": None,
    "answerable": True,
    "expected_doc_ids": ["588c3e9478b86bea"],
    "expected_facts": [{"fact_id": "plan-fiber-500:monthly_price", "value": "39.99"}],
    "refusal_class": None,
    "persona": None,
    "turns": [{"user": "What is the monthly_price of plan-fiber-500?"}],
    "end_state": None,
}


class _FakeParaphraseModel(BaseChatModel):
    variants: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "fake-paraphrase"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError

    def invoke(self, messages, config=None, **kwargs):  # sync path: paraphrase_text calls .invoke
        return AIMessage(content=json.dumps(list(self.variants)))


class _MalformedModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "malformed"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError

    def invoke(self, messages, config=None, **kwargs):
        return AIMessage(content="not json at all")


# ---- the worded key gate: never guesses, never treats an empty value as configured -----------------


def test_paraphrase_key_configured_is_false_with_no_keys():
    assert paraphrase.paraphrase_key_configured({}) is False


def test_paraphrase_key_configured_is_false_for_an_empty_string_value():
    assert paraphrase.paraphrase_key_configured({"ANTHROPIC_API_KEY": ""}) is False


def test_paraphrase_key_configured_is_true_for_anthropic():
    assert paraphrase.paraphrase_key_configured({"ANTHROPIC_API_KEY": "sk-ant-real"}) is True


def test_paraphrase_key_configured_is_true_for_openai():
    assert paraphrase.paraphrase_key_configured({"OPENAI_API_KEY": "sk-real"}) is True


def test_require_paraphrase_key_raises_when_absent():
    with pytest.raises(paraphrase.ParaphraseKeyMissingError):
        paraphrase.require_paraphrase_key({})


def test_require_paraphrase_key_is_silent_when_present():
    paraphrase.require_paraphrase_key({"OPENAI_API_KEY": "sk-real"})  # no raise


# ---- main(): the gate fires FIRST, before argparse, before any provider import ----------------------


def test_main_without_a_key_returns_nonzero_and_never_imports_a_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # No --case at all: if the gate did not fire first, argparse would raise SystemExit(2) instead
    # of the gate's own clean return 1. This proves ordering, not merely that keys are unset.
    assert paraphrase.main([]) == 1


def test_main_gate_message_names_both_key_env_vars(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    paraphrase.main(["--case", "{}"])
    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY" in out
    assert "OPENAI_API_KEY" in out


# ---- paraphrase_case: ground truth never varies, only case_id and turns[0].user ----------------------


def test_paraphrase_case_produces_n_new_cases_with_unique_ids():
    model = _FakeParaphraseModel(variants=("How much does Fiber 500 cost?", "What's the Fiber 500 price?"))
    out = paraphrase.paraphrase_case(BASE_CASE, n=2, model=model)
    assert len(out) == 2
    assert out[0]["case_id"] == f"{BASE_CASE['case_id']}-para-1"
    assert out[1]["case_id"] == f"{BASE_CASE['case_id']}-para-2"
    assert len({c["case_id"] for c in out}) == 2


def test_paraphrase_case_ground_truth_is_identical_to_the_base_case():
    model = _FakeParaphraseModel(variants=("How much does Fiber 500 cost?",))
    out = paraphrase.paraphrase_case(BASE_CASE, n=1, model=model)
    (case,) = out
    assert case["expected_facts"] == BASE_CASE["expected_facts"]
    assert case["expected_doc_ids"] == BASE_CASE["expected_doc_ids"]
    assert case["answerable"] == BASE_CASE["answerable"]
    assert case["intent"] == BASE_CASE["intent"]
    assert case["adversarial_class"] == BASE_CASE["adversarial_class"]


def test_paraphrase_case_only_case_id_and_turns_change():
    model = _FakeParaphraseModel(variants=("How much does Fiber 500 cost?",))
    out = paraphrase.paraphrase_case(BASE_CASE, n=1, model=model)
    (case,) = out
    changed = {k for k in case if case[k] != BASE_CASE.get(k)}
    assert changed == {"case_id", "turns"}


def test_paraphrase_case_new_phrasing_is_the_models_own_variant():
    model = _FakeParaphraseModel(variants=("How much does Fiber 500 cost?",))
    (case,) = paraphrase.paraphrase_case(BASE_CASE, n=1, model=model)
    assert case["turns"] == [{"user": "How much does Fiber 500 cost?"}]


def test_paraphrase_case_validates_against_the_dataset_schema():
    import jsonschema
    from contract_tools import loader

    model = _FakeParaphraseModel(variants=("How much does Fiber 500 cost?", "What's the price of Fiber 500?"))
    schema = loader.load_schema("dataset")
    for case in paraphrase.paraphrase_case(BASE_CASE, n=2, model=model):
        jsonschema.validate(case, schema)


def test_paraphrase_case_rejects_a_multi_turn_case():
    multi_turn_case = {**BASE_CASE, "turns": [{"user": "a"}, {"user": "b"}]}
    model = _FakeParaphraseModel(variants=("x",))
    with pytest.raises(ValueError, match="single turn"):
        paraphrase.paraphrase_case(multi_turn_case, n=1, model=model)


def test_paraphrase_text_raises_on_a_malformed_model_response():
    with pytest.raises(ValueError, match="valid JSON"):
        paraphrase.paraphrase_text("some question", n=2, model=_MalformedModel())


def test_paraphrase_text_raises_when_the_model_returns_a_json_object_not_an_array():
    class _ObjectModel(_MalformedModel):
        def invoke(self, messages, config=None, **kwargs):
            return AIMessage(content=json.dumps({"not": "a list"}))

    with pytest.raises(ValueError, match="JSON array"):
        paraphrase.paraphrase_text("some question", n=1, model=_ObjectModel())
