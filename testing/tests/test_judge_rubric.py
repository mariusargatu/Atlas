"""`judge.rubric`, hermetic (SP8 task 1): the groundedness rubric's scaffolding, absorbed from the
pre rewrite `evals/judge/rubric.py`; the CONTENT is fresh (binary groundedness against cited
retrieved context), never the pre rewrite `RUBRIC_V1`/`V2` helpfulness/account truth pair.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from judge.rubric import RUBRIC_GROUNDEDNESS, Rubric, compare_prompt, prompt, template_hash


def test_rubric_groundedness_names_entailment_not_helpfulness():
    text = RUBRIC_GROUNDEDNESS.system.lower()
    assert "groundedness" in text or "grounded" in text
    assert "entailed" in text or "support" in text
    assert "pass" in text and "fail" in text


def test_rubric_groundedness_names_abstention_as_a_pass():
    assert "abstain" in RUBRIC_GROUNDEDNESS.system.lower()


def test_template_hash_is_stable_for_the_same_rubric():
    assert template_hash(RUBRIC_GROUNDEDNESS) == template_hash(RUBRIC_GROUNDEDNESS)


def test_template_hash_changes_with_the_system_text():
    other = Rubric(version=RUBRIC_GROUNDEDNESS.version, system="a completely different rubric text")
    assert template_hash(RUBRIC_GROUNDEDNESS) != template_hash(other)


def test_template_hash_changes_with_the_version_even_if_text_is_identical():
    other = Rubric(version="groundedness-v2", system=RUBRIC_GROUNDEDNESS.system)
    assert template_hash(RUBRIC_GROUNDEDNESS) != template_hash(other)


def test_prompt_carries_the_rubric_question_context_and_answer():
    messages = prompt(RUBRIC_GROUNDEDNESS, "Is my plan contract-free?", "No, it has a fee.", "chunk: plans have a cancellation fee")
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == RUBRIC_GROUNDEDNESS.system
    assert isinstance(messages[1], HumanMessage)
    body = messages[1].content
    assert "Is my plan contract-free?" in body
    assert "No, it has a fee." in body
    assert "chunk: plans have a cancellation fee" in body


def test_compare_prompt_carries_both_answers_and_context_and_asks_for_a_or_b():
    messages = compare_prompt(RUBRIC_GROUNDEDNESS, "q", "answer A text", "answer B text", "cited chunk")
    assert isinstance(messages[0], SystemMessage)
    assert "A or B" in messages[0].content
    body = messages[1].content
    assert "answer A text" in body
    assert "answer B text" in body
    assert "cited chunk" in body
