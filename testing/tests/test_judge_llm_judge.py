"""`judge.llm_judge`, hermetic (SP8 task 1): the judge routed through the REPLAY gateway (ADR-007,
zero keys, seeded cassettes, the same D19 named seam the agent under test already uses), the
fail closed parse, the order swap position bias probe, and the trace boundary verdict translation.
"""
from __future__ import annotations

import tempfile

from replay.cassette_store import seed_cassette
from replay.gateway import GatewayChatModel

from judge.llm_judge import (
    VERDICT_GROUNDED,
    VERDICT_UNGROUNDED,
    judge_label,
    order_swap,
    translate_verdict,
)
from judge.rubric import RUBRIC_GROUNDEDNESS, compare_prompt, prompt

_MODEL_ID = "gpt-judge"


def _gateway(cassette_dir) -> GatewayChatModel:
    return GatewayChatModel(model_id=_MODEL_ID, cassette_dir=cassette_dir, mode="replay")


# ---- judge_label: a real REPLAY round trip through the gateway -----------------------------------


def test_judge_label_replays_a_pass_verdict_as_one():
    with tempfile.TemporaryDirectory(prefix="judge-llm-") as cdir:
        question, answer, context = "Is my plan contract-free?", "No, it has a cancellation fee.", "chunk: fee applies"
        seed_cassette(cdir, prompt(RUBRIC_GROUNDEDNESS, question, answer, context), {"content": "PASS", "tool_calls": []}, _MODEL_ID)
        label = judge_label(_gateway(cdir), RUBRIC_GROUNDEDNESS, question, answer, context)
    assert label == 1


def test_judge_label_replays_a_fail_verdict_as_zero():
    with tempfile.TemporaryDirectory(prefix="judge-llm-") as cdir:
        question, answer, context = "Is my plan contract-free?", "Yes, totally free forever.", "chunk: fee applies"
        seed_cassette(cdir, prompt(RUBRIC_GROUNDEDNESS, question, answer, context), {"content": "FAIL", "tool_calls": []}, _MODEL_ID)
        label = judge_label(_gateway(cdir), RUBRIC_GROUNDEDNESS, question, answer, context)
    assert label == 0


# ---- fail closed parse: an unparseable reply is never evidence of a pass -------------------------


def test_judge_label_fails_closed_on_an_unparseable_reply():
    with tempfile.TemporaryDirectory(prefix="judge-llm-") as cdir:
        question, answer, context = "q", "a", "c"
        seed_cassette(cdir, prompt(RUBRIC_GROUNDEDNESS, question, answer, context), {"content": "I am not sure how to answer that.", "tool_calls": []}, _MODEL_ID)
        label = judge_label(_gateway(cdir), RUBRIC_GROUNDEDNESS, question, answer, context)
    assert label == 0


def test_judge_label_fails_closed_on_an_empty_reply():
    with tempfile.TemporaryDirectory(prefix="judge-llm-") as cdir:
        question, answer, context = "q", "a", "c"
        seed_cassette(cdir, prompt(RUBRIC_GROUNDEDNESS, question, answer, context), {"content": "", "tool_calls": []}, _MODEL_ID)
        label = judge_label(_gateway(cdir), RUBRIC_GROUNDEDNESS, question, answer, context)
    assert label == 0


# ---- fail closed parse: a PASS prefixed word is not the standalone PASS token ---------------------
# A naive `text.startswith("PASS")` misparses "PASSABLE" as a PASS verdict, since the string does
# start with the four letters "PASS". The parser must read the first STANDALONE token, so a word
# that merely starts with "PASS" without being exactly "PASS" fails closed, same as any other
# unparseable reply.


def test_judge_label_does_not_parse_passable_as_a_pass():
    with tempfile.TemporaryDirectory(prefix="judge-llm-") as cdir:
        question, answer, context = "q", "a", "c"
        seed_cassette(cdir, prompt(RUBRIC_GROUNDEDNESS, question, answer, context), {"content": "PASSABLE", "tool_calls": []}, _MODEL_ID)
        label = judge_label(_gateway(cdir), RUBRIC_GROUNDEDNESS, question, answer, context)
    assert label == 0


def test_judge_label_does_not_parse_a_hyphenated_pass_word_as_a_pass():
    with tempfile.TemporaryDirectory(prefix="judge-llm-") as cdir:
        question, answer, context = "q", "a", "c"
        seed_cassette(cdir, prompt(RUBRIC_GROUNDEDNESS, question, answer, context), {"content": "PASS-ish, arguably", "tool_calls": []}, _MODEL_ID)
        label = judge_label(_gateway(cdir), RUBRIC_GROUNDEDNESS, question, answer, context)
    assert label == 0


def test_judge_label_still_parses_a_bare_pass_with_trailing_punctuation():
    with tempfile.TemporaryDirectory(prefix="judge-llm-") as cdir:
        question, answer, context = "q", "a", "c"
        seed_cassette(cdir, prompt(RUBRIC_GROUNDEDNESS, question, answer, context), {"content": "PASS.", "tool_calls": []}, _MODEL_ID)
        label = judge_label(_gateway(cdir), RUBRIC_GROUNDEDNESS, question, answer, context)
    assert label == 1


# ---- order_swap: consistent vs flipping pairs ------------------------------------------------------


def test_order_swap_is_consistent_on_a_clear_pair():
    q = "Is there a cap on my data?"
    true_ans, false_ans, context = "No.", "No cap at all, unlimited.", "chunk: 50GB monthly cap"
    with tempfile.TemporaryDirectory(prefix="judge-swap-") as cdir:
        def seed_pair(first, second, picks_first):
            seed_cassette(
                cdir, compare_prompt(RUBRIC_GROUNDEDNESS, q, first, second, context),
                {"content": "A" if picks_first else "B", "tool_calls": []}, _MODEL_ID,
            )
        seed_pair(true_ans, false_ans, picks_first=True)   # (true, false) -> A (true wins)
        seed_pair(false_ans, true_ans, picks_first=False)  # (false, true) -> B (true wins again)
        winner_first, winner_second = order_swap(_gateway(cdir), RUBRIC_GROUNDEDNESS, q, true_ans, false_ans, context)
    assert winner_first == winner_second == 0  # "true_ans" (index 0, answer_a) wins regardless of order


def test_order_swap_flips_on_a_position_biased_pair():
    q = "Is my plan uncapped?"
    hard_a, hard_b, context = "Your plan is uncapped.", "There is no data limit on your plan.", "chunk: plan has no data cap"
    with tempfile.TemporaryDirectory(prefix="judge-swap-") as cdir:
        def seed_pair(first, second, picks_first):
            seed_cassette(
                cdir, compare_prompt(RUBRIC_GROUNDEDNESS, q, first, second, context),
                {"content": "A" if picks_first else "B", "tool_calls": []}, _MODEL_ID,
            )
        seed_pair(hard_a, hard_b, picks_first=True)  # (a, b) -> A
        seed_pair(hard_b, hard_a, picks_first=True)  # (b, a) -> A (now b) -> a flip in the a/b frame
        winner_first, winner_second = order_swap(_gateway(cdir), RUBRIC_GROUNDEDNESS, q, hard_a, hard_b, context)
    assert winner_first != winner_second  # the judge's pick tracked reading order, not content


# ---- translate_verdict: the trace boundary, both values -------------------------------------------


def test_translate_verdict_of_a_pass_label_is_grounded():
    assert translate_verdict(1) == VERDICT_GROUNDED == "grounded"


def test_translate_verdict_of_a_fail_label_is_ungrounded():
    assert translate_verdict(0) == VERDICT_UNGROUNDED == "ungrounded"


def test_translate_verdict_is_independent_of_the_prompt_vocabulary():
    # the judge's own prompt asks for PASS/FAIL; the wire vocabulary never leaks that word back.
    assert "pass" not in translate_verdict(1).lower()
    assert "fail" not in translate_verdict(0).lower()
