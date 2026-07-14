"""The living-dataset loop: scrub a production failure, then promote it to a silver golden case.

Scrub redacts a real customer's conversation and synthesises the identity to a seeded account
before anything CI replays touches it. Promotion produces a silver, source="production" GoldenCase,
provisional until a human ratifies it to gold. `promote` refuses an un-scrubbed session (structured
PII or a supplied name), and GoldenCase validation independently refuses a non-seeded identity.
"""
from __future__ import annotations

import pytest

from evals.monitor.feedback import ProdSession, contains_pii, promote, scrub, scrub_text


def _session(turns, expected="reflects THIS customer's real terms", customer_id="cust_real_9087",
             category="policy_question", risk="grounded-but-false", consequence="high",
             oracle="truth_for(customer).has_contract is True",
             captured_at="", model_id="", trace_ref="") -> ProdSession:
    return ProdSession(
        id="prod-4213", turns=tuple(turns), customer_id=customer_id, expected=expected,
        category=category, risk=risk, consequence=consequence, oracle=oracle,
        captured_at=captured_at, model_id=model_id, trace_ref=trace_ref,
    )


# ---- scrub_text: each structured PII type, and the greediest-last ordering ----

def test_scrub_redacts_email():
    assert scrub_text("mail me at dan.okafor@example.co.uk please") == "mail me at [EMAIL] please"


def test_scrub_redacts_a_card_number_as_a_card_not_an_account():
    # 16 digits also match the bare-digit-run account pattern; card must run first.
    assert scrub_text("my card is 4111 1111 1111 1111") == "my card is [CARD]"


def test_scrub_redacts_a_uk_phone_as_a_phone_not_an_account():
    assert scrub_text("call 07700900123") == "call [PHONE]"
    assert scrub_text("call +447700900123") == "call [PHONE]"


def test_scrub_redacts_a_spaced_uk_phone():
    # single space/hyphen groups after the +44/0 prefix still redact as a phone, not slip through
    assert scrub_text("call me on 07911 123456") == "call me on [PHONE]"
    assert scrub_text("ring +44 7911 123456 today") == "ring [PHONE] today"


def test_scrub_redacts_an_account_number():
    assert scrub_text("account 12345678 is overdue") == "account [ACCOUNT] is overdue"


def test_scrub_redacts_a_grouped_account_number():
    # space/hyphen grouped runs of 8+ total digits redact even though no single run is 6+ contiguous
    assert scrub_text("account 4002 1785 overdue") == "account [ACCOUNT] overdue"
    assert scrub_text("sort code and number 12-34-56 78901234") == "sort code and number [ACCOUNT]"


def test_scrub_leaves_an_iso_date_alone():
    # the one carved-out shape: a bare ISO date is metadata, not an account, and must survive
    assert scrub_text("the incident on 2026-07-17 was resolved") == "the incident on 2026-07-17 was resolved"


def test_a_spaced_phone_is_caught_by_the_promote_gate_pre_scrub():
    # contains_pii is the promote-time gate: the unscrubbed spaced phone must read as PII
    assert contains_pii("call me on 07911 123456") is True


def test_scrub_text_rejects_an_empty_or_whitespace_name():
    # an empty name degenerates to \b\b and would match everywhere: fail loudly, do not corrupt text
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        scrub_text("Hi there", names=["Ada", ""])
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        contains_pii("Hi there", names=["   "])


def test_scrub_redacts_a_uk_postcode():
    assert scrub_text("I live at SW1A 1AA") == "I live at [POSTCODE]"


def test_scrub_leaves_ordinary_numbers_alone():
    # a price and a year are not identifiers: 6+ digit runs only.
    assert scrub_text("my £42 bill from 2026") == "my £42 bill from 2026"


def test_scrub_redacts_supplied_names_case_insensitively():
    assert scrub_text("Hi, I'm Daniel", names=["Daniel"]) == "Hi, I'm [NAME]"
    assert scrub_text("thanks daniel", names=["Daniel"]) == "thanks [NAME]"


# ---- contains_pii: the promote-time guard ----

def test_contains_pii_detects_and_clears():
    assert contains_pii("reach me at a@b.com") is True
    assert contains_pii("no identifiers here, just £42") is False
    assert contains_pii("Daniel called", names=["Daniel"]) is True


# ---- scrub the session: redact turns + expected, synthesise the identity ----

def test_scrub_session_redacts_turns_and_remaps_identity():
    s = _session(["I'm Ada, account 12345678, am I free to cancel?"], customer_id="cust_real_9087")
    cleaned = scrub(s, as_customer="cust_legacy_term", names=["Ada"])
    assert cleaned.customer_id == "cust_legacy_term"          # synthesised to a seeded account
    assert "[ACCOUNT]" in cleaned.turns[0] and "[NAME]" in cleaned.turns[0]
    assert "12345678" not in cleaned.turns[0] and "Ada" not in cleaned.turns[0]


# ---- promote: silver production GoldenCase, with the scrub gate ----

def test_promote_produces_a_silver_production_case():
    s = scrub(_session(["am I free to cancel?"]), as_customer="cust_legacy_term")
    case = promote(s, names=(), graders=("answer-true-vs-account",))
    assert case.tier == "silver" and case.source == "production"
    assert case.customer_id == "cust_legacy_term" and case.author_role == "engineer"
    assert case.graders == ("answer-true-vs-account",)
    assert case.id == "prod-4213" and case.category == "policy_question"


def test_promote_refuses_unscrubbed_pii():
    # identity remapped to a seeded account, but a card still in the turn: scrub is the gate.
    s = _session(["pay with 4111 1111 1111 1111"], customer_id="cust_legacy_term")
    with pytest.raises(ValueError, match="un-scrubbed PII"):
        promote(s, names=())


def test_promote_refuses_a_session_whose_turns_still_carry_a_supplied_name():
    # identity remapped and no structured PII, but the name was never actually scrubbed from the
    # turn: promote re-checks the caller's own names list rather than trusting scrub happened.
    s = _session(["thanks Ada, that's clear"], customer_id="cust_legacy_term")
    with pytest.raises(ValueError, match="un-scrubbed PII"):
        promote(s, names=["Ada"])


def test_promote_refuses_a_non_seeded_identity():
    # no PII, but the real customer id was never synthesised: GoldenCase validation fails closed.
    s = _session(["am I free to cancel?"], customer_id="cust_real_9087")
    with pytest.raises(ValueError, match="not a seeded account"):
        promote(s, names=())


def test_promote_can_ratify_to_gold():
    s = scrub(_session(["am I free to cancel?"]), as_customer="cust_legacy_term")
    assert promote(s, names=(), tier="gold", author_role="sme").tier == "gold"


def test_the_scrubbed_case_is_reusable_as_an_eval_case():
    s = scrub(_session(["am I free to cancel?"]), as_customer="cust_legacy_term")
    eval_case = promote(s, names=()).to_eval_case()
    assert eval_case.customer_id == "cust_legacy_term" and eval_case.id == "prod-4213"


# ---- promote: provenance folded into notes (captured_at / model_id / trace_ref) ----

def test_promote_folds_present_provenance_into_the_case_notes():
    s = scrub(
        _session(["am I free to cancel?"], captured_at="2026-07-10",
                  model_id="claude-sonnet-4-5", trace_ref="trace-9c2f1e04"),
        as_customer="cust_legacy_term",
    )
    case = promote(s, names=())
    assert "captured=2026-07-10" in case.notes
    assert "model=claude-sonnet-4-5" in case.notes
    assert "trace=trace-9c2f1e04" in case.notes


def test_promote_leaves_notes_empty_when_no_provenance_is_supplied():
    s = scrub(_session(["am I free to cancel?"]), as_customer="cust_legacy_term")
    assert promote(s, names=()).notes == ""


# ---- notes are part of the session: scrubbed by scrub(), scanned by the promote gate ----

def test_scrub_redacts_pii_in_the_session_notes():
    from dataclasses import replace

    s = replace(_session(["am I free to cancel?"]), notes="triager: customer phone 07911 123456")
    cleaned = scrub(s, as_customer="cust_legacy_term")
    assert "[PHONE]" in cleaned.notes and "07911" not in cleaned.notes


def test_promote_refuses_a_session_whose_notes_still_carry_pii():
    from dataclasses import replace

    # identity remapped and turns clean, but a phone number is sitting in the notes: the gate must
    # scan notes too, or the leak folds straight into the golden case's notes past promotion.
    s = replace(_session(["am I free to cancel?"], customer_id="cust_legacy_term"),
                notes="left a note with 07911 123456")
    with pytest.raises(ValueError, match="un-scrubbed PII"):
        promote(s, names=())
