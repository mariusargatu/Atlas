"""`judge.contract`, hermetic (SP8 task 1): the versioned instrument identity, absorbed verbatim
from the pre rewrite `evals/judge/contract.py`. D15's own rule, word for word: a judge's identity is
``(judge_model_id, rubric_version, prompt_template_hash)``; any change to any one field voids a prior
calibration, so ``fingerprint()`` must move when, and only when, one of the three actually moves.
"""
from __future__ import annotations

import dataclasses

import pytest

from judge.contract import JudgeContract


def test_fingerprint_is_stable_for_the_same_triple():
    base = JudgeContract("gpt-judge", "groundedness-v1", "abc123")
    same = JudgeContract("gpt-judge", "groundedness-v1", "abc123")
    assert base.fingerprint() == same.fingerprint()


def test_fingerprint_moves_when_the_model_id_changes():
    base = JudgeContract("gpt-judge", "groundedness-v1", "abc123")
    other = JudgeContract("claude-judge", "groundedness-v1", "abc123")
    assert base.fingerprint() != other.fingerprint()


def test_fingerprint_moves_when_the_rubric_version_changes():
    base = JudgeContract("gpt-judge", "groundedness-v1", "abc123")
    other = JudgeContract("gpt-judge", "groundedness-v2", "abc123")
    assert base.fingerprint() != other.fingerprint()


def test_fingerprint_moves_when_the_template_hash_changes():
    base = JudgeContract("gpt-judge", "groundedness-v1", "abc123")
    other = JudgeContract("gpt-judge", "groundedness-v1", "def456")
    assert base.fingerprint() != other.fingerprint()


def test_fingerprint_is_a_hex_digest_string():
    contract = JudgeContract("gpt-judge", "groundedness-v1", "abc123")
    fp = contract.fingerprint()
    assert isinstance(fp, str)
    assert len(fp) == 64  # sha256 hexdigest, the same canonical digest() every other identity uses
    assert all(c in "0123456789abcdef" for c in fp)


def test_contract_is_frozen():
    contract = JudgeContract("gpt-judge", "groundedness-v1", "abc123")
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.judge_model_id = "other-model"
