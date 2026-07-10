"""`load.prompt_corpus`, hermetic (SP9 task 6): the golden, fixed prompt corpus the k6 script cycles
through, so every concurrency step drives the SAME controlled token counts (never a freshly
sampled, uncontrolled question) -- the load lane's own precondition for a fair stepped comparison.
Pure JSON to dataclass loading, no network, no k6 binary.
"""
from __future__ import annotations

import json

import pytest

from load.prompt_corpus import (
    DEFAULT_PROMPT_CORPUS_PATH,
    VALID_BUCKETS,
    GoldenPrompt,
    load_prompt_corpus,
    prompt_for_iteration,
)


def test_load_prompt_corpus_reads_the_committed_canonical_file():
    prompts = load_prompt_corpus()
    assert len(prompts) == 6
    assert all(isinstance(p, GoldenPrompt) for p in prompts)
    assert {p.bucket for p in prompts} == VALID_BUCKETS


def test_load_prompt_corpus_preserves_file_order_never_resorted():
    prompts = load_prompt_corpus()
    assert prompts[0].prompt_id == "short-price"
    assert prompts[-1].prompt_id == "long-upgrade"


def test_load_prompt_corpus_rejects_an_empty_corpus(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="empty"):
        load_prompt_corpus(path)


def test_load_prompt_corpus_rejects_a_duplicate_prompt_id(tmp_path):
    path = tmp_path / "dup.json"
    entry = {"prompt_id": "dup", "bucket": "short", "text": "hello there", "approx_tokens": 2}
    path.write_text(json.dumps([entry, entry]))
    with pytest.raises(ValueError, match="duplicate"):
        load_prompt_corpus(path)


def test_load_prompt_corpus_rejects_an_unrecognized_bucket(tmp_path):
    path = tmp_path / "bad_bucket.json"
    entry = {"prompt_id": "x", "bucket": "extra-long", "text": "hello there", "approx_tokens": 2}
    path.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="bucket"):
        load_prompt_corpus(path)


def test_load_prompt_corpus_rejects_a_declared_token_count_that_drifted_from_the_text(tmp_path):
    path = tmp_path / "drifted.json"
    entry = {
        "prompt_id": "x", "bucket": "short",
        "text": "one two three four five six seven eight nine ten",  # 10 words
        "approx_tokens": 2,  # wildly wrong, must not be silently trusted
    }
    path.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="drifted"):
        load_prompt_corpus(path)


def test_load_prompt_corpus_tolerates_a_small_declared_drift(tmp_path):
    """The declared count is a whitespace approximation, not a real tokenizer count -- a SMALL
    drift (within tolerance) is not a hard failure."""
    path = tmp_path / "close_enough.json"
    entry = {
        "prompt_id": "x", "bucket": "short",
        "text": "one two three four five",  # 5 words
        "approx_tokens": 6,  # off by one, within tolerance
    }
    path.write_text(json.dumps([entry]))
    prompts = load_prompt_corpus(path)
    assert prompts[0].approx_tokens == 6


def test_prompt_for_iteration_cycles_deterministically():
    prompts = load_prompt_corpus()
    n = len(prompts)
    assert prompt_for_iteration(prompts, 0) == prompts[0]
    assert prompt_for_iteration(prompts, n) == prompts[0]
    assert prompt_for_iteration(prompts, n + 1) == prompts[1]


def test_prompt_for_iteration_rejects_an_empty_prompt_set():
    with pytest.raises(ValueError, match="at least one"):
        prompt_for_iteration((), 0)


def test_default_prompt_corpus_path_points_at_the_committed_file_next_to_this_module():
    assert DEFAULT_PROMPT_CORPUS_PATH.name == "prompt_corpus.json"
    assert DEFAULT_PROMPT_CORPUS_PATH.exists()
