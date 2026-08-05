from __future__ import annotations

import pytest

from atlas.models.providers import selected_judge_provider, selected_provider


def test_the_judge_provider_defaults_to_the_shared_one(monkeypatch):
    # The default has to stay the shared value, or every existing run silently changes
    # which provider judges its own answers.
    monkeypatch.delenv("JUDGE_MODEL_PROVIDER", raising=False)
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    assert selected_judge_provider() == selected_provider() == "anthropic"


def test_the_judge_provider_overrides_the_shared_one_when_set(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_MODEL_PROVIDER", "anthropic")
    assert selected_provider() == "openai"
    assert selected_judge_provider() == "anthropic"


def test_the_judge_provider_is_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL_PROVIDER", "  Anthropic  ")
    assert selected_judge_provider() == "anthropic"


def test_the_judge_provider_refuses_anything_else(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL_PROVIDER", "azure")
    with pytest.raises(ValueError, match="JUDGE_MODEL_PROVIDER"):
        selected_judge_provider()
