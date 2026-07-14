"""corpus_mutation.scope, hermetic (SP8 task 7): the ephemeral corpus_version this lane's mutated
render lives under. `test_corpus_build.py`'s own frozen artifact rule governs the ONE committed
corpus_version (corpus-0.1.1); this lane's render is a throwaway probe, so its corpus_version must
never look like a committed one, never collide with an actually committed one, and every byte it
writes must be gone once the probe finishes, success or failure.
"""
from __future__ import annotations

import pytest

from corpus_mutation.scope import (
    COMMITTED_CORPUS_ROOT,
    EphemeralCorpusVersion,
    committed_corpus_versions,
    ephemeral_corpus_version,
    is_committed_style_version,
)
from corpus_mutation.selection import FactMutation

_MUTATION = FactMutation(
    contradiction_id="conflict-daniel-contract",
    fact_ref="contract_term-daniel-2025:contract_months",
    old_value=12,
    new_value=24,
    question="Is my plan contract free?",
)
_OTHER_MUTATION = FactMutation(
    contradiction_id="conflict-daniel-contract",
    fact_ref="contract_term-daniel-2025:contract_months",
    old_value=12,
    new_value=99,
    question="Is my plan contract free?",
)


# ---- naming: never looks committed, never collides with an actually committed version -----------


def test_ephemeral_corpus_version_never_matches_the_committed_naming_pattern():
    assert not is_committed_style_version(ephemeral_corpus_version(_MUTATION))


def test_ephemeral_corpus_version_never_collides_with_an_actually_committed_version():
    assert ephemeral_corpus_version(_MUTATION) not in committed_corpus_versions()


def test_committed_corpus_versions_names_the_real_committed_corpus():
    assert "corpus-0.1.1" in committed_corpus_versions()


def test_is_committed_style_version_accepts_the_semver_shape_only():
    assert is_committed_style_version("corpus-0.1.1")
    assert not is_committed_style_version("corpus-mutation-deadbeef0000")
    assert not is_committed_style_version("corpus-0.1")
    assert not is_committed_style_version("not-a-corpus-version")


def test_ephemeral_corpus_version_is_deterministic_for_the_same_mutation():
    assert ephemeral_corpus_version(_MUTATION) == ephemeral_corpus_version(_MUTATION)


def test_ephemeral_corpus_version_differs_for_a_different_mutation():
    assert ephemeral_corpus_version(_MUTATION) != ephemeral_corpus_version(_OTHER_MUTATION)


# ---- EphemeralCorpusVersion: a real directory pair that is ALWAYS cleaned up ---------------------


def test_scope_yields_the_expected_corpus_version_and_two_existing_directories():
    with EphemeralCorpusVersion(_MUTATION) as scope:
        assert scope.corpus_version == ephemeral_corpus_version(_MUTATION)
        assert scope.corpus_root.is_dir()
        assert scope.index_root.is_dir()


def test_scope_directories_are_removed_on_a_clean_exit():
    with EphemeralCorpusVersion(_MUTATION) as scope:
        corpus_root, index_root = scope.corpus_root, scope.index_root
        (corpus_root / "marker.txt").write_text("hello")
    assert not corpus_root.exists()
    assert not index_root.exists()


def test_scope_directories_are_removed_even_when_the_body_raises():
    captured: dict[str, object] = {}
    with pytest.raises(RuntimeError):
        with EphemeralCorpusVersion(_MUTATION) as scope:
            captured["corpus_root"] = scope.corpus_root
            captured["index_root"] = scope.index_root
            (scope.corpus_root / "marker.txt").write_text("hello")
            raise RuntimeError("boom")
    assert not captured["corpus_root"].exists()
    assert not captured["index_root"].exists()


def test_scope_root_never_lands_under_the_real_committed_corpus_directory():
    with EphemeralCorpusVersion(_MUTATION) as scope:
        resolved = scope.corpus_root.resolve()
        assert resolved != COMMITTED_CORPUS_ROOT.resolve()
        assert COMMITTED_CORPUS_ROOT.resolve() not in resolved.parents
