from __future__ import annotations

import dataclasses
import json

import pytest

from atlas.config import ChunkSettings, Settings, SparseSettings, load_settings
from atlas.contracts import DocName, Document
from atlas.corpus.chunk import get_chunker


def test_the_settings_tree_holds_its_two_guarantees(tmp_path) -> None:
    # A leaf that does not reach the identity puts two studies in one bucket, so the
    # comparison reads as a null result rather than as a bug. A key dropped instead of
    # refused publishes a number attached to a configuration nobody ran.
    baseline = Settings()
    changed = dataclasses.replace(baseline, chunk=dataclasses.replace(baseline.chunk, strategy="sentence"))
    assert changed.run_id != baseline.run_id, "a leaf change did not reach the run identity"

    path = tmp_path / "settings.json"
    path.write_text(json.dumps(dataclasses.asdict(baseline)), encoding="utf-8")
    assert load_settings(path).run_id == baseline.run_id
    assert len(baseline.run_id) == 12

    payload = dataclasses.asdict(baseline)
    payload["embedding"]["normalize"] = False  # the misspelling is the real one: normalise/normalize
    misspelled = tmp_path / "misspelled.json"
    misspelled.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(misspelled)

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"embedding": {"backend": "local"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no field"):
        load_settings(unknown)


def test_a_strategy_that_ignores_overlap_may_not_be_configured_with_one() -> None:
    # Only the fixed size chunker reads overlap_tokens, but a settings object carrying a
    # number nothing reads still changes run_id, so two runs producing identical chunks
    # file as two different experiments. The first block stops this passing vacuously.
    doc = Document(name=DocName("t"), kind="filler",
                   text="A dispute must be filed within 60 days of the statement date. " * 12)
    without = get_chunker(ChunkSettings(strategy="fixed", max_tokens=64, overlap_tokens=0))(doc)
    with_overlap = get_chunker(ChunkSettings(strategy="fixed", max_tokens=64, overlap_tokens=32))(doc)
    assert without != with_overlap, "overlap changes nothing even for fixed; test proves nothing"

    for strategy in ("sentence", "recursive"):
        with pytest.raises(ValueError, match="only applied by the fixed size chunker"):
            ChunkSettings(strategy=strategy, max_tokens=256, overlap_tokens=32)


def test_the_shipped_default_is_a_configuration_that_does_not_lie() -> None:
    assert ChunkSettings().overlap_tokens == 0


def test_sparse_settings_refuses_a_b_outside_its_literature_range() -> None:
    # b=3.0 returns a negative BM25 score for a chunk that does contain the query term,
    # and a settings file is how `just compare` is documented to be used.
    SparseSettings(b=0.0)  # the literature's valid range is inclusive at both ends
    SparseSettings(b=1.0)
    with pytest.raises(ValueError, match="must be within \\[0, 1\\]"):
        SparseSettings(b=3.0)
    with pytest.raises(ValueError, match="must be within \\[0, 1\\]"):
        SparseSettings(b=-0.1)
    with pytest.raises(ValueError, match="k1 must be positive"):
        SparseSettings(k1=0.0)


def test_a_setting_added_after_the_store_does_not_rename_the_runs_already_in_it() -> None:
    # `run_id` hashes the settings tree, so without `_ADDED_AFTER_THE_STORE` every new
    # field renames all 18 recorded runs and invalidates the references to their ids
    # across the docs. The constant is deliberate: derived from Settings() it would
    # assert nothing.
    assert Settings().run_id == "6818412cb6ec"


def test_an_exempted_setting_still_separates_runs_once_it_is_actually_set() -> None:
    # Exempting a field while it holds its default must not weaken the guarantee that two
    # runs differing in any setting get different ids.
    base = Settings()
    ids = {base.run_id}
    for group, changed in (
        ("sparse", dataclasses.replace(base.sparse, stopwords="none")),
        ("sparse", dataclasses.replace(base.sparse, query_term_frequency="distinct")),
    ):
        ids.add(dataclasses.replace(base, **{group: changed}).run_id)
    assert len(ids) == 3, "two settings trees that differ share a run_id"


def test_every_exempted_field_names_a_real_field_at_a_real_default() -> None:
    # An entry naming a field that predates the store would silently merge runs that
    # really did differ, but that half cannot be checked mechanically; this checks the
    # half that can.
    from atlas.config import _ADDED_AFTER_THE_STORE, _GROUPS

    for group, names in _ADDED_AFTER_THE_STORE.items():
        assert group in _GROUPS, f"{group} is not a settings group"
        fields = {f.name for f in dataclasses.fields(_GROUPS[group])}
        assert names <= fields, f"{sorted(names - fields)} are not fields of {group}"
