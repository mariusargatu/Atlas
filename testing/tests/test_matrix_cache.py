"""`matrix.cache`, hermetic: the content hash cache (D17) a rerun of the staged runner uses to
recompute only missing cells. No network, no wall clock; `cell_key` is exercised directly against
`determinism.canonical.digest` so the key derivation can never silently drift from the one hashing
scheme the cassette key/run digest already use.
"""
from __future__ import annotations

from determinism.canonical import digest

from matrix.cache import MatrixCache, cell_key


def test_cell_key_matches_the_canonical_digest_directly():
    key = cell_key(corpus_version="corpus-x", dataset_version="0.1.0", component_id="bge-m3", params={"k": 5})
    expected = digest(
        {"corpus_version": "corpus-x", "dataset_version": "0.1.0", "component_id": "bge-m3", "params": {"k": 5}}
    )
    assert key == expected


def test_cell_key_is_order_independent_over_params():
    a = cell_key(corpus_version="c", dataset_version="d", component_id="x", params={"a": 1, "b": 2})
    b = cell_key(corpus_version="c", dataset_version="d", component_id="x", params={"b": 2, "a": 1})
    assert a == b


def test_cell_key_changes_when_any_input_changes():
    base = cell_key(corpus_version="c", dataset_version="d", component_id="x", params={"k": 1})
    assert base != cell_key(corpus_version="c2", dataset_version="d", component_id="x", params={"k": 1})
    assert base != cell_key(corpus_version="c", dataset_version="d2", component_id="x", params={"k": 1})
    assert base != cell_key(corpus_version="c", dataset_version="d", component_id="y", params={"k": 1})
    assert base != cell_key(corpus_version="c", dataset_version="d", component_id="x", params={"k": 2})


def test_get_returns_none_on_a_miss(tmp_path):
    cache = MatrixCache(tmp_path)
    assert cache.get("nope") is None


def test_get_or_compute_calls_compute_exactly_once_per_key_across_two_cache_instances(tmp_path):
    """The property that actually matters: a SECOND `MatrixCache` pointed at the SAME directory (a
    fresh process, in spirit) never calls `compute` again for an already cached key."""
    calls = []

    def compute():
        calls.append(1)
        return {"value": 42}

    cache1 = MatrixCache(tmp_path)
    result1 = cache1.get_or_compute("k1", compute)
    assert result1 == {"value": 42}
    assert len(calls) == 1
    assert cache1.misses == 1
    assert cache1.hits == 0

    cache2 = MatrixCache(tmp_path)  # a fresh instance, same directory: simulates a rerun
    result2 = cache2.get_or_compute("k1", compute)
    assert result2 == {"value": 42}
    assert len(calls) == 1  # compute was NOT called again
    assert cache2.misses == 0
    assert cache2.hits == 1


def test_get_or_compute_recomputes_only_the_missing_cell_not_every_cell(tmp_path):
    calls = {"a": 0, "b": 0}

    def compute_a():
        calls["a"] += 1
        return {"v": "a"}

    def compute_b():
        calls["b"] += 1
        return {"v": "b"}

    cache1 = MatrixCache(tmp_path)
    cache1.get_or_compute("key-a", compute_a)
    cache1.get_or_compute("key-b", compute_b)
    assert calls == {"a": 1, "b": 1}

    # A "rerun" that adds one NEW key (key-c) alongside the two already cached: only the new one
    # should ever call its compute function.
    calls_c = []
    cache2 = MatrixCache(tmp_path)
    cache2.get_or_compute("key-a", compute_a)
    cache2.get_or_compute("key-b", compute_b)
    cache2.get_or_compute("key-c", lambda: (calls_c.append(1), {"v": "c"})[1])
    assert calls == {"a": 1, "b": 1}  # unchanged: neither recomputed
    assert calls_c == [1]  # the genuinely missing cell, and only it, recomputed
    assert cache2.hits == 2
    assert cache2.misses == 1


def test_set_then_get_round_trips_json_plain_values(tmp_path):
    cache = MatrixCache(tmp_path)
    cache.set("k", {"a": [1, 2, 3], "b": "text"})
    assert cache.get("k") == {"a": [1, 2, 3], "b": "text"}
