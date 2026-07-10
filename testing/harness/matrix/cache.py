"""Content hash cache (D17): a rerun of the staged runner recomputes only missing cells.

Keyed `hash(corpus_version, dataset_version, component_id, params)` via `determinism.canonical.digest`
-- the SAME canonical hashing scheme the cassette key and the run digest already use, never a second
hashing recipe. A thin, file backed cache, in the spirit of the record/replay seam (SP12's own future
territory is the GENERALIZED record/replay layer across all five D19 seams; this is SP9's own narrow,
cell level resumability, per the planning digest's own explicit disposition: "a thin cache in the
spirit of the seam, not a claim on SP12's territory").

Deliberately dumb storage: one JSON file per key, mirroring `replay.cassette_store.FileCassetteStore`'s
own shape but for an arbitrary computed cell (not a `Cassette`). Policy free, same as that store: a
miss returns `None`, `get_or_compute` decides what a miss means.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from determinism.canonical import digest


def cell_key(*, corpus_version: str, dataset_version: str, component_id: str, params: Mapping[str, object]) -> str:
    """The one content hash key derivation every stage uses, `hash(corpus_version, dataset_version,
    component_id, params)` per D17's own naming. `digest` (not a bespoke hash here) already sorts
    keys and normalizes scalars, so two logically identical `params` dicts built in a different
    field order still hash identically."""
    return digest(
        {
            "corpus_version": corpus_version,
            "dataset_version": dataset_version,
            "component_id": component_id,
            "params": dict(params),
        }
    )


@dataclass
class MatrixCache:
    """A directory of `<key>.json` cells. `hits`/`misses` are process lifetime counters (mirroring
    `atlas.metrics`'s own request counter shape): a hermetic test reads them directly to assert "the
    rerun recomputed nothing," never inferring it indirectly from timing."""

    directory: Path
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Any | None:
        """`None` on a miss (never raises): a cache is optional infrastructure, not a hard
        dependency the caller must have populated first. Note the one open trap this sentinel
        carries: a cached JSON `null` would read back as `None` too, indistinguishable from a
        miss, and `get_or_compute` would recompute it forever; harmless today since every stage
        that uses this cache always caches a dict, but a future cell type must never legitimately
        cache `null`."""
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set(self, key: str, value: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        # sort_keys for a stable on disk diff and a byte identical write given a byte identical
        # (already plain JSON: lists, not tuples) value, the same discipline `FileCassetteStore`
        # already applies to a Cassette body.
        self._path(key).write_text(json.dumps(value, sort_keys=True))

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        """The one entry point every stage calls: a cache hit never invokes `compute` at all (the
        property "a rerun recomputes only missing cells" depends on this, not merely on the cache
        existing)."""
        cached = self.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        value = compute()
        self.set(key, value)
        return value


__all__ = ["MatrixCache", "cell_key"]
