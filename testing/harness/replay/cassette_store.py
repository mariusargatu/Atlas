"""Cassette persistence: the store seam behind the gateway (ADR-007).

The gateway is a LangChain adapter; *where* a cassette lives is a separate concern, so it lives
here behind a small port. `FileCassetteStore` is the committed on disk store the PR lane replays.
`InMemoryCassetteStore` backs hermetic gateway tests with no temp directory and no I/O. The store
is deliberately policy free: a miss returns `None` and the *gateway* decides that replay turns a
`None` into a `CassetteMiss` hard fail. Persistence does not know about modes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from replay.cassette import Cassette, build_request


class CassetteMiss(RuntimeError):
    """No cassette for a request in replay mode, a hard fail the gateway raises, never a live call."""


@runtime_checkable
class CassetteStore(Protocol):
    """The persistence port. Adapters: `FileCassetteStore` (committed), `InMemoryCassetteStore` (tests)."""

    def load(self, key: str) -> Optional[Cassette]:
        """Return the cassette for `key`, or `None` on a miss (never raises for a miss)."""
        ...

    def save(self, cassette: Cassette) -> None:
        """Persist a cassette under its own content addressed key."""
        ...


class FileCassetteStore:
    """Cassettes as `<key>.json` under one directory, the committed, replayable store."""

    def __init__(self, directory: Path | str) -> None:
        self._dir = Path(directory)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def load(self, key: str) -> Optional[Cassette]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return Cassette.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"corrupt or unreadable cassette at {path}: {exc}") from exc

    def save(self, cassette: Cassette) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        # sort_keys for a stable on disk diff; the body's key order never affects the digest.
        self._path(cassette.key).write_text(
            json.dumps(cassette.to_dict(), indent=2, sort_keys=True)
        )


class InMemoryCassetteStore:
    """A process local store for hermetic gateway tests, record and replay with zero I/O."""

    def __init__(self) -> None:
        self._by_key: dict[str, Cassette] = {}

    def load(self, key: str) -> Optional[Cassette]:
        return self._by_key.get(key)

    def save(self, cassette: Cassette) -> None:
        # Immutable update, replace the mapping rather than mutate it in place.
        self._by_key = {**self._by_key, cassette.key: cassette}


def seed_cassette(cassette_dir, messages, response, model_id: str = "claude-test") -> None:
    """Persist one cassette under its content-addressed key — the seed path tests and demos share.

    The single definition of "pin this model response for this request": tests use it through the
    conftest `seed_cassette` fixture, the eval/drift demos call it directly. A change to the cassette
    schema or the request-key derivation lands here, not in each hand-rolled copy.
    """
    FileCassetteStore(cassette_dir).save(
        Cassette(model_id=model_id, request=build_request(model_id, messages), response=response)
    )


__all__ = [
    "CassetteMiss", "CassetteStore", "FileCassetteStore", "InMemoryCassetteStore", "seed_cassette",
]
