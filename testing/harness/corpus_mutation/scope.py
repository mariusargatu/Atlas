"""The scoped, ephemeral corpus_version this lane's mutated render lives under.

`testing/tests/test_corpus_build.py`'s frozen artifact rule governs the ONE committed corpus_version
(`corpus-0.1.1`): it must stay byte identical, fresh against the registry, forever. This lane's
mutated render is the opposite of that on purpose, a throwaway probe that must never look like a
committed corpus_version, never collide with an actually committed one, and never leave a single
byte behind once the probe finishes, success or failure. `EphemeralCorpusVersion` is the one place
in this package that touches the filesystem for real; `selection.py` stays pure.
"""
from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from corpus_mutation.selection import FactMutation

# The same tree test_corpus_build.py's own COMMITTED constant points at: every subdirectory name
# under here is an actually committed corpus_version (today, just "corpus-0.1.1").
COMMITTED_CORPUS_ROOT = Path("corpus/rendered")

# corpus-0.1.1's own naming shape: "corpus-" + a three part semver. Any real committed corpus_version
# this repo ever ships follows this pattern; this lane's ephemeral names are built to structurally
# never match it (see ephemeral_corpus_version below), so a committed-looking name is a defect this
# test can catch even before checking the actual committed_corpus_versions() listing.
_COMMITTED_VERSION_RE = re.compile(r"^corpus-\d+\.\d+\.\d+$")

__all__ = [
    "COMMITTED_CORPUS_ROOT",
    "EphemeralCorpusVersion",
    "EphemeralScope",
    "committed_corpus_versions",
    "ephemeral_corpus_version",
    "is_committed_style_version",
]


def is_committed_style_version(corpus_version: str) -> bool:
    """True iff `corpus_version` has the shape a COMMITTED corpus_version uses (`corpus-X.Y.Z`).
    Structural, not a filesystem lookup: catches a naming mistake even for a hypothetical future
    committed version this repo has not cut yet."""
    return bool(_COMMITTED_VERSION_RE.match(corpus_version))


def committed_corpus_versions() -> frozenset[str]:
    """Every corpus_version actually committed under `corpus/rendered` today (just the directory
    names; no manifest parsing needed for a simple collision check)."""
    if not COMMITTED_CORPUS_ROOT.is_dir():
        return frozenset()
    return frozenset(p.name for p in COMMITTED_CORPUS_ROOT.iterdir() if p.is_dir())


def ephemeral_corpus_version(mutation: FactMutation) -> str:
    """A `corpus-mutation-<hash>` name, deterministic for a given mutation (no wall clock, no
    randomness: the same discipline every other runtime path in this repo follows), and never in
    the `corpus-X.Y.Z` shape a committed corpus_version uses (see `is_committed_style_version`).
    Deterministic on purpose: a reproducible name makes a failed live/burst run's ephemeral
    corpus_version easy to name and reason about from its own report, even though the directory it
    was rendered under is already gone by the time anyone reads that report."""
    digest_input = f"{mutation.fact_ref}|{mutation.old_value}|{mutation.new_value}"
    digest = hashlib.sha256(digest_input.encode()).hexdigest()[:16]
    return f"corpus-mutation-{digest}"


@dataclass(frozen=True)
class EphemeralScope:
    """What `EphemeralCorpusVersion.__enter__` hands back: a corpus_version name that can never
    collide with a committed one, and two already-created, already-empty directories to render and
    index under (`corpus_root`, the `out_root` a `corpus_tools`/`rag_tools` caller would pass
    alongside `corpus_version`; `index_root`, the analogous root for a built index). Both live under
    the SAME temporary directory, so `EphemeralCorpusVersion.__exit__` removes them together with
    one `TemporaryDirectory.cleanup()` call, regardless of what got written inside either."""

    corpus_version: str
    corpus_root: Path
    index_root: Path


class EphemeralCorpusVersion:
    """Context manager: allocates a fresh temporary directory pair for one mutation probe and
    guarantees its removal on exit, whether the body completes, raises, or does anything in
    between. Nothing under `corpus_root`/`index_root` is ever committed; nothing here writes
    anywhere near the real, committed `corpus/rendered` or `indexes` trees."""

    def __init__(self, mutation: FactMutation) -> None:
        self._mutation = mutation
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> EphemeralScope:
        self._tmp = tempfile.TemporaryDirectory(prefix="atlas-corpus-mutation-")
        root = Path(self._tmp.name)
        corpus_root = root / "rendered"
        index_root = root / "indexes"
        corpus_root.mkdir(parents=True, exist_ok=True)
        index_root.mkdir(parents=True, exist_ok=True)
        return EphemeralScope(
            corpus_version=ephemeral_corpus_version(self._mutation),
            corpus_root=corpus_root,
            index_root=index_root,
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
        return False
