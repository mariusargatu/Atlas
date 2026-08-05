"""Resolves a CollectionSource by name, so swapping corpora is one call rather than a
multi-file patch: a source registers once under its own `.name`, and every call site
reads it back instead of constructing one directly."""

from __future__ import annotations

from typing import Protocol

from atlas.contracts import Collection, CollectionSource, Question
from atlas.corpus.tau2 import Tau2Source


class QuestionedSource(CollectionSource, Protocol):
    """A `CollectionSource` that can also supply its own question set. Stricter than
    `CollectionSource` alone because `load` below needs both."""

    def questions(self) -> tuple[Question, ...]: ...


DEFAULT_SOURCE = "tau2-banking-knowledge"

_SOURCES: dict[str, QuestionedSource] = {DEFAULT_SOURCE: Tau2Source()}


def register_source(source: QuestionedSource) -> None:
    """Adds a source under its own `.name`, typically called at import time by the
    module defining it."""
    _SOURCES[source.name] = source


def get_source(name: str = DEFAULT_SOURCE) -> QuestionedSource:
    if name not in _SOURCES:
        raise ValueError(f"no collection source named {name!r}; known: {sorted(_SOURCES)}")
    return _SOURCES[name]


def load(name: str = DEFAULT_SOURCE, seed: int = 0) -> tuple[Collection, tuple[Question, ...]]:
    """Same shape as `atlas.corpus.tau2.load`, resolved by name instead of importing
    that module's function directly."""
    source = get_source(name)
    return source(seed), source.questions()
