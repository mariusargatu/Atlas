from __future__ import annotations

import pytest

from atlas.contracts import Collection, CollectionSource, DocName, Document, Question, QuestionName
from atlas.corpus.registry import DEFAULT_SOURCE, get_source, load, register_source
from atlas.corpus.tau2 import Tau2Source


def test_the_default_resolves_to_the_real_tau2_source():
    source = get_source()
    assert isinstance(source, Tau2Source) and source.name == DEFAULT_SOURCE


def test_an_unknown_name_is_refused_by_name_rather_than_by_import_error():
    with pytest.raises(ValueError, match="no collection source named"):
        get_source("a-corpus-nobody-registered")


class _TinySource:
    """A second, trivial CollectionSource, to prove register_source really adds an entry
    rather than only ever resolving the one Tau2Source ships with."""

    @property
    def name(self) -> str:
        return "tiny-test-source"

    def __call__(self, seed: int) -> Collection:
        return Collection(documents=(Document(name=DocName("d"), kind="tiny", text="hello"),))

    def questions(self) -> tuple[Question, ...]:
        return (Question(name=QuestionName("q"), text="hi", kind="lookup", required=(DocName("d"),)),)


def test_a_second_source_registers_under_its_own_name_and_the_default_still_resolves():
    source: CollectionSource = _TinySource()
    register_source(source)
    try:
        resolved = get_source("tiny-test-source")
        assert resolved is source
        assert resolved(0).documents[0].text == "hello"
        assert get_source().name == DEFAULT_SOURCE
    finally:
        # The registry is process-wide; a test-only source left in it can leak into
        # later tests.
        from atlas.corpus.registry import _SOURCES
        del _SOURCES["tiny-test-source"]


def test_load_returns_the_same_shape_tau2_loads_returned():
    collection, questions = load()
    assert collection.documents and questions
    assert len(collection.documents) == 698 and len(questions) == 97
