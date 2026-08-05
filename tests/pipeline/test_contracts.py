from __future__ import annotations

import dataclasses
import importlib
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from atlas import contracts, pipeline
from atlas.config import AnswerSettings, Settings
from atlas.contracts import (
    Answer,
    Chunk,
    ChunkId,
    Collection,
    CollectionSource,
    DocName,
    FusedResult,
    Outcome,
    QuestionName,
    Range,
    RerankResult,
    SearchResult,
    Usage,
)
from atlas.corpus.tau2 import Tau2Source
from atlas.models.generate import ModelAnswerWriter, load_answer_template, record_violations
from evals.report import cost_from_ledger
from evals.results import RESULTS_SCHEMA_VERSION, QuestionRow

WORDS = ["alpha", "beta", "gamma"]
# Deliberately not imported from atlas.pipeline: comparing a tuple against itself asserts
# nothing, and renaming a stage has to fail here. `judge` is split off because it is the
# one optional stage.
UNJUDGED_STAGES = ("embed_query", "vector", "keyword", "fuse", "rerank", "answer")
STAGES = (*UNJUDGED_STAGES, "judge")
SHOWN = (ChunkId("card_dispute_policy#0001"), ChunkId("card_dispute_policy#0002"))
UNSEEN = ChunkId("filler.03#0000")


def stub_answer(cited: tuple[ChunkId, ...], outcome: Outcome,
                citation_required: bool = True) -> Answer:
    return Answer(
        question=QuestionName("q0042"), text="Within sixty days.", model="stub",
        cited=cited, outcome=outcome, shown=SHOWN,
        usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
        violations=record_violations(cited, SHOWN, outcome, citation_required),
    )


def test_the_contract_types_hold_their_two_guarantees() -> None:
    # A Range that accepted end before start would let a chunk claim a slice of text it
    # does not hold.
    with pytest.raises(ValueError):
        Range(10, 3)

    chunk = Chunk(ChunkId("d#0001"), DocName("d"), 1, Range(0, 4), "abcd")
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "mutated"
    assert {chunk, chunk} == {chunk}  # correct answers live in frozen sets

    answer = Answer(
        question=QuestionName("q0042"), text="", cited=(), outcome="refused",
        shown=(ChunkId("d#0001"),), model="a setting, never a constant",
        usage=Usage(input_tokens=10, output_tokens=0, cost_usd=0.0),
    )
    assert answer.violations == ()

    empty = SearchResult(query="q", arm="vector", hits=())
    record = contracts.Record(
        run_id="3f9c1a2b7d04", question=QuestionName("q0042"), vector=empty,
        keyword=dataclasses.replace(empty, arm="keyword"),
        fused=FusedResult(query="q", hits=(), arm_ranks={}),
        reranked=RerankResult(query="q", hits=(), input_order=()),
        answer=None, timings=(),
    )
    # A Record with no answer is what a run that only retrieves produces.
    assert record.answer is None and record.judge is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.run_id = "mutated"


@pytest.mark.parametrize("outcome", ["answered", "refused", "unknown", "not_applicable"])
def test_all_four_outcomes_survive_the_answer_writer_with_the_field_intact(outcome: Outcome) -> None:
    answer = stub_answer((SHOWN[0],) if outcome == "answered" else (), outcome)
    assert answer.outcome == outcome and answer.violations == ()


@pytest.mark.parametrize("cited,outcome,fault", [((UNSEEN,), "answered", "not shown"),
                                                 ((), "answered", "no citation")])
def test_each_fault_is_recorded_on_the_answer_and_never_raised(cited, outcome, fault) -> None:
    # Raising instead would kill the adversary that answers fluently while citing a chunk it
    # was never shown, on its first question, so the check it exists to trip never runs.
    answer = stub_answer(cited, outcome)
    assert isinstance(answer, Answer) and any(fault in v for v in answer.violations)


@pytest.mark.parametrize("outcome", ["refused", "unknown", "not_applicable"])
def test_a_refusal_that_names_the_passages_it_read_is_not_a_fault(outcome: Outcome) -> None:
    # It was a fault once, and fired on 39 of 97 answers, every one a refusal citing the
    # passages it had just read and found wanting. The prompt never asked a refusal to cite
    # nothing, so the rule was stricter than the instruction it checked.
    answer = stub_answer((SHOWN[0],), outcome)
    assert answer.violations == ()


@pytest.mark.parametrize("outcome", ["answered", "refused", "unknown", "not_applicable"])
def test_citing_something_never_shown_is_a_fault_under_every_outcome(outcome: Outcome) -> None:
    answer = stub_answer((UNSEEN,), outcome)
    assert any("not shown" in v for v in answer.violations)


@pytest.mark.parametrize("required,expected_faults", [(True, 1), (False, 0)])
def test_answering_with_nothing_cited_is_a_fault_only_when_citations_are_required(
        required: bool, expected_faults: int) -> None:
    # Without this pair, an implementation that ignores the setting and always records the
    # fault passes every other test here.
    assert len(stub_answer((), "answered", citation_required=required).violations) == expected_faults


def test_the_writer_sets_shown_to_exactly_the_chunks_it_was_given(two_chunks, lookup_question,
                                                                    fake_model) -> None:
    writer = ModelAnswerWriter(AnswerSettings(), client=fake_model(
        text="Within sixty days.", cited=[SHOWN[0]], outcome="answered", reason=""))
    answer = writer(lookup_question, two_chunks)
    # A writer that fills this field from the model's reply, or from the whole chunk set
    # rather than the prompt, turns every citation check into a set compared against itself.
    assert answer.shown == tuple(f.name for f in two_chunks)
    assert answer.violations == ()


def test_the_writer_records_the_fault_itself_rather_than_leaving_it_to_the_caller(
        two_chunks, lookup_question, fake_model) -> None:
    # The test above proves record_violations on its own. Without this one, a writer that
    # never calls it ships green and every violation count at the end of a run is zero.
    writer = ModelAnswerWriter(AnswerSettings(), client=fake_model(
        text="Within thirty days.", cited=[UNSEEN], outcome="answered", reason=""))
    answer = writer(lookup_question, two_chunks)
    assert isinstance(answer, Answer) and any("not shown" in v for v in answer.violations)


def test_the_writer_takes_its_model_name_from_settings(two_chunks, lookup_question,
                                                       fake_model) -> None:
    settings = replace(AnswerSettings(), model="a name that exists only inside this test")
    writer = ModelAnswerWriter(settings, client=fake_model(
        text="Within sixty days.", cited=[SHOWN[0]], outcome="answered", reason=""))
    assert writer.model == settings.model
    assert writer(lookup_question, two_chunks).model == settings.model
    source = Path("src/atlas/models/generate.py").read_text(encoding="utf-8")
    assert AnswerSettings().model not in source   # a withdrawn model is a settings edit, not a patch


def test_the_prompt_template_version_agrees_with_the_settings_and_reaches_the_run_identity() -> None:
    assert load_answer_template().version == AnswerSettings().prompt_version
    base = Settings()
    moved = replace(base, answer=replace(base.answer, prompt_version="9.9.9"))
    assert moved.run_id != base.run_id


def test_the_answer_prompt_names_the_domain_the_collection_source_names() -> None:
    # The prompt once told the model it was answering a *broadband* question, over 698
    # banking documents, in every generation call this repository made. A version field
    # cannot catch that: it says two runs differ, never that one describes the wrong
    # industry. The second assertion stops this being fooled by a body that hardcodes a
    # different word than the front matter declares.
    template = load_answer_template()
    assert template.domain in set(Tau2Source().name.split("-")), (
        f"the answer prompt tells the model it is answering a {template.domain!r} question "
        f"and the collection it will be shown is {Tau2Source().name!r}"
    )
    rendered = template.body.render(question="q", chunks=[], domain=template.domain)
    assert template.domain in rendered, (
        "the front matter declares a domain the prompt body never renders, so the check "
        "above is reading a value the model never sees"
    )


def test_every_passage_the_writer_was_given_reaches_the_rendered_prompt() -> None:
    # Jinja2's default undefined is silent, so a template iterating one name while the
    # writer renders under another yields an empty Passages section rather than an error:
    # the model refuses everything and the refusals look like a retrieval failure.
    # `answer.shown` cannot catch it, being built from the argument the writer was handed
    # rather than the text it sent. This reads the rendered string the model actually sees.
    template = load_answer_template()
    passages = [("doc_x#0000", "Disputes may be filed within sixty days."),
                ("doc_y#0003", "A provisional credit is issued within ten days.")]
    rendered = template.body.render(question="how do I dispute a charge?",
                                    domain=template.domain, chunks=passages)
    for name, text in passages:
        assert name in rendered, f"{name} was shown to the writer and never reached the prompt"
        assert text in rendered, f"the text of {name} never reached the prompt"


@pytest.fixture
def tracer_absent(monkeypatch):
    # Without the restore the fallback versions stay in sys.modules for the rest of the
    # session, and the next test that expects the tracer wired fails for a reason that has
    # nothing to do with itself.
    from atlas import trace
    monkeypatch.setitem(sys.modules, "langfuse", None)
    importlib.reload(trace)
    importlib.reload(pipeline)
    yield
    monkeypatch.undo()
    importlib.reload(trace)
    importlib.reload(pipeline)


def test_a_grader_reaches_the_record_and_is_timed_as_its_own_stage(small_run):
    # Record.judge once existed, was announced by a schema version bump and read by
    # cost_summary, and was None in every run ever executed because nothing could fill it.
    class _Verdict:
        usage = Usage(input_tokens=11, output_tokens=3, cost_usd=0.25)

    seen: dict[str, object] = {}

    def grader(question, answer, texts):
        # A grader handed an empty mapping fails every answer under a guide whose every
        # rule is about the passages.
        seen["texts"] = texts
        seen["shown"] = answer.shown
        return _Verdict()

    prepared = pipeline.PreparedCorpus.build(small_run.settings, small_run.chunks)
    record = prepared.ask(small_run.question, grader=grader)

    assert record.judge is not None, "the grader ran but never reached the record"
    assert record.judge.usage.cost_usd == 0.25
    assert set(seen["texts"]) == set(seen["shown"]) == set(record.answer.shown)
    assert all(seen["texts"].values()), "a passage resolved to empty text"
    assert tuple(t.stage for t in record.timings) == STAGES
    # Asserted through the ledger rather than through Record, because the ledger is what a
    # run row is actually built from once cost moved to disk.
    row = QuestionRow(
        schema=RESULTS_SCHEMA_VERSION, run_id="r", question=record.question,
        generated=True, judged=True, rankings={}, correct=(), primary=(), timings_ms={},
        judge_cost_usd=record.judge.usage.cost_usd,
    )
    assert cost_from_ledger([row]).by_stage["judge"] == 0.25


def test_the_pipeline_produces_a_complete_record_when_the_tracer_is_absent(tracer_absent, small_run):
    # Timings come from a plain performance counter, never from the tracer, so a reader with
    # no container runtime still gets a full record.
    from atlas import trace
    started = time.perf_counter()
    record = pipeline.run_question(small_run.settings, small_run.question, small_run.chunks)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert tuple(t.stage for t in record.timings) == UNJUDGED_STAGES
    # Not a strictly positive time per stage: blending ten hits can legitimately read zero on
    # a coarse clock. What must hold is that every stage reported a real finite number and
    # that the numbers fit inside the call that made them.
    assert all(math.isfinite(t.wall_ms) and t.wall_ms >= 0.0 for t in record.timings)
    assert 0.0 < sum(t.wall_ms for t in record.timings) <= elapsed_ms
    assert record.fused.hits and record.reranked.input_order != () and record.answer is not None
    assert record.run_id == small_run.settings.run_id
    assert trace.observe(len)([1, 2]) == 2                  # the decorator became identity
    assert trace.flush() is None                            # and the flush became a no operation


def test_the_corpus_source_satisfies_the_collection_source_protocol() -> None:
    # isinstance on a runtime protocol only checks that the attributes exist, so what the
    # source returns is checked too.
    source: CollectionSource = Tau2Source()
    assert isinstance(source, CollectionSource)
    produced = source(0)
    assert isinstance(produced, Collection)
    assert produced.documents, "the source produced no documents, so this test proves nothing"
    assert source.name == "tau2-banking-knowledge"


