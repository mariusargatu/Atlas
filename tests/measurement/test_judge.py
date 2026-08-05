from __future__ import annotations

import hashlib
import statistics
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from pathlib import Path

import jinja2
import pytest

from atlas.config import Settings
from atlas.contracts import Answer, ChunkId, QuestionName, Usage
from atlas.corpus.chunk import cut
from atlas.models.generate import ANSWER_TEMPLATE_PATH
from atlas.pipeline import PreparedCorpus
from evals.calibration import NOISE_FLOOR_PATH, recorded_noise_floor
from evals.judge import (
    JudgeModel,
    JudgeVerdict,
    judge,
    load_versioned_front_matter,
    render_judge_prompt,
)
from evals.stats import resample_interval

_PADDING_SENTENCE = "This sentence exists only to add length without adding meaning. "
BIASES = ("swapped_order", "padded_verbosity", "reformatted", "reattributed")


@pytest.fixture
def measured_noise_floor() -> float:
    """How often two readings of the same answer disagree, which every threshold below is
    a multiple of. Both gates matter: every assertion below is `rate <= 2 * floor`, so a
    floor of 0.0 fails on a single honest disagreement and a floor at or above 0.5 puts the
    threshold past 1.0, where nothing can fail it.
    """
    if not Path(NOISE_FLOOR_PATH).exists():
        # Fails rather than skips: tests/no_skip_guard.py fails any session that skips
        # outside a registered marker, so a skip here reports as a broken suite instead of
        # a missing prerequisite.
        pytest.fail(
            f"no recorded noise floor at {NOISE_FLOOR_PATH}. These relations are all "
            "multiples of the judge's own noise, so they mean nothing until it is "
            "measured: run evals.calibration.noise_floor against the real judge and "
            "record it with record_noise_floor."
        )
    # `flip_floor`, not `spread` and not `unanimity_rate`. Everything below bounds a
    # `flip_rate`, which reads each answer exactly twice, so the floor has to be a rate per
    # pair of readings. `spread` cancels when questions flip in opposite directions;
    # `1 - unanimity_rate` grows with the repeat count, so measuring the judge more
    # carefully would raise the bar it has to clear.
    floor = recorded_noise_floor().at_configured.flip_floor
    assert floor > 0.0, (
        f"the recorded floor is {floor}, so every threshold below is zero and any "
        "disagreement at all fails. That is the signature of a deterministic stand-in "
        "judge, not a measurement of a real one."
    )
    assert floor < 0.5, (
        f"the recorded floor is {floor:.3f}, so every threshold below is 2 * {floor:.3f} = "
        f"{2 * floor:.3f}. A flip rate cannot exceed 1.0, so none of these relations can "
        f"fail and passing them would mean nothing. Two readings of the same answer "
        f"disagreed {floor:.0%} of the time; fix that before measuring bias against it."
    )
    # The floor and the rates it bounds are measured on disjoint questions: the recorded
    # floor covers task_073 to task_102 and `small` is task_001 to task_040. A relation
    # that fails narrowly may be reporting that difference rather than a real bias, so
    # re-record the floor over `small` before believing one.
    return floor


def _biasable_answer(cited: tuple[ChunkId, ...], shown: tuple[ChunkId, ...],
                     text: str = "Within sixty days, per policy A and policy B.") -> Answer:
    return Answer(
        question=QuestionName("q_bias"), text=text, cited=cited, outcome="answered",
        shown=shown, model="stub", usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
    )


def test_swapped_order_reverses_a_multi_citation_answer_and_leaves_a_single_one_alone():
    # The four bias mutations are otherwise reached only through the `reporting` relations,
    # which nothing runs by default, so their model-free logic would go unexercised.
    multi = _biasable_answer(cited=(ChunkId("a"), ChunkId("b")), shown=(ChunkId("a"), ChunkId("b")))
    assert _apply_bias("swapped_order", multi).cited == (ChunkId("b"), ChunkId("a"))
    single = _biasable_answer(cited=(ChunkId("a"),), shown=(ChunkId("a"), ChunkId("b")))
    assert _apply_bias("swapped_order", single).cited == single.cited


def test_padded_verbosity_lengthens_the_text_and_changes_nothing_else():
    answer = _biasable_answer(cited=(ChunkId("a"),), shown=(ChunkId("a"),))
    padded = _apply_bias("padded_verbosity", answer)
    assert padded.text.startswith(answer.text) and len(padded.text) > len(answer.text)
    assert padded.cited == answer.cited and padded.outcome == answer.outcome


def test_reformatted_changes_layout_and_never_a_sentence():
    answer = _biasable_answer(cited=(ChunkId("a"),), shown=(ChunkId("a"),),
                              text="First fact stated plainly. Second fact stated plainly.")
    reformatted = _apply_bias("reformatted", answer)
    assert reformatted.text != answer.text
    assert "First fact stated plainly." in reformatted.text
    assert "Second fact stated plainly." in reformatted.text
    assert "\n\n" in reformatted.text and ". " not in reformatted.text


def test_reattributed_cites_a_different_shown_passage_or_leaves_a_saturated_answer_alone():
    answer = _biasable_answer(cited=(ChunkId("a"),), shown=(ChunkId("a"), ChunkId("b")))
    assert _apply_bias("reattributed", answer).cited == (ChunkId("b"),)
    # No uncited alternative exists, so the mutation is a no-op rather than inventing a
    # passage the answer was never shown.
    saturated = _biasable_answer(cited=(ChunkId("a"), ChunkId("b")), shown=(ChunkId("a"), ChunkId("b")))
    assert _apply_bias("reattributed", saturated) == saturated


def test_the_judge_returns_a_frozen_typed_verdict(
    one_question, one_answer, stub_judge, shown_texts
):
    verdict = judge(one_question, one_answer, stub_judge, Settings(), shown_texts)
    assert isinstance(verdict, JudgeVerdict) and verdict.verdict in ("pass", "fail")
    assert verdict.rubric_version and verdict.prompt_version
    assert verdict.model == Settings().judge.model and isinstance(verdict.usage, Usage)
    # Assigning to a name a slotted class does not declare also raises, so the field has to
    # be shown to exist before frozenness means anything.
    assert "reason" in {f.name for f in fields(JudgeVerdict)}
    with pytest.raises(FrozenInstanceError):
        verdict.reason = "edited"


@pytest.mark.parametrize("field", ["rubric_version", "prompt_version"])
def test_both_versions_feed_the_run_identity(field):
    # Editing a guide or a prompt without moving the run identity makes two runs that
    # cannot be compared look like they can be.
    base = Settings()
    moved = replace(base, judge=replace(base.judge, **{field: "9.9.9"}))
    assert getattr(moved.judge, field) != getattr(base.judge, field)   # the edit landed
    assert moved.run_id != base.run_id


@pytest.mark.parametrize(
    ("path_setting", "version_setting"),
    [("rubric_path", "rubric_version"), ("prompt_path", "prompt_version")],
)
def test_each_versioned_file_agrees_with_the_settings(path_setting, version_setting):
    judge_settings = Settings().judge
    on_disk = load_versioned_front_matter(getattr(judge_settings, path_setting))
    assert on_disk.version == getattr(judge_settings, version_setting)


# What each versioned file's body hashed to when its version was last set, so a body cannot
# be edited without moving its version in the same change. The check above compares front
# matter against settings, which are edited together and stay green when both say 1.0.0 over
# a rewritten body. Two materially different prompts have shared one version string that way.
#
# To change a prompt: edit it, run the suite, and paste the printed digest here under a new
# version. The paste is the point, since it cannot happen by accident.
VERSIONED_BODIES = {
    "data/prompts/judge.md.j2": (
        "3.0.0", "993f04235c25845949fb951d0c73e2c6c73cd027323109211efb77df1727858b",
    ),
    "data/rubric.md": (
        "1.0.0", "d44a6f80501a4632367eb482e3379a4f30e1cf8285d20fc6e07363a0a79b0f82",
    ),
    # 4.0.0 fences the question as well as the passages: this corpus's question field is
    # tau2's user-simulator script, and the writer was following those instructions in 39
    # of 97 recorded answers.
    "data/prompts/answer.md.j2": (
        "4.0.0", "8358cbb162cfabd23945f1bb2dca2e573902703930c94f1ee738370be5e8ba2a",
    ),
}


@pytest.mark.parametrize(("path", "expected"), sorted(VERSIONED_BODIES.items()))
def test_a_versioned_body_cannot_change_without_its_version_moving(path, expected):
    expected_version, expected_digest = expected
    on_disk = load_versioned_front_matter(path)
    digest = hashlib.sha256(on_disk.body.encode()).hexdigest()
    assert (on_disk.version, digest) == (expected_version, expected_digest), (
        f"{path} now hashes to {digest} at version {on_disk.version!r}. If you meant to "
        f"change it, move the version in its front matter and in src/atlas/config.py, and "
        f"put ({on_disk.version!r}, {digest!r}) here in the same change. Two bodies under "
        f"one version make two runs that cannot be compared look like they can be."
    )


def test_the_pinned_bodies_cover_every_versioned_file_the_settings_name():
    """The pin above is only a guard while it names every file that carries a version, so a
    third prompt added tomorrow has to join it."""
    settings = Settings()
    # The answering template is reached by a module constant rather than a settings field,
    # so it is named relative to the repository root to match the judge's two.
    named = {
        settings.judge.prompt_path,
        settings.judge.rubric_path,
        str(ANSWER_TEMPLATE_PATH.relative_to(ANSWER_TEMPLATE_PATH.parents[2])),
    }
    assert named <= set(VERSIONED_BODIES), f"unpinned versioned file(s): {named - set(VERSIONED_BODIES)}"


@dataclass(frozen=True, slots=True)
class Rate:
    rate: float
    low: float
    high: float


def _apply_bias(bias: str, answer: Answer) -> Answer:
    if bias == "swapped_order":
        # Each mutation preserves correctness, so the verdict should not move.
        return replace(answer, cited=tuple(reversed(answer.cited))) if len(answer.cited) > 1 else answer
    if bias == "padded_verbosity":
        return replace(answer, text=answer.text + " " + _PADDING_SENTENCE * 3)
    if bias == "reformatted":
        return replace(answer, text=answer.text.replace(". ", ".\n\n"))
    if bias == "reattributed":
        alternative = next((name for name in answer.shown if name not in answer.cited), None)
        return replace(answer, cited=(alternative,)) if alternative is not None else answer
    raise ValueError(f"no bias named {bias!r}")


def flip_rate(questions, answers, judge_fn, bias: str | None, seed: int = 0) -> Rate:
    """Rate at which a verdict flips when the bias is applied, with a seeded interval.
    Pass bias=None for the unbiased control."""
    diffs = []
    for question in questions:
        answer = answers[question.name]
        first = judge_fn(question, answer).verdict
        other_answer = _apply_bias(bias, answer) if bias is not None else answer
        second = judge_fn(question, other_answer).verdict
        diffs.append(1.0 if first != second else 0.0)
    rate = statistics.fmean(diffs)
    low, high = resample_interval(diffs, seed)
    return Rate(rate=rate, low=low, high=high)


_RELATIONS_LOG: list[tuple[str, Rate]] = []


def record_relation(name: str, rate: Rate) -> None:
    _RELATIONS_LOG.append((name, rate))
    print(f"judge relation {name}: rate={rate.rate:.3f} interval=({rate.low:.3f}, {rate.high:.3f})")


@pytest.fixture
def answers(small, documents):
    settings = Settings()
    prepared = PreparedCorpus.build(settings, cut(documents, settings.chunk))
    return {question.name: prepared.ask(question).answer for question in small}


@pytest.fixture
def corpus_texts(documents):
    """Every chunk's text, keyed by name, cut exactly as the answering run cut it: an
    Answer carries only names, and the judge has to see the passages the writer saw."""
    return {f.name: f.text for f in cut(documents, Settings().chunk)}


@pytest.fixture
def real_judge(corpus_texts):
    settings = Settings()
    client = JudgeModel(settings.judge)

    def _call(question, answer):
        return judge(question, answer, client, settings, corpus_texts)

    return _call


@pytest.mark.reporting
def test_the_judge_disagreeing_with_itself_stays_inside_two_noise_floors(
    measured_noise_floor, small, answers, real_judge
):
    # The control for the four cases below. Without it, a bias case that fails on ordinary
    # judge noise reads as a discovery about bias.
    floor = measured_noise_floor
    control = flip_rate(small, answers, real_judge, bias=None)
    record_relation("unbiased_control", control)
    assert control.rate <= 2 * floor, (
        "the judge disagrees with itself by more than twice its own measured noise floor, "
        "so nothing below can be attributed to the change under test.")


@pytest.mark.reporting
@pytest.mark.parametrize("bias", BIASES)
def test_the_judge_verdict_survives_a_change_that_should_not_matter(
    measured_noise_floor, bias, small, answers, real_judge
):
    # Recorded before the assertion runs, so a failure is written down with its size
    # rather than lost as a red mark.
    observed = flip_rate(small, answers, real_judge, bias=bias)
    record_relation(bias, observed)
    assert observed.rate <= 2 * measured_noise_floor


def test_everything_the_rubric_grades_reaches_the_rendered_judge_prompt() -> None:
    """The rubric names citations and outcome; the prompt has to actually carry them.

    Jinja2's default undefined is silent, so a variable that stops being passed turns a
    rubric rule unenforceable without anything going red. Rendering only `answer.text` once
    left rule 2 dead and the two `Answer.cited` bias relations unable to move a verdict.
    """
    answer = Answer(
        question=QuestionName("q"), text="the answer body",
        cited=(ChunkId("chunk_alpha"),), shown=(ChunkId("chunk_alpha"), ChunkId("chunk_beta")),
        outcome="answered", violations=(), model="m",
        usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
    )
    rendered = render_judge_prompt(
        "RUBRIC BODY", "the question text", answer,
        [(ChunkId("chunk_alpha"), "alpha text"), (ChunkId("chunk_beta"), "beta text")],
    )
    for required in ("RUBRIC BODY", "the question text", "the answer body",
                     "chunk_beta", "alpha text", "answered"):
        assert required in rendered, f"{required!r} never reached the judge's prompt"

    # Twice, not once: a cited id is also a shown id, so a plain membership check is
    # satisfied by the passages loop alone even when the citations loop is undefined.
    assert rendered.count("chunk_alpha") >= 2, (
        "the cited chunk appears only once, so it reached the passages section and not "
        "the citations section -- the judge cannot apply rubric rule 2 against this prompt"
    )
    # The silence itself: an undefined variable renders as nothing at all, which is why the
    # assertions above name every field rather than trust that a missing one would raise.
    assert jinja2.Template("[{{ missing }}]").render() == "[]"
