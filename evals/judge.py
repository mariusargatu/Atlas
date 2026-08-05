"""The structured judge: a versioned scoring guide and a versioned prompt,
producing a frozen verdict a consumer reads by field rather than by parsing
prose. Both version strings feed the run identity, so editing either file
without moving the setting that names it makes two runs that cannot be
compared look like they can be.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import jinja2
import yaml

from atlas.config import JudgeSettings, Settings
from atlas.contracts import Answer, ChunkId, Question, QuestionName, Usage
from atlas.models.providers import JsonReply, get_judge_client

Verdict = Literal["pass", "fail"]


@dataclass(frozen=True, slots=True)
class VersionedFile:
    version: str
    body: str


def load_versioned_front_matter(path: str) -> VersionedFile:
    text = Path(path).read_text(encoding="utf-8")
    _, front_matter, body = text.split("---", 2)
    version = yaml.safe_load(front_matter)["version"]
    return VersionedFile(version=version, body=body)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    question: QuestionName
    verdict: Verdict
    reason: str
    rubric_version: str
    prompt_version: str
    model: str
    usage: Usage


class JudgeReply(Protocol):
    # Read-only: every reply that satisfies this is a frozen dataclass, which cannot
    # satisfy a protocol declaring a settable attribute.
    @property
    def verdict(self) -> Verdict: ...
    @property
    def reason(self) -> str: ...
    @property
    def usage(self) -> Usage: ...


class JudgeClient(Protocol):
    def complete(self, prompt: str) -> JudgeReply: ...


def shown_passages(
    answer: Answer, texts: Mapping[ChunkId, str]
) -> tuple[tuple[ChunkId, str], ...]:
    """Every passage the writer was shown, named and in the order it was shown.

    Raises on a name it cannot resolve rather than dropping it: a judge shown fewer
    passages than the writer saw will call a genuinely supported claim unsupported, and
    that verdict is indistinguishable from an honest one.
    """
    missing = [name for name in answer.shown if name not in texts]
    if missing:
        raise KeyError(
            f"no text for shown passage(s) {', '.join(missing)}: a judge must see exactly "
            "what the answer writer saw, so a passage that cannot be resolved is not skippable"
        )
    return tuple((name, texts[name]) for name in answer.shown)


def render_judge_prompt(
    rubric: str, question: str, answer: Answer,
    passages: Sequence[tuple[ChunkId, str]], body: str | None = None,
) -> str:
    """The judge's prompt, rendered. Separate from `judge` so it can be checked without a
    model: jinja2's default undefined is silent, so a variable that stops being passed
    produces a prompt missing a section rather than an error.

    `cited` and `outcome` are passed because the rubric grades them directly (rule 2:
    every citation names a passage that was actually shown).
    """
    if body is None:
        body = load_versioned_front_matter(JudgeSettings().prompt_path).body
    rendered: str = jinja2.Template(body).render(
        rubric=rubric, question=question, answer=answer.text,
        cited=list(answer.cited), outcome=answer.outcome, passages=passages,
    )
    return rendered


def judge(
    question: Question,
    answer: Answer,
    client: JudgeClient,
    settings: Settings,
    texts: Mapping[ChunkId, str],
) -> JudgeVerdict:
    """Grades one answer against the versioned scoring guide.

    `texts` is required, not defaulted: `Answer` carries chunk names, not text, and every
    rule in the guide is about the shown passages, so a judge given none would fail every
    answer by rule 3.
    """
    rubric = load_versioned_front_matter(settings.judge.rubric_path)
    template = load_versioned_front_matter(settings.judge.prompt_path)
    prompt = render_judge_prompt(
        rubric.body, question.text, answer, shown_passages(answer, texts), template.body,
    )
    reply = client.complete(prompt)
    return JudgeVerdict(
        question=question.name, verdict=reply.verdict, reason=reply.reason,
        rubric_version=rubric.version, prompt_version=template.version,
        model=settings.judge.model, usage=reply.usage,
    )


@dataclass(frozen=True, slots=True)
class _JudgeReply:
    verdict: Verdict
    reason: str
    usage: Usage


class JudgeModel:
    """The real model client judge() talks to when actually scoring.

    Runs on the same seam the answering stage does, constrained to JUDGE_SCHEMA, so a
    verdict is read by field rather than parsed from prose and priced the same way an
    answer is. Retries and timeouts come from the SDK rather than a second backoff loop
    written by hand here.

    `JudgeSettings.repeats` runs this several times over the same answer to establish a
    noise floor, making a judge the most expensive thing in the repository to call.
    """

    def __init__(self, settings: JudgeSettings, randomness: float = 0.0) -> None:
        self._model = settings.model
        self._client = get_judge_client(settings, randomness)

    @property
    def randomness_applied(self) -> bool | None:
        """Whether the randomness this client was built with actually reached the model.

        None until the first request, and on a provider that has no opinion. False when the
        model rejected the sampling temperature and the client retried without it, which
        matters to exactly one caller: a noise floor is two columns differing only in this
        value, and if it was dropped the two columns are the same request twice.
        """
        return getattr(self._client, "randomness_applied", None)

    def complete(self, prompt: str) -> _JudgeReply:
        return self.to_reply(self._client.complete_json(prompt))

    def to_reply(self, reply: JsonReply) -> _JudgeReply:
        """The one function that turns a provider reply into a structured verdict, so
        there is no second implementation to drift against.

        Raises on a declined request rather than returning a verdict. A judge whose
        request was refused has not decided the answer was bad, and scoring that as a
        fail would record the provider's safety classifier as a quality measurement.
        """
        if reply.payload is None:
            raise RuntimeError(
                f"the judge provider declined to grade with model {self._model!r}; "
                "no verdict was reached, and a decline is not a fail"
            )
        verdict: Verdict = reply.payload["verdict"]
        return _JudgeReply(
            verdict=verdict, reason=reply.payload.get("reason", ""), usage=reply.usage
        )
