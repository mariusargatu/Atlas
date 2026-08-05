from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jinja2
import yaml

from atlas.config import AnswerSettings
from atlas.contracts import Answer, Chunk, ChunkId, Outcome, Question, Usage

ANSWER_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "data" / "prompts" / "answer.md.j2"


def record_violations(
    cited: tuple[ChunkId, ...],
    shown: tuple[ChunkId, ...],
    outcome: Outcome,
    citation_required: bool,
) -> tuple[str, ...]:
    """The two free checks on a generated answer: every citation names a passage that was
    shown, and an answer claiming to have answered cites something. Neither needs a
    reference answer, a judge model, or any money, so both record a violation rather than
    raising; the caller decides what to do with a non-empty result.

    Deliberately does not require refusals to cite nothing: a refusal naming the
    passages it read before concluding they lack the answer is showing its work, not
    contradicting itself.
    """
    violations: list[str] = []
    shown_set = set(shown)
    unshown = [c for c in cited if c not in shown_set]
    if unshown:
        violations.append(f"cited chunks not shown: {', '.join(unshown)}")
    if outcome == "answered" and not cited and citation_required:
        violations.append("answered with no citation")
    return tuple(violations)


@dataclass(frozen=True, slots=True)
class AnswerTemplate:
    version: str
    # Declared in the front matter and rendered into the body, so the domain is written
    # in exactly one place and a check can compare it against the collection actually
    # shown to the model.
    domain: str
    body: jinja2.Template


def load_answer_template() -> AnswerTemplate:
    """Reads the front matter version and compiles the rest as a template. The version
    should be checked against AnswerSettings.prompt_version elsewhere, rather than
    assumed equal, since an edited template without a version bump makes two
    incomparable runs look comparable."""
    text = ANSWER_TEMPLATE_PATH.read_text(encoding="utf-8")
    _, front_matter, body = text.split("---", 2)
    declared = yaml.safe_load(front_matter)
    return AnswerTemplate(
        version=declared["version"], domain=declared["domain"], body=jinja2.Template(body)
    )


class StructuredReply(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def cited(self) -> Sequence[ChunkId]: ...
    @property
    def outcome(self) -> Outcome: ...
    @property
    def reason(self) -> str: ...
    @property
    def usage(self) -> Usage: ...


class ModelClient(Protocol):
    def complete(self, prompt: str) -> StructuredReply: ...


class ModelAnswerWriter:
    """Asks a client for a structured result, never parsed prose. `shown` is set from
    what this instance actually handed over, never trusted from the reply, since it's
    the evidence every citation check reads."""

    def __init__(self, settings: AnswerSettings, client: ModelClient) -> None:
        self._settings = settings
        self._client = client
        self._template = load_answer_template()
        self.model = settings.model

    def __call__(self, q: Question, shown: Sequence[Chunk]) -> Answer:
        prompt = self._template.body.render(
            question=q.text, domain=self._template.domain,
            chunks=[(f.name, f.text) for f in shown],
        )
        reply = self._client.complete(prompt)
        shown_names = tuple(f.name for f in shown)
        cited = tuple(reply.cited)
        return Answer(
            question=q.name, text=reply.text, cited=cited, outcome=reply.outcome,
            shown=shown_names, model=self.model, usage=reply.usage,
            violations=record_violations(
                cited, shown_names, reply.outcome, self._settings.citation_required
            ),
        )


def get_answer_writer(settings: AnswerSettings) -> ModelAnswerWriter:
    """The answering stage's seam, so `atlas.pipeline` has one thing to default to.
    Imported inside the function so `record_violations` and the template loader stay
    testable without an SDK present."""
    from atlas.models.providers import get_answer_client

    return ModelAnswerWriter(settings, get_answer_client(settings))
