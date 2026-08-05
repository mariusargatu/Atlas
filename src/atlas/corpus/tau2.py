"""The tau2-bench banking knowledge domain as Atlas's collection source.

Sierra Research's tau2-bench (MIT): 698 authored documents and 97 tasks, each naming
the documents (`required_documents`) an agent must consult as a third-party gold set.
Gold truth here is granular only to the document, never to a character range, so
measurements needing finer truth need a different collection. See
docs/where-the-answers-come-from.md.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas.contracts import Collection, DocName, Document, Question, QuestionName

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "tau2" / "banking_knowledge"
DOCUMENT_KIND = "tau2_knowledge"

# Identity/plumbing argument keys that say nothing about which document holds the
# answer. An exclusion list rather than inclusion because domain keys differ per tool
# and a fixed inclusion list would silently stop matching when tau2 adds a tool.
_PLUMBING_ARGUMENT_KEYS = frozenset({
    "agent_tool_name", "discoverable_tool_name", "user_id", "name", "address", "email",
    "phone_number", "date_of_birth", "time_verified", "requestor", "action_id",
})

# Below this, a folded value matches document ids by accident: "usd" appears inside
# any id containing "used", and a value of two characters matches most of the corpus.
_MIN_MATCH_CHARACTERS = 6


def _fold(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def _collect_content_arguments(actions: Any, into: list[str]) -> None:
    """Every string argument that describes the domain, following nested `arguments`
    (tau2 nests the real call inside that key under the tool dispatch wrapper)."""
    for action in actions:
        _collect_from_mapping(action.get("arguments") or {}, into)


def _collect_from_mapping(payload: Mapping[str, Any], into: list[str]) -> None:
    for key, value in payload.items():
        if isinstance(value, Mapping):
            _collect_from_mapping(value, into)
        elif key not in _PLUMBING_ARGUMENT_KEYS and isinstance(value, str):
            into.append(value)


@dataclass(frozen=True, slots=True)
class Tau2Source:
    """A `CollectionSource` over the vendored tau2 knowledge domain. Takes a seed to
    satisfy the protocol and ignores it: the corpus is a fixed third-party artifact,
    not regenerated per run."""

    root: Path = DEFAULT_ROOT
    name: str = "tau2-banking-knowledge"

    def __call__(self, seed: int = 0) -> Collection:
        return Collection(documents=self.documents())

    def documents(self) -> tuple[Document, ...]:
        docs = []
        for path in sorted((self.root / "documents").glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            text = f"# {raw['title']}\n\n{raw['content']}"
            docs.append(Document(name=DocName(raw["id"]), kind=DOCUMENT_KIND, text=text))
        return tuple(docs)

    def primary_documents(self, task: dict[str, Any]) -> tuple[DocName, ...]:
        """Which of a task's required documents actually carry its answer, derived (never
        annotated) from `evaluation_criteria.actions`: a document is primary when an
        action argument matches its id. Covers 22 of 97 tasks; the rest return empty,
        which every consumer treats as "unknown" rather than "none". See
        docs/where-the-answers-come-from.md.
        """
        values: list[str] = []
        _collect_content_arguments((task.get("evaluation_criteria") or {}).get("actions") or (),
                                   values)
        if not values:
            return ()
        # Substring match on folded (case/punctuation-stripped) text: a document id like
        # gold_rewards_card carries a category prefix and numeric suffix around the part
        # that matches the argument's "Gold Rewards Card".
        folded = [_fold(v) for v in values if len(_fold(v)) >= _MIN_MATCH_CHARACTERS]
        return tuple(
            DocName(doc) for doc in task.get("required_documents") or ()
            if any(value in _fold(doc) for value in folded)
        )

    def questions(self) -> tuple[Question, ...]:
        """One question per tau2 task. Query text is the customer scenario, not a query
        written for retrieval: tau2 is a multi-turn benchmark with no short query field
        to borrow."""
        raw = json.loads((self.root / "tasks.json").read_text(encoding="utf-8"))
        questions = []
        for task in raw:
            required = tuple(DocName(doc) for doc in task.get("required_documents") or ())
            if not required:
                continue
            scenario = task["user_scenario"]["instructions"]
            text = scenario if isinstance(scenario, str) else scenario.get("reason_for_call", "")
            questions.append(Question(
                name=QuestionName(task["id"]), text=text, kind="across_documents",
                required=required, primary=self.primary_documents(task),
            ))
        return tuple(questions)


def load(root: Path = DEFAULT_ROOT) -> tuple[Collection, tuple[Question, ...]]:
    source = Tau2Source(root=root)
    return source(0), source.questions()
