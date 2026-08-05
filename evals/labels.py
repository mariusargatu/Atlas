"""The label store: verdicts on answers, recorded beside the judge's own.

A human label cannot be regenerated at any price, unlike everything else in this
repository, so the store is append only. Every row names its rater: `agreement_against_
labels` refuses a mixed set, since a store mixing model and human verdicts would report
"human agreement" for a number partly measuring a model agreeing with itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from atlas.contracts import QuestionName

LABEL_SCHEMA_VERSION = "3.0.0"

# The rater a `just label` session records. Anything else is a model, and the two must
# never be pooled into one figure.
HUMAN_RATER = "human"


@dataclass(frozen=True, slots=True)
class Label:
    schema: str
    question: QuestionName
    rubric_version: str
    verdict: str
    reason: str
    run_id: str
    # The judge's own verdict on this answer, recorded at labelling time so the row is
    # self contained: agreement no longer needs to join against the (gitignored) run
    # ledger by run_id. The labeller is never shown this value; seeing it first would make
    # them an anchored rater rather than an independent one.
    judge_verdict: str
    # Who graded this answer: HUMAN_RATER for a `just label` session, or the model id for
    # a model-produced verdict. Defaulted so the distinction is enforced only when reading
    # the store, never at a construction site that could forget to set it.
    rater: str = HUMAN_RATER


def append_label(path: str | Path, label: Label) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(label)) + "\n")


def read_labels(path: str | Path) -> tuple[Label, ...]:
    path = Path(path)
    if not path.exists():
        return ()
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload["schema"] != LABEL_SCHEMA_VERSION:
            raise ValueError(
                f"label store holds schema {payload['schema']!r}, expected {LABEL_SCHEMA_VERSION!r}"
            )
        labels.append(Label(**payload))
    return tuple(labels)
