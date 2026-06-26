"""The trace boundary: where a computed judge verdict crosses into the span tree.

D29 runs the judge as a batch teardown stage, one pass over a burst's already recorded spans and
dataset cases, verdicts written back as span annotations -- never wired into the live running graph
(`backend/atlas/orchestration/atlas_graph.py` never imports this package, or any of `judge/`, at
all). This module is the ONE place a verdict crosses into the `Tracer` port, opening a
``kind="judge"`` span (``atlas.judge.id``/``atlas.judge.rubric_version``/``atlas.judge.verdict``,
span_kind ``EVALUATOR``) under whatever parent the caller names (typically the graded turn's own
root span).

``judge_id``/``rubric_version``/``verdict`` are the INFORMAL kwarg names
`backend/atlas/adapters/trace_translation.py`'s translation table maps to the frozen
``atlas.judge.*`` attribute names. This module never imports that backend module -- harness code may
import backend only through the `Tracer` protocol itself (`open`/`annotate`/`close`), never the
translation table directly (`test_import_lint.py`'s own boundary runs the other direction: backend
must never import harness, but this module still keeps to the protocol seam by convention, the same
one every other span opening caller in this repo uses). The informal vocabulary below is the shared
contract between the two modules, cross checked by `testing/tests/test_judge_emission.py` and
`testing/tests/test_trace_translation.py`, never by a Python import.

SP8 Task 4 remainder: this is also the ONE place `atlas.metrics`'s judge counter pair
(`atlas_judge_pass_total`/`atlas_judge_fail_total`) increments -- a grounded verdict bumps
`record_judge_pass`, an ungrounded verdict `record_judge_fail`, a thin call right beside the span
open above, never a second, independent counting mechanism. Importing `atlas.metrics` here is the
normal, allowed direction (harness code may import backend; the import lint's own boundary runs the
other way, backend must never import harness/evals), the same direction `atlas.adapters.label_store`
already imports `determinism.canonical` across.
"""
from __future__ import annotations

from typing import Optional

from atlas.metrics import record_judge_fail, record_judge_pass

from judge.contract import JudgeContract
from judge.llm_judge import VERDICT_GROUNDED, VERDICT_UNGROUNDED

_SPAN_NAME = "judge_verdict"
_SPAN_KIND = "judge"

_VALID_VERDICTS = frozenset({VERDICT_GROUNDED, VERDICT_UNGROUNDED})

# The wire verdict decides which counter moves; translation into the trace boundary already
# happened at `llm_judge.translate_verdict`, so this is a lookup, never a second translation.
_RECORD_BY_VERDICT = {VERDICT_GROUNDED: record_judge_pass, VERDICT_UNGROUNDED: record_judge_fail}


def emit_verdict(tracer, parent: Optional[int], contract: JudgeContract, verdict: str) -> int:
    """Open the judge's own span under ``parent``, carrying its versioned identity and verdict, and
    increment the matching Prometheus counter.

    ``verdict`` must already be the WIRE vocabulary (``llm_judge.translate_verdict``'s own output,
    ``grounded``/``ungrounded``) -- this function performs no translation of its own, only carries
    the value across the trace boundary, and fails closed on anything else rather than let an
    untranslated or mistyped verdict reach a real span or move a counter."""
    if verdict not in _VALID_VERDICTS:
        raise ValueError(
            f"emit_verdict got verdict={verdict!r}; expected one of {sorted(_VALID_VERDICTS)} "
            "(llm_judge.translate_verdict's own output, never a hand typed string)"
        )
    span_id = tracer.open(
        _SPAN_NAME, _SPAN_KIND, parent,
        judge_id=contract.fingerprint(),
        rubric_version=contract.rubric_version,
        verdict=verdict,
    )
    _RECORD_BY_VERDICT[verdict]()
    return span_id


__all__ = ["emit_verdict"]
