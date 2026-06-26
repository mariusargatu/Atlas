"""``EvalCase``: one seeded task the eval harness drives the agent through.

Deliberately minimal for this pass. The rich case format (tags, difficulty, provenance, expected
trajectory) is deliberately not built here and would extend this later. Keeping the surface
small now avoids preempting that design. A case names what to run (the turns, the identity) and
which graders apply by name. The runner is handed the grader instances, so a case stays pure data.

Identity rides in ``customer_id`` and is fed into the ``session`` channel (outside the model), never
as a tool argument the model can fill, the invariant the whole system turns on.

Two fields a reader will look for are named here but not built now: an expected
trajectory with a ``TrajectoryGrader`` (match modes over the tool calls, the agentevals standard),
and a ``UserSimulator`` (a scripted user or one driven by an LLM for the live lane, the tau-bench pattern).
Until then the drift lane's ``DecisionRecord`` already captures tool order, and ``runner._drive``
replays the case's ``turns`` verbatim, the correct hermetic default.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    """A single eval task: a thread of user turns for one signed in customer.

    The first three fields are the mechanics. ``name`` and ``risk`` make the case read as a spec a
    reviewer can act on. An SDET writes the case, and a CTO reads the report grouped by ``risk``. A case
    without a ``name``/``risk`` still runs, it just reports under its ``id``.

    - ``id``: stable identifier, the key results are reported under.
    - ``turns``: the user utterances to drive, in order, on one conversation thread.
    - ``customer_id``: the session identity (e.g. ``"cust_legacy_term"`` for the cold open).
    - ``expected``: what *correct* means here, in the SME's words (prose, not a frozen value). This is
      the human verified oracle of the golden set, the one field an SME holds the pen on. The grader
      that turns this prose into an executable check against the source of truth is deliberately not
      built here. The case carries the SME's answer first class so nothing downstream invents it.
    - ``name``: a one line, human readable title (e.g. "contracted customer asks to cancel").
    - ``risk``: the business risk this case guards, the bucket the outcome rolls up under (e.g.
      "fee-claim-safety", "data-isolation", "unauthorized-write"). This is the word a CTO reads.
    - ``graders``: the grader names that apply (declarative). ``run_suite`` resolves these against a
      ``{name: Grader}`` registry, so a mixed risk suite grades each case with only the rules it
      names. The registry grows as new graders are added, and the per case resolution is wired today.
    - ``expected_doc_ids``/``expected_tool_calls``: the machine readable form of ``expected``, for
      case parameterised graders (retrieval ids, tool calls). Empty for a case that declares no
      structured expectation. NOT provenance: provenance stays on the dataset record, per
      ``golden_case.to_eval_case``.
    """

    id: str
    turns: tuple[str, ...]
    customer_id: str
    expected: str = ""
    name: str = ""
    risk: str = ""
    graders: tuple[str, ...] = ()
    expected_doc_ids: tuple[str, ...] = ()
    expected_tool_calls: tuple[Mapping[str, object], ...] = ()


__all__ = ["EvalCase"]
