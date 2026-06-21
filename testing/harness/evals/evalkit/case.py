"""``EvalCase``: one seeded task the eval harness drives the agent through.

Deliberately MINIMAL for this pass. The golden-dataset article (04) owns the rich case format
(tags, difficulty, provenance, expected trajectory) and will extend this; keeping the surface
small now avoids pre-empting that design. A case names WHAT to run (the turns, the identity) and
WHICH graders apply by name; the runner is handed the grader instances, so a case stays pure data.

Identity rides in ``customer_id`` and is fed into the non-model ``session`` channel, never as a
tool argument the model can fill, the invariant the whole system turns on (principle: identity
comes from the session, never the model).

Deferred to 04 — deliberate, not an oversight
---------------------------------------------
Two capabilities a 2026 reader will look for are real standards but belong to 04's rich case
format, so they are named here rather than half-built now:

- **An expected trajectory + a ``TrajectoryGrader`` with match modes** — strict / unordered /
  subset / superset over the tool calls, plus tool-call precision/recall. This is now table stakes
  for agent evals (LangChain ``agentevals`` ships exactly those four modes; LangSmith has trajectory
  match). It needs a ``trajectory`` field on the case, which is part of 04's format. Until then the
  drift lane's ``DecisionRecord`` already captures the tool order, and a grader can read it through
  ``GradeContext.trace`` (``tracing`` exposes ``tool_order()``). See ``graders.py``.

- **A ``UserSimulator``** — a scripted user by default, and an LLM user-simulator with information
  asymmetry driven *through the gateway* (so it is cassette-replayable) for the LIVE lane. This is
  the tau-bench pattern. Today ``runner._drive`` replays the case's ``turns`` verbatim, which is the
  correct hermetic default; turning a turn list into a simulated user is a case-authoring concern 04
  owns, so ``turns`` stays literal data here. See ``runner.py``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    """A single eval task: a thread of user turns for one signed-in customer.

    The first three fields are the mechanics; ``name`` and ``risk`` make the case read as a spec a
    reviewer can act on. An SDET writes the case; a CTO reads the report grouped by ``risk``. A case
    without a ``name``/``risk`` still runs, it just reports under its ``id``.

    - ``id``: stable identifier, the key results are reported under.
    - ``turns``: the user utterances to drive, in order, on one conversation thread.
    - ``customer_id``: the session identity (e.g. ``"cust_legacy_term"`` for the cold open).
    - ``name``: a one-line, human-readable title (e.g. "contracted customer asks to cancel").
    - ``risk``: the business risk this case guards, the bucket the outcome rolls up under (e.g.
      "fee-claim-safety", "data-isolation", "unauthorized-write"). This is the word a CTO reads.
    - ``graders``: the grader names that apply (declarative). ``run_suite`` resolves these against a
      ``{name: Grader}`` registry, so a mixed-risk suite grades each case with only the rules it
      names. 04 grows this registry; the per-case resolution is wired today.
    """

    id: str
    turns: tuple[str, ...]
    customer_id: str
    name: str = ""
    risk: str = ""
    graders: tuple[str, ...] = ()


__all__ = ["EvalCase"]
