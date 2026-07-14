"""Trajectory grading: judge the agent as a path of decisions, not a single answer (doc 08).

Once the model can act, the final message stops being the thing to test. The unit becomes the
trajectory: which tools it called, in what order, with what arguments, whether it confirmed before
anything irreversible, and whether it stopped. This subpackage grades that path DETERMINISTICALLY
over the recorded span tree, so it gates the hermetic lane while DeepEval's judged agentic metrics run
beside it (the operator ``__main__``).

- ``model``      the ``Trajectory`` / ``ToolCall`` / ``Verdict`` shapes.
- ``from_trace`` builds a ``Trajectory`` from a trace, reusing ``tracing``'s ONE definition of the
  trajectory (``tool_calls`` / ``guard_outcomes`` / ``write_applied``) so a grader and the drift lane
  can never disagree on what was called.
- ``atom``       the single tool call: right tool for the intent, arguments in bounds, id scoped to
  the session. Reuses ``binding`` and ``guard``, so an eval never grades more leniently than runtime.
- ``path``       the whole path: at most one write, no orphan action, terminated, within budget, plus
  ``grade_trajectory`` at the three levels (end-to-end, path, node) and the two metric families.

Confirmation and idempotency are properties of the propose-confirm-execute gate itself
(``domain.confirmation`` + the actions backend), asserted directly in ``test_trajectory``.
"""
