"""The trace boundary for the matrix's own cost figures (SP9 task 5, ADR-029's cost trio close).

D29 runs the judge as a batch teardown stage over already recorded spans (`judge/emission.py`'s own
module docstring); this module is that SAME shape for cost: the ONE place a computed generator cost
crosses into the `Tracer` port, opening a ``kind="llm"`` span (``atlas.cost.input_tokens``/
``output_tokens``/``usd``, translated by `backend/atlas/adapters/trace_translation.py`'s key table
under ``kind == "llm"``, alongside the SAME ``model`` -> ``gen_ai.request.model`` rename `atlas_
graph.py`'s own ``agent`` span already uses) under whatever parent the caller names.

Never wired into the live running graph (`atlas_graph.py` never imports `matrix/`, the same
disposition `judge/emission.py`'s own docstring states for the judge): a real matrix run emits this
AFTER a real generator call returns real `usage_metadata` (`matrix.spend_gate.cost_from_usage`
computing the dollar figure), never speculatively, and never gates anything -- it is a report time
annotation, not a runtime decision.
"""
from __future__ import annotations

from typing import Optional

_SPAN_NAME = "generator_cost"
_SPAN_KIND = "llm"


def emit_cost(
    tracer, parent: Optional[int], *, model_id: str, input_tokens: int, output_tokens: int, usd: float,
) -> int:
    """Open the cost span under ``parent``, carrying the generator's own model id and the three
    usage accounting figures. ``usd`` is a REAL, present number even when it is zero (Ollama): "cost
    unavailable" (an old cassette, no usage_metadata at all) is `matrix.spend_gate.cost_from_usage`
    returning `None` BEFORE this function is ever called -- this function is never invoked at all in
    that case, so it never has to represent "unavailable" as a span attribute value itself."""
    return tracer.open(
        _SPAN_NAME, _SPAN_KIND, parent,
        model=model_id, input_tokens=input_tokens, output_tokens=output_tokens, usd=usd,
    )


__all__ = ["emit_cost"]
