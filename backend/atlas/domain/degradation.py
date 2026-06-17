"""The degradation ladder (SP4 task 4, D21): a pure domain constant mirroring the trace enum
(`contracts/trace/schema.json`'s `atlas.degradation.mode`, `contracts/sse/schema.json`'s
`degradation` event) so the graph, and later the trace/SSE layers, key off the SAME five rungs and
the SAME ordering, never a hand kept in sync copy. Pure: no framework, no client, no I/O (D8's own
discipline, see `atlas/ports/knowledge.py`'s module docstring), just a tuple and the one comparison
every transition needs.

`"none"` is the graph state's own default (`AtlasState.degradation_mode`, `atlas_graph.py`), not
itself a rung: a turn that never degraded carries it unchanged. `DEGRADATION_LADDER` only names the
five rungs a transition can escalate the state TO, ordered least to most severe, matching the trace
contract's own enum order (`none, retry, provider_fallback, drop_rerank, lexical_only, refusal`)
with `none` dropped, since a transition never "escalates to none".
"""
from __future__ import annotations

DEGRADATION_MODE_NONE = "none"

DEGRADATION_LADDER: tuple[str, ...] = (
    "retry",
    "provider_fallback",
    "drop_rerank",
    "lexical_only",
    "refusal",
)

# The MCP knowledge server's own envelope discriminator (SP4 task 4): `search_knowledge`'s
# ordinary, undegraded result stays the bare passages array, byte identical to every pre ladder
# caller's expectation; only a degraded result is wrapped in an object carrying this key, so
# `atlas_graph._knowledge_call` (the only reader) tells the two shapes apart without guessing from
# content. Single sourced here, not a literal repeated in both the server and the graph, the same
# discipline `knowledge_server.DEPLOYED_K` already follows.
DEGRADED_RESULT_KEY = "atlas_degraded"

_RUNG_RANK = {mode: rank for rank, mode in enumerate(DEGRADATION_LADDER)}


def escalate(current: str, candidate: str) -> str:
    """Last rung wins, but only upward (Global Constraints: "a lower rung never overwrites a
    higher one"): returns `candidate` when it outranks `current` on the ladder, else `current`
    unchanged. `"none"` (rank -1, not itself a rung) always loses to any real rung. An unranked
    string (a typo, never expected in practice) also ranks -1, so it can neither win against a real
    rung nor silently downgrade one -- fail closed, the same discipline
    `resilience.is_retryable_status` uses for an unrecognized status code (default to the safer
    outcome, never the more permissive one)."""
    return candidate if _RUNG_RANK.get(candidate, -1) > _RUNG_RANK.get(current, -1) else current
