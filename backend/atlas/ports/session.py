"""The session (identity) port, the load bearing seam of the whole series.

``customer_id`` comes from the authenticated session, NEVER from the model and never from a tool
argument (principle 1). The CI adapter is a seeded fixture; dev/prod resolves a bearer token at the
edge. Whatever the adapter, identity is injected into MCP call context out of band of the schema.
"""
from __future__ import annotations

from typing import Protocol


class SessionResolver(Protocol):
    def customer_id_for(self, session_id: str) -> str: ...
