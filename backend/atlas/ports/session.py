"""The session (identity) port, the load bearing seam of the whole runtime.

``customer_id`` comes from the authenticated session, never from the model and never from a tool
argument. The CI adapter is a seeded fixture; dev/prod resolves a bearer token at the
edge. Whatever the adapter, identity is injected into MCP call context out of band of the schema.
"""
from __future__ import annotations

from typing import Protocol


class SessionResolver(Protocol):
    def customer_id_for(self, session_id: str) -> str: ...
