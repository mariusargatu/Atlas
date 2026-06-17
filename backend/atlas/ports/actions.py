"""The actions port, the write surface (highest consequence).

Generic over the write tool and keyed by an orchestrator minted ``idempotency_key`` (never a model
argument), so a timed out retry applies exactly once. CI adapter is the stateful in memory fake;
dev/prod simulates provisioning against Postgres behind the ``actions`` MCP server.
"""
from __future__ import annotations

from typing import Protocol

from atlas.domain.actions import ActionResult


class ActionExecutor(Protocol):
    def apply(self, tool: str, customer_id: str, args: dict, idempotency_key: str) -> ActionResult: ...
