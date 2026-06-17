"""The account port, read only access to the customer's source of truth (the oracle's left half).

Graded on scope (only this customer) and freshness (current state). CI adapter is the seeded
in memory fake. Dev/prod is Postgres behind the ``account`` MCP server.
"""
from __future__ import annotations

from typing import Protocol

from atlas.domain.accounts import Account


class AccountReader(Protocol):
    def get_account(self, customer_id: str) -> Account: ...
