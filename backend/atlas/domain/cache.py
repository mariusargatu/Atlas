"""The semantic cache, the cheapest component, the nastiest isolation bug.

A customer specific answer must be keyed per customer, or the cache serves one person's bill
to another (the Asana cross tenant failure rebuilt for performance). Only generic answers are
shared. `NaiveCache` is the buggy version kept to prove the regression test bites.
"""
from __future__ import annotations


class PerCustomerCache:
    def __init__(self) -> None:
        self._store: dict[tuple, str] = {}

    @staticmethod
    def _key(customer_id: str, question: str, generic: bool) -> tuple:
        return ("__generic__", question) if generic else (customer_id, question)

    def put(self, customer_id: str, question: str, answer: str, *, generic: bool) -> None:
        self._store[self._key(customer_id, question, generic)] = answer

    def get(self, customer_id: str, question: str, *, generic: bool) -> str | None:
        return self._store.get(self._key(customer_id, question, generic))

    def invalidate(self, customer_id: str) -> None:
        """Drop this customer's cached answers after a confirmed write changed their account, so a
        repeat question is recomputed instead of served stale. Generic (customer independent) answers
        are left alone: a write to one customer does not change a population wide reply. Rebinds the
        store rather than mutating entries in place."""
        self._store = {k: v for k, v in self._store.items() if k[0] != customer_id}


class NaiveCache:
    """Keyed only by the question, leaks one customer's answer to another. Its `invalidate` is a
    no-op for the same reason it leaks: with no customer in the key it cannot drop one customer's
    entries, so it also serves stale data after that customer's own write."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def put(self, customer_id: str, question: str, answer: str, *, generic: bool) -> None:
        self._store[question] = answer

    def get(self, customer_id: str, question: str, *, generic: bool) -> str | None:
        return self._store.get(question)

    def invalidate(self, customer_id: str) -> None:
        return None
