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


class NaiveCache:
    """Keyed only by the question, leaks one customer's answer to another."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def put(self, customer_id: str, question: str, answer: str, *, generic: bool) -> None:
        self._store[question] = answer

    def get(self, customer_id: str, question: str, *, generic: bool) -> str | None:
        return self._store.get(question)
