"""The semantic cache port, the cheapest component with the nastiest isolation bug.

Customer specific answers MUST be keyed per customer; only generic, customer independent answers
are shared across the population (`02-app-spec.md`). A cache hit still passes the render guard.
"""
from __future__ import annotations

from typing import Optional, Protocol


class AnswerCache(Protocol):
    def get(self, customer_id: str, question: str, *, generic: bool = False) -> Optional[str]: ...
    def put(self, customer_id: str, question: str, answer: str, *, generic: bool = False) -> None: ...
