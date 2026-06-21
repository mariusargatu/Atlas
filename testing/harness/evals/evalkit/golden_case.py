"""The golden dataset's two shapes: a loose draft you ingest, a strict case you commit.

The seed is authored by hand and onboarded through a forgiving door, then hardened into a typed,
validated record before anything depends on it. Two types make that pipeline explicit:

- ``GoldenDraft`` is what a bootstrap source (the CSV, or a script run once) yields. Only the SME core
  is required (who, what they asked, what correct means). Every engineer added field is optional, so a
  half authored case still loads instead of failing the import. This is the "optional types on inject".

- ``GoldenCase`` is the canonical record. Every field is required and validated in ``__post_init__``:
  the enums must be in range, the identity must name a real seeded account, the turns must not be
  empty. An invalid case cannot be constructed, so the committed seed (``datasets/seed.py``) is
  validated the moment it is imported. This is the "required typed format" the rest of the system
  consumes.

``enrich`` is the one way door from the first to the second: a draft plus the engineer's metadata
becomes a ``GoldenCase``, or raises. ``oracle`` stays a prose pointer to the source of truth here.
Turning that prose into an executable check against the account+catalog is deliberately deferred
to structured-claim extraction, not built in the dataset layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from atlas.domain.accounts import SEED

from evals.evalkit.case import EvalCase

# The controlled vocabularies. Kept as ``Literal`` so a typo is a type error at author time and a
# ValueError at construction time, the two places a stringly typed CSV could not catch it.
Category = Literal["policy_question", "account_read", "action"]
Consequence = Literal["low", "high"]
Source = Literal["hand_written", "production", "generated"]
AuthorRole = Literal["sme", "engineer"]
Tier = Literal["gold", "silver"]


@dataclass(frozen=True)
class GoldenDraft:
    """Loose ingest target. SME core required, engineer enrichment optional (``None``/empty).

    A draft is deliberately NOT runnable or gradeable on its own. It is the raw material ``enrich``
    turns into a ``GoldenCase``. Keeping it separate is what lets the inbound door stay forgiving
    without letting an unenriched case leak into the set that gates a release.
    """

    id: str
    turns: tuple[str, ...]
    customer_id: str
    expected: str = ""
    notes: str = ""


@dataclass(frozen=True)
class GoldenCase:
    """The canonical, validated dataset record. All fields required, and invalid construction raises.

    Carries the SME's verified answer (``expected``), the engineer's risk framing (``risk``,
    ``consequence``), the source of truth pointer (``oracle``, prose), and provenance (``source``,
    ``author_role``, ``tier``) so the set can be sliced and a regression debugged by surface.
    """

    id: str
    turns: tuple[str, ...]
    customer_id: str
    expected: str
    category: Category
    risk: str
    consequence: Consequence
    oracle: str
    source: Source
    author_role: AuthorRole
    tier: Tier
    notes: str = ""
    graders: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("id", "customer_id", "expected", "risk", "oracle"):
            if not (getattr(self, field_name) or "").strip():
                raise ValueError(f"{self.id or '<no id>'}: {field_name} is required and empty")
        if not self.turns:
            raise ValueError(f"{self.id}: at least one turn is required")
        if self.customer_id not in SEED:
            raise ValueError(
                f"{self.id}: customer_id {self.customer_id!r} is not a seeded account "
                f"(known: {sorted(SEED)})"
            )
        for field_name, type_ in (
            ("category", Category), ("consequence", Consequence),
            ("source", Source), ("author_role", AuthorRole), ("tier", Tier),
        ):
            value, allowed = getattr(self, field_name), get_args(type_)
            if value not in allowed:
                raise ValueError(f"{self.id}: {field_name}={value!r} not in {allowed}")

    def to_eval_case(self) -> EvalCase:
        """Project to the runner's input. The runner needs WHAT to run and WHICH graders. The rich
        provenance stays on the dataset record, so the run spec stays minimal (principle: a case is
        pure data). ``risk`` rides along because the report rolls up under it."""
        return EvalCase(
            id=self.id,
            turns=self.turns,
            customer_id=self.customer_id,
            expected=self.expected,
            name=self.id,
            risk=self.risk,
            graders=self.graders,
        )


def enrich(draft: GoldenDraft, **meta) -> GoldenCase:
    """Promote a loose ``GoldenDraft`` to a validated ``GoldenCase`` with engineer supplied ``meta``.

    The draft carries the SME core. ``meta`` supplies the required enrichment (``category``, ``risk``,
    ``consequence``, ``oracle``, ``source``, ``author_role``, ``tier``) and may override ``graders``/
    ``notes``. ``GoldenCase`` validation is the gate: a missing or out of range field raises here,
    so an unenriched draft can never silently become a gold case.
    """
    return GoldenCase(
        id=draft.id,
        turns=draft.turns,
        customer_id=draft.customer_id,
        expected=draft.expected,
        notes=meta.pop("notes", draft.notes),
        **meta,
    )


__all__ = [
    "Category", "Consequence", "Source", "AuthorRole", "Tier",
    "GoldenDraft", "GoldenCase", "enrich",
]
