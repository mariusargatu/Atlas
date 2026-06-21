"""Ingest the golden seed from CSV into loose ``GoldenDraft`` records.

The CSV is the INTERMEDIARY, the spreadsheet an SME reads and edits and the blog uses to explain how
a case is authored. It is not the canonical artifact. ``datasets/seed.py`` is, holding validated
``GoldenCase`` literals. This loader is the forgiving inbound door: it parses the SME core, validates
the cheap structural things a typo would break (unknown account, empty turns, duplicate id), and
hands back drafts for ``enrich`` to harden into the required typed format.

CSV columns (the SME facing minimum):

  id          stable identifier, unique across the set
  customer_id the session identity, validated against the account seed
  turns       the user utterances, separated by ``||`` for a case of many turns
  expected    what *correct* means here, in the SME's words (prose, not a frozen value)
  notes       authoring context for the human reader
"""
from __future__ import annotations

import csv
from pathlib import Path

from atlas.domain.accounts import SEED

from evals.evalkit.golden_case import GoldenDraft

_REQUIRED_COLUMNS = ("id", "customer_id", "turns", "expected")
_TURN_SEPARATOR = "||"


def _split_turns(raw: str) -> tuple[str, ...]:
    """``"a||b"`` -> ``("a", "b")``. Whitespace stripped, empty segments dropped."""
    return tuple(part.strip() for part in raw.split(_TURN_SEPARATOR) if part.strip())


def _row_to_draft(row: dict[str, str], *, line: int) -> GoldenDraft:
    case_id = (row.get("id") or "").strip()
    if not case_id:
        raise ValueError(f"row {line}: missing id")

    customer_id = (row.get("customer_id") or "").strip()
    if customer_id not in SEED:
        raise ValueError(
            f"row {line} ({case_id!r}): customer_id {customer_id!r} is not a seeded account "
            f"(known: {sorted(SEED)})"
        )

    turns = _split_turns(row.get("turns") or "")
    if not turns:
        raise ValueError(f"row {line} ({case_id!r}): no turns")

    return GoldenDraft(
        id=case_id,
        turns=turns,
        customer_id=customer_id,
        expected=(row.get("expected") or "").strip(),
        notes=(row.get("notes") or "").strip(),
    )


def load_golden_drafts(path: str | Path) -> tuple[GoldenDraft, ...]:
    """Read the golden CSV at ``path`` into loose ``GoldenDraft`` records, validating as it goes.

    Raises ``FileNotFoundError`` if the file is missing, and ``ValueError`` on a malformed set:
    a missing required column, a missing id, a ``customer_id`` that names no seeded account, an
    empty turn list, or a duplicate id. A bad set fails loud at load instead of silently ingesting
    nothing, which is the failure mode a stringly typed CSV invites.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"golden dataset not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [c for c in _REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{csv_path}: missing required column(s): {missing}")

        drafts: list[GoldenDraft] = []
        seen: set[str] = set()
        for offset, row in enumerate(reader):
            draft = _row_to_draft(row, line=offset + 2)  # +2: header is line 1, rows are 1 indexed
            if draft.id in seen:
                raise ValueError(f"duplicate id {draft.id!r} in {csv_path}")
            seen.add(draft.id)
            drafts.append(draft)

    if not drafts:
        raise ValueError(f"{csv_path}: no cases")
    return tuple(drafts)


__all__ = ["load_golden_drafts"]
