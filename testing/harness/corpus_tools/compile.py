"""Compile the registry into SQLite so golden answers are lookups, not annotations (HLD D4)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from corpus_tools.registry import Registry

_SCHEMA = """
create table entities (id text primary key, kind text not null, render integer not null);
create table entity_fields (entity_id text not null references entities(id), field text not null, value text not null, primary key (entity_id, field));
create table edges (relation text not null, src text not null references entities(id), dst text not null references entities(id), fields_json text not null default '{}');
create table contradictions (id text primary key, conflict_type text not null, hops integer not null, winning_fact text not null, losing_fact text not null, resolution_rule text not null, question_hint text not null);
"""
# fields_json (added beyond the brief's literal schema) carries json.dumps(edge.fields, sort_keys=True)
# so integrity_report can read an edge's override_amount for the mirrored-override check below.


def compile_registry(reg: Registry, db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.executemany(
            "insert into entities values (?, ?, ?)",
            [(e.id, e.kind, int(e.render)) for e in reg.entities],
        )
        conn.executemany(
            "insert into entity_fields values (?, ?, ?)",
            [(e.id, f, str(v)) for e in reg.entities for f, v in sorted(e.fields.items())],
        )
        conn.executemany(
            "insert into edges values (?, ?, ?, ?)",
            [
                (edge.relation, edge.src, edge.dst, json.dumps(edge.fields, sort_keys=True))
                for edge in reg.edges
            ],
        )
        conn.executemany(
            "insert into contradictions values (?, ?, ?, ?, ?, ?, ?)",
            [
                (c.id, c.conflict_type, c.hops, c.winning_fact, c.losing_fact, c.resolution_rule, c.question_hint)
                for c in reg.contradictions
            ],
        )


def lookup_fact(db_path: Path, fact_ref: str) -> str:
    entity_id, _, fact_field = fact_ref.partition(":")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "select value from entity_fields where entity_id = ? and field = ?", (entity_id, fact_field)
        ).fetchone()
    if row is None:
        raise KeyError(fact_ref)
    return row[0]


def _resolves(conn: sqlite3.Connection, fact_ref: str) -> bool:
    entity_id, _, fact_field = fact_ref.partition(":")
    return (
        conn.execute(
            "select 1 from entity_fields where entity_id = ? and field = ?", (entity_id, fact_field)
        ).fetchone()
        is not None
    )


def integrity_report(db_path: Path) -> tuple[str, ...]:
    violations: list[str] = []
    with sqlite3.connect(db_path) as conn:
        planless = conn.execute(
            "select id from entities where kind = 'plan' and render = 1 and id not in (select src from edges where relation = 'available_in') order by id"
        ).fetchall()
        violations += [f"plan {pid} has no available_in edge" for (pid,) in planless]
        promoless = conn.execute(
            "select id from entities where kind = 'promotion' and id not in (select src from edges where relation = 'applies_to') order by id"
        ).fetchall()
        violations += [f"promotion {pid} has no applies_to edge" for (pid,) in promoless]
        for cid, winning, losing in conn.execute(
            "select id, winning_fact, losing_fact from contradictions order by id"
        ).fetchall():
            for ref in (winning, losing):
                if not _resolves(conn, ref):
                    violations.append(f"contradiction {cid}: fact {ref} does not resolve")
        superseded = conn.execute("select src, dst from edges where relation = 'supersedes' order by src").fetchall()
        for src, dst in superseded:
            covered = conn.execute(
                "select 1 from contradictions where winning_fact like ? and losing_fact like ?",
                (f"{src}:%", f"{dst}:%"),
            ).fetchone()
            if covered is None:
                violations.append(f"supersedes edge {src} -> {dst} has no contradiction record")
        # Mirrored override check (added beyond the brief's literal code): overrides_fee edges may
        # carry an override_amount that duplicates the src region's own equipment_rental_override_amount
        # field (conflict-promo-price-north relies on the region entity's copy); if both exist they
        # must agree, or the two copies have silently drifted apart.
        overrides = conn.execute(
            "select src, dst, fields_json from edges where relation = 'overrides_fee' order by src, dst"
        ).fetchall()
        for src, dst, fields_json in overrides:
            edge_fields = json.loads(fields_json)
            if "override_amount" not in edge_fields:
                continue
            override_amount = edge_fields["override_amount"]
            mirror_row = conn.execute(
                "select value from entity_fields where entity_id = ? and field = 'equipment_rental_override_amount'",
                (src,),
            ).fetchone()
            if mirror_row is not None and mirror_row[0] != override_amount:
                violations.append(
                    f"overrides_fee edge {src} -> {dst}: override_amount {override_amount!r} "
                    f"!= {src}:equipment_rental_override_amount {mirror_row[0]!r}"
                )
    return tuple(violations)
