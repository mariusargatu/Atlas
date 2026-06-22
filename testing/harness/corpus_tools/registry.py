"""Load and validate the fact registry: the root artifact everything else derives from (HLD D4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ENTITY_KINDS: tuple[str, ...] = ("plan", "region", "fee", "device", "contract_term", "promotion", "policy")
EDGE_RELATIONS: tuple[str, ...] = ("available_in", "applies_to", "overrides_fee", "compatible_with", "supersedes")
CONFLICT_TYPES: tuple[str, ...] = ("temporal", "inter_doc")


class RegistryError(ValueError):
    """A registry file violates the schema or internal consistency."""


@dataclass(frozen=True)
class Entity:
    id: str
    kind: str
    render: bool
    fields: dict


@dataclass(frozen=True)
class Edge:
    relation: str
    src: str
    dst: str
    fields: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Contradiction:
    id: str
    conflict_type: str
    hops: int
    winning_fact: str
    losing_fact: str
    resolution_rule: str
    question_hint: str


@dataclass(frozen=True)
class Registry:
    entities: tuple[Entity, ...]
    edges: tuple[Edge, ...]
    contradictions: tuple[Contradiction, ...]

    def entity(self, entity_id: str) -> Entity:
        for e in self.entities:
            if e.id == entity_id:
                return e
        raise KeyError(entity_id)

    def by_kind(self, kind: str) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.kind == kind)


def _entity(raw: dict) -> Entity:
    entity_id = raw.get("id", "<missing id>")
    kind = raw.get("kind")
    if kind not in ENTITY_KINDS:
        raise RegistryError(f"entity {entity_id}: unknown kind {kind!r}")
    if not isinstance(raw.get("fields"), dict):
        raise RegistryError(f"entity {entity_id}: fields must be a mapping")
    return Entity(id=entity_id, kind=kind, render=bool(raw.get("render", True)), fields=dict(raw["fields"]))


def _edge(raw: dict, ids: set[str]) -> Edge:
    relation = raw.get("relation")
    if relation not in EDGE_RELATIONS:
        raise RegistryError(f"edge {raw}: unknown relation {relation!r}")
    for endpoint in ("src", "dst"):
        if raw.get(endpoint) not in ids:
            raise RegistryError(f"edge {relation}: {raw.get(endpoint)!r} does not exist")
    return Edge(relation=relation, src=raw["src"], dst=raw["dst"], fields=dict(raw.get("fields", {})))


def _fact_ref_ok(ref: str, registry: Registry) -> bool:
    entity_id, _, fact_field = ref.partition(":")
    try:
        return fact_field in registry.entity(entity_id).fields
    except KeyError:
        return False


def _contradiction(raw: dict, reg: Registry) -> Contradiction:
    cid = raw.get("id", "<missing id>")
    for required in ("conflict_type", "hops", "winning_fact", "losing_fact", "resolution_rule"):
        if required not in raw:
            raise RegistryError(f"contradiction {cid}: missing {required}")
    if raw["conflict_type"] not in CONFLICT_TYPES:
        raise RegistryError(f"contradiction {cid}: unknown conflict_type {raw['conflict_type']!r}")
    if raw["hops"] not in (1, 2):
        raise RegistryError(f"contradiction {cid}: hops must be 1 or 2")
    for fact_key in ("winning_fact", "losing_fact"):
        if not _fact_ref_ok(raw[fact_key], reg):
            raise RegistryError(f"contradiction {cid}: {fact_key} {raw[fact_key]!r} does not dereference")
    return Contradiction(
        id=cid,
        conflict_type=raw["conflict_type"],
        hops=int(raw["hops"]),
        winning_fact=raw["winning_fact"],
        losing_fact=raw["losing_fact"],
        resolution_rule=str(raw["resolution_rule"]),
        question_hint=str(raw.get("question_hint", "")),
    )


def load_registry(paths: list[Path]) -> Registry:
    raw_entities: list[dict] = []
    raw_edges: list[dict] = []
    raw_contradictions: list[dict] = []
    for path in paths:
        doc = yaml.safe_load(path.read_text()) or {}
        raw_entities += doc.get("entities", [])
        raw_edges += doc.get("edges", [])
        raw_contradictions += doc.get("contradictions", [])

    entities = tuple(_entity(raw) for raw in raw_entities)
    ids = {e.id for e in entities}
    if len(ids) != len(entities):
        seen: set[str] = set()
        dup = next(e.id for e in entities if e.id in seen or seen.add(e.id))
        raise RegistryError(f"duplicate entity id: {dup}")
    edges = tuple(_edge(raw, ids) for raw in raw_edges)
    partial = Registry(entities=entities, edges=edges, contradictions=())
    contradictions = tuple(_contradiction(raw, partial) for raw in raw_contradictions)
    return Registry(entities=entities, edges=edges, contradictions=contradictions)
