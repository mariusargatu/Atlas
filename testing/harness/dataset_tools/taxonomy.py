"""contracts/dataset/taxonomy.yaml: the versioned failure taxonomy (D34) every promoted dataset
case's `failure_class` field is checked against.

`contracts/dataset/schema.json`'s own `failure_class` field is typed `["string", "null"]`, a free
string, never a JSON Schema enum (see that schema file, and the SP8 digest's own design question 6).
A schema enum would force every new failure code a later triage pass names to be a MAJOR bump on the
CASE shape itself; the taxonomy instead carries its OWN semver (`taxonomy_version`, below),
independent of `contracts/dataset/schema.json`'s `x-contract-version` (0.1.0, never touched by this
module or by taxonomy.yaml). Consequently `failure_class` is validated against this loader's own
codes by APPLICATION code (`Taxonomy.validate_failure_class`, `_code`, `load_taxonomy` below), never
by a schema enum -- `test_dataset_taxonomy.py`'s own tests are that hermetic proof, both directions.

`contract_tools.changelog_gate`'s hermetic `check_consistency` never sees this file: it only compares
`contract_tools.loader.contract_versions()`, which reads `x-contract-version` off
`contracts/{trace,dataset,manifest,sse}/schema.json` (`contract_tools.loader.FAMILIES`).
`taxonomy.yaml` is not one of those four families and this module never changes
`contracts/dataset/schema.json`, so a `taxonomy.yaml` edit alone needs no CHANGELOG.md entry and
never trips the changelog gate.

Structured like `corpus/registry/core.yaml` (HLD D4's own hand authored root artifact, loaded by
`corpus_tools.registry.load_registry`): one top level semver plus a list of codes, each an
id/description/example triple, loaded and validated the same fail closed way that module validates
the fact registry -- a duplicate id, a malformed semver, or an incomplete code entry raises
`TaxonomyError` immediately; `load_taxonomy` never returns a partial or best effort `Taxonomy`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

TAXONOMY_PATH = Path("contracts/dataset/taxonomy.yaml")

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_REQUIRED_CODE_FIELDS = ("id", "description", "example")


class TaxonomyError(ValueError):
    """`taxonomy.yaml` violates its own structural contract: a malformed `taxonomy_version`, a
    duplicate code id, an incomplete code entry, or an unknown `failure_class` value checked against
    an already loaded `Taxonomy`. Fails closed in every case: never a partial result, never a silent
    coercion."""


@dataclass(frozen=True)
class FailureCode:
    id: str
    description: str
    example: str


@dataclass(frozen=True)
class Taxonomy:
    version: str
    codes: tuple[FailureCode, ...]

    @property
    def code_ids(self) -> frozenset[str]:
        return frozenset(c.id for c in self.codes)

    def is_known(self, code: str) -> bool:
        return code in self.code_ids

    def validate_failure_class(self, failure_class: str | None) -> None:
        """`failure_class` is the dataset contract's own free string field (`["string", "null"]`,
        never a schema enum, see this module's docstring). `None` (a case with no failure class at
        all, e.g. a plain factoid) always passes: this only ever rejects a non null value that is
        not one of THIS taxonomy's known codes, fail closed via `TaxonomyError`."""
        if failure_class is None:
            return
        if failure_class not in self.code_ids:
            raise TaxonomyError(
                f"unknown failure_class {failure_class!r}: not a code in taxonomy.yaml "
                f"(taxonomy_version {self.version}); known codes: {sorted(self.code_ids)}"
            )


def _code(raw: dict) -> FailureCode:
    for required in _REQUIRED_CODE_FIELDS:
        if not raw.get(required):
            raise TaxonomyError(f"taxonomy code entry missing {required!r}: {raw!r}")
    return FailureCode(id=raw["id"], description=raw["description"], example=raw["example"])


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Taxonomy:
    """Loads and validates `taxonomy.yaml`: `taxonomy_version` must be a well formed
    MAJOR.MINOR.PATCH semver, every code id must be unique, and at least one code must be declared.
    Fails closed (`TaxonomyError`) on any violation, mirroring
    `corpus_tools.registry.load_registry`'s own discipline for the fact registry."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    version = doc.get("taxonomy_version")
    if not isinstance(version, str) or not _SEMVER_RE.match(version):
        raise TaxonomyError(
            f"taxonomy_version {version!r} is not a well formed MAJOR.MINOR.PATCH semver"
        )
    raw_codes = doc.get("codes") or []
    codes = tuple(_code(raw) for raw in raw_codes)
    if not codes:
        raise TaxonomyError("taxonomy.yaml declares zero codes")
    ids = [c.id for c in codes]
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        dup = next(i for i in ids if i in seen or seen.add(i))
        raise TaxonomyError(f"duplicate failure code id: {dup!r}")
    return Taxonomy(version=version, codes=codes)


__all__ = ["TAXONOMY_PATH", "FailureCode", "Taxonomy", "TaxonomyError", "load_taxonomy"]
