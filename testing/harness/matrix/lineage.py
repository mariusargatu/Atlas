"""Every matrix cell's lineage row, in EXACTLY `contracts/manifest/schema.json`'s shape (D26's 12
field attribution tuple). This module is a caller of that already frozen contract, never a second,
matrix specific lineage shape: `contract_tools.loader.LINEAGE_FIELDS` is the single source of truth
for the field NAMES, and `contract_tools.loader.contract_versions()` is the single source of truth
for the `contract_versions` sub object every row also carries.

The schema requires `model_snapshot` and `embedding_model` as populated OBJECTS on every row (only
`judge_id` is nullable per the schema), so a retrieval only cell (stage 1/2, no generation model) and the
BM25 lexical baseline (no real embedder at all) both need an honest placeholder rather than a bare
`None` the schema would reject. `NOT_APPLICABLE_MODEL_SNAPSHOT`/`NOT_APPLICABLE_EMBEDDING_MODEL` are
that placeholder, named plainly ("not-applicable") so a reader can never mistake either for a real
pinned model -- the same "never silent" doctrine the degradation ladder and the contract narrowing
rules already apply elsewhere in this repo, applied here to an unavoidable schema shaped gap instead.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Optional

from contract_tools.loader import LINEAGE_FIELDS, contract_versions

#: The schema's own `git_sha` pattern (`contracts/manifest/schema.json`), restated here (not
#: parsed again from the schema file at every call) so a malformed git_sha fails fast, at construction,
#: rather than surfacing later as an opaque `jsonschema.ValidationError` far from the call site.
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")

NOT_APPLICABLE = "not-applicable"
NOT_APPLICABLE_MODEL_SNAPSHOT: dict = {
    "provider": "none",
    "model_id": NOT_APPLICABLE,
    "revision": NOT_APPLICABLE,
}
NOT_APPLICABLE_EMBEDDING_MODEL: dict = {"id": NOT_APPLICABLE, "revision": NOT_APPLICABLE}


def build_manifest_row(
    *,
    run_id: str,
    git_sha: str,
    prompt_hash: str,
    model_snapshot: Optional[Mapping[str, str]],
    request_params: Mapping[str, object],
    embedding_model: Optional[Mapping[str, str]],
    index_build_id: str,
    corpus_version: str,
    chunker_config_hash: str,
    retrieval_config: Mapping[str, object],
    dataset_version: str,
    judge_id: Optional[str],
    taxonomy_version: Optional[str] = None,
) -> dict:
    """One cell's lineage record, in exactly `contracts/manifest/schema.json`'s shape.

    `model_snapshot`/`embedding_model` default to the documented not-applicable sentinel when the
    caller has no real value for them (a retrieval only stage has no generation model; the BM25
    lexical baseline has no real embedder); `judge_id` stays a bare `None` when absent (the ONE
    field the schema itself declares nullable). Raises `ValueError` on a malformed `git_sha` rather
    than emitting a row the contract would reject downstream.
    """
    if not GIT_SHA_PATTERN.match(git_sha):
        raise ValueError(f"git_sha must match {GIT_SHA_PATTERN.pattern!r}, got {git_sha!r}")
    row: dict = {
        "run_id": run_id,
        "git_sha": git_sha,
        "prompt_hash": prompt_hash,
        "model_snapshot": dict(model_snapshot) if model_snapshot is not None else dict(NOT_APPLICABLE_MODEL_SNAPSHOT),
        "request_params": dict(request_params),
        "embedding_model": dict(embedding_model) if embedding_model is not None else dict(NOT_APPLICABLE_EMBEDDING_MODEL),
        "index_build_id": index_build_id,
        "corpus_version": corpus_version,
        "chunker_config_hash": chunker_config_hash,
        "retrieval_config": dict(retrieval_config),
        "dataset_version": dataset_version,
        "judge_id": judge_id,
        "contract_versions": contract_versions(),
    }
    if taxonomy_version is not None:
        row["taxonomy_version"] = taxonomy_version
    missing = set(LINEAGE_FIELDS) - set(row)
    assert not missing, f"lineage row dropped D26 field(s): {sorted(missing)}"  # never silently short
    return row


__all__ = [
    "GIT_SHA_PATTERN",
    "NOT_APPLICABLE",
    "NOT_APPLICABLE_EMBEDDING_MODEL",
    "NOT_APPLICABLE_MODEL_SNAPSHOT",
    "build_manifest_row",
]
