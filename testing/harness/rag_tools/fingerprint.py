"""EmbeddingFingerprint and content addressed index identity (HLD D9).

`EmbeddingFingerprint` is the fail closed check between an index build and the model that served
it: model id, revision, dim, normalize, and prefixes are pinned in `models.lock` (D26 alias
rejection discipline extended to embedding models, per the SP3 planning digest section 7 open
decision 5) and loaded once via `from_models_lock`; `server_version` is filled in later, at query
time, from the live TEI server's own version string, and is deliberately excluded from the identity
hash below, because a server patch version bump is not a model identity change.

`index_build_id` and `index_name` turn `(corpus_version, chunker_hash, fingerprint, index_params)`
into an index build's content addressed identity (`index_build_id`) and a human readable rendering
of the same inputs (`index_name`). Builds are immutable; the active index is selected by pinning
`index_build_id` in typed settings, not by name.

**API embedder shape (SP9 task 3).** `local-tei` entries pin a real 40 hex Hugging Face commit sha,
the only identity a self-hosted model has. An API embedder (e.g. OpenAI `text-embedding-3-small`)
has no git sha to pin to at all, so the documented shape for a non `local-tei` provider is: the
pinned identity IS the model id string itself, and `revision` must equal `model_id` exactly (a
self pin). The alias rejection discipline still applies in full to that self pin (`"latest"`/
`"main"` are rejected exactly as they are for a TEI revision); what changes is which string plays
the pinned-identity role, never whether floating is allowed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_HASH_HEX_LEN = 16
_ALIAS_REVISIONS = frozenset({"latest", "main"})
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class EmbeddingFingerprint:
    """The identity of one embedding model as actually pinned and configured.

    `server_version` is not part of the model's identity (it is filled in from the live serving
    stack at query time, `None` at load); `fingerprint_hash` excludes it on purpose so a server
    restart with a new TEI patch version does not silently invalidate every index built against the
    same pinned model.
    """

    model_id: str
    revision: str
    dim: int
    normalize: bool
    query_prefix: str
    document_prefix: str
    provider: str
    server_version: str | None = None


def from_models_lock(path: str | Path, model_id: str) -> EmbeddingFingerprint:
    """Load one `embedding` entry from `models.lock` by `model_id`. `server_version` is always
    `None` on load (nothing in the lock file names a live server); the caller fills it in from the
    running TEI instance, if and when one exists.

    Fails closed (D26/D18 discipline): an entry missing entirely, or pinned to a floating alias
    revision (`"latest"`, `"main"`) instead of a real 40 hex Hugging Face commit sha, raises
    `ValueError` rather than silently loading an unpinned model.
    """
    data = json.loads(Path(path).read_text())
    entry = next((e for e in data.get("embedding", []) if e["model_id"] == model_id), None)
    if entry is None:
        raise ValueError(f"models.lock ({path}) has no embedding entry for model_id={model_id!r}")

    revision = entry["revision"]
    _validate_revision(revision, model_id, entry["provider"])

    return EmbeddingFingerprint(
        model_id=entry["model_id"],
        revision=revision,
        dim=entry["dim"],
        normalize=entry["normalize"],
        query_prefix=entry["query_prefix"],
        document_prefix=entry["document_prefix"],
        provider=entry["provider"],
        server_version=None,
    )


def _validate_revision(revision: str, model_id: str, provider: str) -> None:
    """`local-tei`: the existing D26/D18 discipline unchanged (a real 40 hex Hugging Face commit
    sha, never `"latest"`/`"main"`). Any other provider (an API embedder, no git sha to pin to at
    all): the documented shape is `revision == model_id` (the model id string IS the pin), with the
    exact same alias rejection applied to that self pin -- see the module docstring."""
    if provider == "local-tei":
        if revision in _ALIAS_REVISIONS:
            raise ValueError(
                f"models.lock pins {model_id!r} to alias revision {revision!r}; pin a real 40 hex "
                "Hugging Face commit sha instead (D26/D18: aliases float, pins do not)"
            )
        if not _REVISION_RE.match(revision):
            raise ValueError(
                f"models.lock pins {model_id!r} to revision {revision!r}, which is not a 40 lowercase "
                "hex character Hugging Face commit sha"
            )
        return

    if revision in _ALIAS_REVISIONS:
        raise ValueError(
            f"models.lock pins {model_id!r} ({provider}) to alias revision {revision!r}; an API "
            "embedder has no git sha, so pin the exact API model id string instead (D26/D18: "
            "aliases float, pins do not, whichever string plays the pinned-identity role)"
        )
    if revision != model_id:
        raise ValueError(
            f"models.lock pins {model_id!r} ({provider}) to revision {revision!r}; a non local-tei "
            "provider has no git sha to pin to, so the documented shape is revision == model_id "
            "(the model id string itself is the pin, see fingerprint.py's module docstring)"
        )


def fingerprint_hash(fp: EmbeddingFingerprint) -> str:
    """sha256 over the identity fields only, joined with `\"|\"`, first 16 hex chars.
    `server_version` is excluded on purpose: two fingerprints differing only there hash identically."""
    fields = (
        fp.model_id,
        fp.revision,
        str(fp.dim),
        str(fp.normalize),
        fp.query_prefix,
        fp.document_prefix,
        fp.provider,
    )
    return _sha256_16("|".join(fields))


def index_build_id(
    corpus_version: str,
    chunker_hash: str,
    fp: EmbeddingFingerprint,
    index_params: dict,
) -> str:
    """sha256(corpus_version | chunker_hash | fp.model_id | fp.revision |
    json.dumps(index_params, sort_keys=True)), first 16 hex chars. An index build's canonical,
    content addressed identity (D9): any change to the corpus, the chunker, the pinned embedding
    model/revision, or the index build parameters yields a different id."""
    fields = (
        corpus_version,
        chunker_hash,
        fp.model_id,
        fp.revision,
        json.dumps(index_params, sort_keys=True),
    )
    return _sha256_16("|".join(fields))


def index_name(corpus_version: str, model_id: str, chunker_hash: str) -> str:
    """`{corpus_version}-{model_short}-{chunker_hash[:8]}`, a human readable rendering of
    `index_build_id`'s inputs (never the identity itself). `model_short` is the model name after
    the slash, lowercased, with every run of non alphanumeric characters collapsed to one hyphen
    (e.g. `BAAI/bge-m3` -> `bge-m3`)."""
    return f"{corpus_version}-{_model_short_name(model_id)}-{chunker_hash[:8]}"


def _model_short_name(model_id: str) -> str:
    name = model_id.rsplit("/", 1)[-1].lower()
    return _NON_ALNUM_RE.sub("-", name).strip("-")


def _sha256_16(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_HASH_HEX_LEN]


__all__ = [
    "EmbeddingFingerprint",
    "fingerprint_hash",
    "from_models_lock",
    "index_build_id",
    "index_name",
]
