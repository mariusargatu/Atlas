"""One source for everything the test suite needs to know about the COMMITTED corpus and index.

Sibling to `catalog_expectations.py`, but built on the opposite principle, deliberately. The
catalog fixture hand types its values so a test can genuinely disagree with production. That works
because `atlas.domain.catalog` is hand authored: there is a human decision to disagree with.

The corpus and the index are not hand authored, they are BUILD OUTPUTS. `manifest.json` and
`build_manifest.json` are committed alongside the artifacts they describe, and they are already the
thing every consumer reads at runtime (`rag_tools.ingest`, `dataset_tools.provenance_join`,
`atlas.config`, `atlas.adapters.pgvector_retriever`). Hand typing `45` or
`corpus-0.1.1-bge-m3-03f983e0` into a test asserts only that a human remembered to edit a literal;
it cannot catch a build whose manifest and artifacts disagree, which is the failure that actually
matters. So this module READS the manifests, and the real invariants are pinned as their own tests:

  * `test_corpus_build.py` -- the manifest agrees with the rendered tree it describes
  * `test_ingest.py`       -- the committed index identity recomputes from its own declared inputs

Registry source paths are re-exported from `corpus_tools.build`, which owns them; nine test modules
used to redeclare the same three `Path(...)` literals.
"""
from __future__ import annotations

import json
from pathlib import Path

from corpus_tools.build import CORE, GENERATED, TEMPLATES

__all__ = [
    "CORE",
    "GENERATED",
    "TEMPLATES",
    "COMMITTED_CORPUS_DIR",
    "COMMITTED_INDEX_DIR",
    "CORPUS_VERSION",
    "CORPUS_CONTENT_HASH",
    "DOC_COUNT",
    "CHUNK_COUNT",
    "INDEX_BUILD_ID",
    "INDEX_PARAMS",
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_REVISION",
    "corpus_manifest",
    "index_manifest",
    "index_fingerprint",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The one built corpus and the one built index committed to this repo. Both are discovered from
#: the filesystem rather than named, so adding a second build does not silently leave every test
#: asserting against the first one by a stale literal.
_RENDERED_ROOT = _REPO_ROOT / "corpus" / "rendered"
_INDEX_ROOT = _REPO_ROOT / "indexes"


def _sole_child(root: Path, what: str) -> Path:
    children = sorted(p for p in root.iterdir() if p.is_dir())
    if len(children) != 1:
        raise AssertionError(
            f"expected exactly one committed {what} under {root}, found {[p.name for p in children]}. "
            f"Update {__name__} to name which one the suite pins."
        )
    return children[0]


COMMITTED_CORPUS_DIR = _sole_child(_RENDERED_ROOT, "corpus")
COMMITTED_INDEX_DIR = _sole_child(_INDEX_ROOT, "index")


def corpus_manifest() -> dict:
    """The committed corpus's own `manifest.json`, the artifact `rag_tools.ingest.load_corpus_docs`
    and `dataset_tools.provenance_join.load_corpus_index` both already read."""
    return json.loads((COMMITTED_CORPUS_DIR / "manifest.json").read_text())


def index_manifest() -> dict:
    """The committed index's own `build_manifest.json`, the artifact `atlas.config.from_env` and
    `PgvectorRetriever._load_build_manifest` both already read."""
    return json.loads((COMMITTED_INDEX_DIR / "build_manifest.json").read_text())


def index_fingerprint() -> dict:
    """The committed index's `fingerprint.json`, which `PgvectorRetriever` verifies against the
    live TEI server's `/info` at construction."""
    return json.loads((COMMITTED_INDEX_DIR / "fingerprint.json").read_text())


_CORPUS = corpus_manifest()
_INDEX = index_manifest()
_FINGERPRINT = index_fingerprint()

CORPUS_VERSION: str = _CORPUS["corpus_version"]
CORPUS_CONTENT_HASH: str = _CORPUS["content_hash"]
DOC_COUNT: int = _CORPUS["doc_count"]

CHUNK_COUNT: int = _INDEX["chunk_count"]
INDEX_BUILD_ID: str = _INDEX["index_build_id"]
INDEX_PARAMS: dict = _INDEX["index_params"]

EMBEDDING_MODEL_ID: str = _FINGERPRINT["model_id"]
EMBEDDING_REVISION: str = _FINGERPRINT["revision"]
