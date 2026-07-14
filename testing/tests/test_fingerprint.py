"""EmbeddingFingerprint and content addressed index identity (HLD D9).

`EmbeddingFingerprint` is the fail closed check between an index build and the model that served
it: `from_models_lock` loads the pinned model facts (never `server_version`, which only exists at
runtime), `fingerprint_hash` digests the identity fields only, and `index_build_id`/`index_name`
turn `(corpus_version, chunker_hash, fingerprint, index_params)` into the index's content addressed
identity and its human readable rendering.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from rag_tools.fingerprint import (
    EmbeddingFingerprint,
    fingerprint_hash,
    from_models_lock,
    index_build_id,
    index_name,
)

REAL_LOCK_PATH = Path("models.lock")


def _write_lock(
    tmp_path: Path, *, revision: str, model_id: str = "BAAI/bge-m3", provider: str = "local-tei"
) -> Path:
    lock = {
        "embedding": [
            {
                "provider": provider,
                "model_id": model_id,
                "revision": revision,
                "dim": 1024,
                "normalize": True,
                "query_prefix": "",
                "document_prefix": "",
            }
        ],
        "reranker": [],
        "generator": [],
    }
    path = tmp_path / "models.lock"
    path.write_text(json.dumps(lock))
    return path


def _fingerprint(**overrides) -> EmbeddingFingerprint:
    defaults = dict(
        model_id="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        dim=1024,
        normalize=True,
        query_prefix="",
        document_prefix="",
        provider="local-tei",
        server_version=None,
    )
    defaults.update(overrides)
    return EmbeddingFingerprint(**defaults)


# --- from_models_lock -----------------------------------------------------------------------------


def test_from_models_lock_loads_the_real_bge_m3_entry() -> None:
    fp = from_models_lock(REAL_LOCK_PATH, "BAAI/bge-m3")
    assert fp.model_id == "BAAI/bge-m3"
    assert fp.revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert fp.dim == 1024
    assert fp.normalize is True
    assert fp.query_prefix == ""
    assert fp.document_prefix == ""
    assert fp.provider == "local-tei"


def test_from_models_lock_leaves_server_version_none() -> None:
    fp = from_models_lock(REAL_LOCK_PATH, "BAAI/bge-m3")
    assert fp.server_version is None


def test_from_models_lock_rejects_latest_alias(tmp_path) -> None:
    path = _write_lock(tmp_path, revision="latest")
    with pytest.raises(ValueError, match="latest"):
        from_models_lock(path, "BAAI/bge-m3")


def test_from_models_lock_rejects_main_alias(tmp_path) -> None:
    path = _write_lock(tmp_path, revision="main")
    with pytest.raises(ValueError, match="main"):
        from_models_lock(path, "BAAI/bge-m3")


def test_from_models_lock_rejects_unknown_model_id(tmp_path) -> None:
    path = _write_lock(tmp_path, revision="5617a9f61b028005a4858fdac845db406aefb181")
    with pytest.raises(ValueError, match="unknown-model"):
        from_models_lock(path, "unknown-model")


def test_from_models_lock_accepts_a_real_40_hex_revision(tmp_path) -> None:
    path = _write_lock(tmp_path, revision="5617a9f61b028005a4858fdac845db406aefb181")
    fp = from_models_lock(path, "BAAI/bge-m3")
    assert fp.revision == "5617a9f61b028005a4858fdac845db406aefb181"


# --- from_models_lock: the API embedder shape (SP9 task 3, no git sha to pin to) --------------------


def test_from_models_lock_accepts_the_api_shape_revision_equal_to_model_id(tmp_path) -> None:
    path = _write_lock(
        tmp_path, model_id="text-embedding-3-small", revision="text-embedding-3-small", provider="openai"
    )
    fp = from_models_lock(path, "text-embedding-3-small")
    assert fp.revision == "text-embedding-3-small"
    assert fp.provider == "openai"


def test_from_models_lock_rejects_latest_alias_for_an_api_provider(tmp_path) -> None:
    path = _write_lock(tmp_path, model_id="text-embedding-3-small", revision="latest", provider="openai")
    with pytest.raises(ValueError, match="latest"):
        from_models_lock(path, "text-embedding-3-small")


def test_from_models_lock_rejects_main_alias_for_an_api_provider(tmp_path) -> None:
    path = _write_lock(tmp_path, model_id="text-embedding-3-small", revision="main", provider="openai")
    with pytest.raises(ValueError, match="main"):
        from_models_lock(path, "text-embedding-3-small")


def test_from_models_lock_rejects_an_api_revision_that_does_not_match_model_id(tmp_path) -> None:
    path = _write_lock(
        tmp_path, model_id="text-embedding-3-small", revision="text-embedding-3-large", provider="openai"
    )
    with pytest.raises(ValueError, match="revision == model_id"):
        from_models_lock(path, "text-embedding-3-small")


def test_from_models_lock_still_rejects_a_non_40_hex_revision_for_local_tei(tmp_path) -> None:
    # An API-shape revision (self-pin) is only accepted for a non local-tei provider; a local-tei
    # entry pinned to anything other than a real 40 hex sha still fails closed exactly as before.
    path = _write_lock(tmp_path, revision="BAAI/bge-m3")  # provider defaults to local-tei
    with pytest.raises(ValueError, match="40 lowercase"):
        from_models_lock(path, "BAAI/bge-m3")


# --- the real committed models.lock: both embedders load and stay distinct (SP9 task 3) -------------


def test_from_models_lock_loads_the_real_openai_text_embedding_3_small_entry() -> None:
    fp = from_models_lock(REAL_LOCK_PATH, "text-embedding-3-small")
    assert fp.model_id == "text-embedding-3-small"
    assert fp.revision == "text-embedding-3-small"
    assert fp.provider == "openai"
    assert fp.dim == 1536


def test_embedding_fingerprint_distinguishes_bge_m3_from_the_openai_embedder() -> None:
    bge_m3 = from_models_lock(REAL_LOCK_PATH, "BAAI/bge-m3")
    openai_small = from_models_lock(REAL_LOCK_PATH, "text-embedding-3-small")
    assert fingerprint_hash(bge_m3) != fingerprint_hash(openai_small)
    assert index_build_id("corpus-0.1.1", "chunkerhash123", bge_m3, {}) != index_build_id(
        "corpus-0.1.1", "chunkerhash123", openai_small, {}
    )


# --- fingerprint_hash: stability + per field sensitivity --------------------------------------------


def test_fingerprint_hash_is_stable_across_repeated_calls() -> None:
    fp = _fingerprint()
    assert fingerprint_hash(fp) == fingerprint_hash(fp)


def test_fingerprint_hash_is_16_hex_chars() -> None:
    digest = fingerprint_hash(_fingerprint())
    assert len(digest) == 16
    int(digest, 16)  # raises if not hex


def test_fingerprint_hash_flips_with_model_id() -> None:
    baseline = fingerprint_hash(_fingerprint())
    changed = fingerprint_hash(_fingerprint(model_id="BAAI/bge-reranker-v2-m3"))
    assert baseline != changed


def test_fingerprint_hash_flips_with_revision() -> None:
    baseline = fingerprint_hash(_fingerprint())
    changed = fingerprint_hash(_fingerprint(revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"))
    assert baseline != changed


def test_fingerprint_hash_flips_with_dim() -> None:
    baseline = fingerprint_hash(_fingerprint())
    changed = fingerprint_hash(_fingerprint(dim=768))
    assert baseline != changed


def test_fingerprint_hash_flips_with_normalize() -> None:
    baseline = fingerprint_hash(_fingerprint())
    changed = fingerprint_hash(_fingerprint(normalize=False))
    assert baseline != changed


def test_fingerprint_hash_flips_with_query_prefix() -> None:
    baseline = fingerprint_hash(_fingerprint())
    changed = fingerprint_hash(_fingerprint(query_prefix="query: "))
    assert baseline != changed


def test_fingerprint_hash_flips_with_document_prefix() -> None:
    baseline = fingerprint_hash(_fingerprint())
    changed = fingerprint_hash(_fingerprint(document_prefix="passage: "))
    assert baseline != changed


def test_fingerprint_hash_flips_with_provider() -> None:
    baseline = fingerprint_hash(_fingerprint())
    changed = fingerprint_hash(_fingerprint(provider="api-voyage"))
    assert baseline != changed


def test_fingerprint_hash_excludes_server_version() -> None:
    a = _fingerprint(server_version=None)
    b = _fingerprint(server_version="tei-1.5.0")
    assert fingerprint_hash(a) == fingerprint_hash(b)


# --- index_build_id ---------------------------------------------------------------------------------


def test_index_build_id_is_stable_across_repeated_calls() -> None:
    fp = _fingerprint()
    params = {"m": 16, "ef_construction": 128}
    assert index_build_id("corpus-0.1.1", "chunkerhash123", fp, params) == index_build_id(
        "corpus-0.1.1", "chunkerhash123", fp, params
    )


def test_index_build_id_is_16_hex_chars() -> None:
    digest = index_build_id("corpus-0.1.1", "chunkerhash123", _fingerprint(), {})
    assert len(digest) == 16
    int(digest, 16)


def test_index_build_id_flips_with_corpus_version() -> None:
    fp = _fingerprint()
    baseline = index_build_id("corpus-0.1.1", "chunkerhash123", fp, {})
    changed = index_build_id("corpus-9.9.9", "chunkerhash123", fp, {})
    assert baseline != changed


def test_index_build_id_flips_with_chunker_hash() -> None:
    fp = _fingerprint()
    baseline = index_build_id("corpus-0.1.1", "chunkerhash123", fp, {})
    changed = index_build_id("corpus-0.1.1", "chunkerhashXYZ", fp, {})
    assert baseline != changed


def test_index_build_id_flips_with_fingerprint_model_id() -> None:
    baseline = index_build_id("corpus-0.1.1", "chunkerhash123", _fingerprint(), {})
    changed = index_build_id(
        "corpus-0.1.1", "chunkerhash123", _fingerprint(model_id="BAAI/bge-reranker-v2-m3"), {}
    )
    assert baseline != changed


def test_index_build_id_flips_with_fingerprint_revision() -> None:
    baseline = index_build_id("corpus-0.1.1", "chunkerhash123", _fingerprint(), {})
    changed = index_build_id(
        "corpus-0.1.1",
        "chunkerhash123",
        _fingerprint(revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
        {},
    )
    assert baseline != changed


def test_index_build_id_flips_with_index_params() -> None:
    fp = _fingerprint()
    baseline = index_build_id("corpus-0.1.1", "chunkerhash123", fp, {"m": 16})
    changed = index_build_id("corpus-0.1.1", "chunkerhash123", fp, {"m": 32})
    assert baseline != changed


def test_index_build_id_is_insensitive_to_index_params_key_order() -> None:
    fp = _fingerprint()
    a = index_build_id("corpus-0.1.1", "chunkerhash123", fp, {"m": 16, "ef_construction": 128})
    b = index_build_id("corpus-0.1.1", "chunkerhash123", fp, {"ef_construction": 128, "m": 16})
    assert a == b


def test_index_build_id_ignores_fingerprint_server_version() -> None:
    fp_a = _fingerprint(server_version=None)
    fp_b = _fingerprint(server_version="tei-1.5.0")
    assert index_build_id("corpus-0.1.1", "chunkerhash123", fp_a, {}) == index_build_id(
        "corpus-0.1.1", "chunkerhash123", fp_b, {}
    )


# --- index_name --------------------------------------------------------------------------------------


def test_index_name_format_on_the_real_bge_m3_example() -> None:
    name = index_name("corpus-0.1.1", "BAAI/bge-m3", "abcdef1234567890")
    assert name == "corpus-0.1.1-bge-m3-abcdef12"


def test_index_name_collapses_non_alphanumerics_in_model_short_name() -> None:
    name = index_name("corpus-0.1.1", "voyage-ai/voyage-4-lite", "abcdef1234567890")
    assert name == "corpus-0.1.1-voyage-4-lite-abcdef12"


def test_index_name_lowercases_the_model_short_name() -> None:
    name = index_name("corpus-0.1.1", "BAAI/BGE-M3", "abcdef1234567890")
    assert name == "corpus-0.1.1-bge-m3-abcdef12"


def test_index_name_uses_only_the_first_eight_chunker_hash_chars() -> None:
    name = index_name("corpus-0.1.1", "BAAI/bge-m3", "abcdef12ZZZZZZZZ")
    assert name.endswith("abcdef12")
    assert "ZZZZZZZZ" not in name
