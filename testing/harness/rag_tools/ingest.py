"""The ingestion pipeline (D5, D9): read the rendered corpus, chunk it (`rag_tools.chunker`), embed
every chunk via a live TEI server, L2 normalize per the pinned `EmbeddingFingerprint`, and write the
immutable index build artifact (`chunks.parquet` + `fingerprint.json` + `build_manifest.json`) plus
a Postgres loader that creates the hybrid retrieval schema (pgvector HNSW + a generated tsvector
column) and loads the parquet.

Reads ONLY `docs/` and `provenance/*.json` as chunk content; `manifest.json` is consulted for
`doc_version` (the per doc `content_hash`, per the SP3 digest's open decision 8) and the corpus wide
`content_hash`, never embedded as content itself (the SP2 backlog note this task must not violate).

Exactly two I/O boundaries: `httpx` to TEI (embed, batched, no retries: a batch failure fails the
whole build loud, matching D9's fail closed discipline extended to build time) and `psycopg` to
Postgres (schema + load). Everything else here (schema assembly, manifest field construction,
L2 normalization) is pure and unit tested without either.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from atlas.domain.retrieval import l2_normalize, vector_literal
from rag_tools import chunker
from rag_tools.fingerprint import (
    EmbeddingFingerprint,
    from_models_lock,
    index_build_id,
    index_name,
)

CORPUS_ROOT = Path("corpus/rendered")
INDEX_ROOT = Path("indexes")
MODELS_LOCK = Path("models.lock")
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
INDEX_PARAMS = {"m": 16, "ef_construction": 128}
DEFAULT_TEI_EMBED_URL = "http://localhost:8081"
DEFAULT_POSTGRES_DSN = "postgresql://atlas:atlas-dev-password@localhost:5433/atlas"
EMBED_BATCH_SIZE = 16  # chunks per TEI request: keeps request bodies small and any failure localized


# --- corpus reading (pure: filesystem only, no client) ---------------------------------------------


def load_corpus_docs(corpus_version: str, corpus_root: Path = CORPUS_ROOT) -> tuple[dict, ...]:
    """Read every doc's text + provenance sidecar + `doc_version` (manifest's per doc content_hash)
    for one corpus_version, sorted by `doc_id`. Reads `docs/` and `provenance/*.json` as content;
    `manifest.json` is consulted only for `doc_version`, never embedded as chunk content itself."""
    corpus_dir = corpus_root / corpus_version
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    docs = []
    for doc_id, doc_version in sorted(manifest["docs"].items()):
        text = (corpus_dir / "docs" / f"{doc_id}.txt").read_text()
        sidecar = json.loads((corpus_dir / "provenance" / f"{doc_id}.json").read_text())
        docs.append(
            {
                "doc_id": doc_id,
                "doc_type": sidecar["doc_type"],
                "text": text,
                "doc_version": doc_version,
                "placements": sidecar["placements"],
            }
        )
    return tuple(docs)


def chunk_corpus(corpus_version: str, corpus_root: Path = CORPUS_ROOT) -> tuple[chunker.ChunkRecord, ...]:
    """Chunk every doc in one corpus_version, returning records sorted by `chunk_id` (the parquet's
    committed row order)."""
    records: list[chunker.ChunkRecord] = []
    for doc in load_corpus_docs(corpus_version, corpus_root):
        records.extend(
            chunker.chunk_document(
                doc_id=doc["doc_id"],
                doc_type=doc["doc_type"],
                text=doc["text"],
                doc_version=doc["doc_version"],
                corpus_version=corpus_version,
                placements=doc["placements"],
            )
        )
    return tuple(sorted(records, key=lambda r: r.chunk_id))


# --- embedding (the one client boundary: TEI over HTTP) --------------------------------------------


def fetch_server_version(base_url: str, *, client: httpx.Client | None = None) -> str:
    """GET /info and return TEI's own `version` string. Fails loud (raise_for_status): a build
    against a TEI server that can't answer /info should not silently record `server_version=None`."""
    owns_client = client is None
    client = client or httpx.Client(base_url=base_url, timeout=30.0)
    try:
        response = client.get("/info")
        response.raise_for_status()
        return response.json()["version"]
    finally:
        if owns_client:
            client.close()


def embed_texts(
    base_url: str,
    texts: Sequence[str],
    *,
    batch_size: int = EMBED_BATCH_SIZE,
    client: httpx.Client | None = None,
) -> list[list[float]]:
    """POST /embed in fixed size batches, in input order. No retries: `raise_for_status` on any
    batch failure propagates immediately (fail loud, per the plan's global constraints), rather than
    silently skipping or padding a failed batch with zeros."""
    owns_client = client is None
    client = client or httpx.Client(base_url=base_url, timeout=120.0)
    try:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            response = client.post("/embed", json={"inputs": batch})
            response.raise_for_status()
            vectors.extend(response.json())
        return vectors
    finally:
        if owns_client:
            client.close()


# --- parquet assembly (pure: pyarrow only, no client) -----------------------------------------------


def parquet_schema(dim: int) -> pa.Schema:
    """Every `ChunkRecord` field, in declaration order, plus `embedding` as a fixed size list column
    (`dim` floats, float32) appended last."""
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("parent_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("doc_version", pa.string()),
            pa.field("doc_type", pa.string()),
            pa.field("heading_path", pa.list_(pa.string())),
            pa.field("char_span", pa.list_(pa.int64(), 2)),
            pa.field("token_count", pa.int64()),
            pa.field("content_hash", pa.string()),
            pa.field("entity_ids", pa.list_(pa.string())),
            pa.field("chunker_version", pa.string()),
            pa.field("corpus_version", pa.string()),
            pa.field("doc_title", pa.string()),
            pa.field("text", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), dim)),
        ]
    )


def build_table(
    records: Sequence[chunker.ChunkRecord],
    vectors: Sequence[Sequence[float]],
    *,
    dim: int,
) -> pa.Table:
    """Assemble the chunks table: one row per `(record, vector)` pair, in the given order (the
    caller is responsible for chunk_id ordering; this function does not re-sort)."""
    if len(records) != len(vectors):
        raise ValueError(f"records ({len(records)}) and vectors ({len(vectors)}) must be the same length")
    schema = parquet_schema(dim)
    columns: dict[str, list] = {
        "chunk_id": [r.chunk_id for r in records],
        "parent_id": [r.parent_id for r in records],
        "doc_id": [r.doc_id for r in records],
        "doc_version": [r.doc_version for r in records],
        "doc_type": [r.doc_type for r in records],
        "heading_path": [list(r.heading_path) for r in records],
        "char_span": [list(r.char_span) for r in records],
        "token_count": [r.token_count for r in records],
        "content_hash": [r.content_hash for r in records],
        "entity_ids": [list(r.entity_ids) for r in records],
        "chunker_version": [r.chunker_version for r in records],
        "corpus_version": [r.corpus_version for r in records],
        "doc_title": [r.doc_title for r in records],
        "text": [r.text for r in records],
        "embedding": [list(v) for v in vectors],
    }
    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def write_parquet(path: Path, table: pa.Table) -> None:
    pq.write_table(table, path, compression="snappy")


# --- manifest field construction (pure) --------------------------------------------------------------


def build_fingerprint_dict(fp: EmbeddingFingerprint) -> dict:
    return asdict(fp)


def build_index_manifest(
    *,
    corpus_version: str,
    corpus_content_hash: str,
    chunker_hash_value: str,
    fp: EmbeddingFingerprint,
    reranker_lock_entry: Mapping[str, str],
    index_params: Mapping[str, int],
    doc_count: int,
    chunk_count: int,
) -> dict:
    """`build_manifest.json`'s fields: index identity (`index_build_id`), the index params baked
    into the HNSW index, the corpus's content_hash (not the corpus_version string alone: a tag can
    be trusted less than its content), the chunker's own hash, both models.lock entries this build
    was pinned against (embedding, which IS part of `index_build_id`; reranker, which is a
    query time knob recorded here for provenance only), and doc/chunk counts."""
    return {
        "index_build_id": index_build_id(corpus_version, chunker_hash_value, fp, dict(index_params)),
        "index_params": dict(index_params),
        "corpus_version": corpus_version,
        "corpus_content_hash": corpus_content_hash,
        "chunker_hash": chunker_hash_value,
        "chunker_version": chunker.CHUNKER_VERSION,
        "models_lock": {
            "embedding": {"model_id": fp.model_id, "revision": fp.revision},
            "reranker": dict(reranker_lock_entry),
        },
        "doc_count": doc_count,
        "chunk_count": chunk_count,
    }


def load_reranker_lock_entry(models_lock_path: Path, model_id: str) -> dict:
    data = json.loads(Path(models_lock_path).read_text())
    entry = next((e for e in data.get("reranker", []) if e["model_id"] == model_id), None)
    if entry is None:
        raise ValueError(f"models.lock ({models_lock_path}) has no reranker entry for model_id={model_id!r}")
    return {"model_id": entry["model_id"], "revision": entry["revision"]}


# --- full build (the orchestrating function: I/O + the pure pieces above) --------------------------


def build_index(
    *,
    corpus_version: str,
    tei_embed_url: str = DEFAULT_TEI_EMBED_URL,
    corpus_root: Path = CORPUS_ROOT,
    index_root: Path = INDEX_ROOT,
    models_lock_path: Path = MODELS_LOCK,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    reranker_model_id: str = RERANKER_MODEL_ID,
    index_params: Mapping[str, int] = INDEX_PARAMS,
) -> Path:
    """Build one index: chunk the corpus, embed every chunk's `embed_text` via TEI, L2 normalize
    (per the fingerprint's `normalize` flag), and write `chunks.parquet` + `fingerprint.json` +
    `build_manifest.json` under `index_root/{index_name}/`. Returns the output directory."""
    records = chunk_corpus(corpus_version, corpus_root)
    fp = from_models_lock(models_lock_path, embedding_model_id)

    with httpx.Client(base_url=tei_embed_url, timeout=120.0) as client:
        server_version = fetch_server_version(tei_embed_url, client=client)
        raw_vectors = embed_texts(tei_embed_url, [r.embed_text for r in records], client=client)

    vectors = [l2_normalize(v) if fp.normalize else list(v) for v in raw_vectors]
    fp = replace(fp, server_version=server_version)

    chunker_hash_value = chunker.chunker_hash()
    corpus_manifest = json.loads((corpus_root / corpus_version / "manifest.json").read_text())
    reranker_lock_entry = load_reranker_lock_entry(models_lock_path, reranker_model_id)

    out_dir = Path(index_root) / index_name(corpus_version, fp.model_id, chunker_hash_value)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_table(records, vectors, dim=fp.dim)
    write_parquet(out_dir / "chunks.parquet", table)

    (out_dir / "fingerprint.json").write_text(json.dumps(build_fingerprint_dict(fp), indent=2, sort_keys=True) + "\n")

    doc_count = len({r.doc_id for r in records})
    manifest = build_index_manifest(
        corpus_version=corpus_version,
        corpus_content_hash=corpus_manifest["content_hash"],
        chunker_hash_value=chunker_hash_value,
        fp=fp,
        reranker_lock_entry=reranker_lock_entry,
        index_params=index_params,
        doc_count=doc_count,
        chunk_count=len(records),
    )
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return out_dir


# --- Postgres loader: schema (pgvector HNSW + generated tsvector) + parquet load -------------------


_INSERT_SQL = """
    INSERT INTO chunks (
        chunk_id, parent_id, doc_id, doc_version, doc_type, heading_path,
        char_span_start, char_span_end, token_count, content_hash, entity_ids,
        chunker_version, corpus_version, doc_title, text, index_build_id, embedding
    ) VALUES (
        %(chunk_id)s, %(parent_id)s, %(doc_id)s, %(doc_version)s, %(doc_type)s, %(heading_path)s,
        %(char_span_start)s, %(char_span_end)s, %(token_count)s, %(content_hash)s, %(entity_ids)s,
        %(chunker_version)s, %(corpus_version)s, %(doc_title)s, %(text)s, %(index_build_id)s,
        %(embedding)s::vector
    )
    ON CONFLICT (chunk_id) DO NOTHING;
"""
# `ON CONFLICT (chunk_id) DO NOTHING` stays keyed on `chunk_id` alone, not `(chunk_id,
# index_build_id)`, on purpose (SP3 final review, table scoping): a rerun of the SAME build is an
# idempotent no op re insert, and this table can carry more than one build's rows side by side.
# `chunk_id` hashes `(corpus_version, doc_id, doc_version, span)` (`chunker._compute_chunk_id`), so
# it is unique ACROSS builds that differ in corpus_version, but NOT across two builds of the SAME
# corpus_version that differ only in something the hash does not see. An embedding model change is
# the real example: rebuilding the same corpus_version under a different embedding model produces
# an index with a DIFFERENT index_build_id but the IDENTICAL set of chunk_ids, so every row of that
# second build collides with the first build's already loaded rows, `DO NOTHING` keeps the first
# build's rows untouched, and the second build's index_build_id ends up backing zero rows. That is
# a fail empty result for every search against it, not a loud error, unless something checks for
# it. `load_parquet` below runs a post load row count check for exactly this reason.


def create_schema(conn, *, dim: int, index_params: Mapping[str, int]) -> None:
    """Create the hybrid retrieval schema (idempotent: `IF NOT EXISTS` throughout): a `chunks` table
    with a pgvector `vector(dim)` column, an `index_build_id` column scoping every row to the build
    that produced it (SP3 final review: `chunks` is a single physical table that can hold more than
    one index build's rows at once, see the `_INSERT_SQL` comment above), a `tsvector` column
    generated from `text`, a GIN index over the tsvector, a plain btree index over `index_build_id`
    (both SQL arms in `atlas.adapters.pgvector_retriever` filter on it), and an HNSW index over the
    vector column with the given `m`/`ef_construction`.

    `dim`/`m`/`ef_construction` are interpolated directly into the DDL text rather than bound as
    query parameters: Postgres type modifiers and `WITH (...)` index storage params are not
    ordinary bind positions. All three come from this build's own typed, internally sourced
    `EmbeddingFingerprint`/`index_params`, never from external input.
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive int, got {dim!r}")
    m = index_params["m"]
    ef_construction = index_params["ef_construction"]
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id text PRIMARY KEY,
                parent_id text NOT NULL,
                doc_id text NOT NULL,
                doc_version text NOT NULL,
                doc_type text NOT NULL,
                heading_path text[] NOT NULL,
                char_span_start integer NOT NULL,
                char_span_end integer NOT NULL,
                token_count integer NOT NULL,
                content_hash text NOT NULL,
                entity_ids text[] NOT NULL,
                chunker_version text NOT NULL,
                corpus_version text NOT NULL,
                doc_title text NOT NULL,
                text text NOT NULL,
                index_build_id text NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
                embedding vector({dim}) NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);")
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_index_build_id_idx ON chunks (index_build_id);")
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction});
            """
        )
    conn.commit()


def _row_to_insert_params(row: Mapping[str, object], *, build_id: str) -> dict:
    char_span = row["char_span"]
    return {
        "chunk_id": row["chunk_id"],
        "parent_id": row["parent_id"],
        "doc_id": row["doc_id"],
        "doc_version": row["doc_version"],
        "doc_type": row["doc_type"],
        "heading_path": list(row["heading_path"]),
        "char_span_start": char_span[0],
        "char_span_end": char_span[1],
        "token_count": row["token_count"],
        "content_hash": row["content_hash"],
        "entity_ids": list(row["entity_ids"]),
        "chunker_version": row["chunker_version"],
        "corpus_version": row["corpus_version"],
        "doc_title": row["doc_title"],
        "text": row["text"],
        "index_build_id": build_id,
        "embedding": vector_literal(row["embedding"]),
    }


class LoadCountMismatchError(RuntimeError):
    """Raised by `load_parquet` when the post load row count for `build_id` does not match the
    parquet's own row count (see the softened comment above `_INSERT_SQL`): the fail empty
    collision the SP4 task 3 reviewer named, surfaced loud instead of silently serving an index
    that backs zero (or fewer than expected) rows under this build_id."""


def load_parquet(
    conn, parquet_path: Path, *, dim: int, build_id: str, index_params: Mapping[str, int] = INDEX_PARAMS
) -> int:
    """Create the schema (idempotent), then load every row from `parquet_path`, stamping every row's
    `index_build_id` column with `build_id` (SP3 final review: the caller reads this from the same
    index directory's `build_manifest.json`, mirroring how `dim` is read from `fingerprint.json`;
    see `main()` below and the live test fixtures for the read). Returns the row count loaded (==
    the parquet's row count; `ON CONFLICT DO NOTHING` means a rerun over the same chunk_ids reports
    the same count without duplicating rows).

    Post load count check (SP4 task 3): after the insert, counts how many rows actually carry THIS
    `build_id` and compares that against the parquet's own row count. The two numbers can honestly
    differ even on a correct load only if something upstream is already wrong, so this is not a
    fuzzy heuristic: `ON CONFLICT (chunk_id) DO NOTHING` silently keeps an existing row when its
    chunk_id already exists under a DIFFERENT build_id (the comment above `_INSERT_SQL` names the
    real cause, an embedding model only rebuild of the same corpus_version), and without this check
    that collision would leave this build_id's searches silently empty instead of raising."""
    create_schema(conn, dim=dim, index_params=index_params)
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    params = [_row_to_insert_params(row, build_id=build_id) for row in rows]
    with conn.cursor() as cur:
        cur.executemany(_INSERT_SQL, params)
        cur.execute("SELECT count(*) FROM chunks WHERE index_build_id = %(build_id)s;", {"build_id": build_id})
        (loaded_count,) = cur.fetchone()
    conn.commit()
    if loaded_count != len(params):
        raise LoadCountMismatchError(
            f"Loaded {loaded_count} row(s) carrying index_build_id={build_id!r} after inserting from "
            f"{parquet_path}, expected {len(params)} (the parquet's own row count). ON CONFLICT "
            "(chunk_id) DO NOTHING silently keeps an existing row when its chunk_id already exists "
            "under a DIFFERENT index_build_id, the known cause being a prior build of the same "
            "corpus_version under a different embedding model; that would otherwise leave this "
            "build_id's searches silently empty instead of raising. Investigate the collision "
            "before serving from this build_id."
        )
    return loaded_count


# --- CLI: `task rag:ingest` -------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the retrieval index and load it into Postgres.")
    parser.add_argument("--corpus-version", default="corpus-0.1.1")
    parser.add_argument("--tei-embed-url", default=DEFAULT_TEI_EMBED_URL)
    parser.add_argument("--postgres-dsn", default=DEFAULT_POSTGRES_DSN)
    parser.add_argument("--index-root", default=str(INDEX_ROOT))
    parser.add_argument("--skip-load", action="store_true", help="build the artifact only, skip the Postgres load")
    parser.add_argument(
        "--load-existing", metavar="INDEX_DIR", default=None,
        help="skip the build entirely (no TEI call, no corpus/rendered read) and load an "
             "already-built index directory's chunks.parquet straight into Postgres (idempotent: "
             "IF NOT EXISTS schema, ON CONFLICT DO NOTHING inserts). This is what the compose init "
             "service runs against the committed indexes/<name>/ tree, so the served backend never "
             "has to build anything at startup.",
    )
    args = parser.parse_args(argv)

    if args.load_existing:
        out_dir = Path(args.load_existing)
    else:
        out_dir = build_index(
            corpus_version=args.corpus_version,
            tei_embed_url=args.tei_embed_url,
            index_root=Path(args.index_root),
        )
        print(f"wrote index build to {out_dir}")

    if not args.skip_load:
        import psycopg

        fp = json.loads((out_dir / "fingerprint.json").read_text())
        manifest = json.loads((out_dir / "build_manifest.json").read_text())
        with psycopg.connect(args.postgres_dsn) as conn:
            row_count = load_parquet(conn, out_dir / "chunks.parquet", dim=fp["dim"], build_id=manifest["index_build_id"])
        print(f"loaded {row_count} rows into postgres")


if __name__ == "__main__":
    main()
