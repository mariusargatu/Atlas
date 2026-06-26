"""The batch answer generation step (SP8 Task 4, label collection half, pulled early): runs the
real Atlas graph over a label set drawn from SP7's seed cases
(`testing/harness/dataset_tools/seed_cases.jsonl`) and produces question+answer+retrieved_chunks
items -- exactly what a human adjudicator labels on the HITL page.

THE LIVE DEPENDENCY, handled honestly (per the plan's own "code first" directive): generating REAL
answers needs a real retriever (embeddings via TEI, or the hermetic `InMemoryRetriever`'s toy
corpus) and, for anything beyond the toy corpus, a real generation provider (keys in `.env`). This
module supports both without branching on which one is "the real path":

  - `mode="replay"` (the default, hermetic): `GatewayChatModel` replays committed cassettes, zero
    keys, zero egress. The CLI's own defaults (`FIXTURE_SEED_CASES`, `FIXTURE_CASSETTE_DIR`) point
    at a small, committed, clearly marked fixture set (three cases against the toy `CORPUS`), so a
    bare `task label:generate` reproduces `label_items.fixture.jsonl` byte for byte with zero keys
    and zero network -- never a live model or a live index.
  - `mode="record"`/`mode="live"`: `GatewayChatModel` calls a real provider (`replay.providers`,
    the SAME provider selection `server.py` uses). Paired with `retriever="pgvector"` (a running
    `docker compose up` stack, keyless retrieval, slow under Rosetta but fine for a one time batch)
    and `SEED_CASES` (SP7's real seed set), this is the REAL label generation path -- `task
    label:generate-live`, documented in this task's own report, never run by the hermetic gate.

Fixed seed order, no managed queue (D30): `load_seed_cases` returns cases in the SAME order the
JSONL file lists them, never shuffled, never sampled at random -- the SAME order the HITL page
serves items in, so "item 47" means the same thing across a run, a restart, and a rerun.

Never fabricates a label item: a case whose turn does not resolve to a `final_response` (a pending
write proposal awaiting confirmation, or a recursion exhausted turn with only the generic handoff)
is SKIPPED, never padded with invented text. `retrieved_chunks_from_messages` reads the SAME
`KNOWLEDGE_TOOLS` filter `chat_app.py`'s own `_citations_from_messages` uses, so "what counts as a
retrieved chunk" cannot silently drift between the product edge and this lane.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from atlas.domain.binding import KNOWLEDGE_TOOLS
from atlas.orchestration.atlas_graph import thread_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASET_TOOLS = _REPO_ROOT / "testing" / "harness" / "dataset_tools"
SEED_CASES = _DATASET_TOOLS / "seed_cases.jsonl"  # SP7's real seed set, the live generation path
FIXTURE_SEED_CASES = _DATASET_TOOLS / "label_fixture_cases.jsonl"  # three cases against the toy CORPUS
FIXTURE_CASSETTE_DIR = _REPO_ROOT / "testing" / "harness" / "cassettes" / "labeling"
# Cassette key coincidence, harmless by construction (HITL review, fc4d65d): this directory and
# cassettes/e2e/ both hold a same named file because both decode the identical request
# (model_id="claude-test", the single message "Is my plan contract-free?") into two different
# recorded responses. Harmless today because every GatewayChatModel is constructed with one
# explicit cassette_dir and the two directories are never merged; would only matter if a future
# caller pointed a gateway at a merged or wrong directory.
FIXTURE_OUT = _DATASET_TOOLS / "label_items.fixture.jsonl"

_GENERATION_CUSTOMER = "cust_current"  # a seeded identity with no account state dependence: these
# cases are registry/catalog fact lookups (plan names, prices, fee amounts), never account
# specific, so any seeded customer answers identically -- Sarah (cust_current) is the default the
# hermetic suite already leans on for the "no legacy contract" branch.


def load_seed_cases(path: Path, limit: int | None = None) -> list[dict]:
    """Reads cases in FILE ORDER (fixed seed order, D30: no managed queue, no shuffling). `limit`
    takes the first N in that same order, never a random sample. Blank lines are skipped."""
    cases: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases[:limit] if limit is not None else cases


def _decode_score(value) -> float:
    """`score` arrives through `serialize_tool_result` (`determinism.canonical.canonical_json`),
    which tags a float as `"F:<repr>"` (and a Decimal as `"D:<value>"`) so a Decimal and a float
    that print alike never key apart (`canonical()`'s own docstring). Nothing downstream of the
    knowledge tool ever needed to read `score` back before this module (`chat_app.py`'s own
    `_citations_from_messages` only reads `doc_id`/`entity_ids`), so this is the first reader that
    untags it -- a plain float pass through for anything not tagged, so a hand built test fixture
    that never went through canonicalization still works."""
    if isinstance(value, str) and (value.startswith("F:") or value.startswith("D:")):
        try:
            return float(value[2:])
        except ValueError:
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def retrieved_chunks_from_messages(messages) -> list[dict]:
    """The knowledge tool's own passages (`doc_id`/`chunk_id`/`score`/`text`), deduplicated by
    `(doc_id, chunk_id)` across every `search_knowledge` call in the turn. Mirrors `chat_app.py`'s
    own `_citations_from_messages` filter (`KNOWLEDGE_TOOLS`), kept intentionally richer here (full
    `text`, not just `doc_id`/`entity_ids`): the HITL page needs the passage TEXT to render and
    highlight, the SSE citation event never carries it."""
    chunks: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for msg in messages:
        if getattr(msg, "name", None) not in KNOWLEDGE_TOOLS:
            continue
        try:
            passages = json.loads(getattr(msg, "content", "") or "")
        except json.JSONDecodeError:
            continue
        if not isinstance(passages, list):
            continue
        for passage in passages:
            if not isinstance(passage, dict):
                continue
            doc_id = passage.get("doc_id")
            chunk_id = passage.get("chunk_id") or ""
            key = (doc_id, chunk_id)
            if not doc_id or key in seen:
                continue
            seen.add(key)
            chunks.append({
                "doc_id": doc_id, "chunk_id": chunk_id, "text": passage.get("text", ""),
                "score": _decode_score(passage.get("score", 0.0)),
            })
    return chunks


def build_generation_graph(mode: str, cassette_dir: Path, *, retriever=None):
    """Builds one fresh Atlas graph for batch label generation. `mode` follows `server.py`'s own
    `ATLAS_MODE` vocabulary (`replay`/`record`/`live`); `retriever` defaults to `None`, which
    `build_atlas_graph` itself resolves to the hermetic `InMemoryRetriever` -- an operator wanting
    the real ~200 item set passes `select_retriever("pgvector")` explicitly (this function never
    reads `ATLAS_RETRIEVER` itself, so a hermetic caller can never accidentally reach the network
    through an inherited env var). Returns `(graph, tracer)`: an `InMemoryTracer` so every turn gets
    a real, unique `trace_root` (never the `NullTracer` sentinel `chat_app.py` has to fall back
    around) -- the label item's `trace_id` IS this value, `str()`-ed, the same identity a served
    turn's `message_start` event would carry."""
    from determinism.checkpointer import new_checkpointer
    from determinism.sources import IdFactory
    from replay.gateway import GatewayChatModel
    from tracing import InMemoryTracer

    from atlas.domain.actions import ActionsBackend
    from atlas.orchestration.atlas_graph import build_atlas_graph

    tracer = InMemoryTracer()
    if mode == "replay":
        gw = GatewayChatModel(model_id="claude-test", cassette_dir=Path(cassette_dir), mode="replay")
    else:
        from replay.providers import build_chat_model, provider_tag

        gw = GatewayChatModel(
            model_id=provider_tag(), cassette_dir=Path(cassette_dir), mode=mode, inner=build_chat_model()
        )
    backend = ActionsBackend(IdFactory("ref"))
    graph = build_atlas_graph(
        gw, IdFactory("idem"), backend, new_checkpointer(), retriever=retriever, tracer=tracer
    )
    return graph, tracer


async def generate_label_items(
    graph, cases: list[dict], *, customer_id: str = _GENERATION_CUSTOMER, source: str | None = None
) -> list[dict]:
    """Runs `graph` over `cases` IN ORDER (fixed seed order, carried straight through from
    `load_seed_cases`) and returns one well formed label item per case that actually answers:

        {"case_id", "trace_id", "question", "answer", "retrieved_chunks", "registry_facts"}

    A case whose turn ends without a `final_response` (a pending write, or a truncated turn) is
    skipped -- see this module's own docstring, "never fabricates a label item." `registry_facts`
    passes `expected_facts` straight through from the case (SP7's dataset contract field), values
    stringified so the HITL page can highlight them against retrieved chunk text uniformly.

    `source`, when given, is stamped on every item as `"source"` -- how the committed fixture
    (`label_items.fixture.jsonl`) marks itself `"fixture"` so nothing downstream can mistake a
    canned demo item for a real one. `None` (the default, the real generation path) omits the
    field entirely."""
    items: list[dict] = []
    for case in cases:
        question = case["turns"][0]["user"]
        state = {"messages": [HumanMessage(question)], "session": {"customer_id": customer_id}}
        result = await graph.ainvoke(state, thread_config(f"label::{case['case_id']}"))
        answer = result.get("final_response")
        if answer is None:
            continue
        trace_id = str(result.get("trace_root"))
        item = {
            "case_id": case["case_id"],
            "trace_id": trace_id,
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks_from_messages(result.get("messages") or []),
            "registry_facts": [
                {"fact_id": f["fact_id"], "value": str(f["value"])} for f in case.get("expected_facts") or []
            ],
        }
        if source is not None:
            item["source"] = source
        items.append(item)
    return items


def write_label_items(items: list[dict], out_path: Path) -> None:
    """One JSON object per line, sorted keys, no whitespace drift -- deterministic bytes from the
    same input list every time (the "byte reproducible" property this file's own consumers, the
    backend label route and its hermetic tests, depend on). Deliberately PLAIN `json.dumps`, not
    `determinism.canonical.canonical_json`: that helper's float/Decimal type tagging
    (`"F:0.0"`/`"D:35.00"`) exists for the cassette key's WORM digest, where a Decimal and a float
    that print alike must never collide -- `retrieved_chunks_from_messages` already untags `score`
    (`_decode_score`) back to a plain float BEFORE it reaches this function, so tagging it again here
    would undo that and hand the backend's `score: float` field a string it cannot parse."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
            fh.write("\n")


__all__ = [
    "FIXTURE_CASSETTE_DIR",
    "FIXTURE_OUT",
    "FIXTURE_SEED_CASES",
    "SEED_CASES",
    "build_generation_graph",
    "generate_label_items",
    "load_seed_cases",
    "retrieved_chunks_from_messages",
    "write_label_items",
]
