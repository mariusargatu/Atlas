"""`task rag:smoke` (SP3 task 7): the D36 tier 2 compose acceptance check.

Asks the Daniel question ("is my plan contract free", the corpus's planted grounding conflict,
SP2's `conflict-daniel-contract`) against a running `docker compose up` stack, and prints exactly
what happened. Four parts, run in this order:

RETRIEVAL (always runs, no key needed): queries the real `PgvectorRetriever` directly, using the
    SAME defaults the adapter itself falls back to when its env vars are unset -- which are the
    compose stack's host published ports (postgres:5433, tei-embed:8081, tei-rerank:8082) and the
    committed index dir. This mirrors the exact call the served backend's `search_knowledge` MCP
    tool makes (`DEPLOYED_K`, a bare `RetrievalConfig()`), so the doc ids printed here are what the
    real chat pipeline's retrieval step would surface too. Real corpus-0.1.1 doc ids come back
    (`doc-...`), never the hermetic toy corpus's (`plan-current-page`, ...).

CHAT ENDPOINT (always attempted): logs in as Daniel (`cust_legacy_term`) and POSTs the question to
    the real served `/chat` (backend/atlas/chat_app.py's actual route, over HTTP). The compose
    backend stays in its default `ATLAS_MODE=replay` (D36 tier 2: no key needed for retrieval), and
    there is no committed cassette for this fresh question, so the turn is EXPECTED to 503 with a
    replay cassette miss -- that miss is itself evidence for the boundary D36 tier 2 draws:
    retrieval is real and free, generation needs either a recorded cassette or a live provider.

CHAT STREAM (SP4 task 6, always attempted, no key needed): the same question over `/chat/stream`
    (the SSE surface), printing the event TYPE sequence it received. Against the compose backend's
    default replay mode, the cassette miss above happens again here, but the streaming contract
    turns it into an IN BAND sequence rather than an HTTP status code: `message_start`, `error`
    (`code=cassette_miss`), `message_end` (`finish_reason=error`) -- the terminal guarantee holding
    live, over the real served endpoint, not just under a replayed fake in the hermetic suite.

GENERATION (only if a provider key is present in `.env`): since dcc76a1 the agent graph DOES bind
    the intent scoped tool surface to a live model (`_generate_message` in live/record mode), so a
    full live agentic turn through `/chat` belongs to the live suite, which exercises it with real
    tool_calls. This half keeps a narrower, cheaper check on purpose. Instead: take
    the SAME retrieved passages from the retrieval half and ask the live provider a single grounded
    completion over them, `max_tokens` tiny -- genuine live generation, grounded in real local
    retrieval, which is the thing D36 tier 2's "generation is the only keyed step" is asking to
    prove.

This script never reads `.env` itself and never prints a key's VALUE: `task rag:smoke` sources
`.env` into the process environment via Task's own `dotenv:` loading, and this script only checks
whether a recognized provider key env var is non-empty.
"""
from __future__ import annotations

import json
import os

import httpx

from atlas.adapters.pgvector_retriever import PgvectorRetriever
from atlas.domain.retrieval import RetrievalConfig
from atlas.mcp_servers.knowledge_server import DEPLOYED_K

BASE_URL = os.environ.get("ATLAS_SMOKE_BASE_URL", "http://localhost:8000")
CUSTOMER_ID = "cust_legacy_term"  # Daniel
QUESTION = "is my plan contract free"
# DEPLOYED_K is single sourced from atlas.mcp_servers.knowledge_server (SP4 task 1): this script
# already imports PgvectorRetriever straight from `atlas.adapters` below, so importing the constant
# alongside it is no new dependency direction (harness may import backend code, only the reverse is
# lint-forbidden; see test_import_lint.py). Previously a hand kept in sync literal; the SP3 final
# review named this drift risk and deferred the dedup to SP4, done here.
MAX_TOKENS = 64  # tiny: a smoke check proves the wiring works, it is not a quality eval
_KEY_VAR_BY_PROVIDER = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def _keyed_provider() -> tuple[str, str] | None:
    """Which provider + key env var name to use for the generation half, or None if unkeyed. Only
    ever checks PRESENCE (`bool(os.environ.get(...))`); the value itself is never read into a
    variable this script inspects, logs, or forwards anywhere but straight into the provider SDK."""
    provider = os.environ.get("MODEL_PROVIDER", "ollama")
    key_var = _KEY_VAR_BY_PROVIDER.get(provider)
    if key_var and os.environ.get(key_var):
        return provider, key_var
    return None


def _retrieval_half() -> list[dict]:
    print("\n== retrieval half (always runs, no key needed) ==")
    retriever = PgvectorRetriever()
    try:
        chunks = retriever.search_chunks(QUESTION, k=DEPLOYED_K, config=RetrievalConfig())
    finally:
        retriever.close()
    print(f"query: {QUESTION!r}")
    for c in chunks:
        print(f"  doc_id={c.doc_id!r} score={c.score:.6f}")
    print(f"retrieved doc ids ({len(chunks)}): {[c.doc_id for c in chunks]}")
    return [{"doc_id": c.doc_id, "text": c.text} for c in chunks]


def _chat_endpoint_half() -> None:
    print(f"\n== chat endpoint half (the real served backend at {BASE_URL}) ==")
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        login = client.post("/auth/login", json={"customer_id": CUSTOMER_ID})
        login.raise_for_status()
        token = login.json()["access_token"]
        r = client.post(
            "/chat",
            json={"message": QUESTION, "thread_id": "rag-smoke"},
            headers={"authorization": f"Bearer {token}"},
        )
    if r.status_code == 503:
        print("chat endpoint: 503 replay cassette miss -- EXPECTED with no key/no matching cassette:")
        print(f"  {r.json().get('error', '')}")
        print("  this is the D36 tier 2 boundary itself: retrieval ran for free above, generation")
        print("  needs either a recorded cassette (not this fresh a question) or a live provider.")
    elif r.status_code == 200:
        print(f"chat endpoint: 200 final_response={r.json().get('final_response')!r}")
    else:
        print(f"chat endpoint: unexpected status {r.status_code}: {r.text}")


def _chat_stream_half() -> None:
    """SP4 task 6: the SSE surface, over the real served backend. Prints the event TYPE sequence
    only (never the token text/citation ids themselves, to keep this script's output stable and
    short regardless of what the corpus/model happen to answer) -- the sequence shape IS the
    thing this half exists to demonstrate: `message_start` first, `message_end` last, always."""
    print(f"\n== chat stream half (SSE, the real served backend at {BASE_URL}) ==")
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        login = client.post("/auth/login", json={"customer_id": CUSTOMER_ID})
        login.raise_for_status()
        token = login.json()["access_token"]
        event_types: list[str] = []
        with client.stream(
            "POST", "/chat/stream",
            json={"message": QUESTION, "thread_id": "rag-smoke-stream"},
            headers={"authorization": f"Bearer {token}"},
        ) as r:
            if r.status_code != 200:
                print(f"chat stream: unexpected status {r.status_code}: {r.read()!r}")
                return
            for line in r.iter_lines():
                if line.startswith("data: "):
                    event_types.append(json.loads(line[len("data: "):])["event"])
    print(f"event sequence: {event_types}")
    if event_types[:1] == ["message_start"] and event_types[-1:] == ["message_end"]:
        print("  message_start first, message_end last: the terminal guarantee held, live.")
    if event_types[-2:] == ["error", "message_end"]:
        print("  ends in error -> message_end: EXPECTED, no committed cassette for this fresh question")
        print("  (same D36 tier 2 boundary the chat endpoint half above hits as a 503, in band here)")


def _generation_half(passages: list[dict], provider: str) -> None:
    print(f"\n== generation half (MODEL_PROVIDER={provider}, max_tokens={MAX_TOKENS}) ==")
    from langchain_core.messages import HumanMessage, SystemMessage
    from replay.providers import DEFAULT_MODEL_IDS, build_chat_model

    # Explicit, provider-matched model_id (not build_chat_model's own MODEL_ID env fallback): a
    # `.env` can carry a MODEL_ID left over from a different MODEL_PROVIDER (e.g. ollama's
    # "qwen2.5:7b" while MODEL_PROVIDER now says "openai"), which this script must not trust blindly
    # since its whole job is proving generation works for the DETECTED provider.
    model = build_chat_model(provider=provider, model_id=DEFAULT_MODEL_IDS.get(provider), max_tokens=MAX_TOKENS)
    context = "\n\n".join(f"[{p['doc_id']}] {p['text']}" for p in passages)
    prompt = [
        SystemMessage("Answer the customer's question using ONLY the passages below."),
        HumanMessage(f"Passages:\n{context}\n\nQuestion: {QUESTION}"),
    ]
    result = model.invoke(prompt)
    print(f"generated answer (grounded in the real retrieval above): {result.content!r}")


def main() -> None:
    print(f"Atlas rag:smoke -- D36 tier 2 compose acceptance ({BASE_URL})")
    passages = _retrieval_half()
    _chat_endpoint_half()
    _chat_stream_half()

    keyed = _keyed_provider()
    if keyed is None:
        print("\n== generation half: SKIPPED (no recognized provider key present in .env) ==")
        print("ran retrieval + chat endpoint + chat stream (no key needed for any of the three).")
        return

    provider, key_var = keyed
    print(f"\n({key_var} is present in .env; its value is never read or printed by this script)")
    _generation_half(passages, provider)
    print("\nran ALL FOUR: retrieval, chat endpoint, chat stream, and generation.")


if __name__ == "__main__":
    main()
