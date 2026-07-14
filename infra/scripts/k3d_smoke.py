#!/usr/bin/env python3
"""`task k3d:smoke` (SP5 task 4): the rag smoke, ported to the k3d tier and run against the real
Traefik ingress rather than a directly published backend port. Mirrors testing/harness/rag_tools/
smoke.py's own four part structure (retrieval half always runs, chat endpoint/stream halves always
attempted through HTTP, generation half only if a provider key is present), adapted for what the k3d
tier actually exposes to the host:

RETRIEVAL (always runs, no key needed): the SAME direct PgvectorRetriever call rag_tools.smoke.py's
    own retrieval half makes, just pointed at the k3d tier instead of compose. CNPG's Service is
    ClusterIP only (never published to the host, unlike compose's own postgres:5433), so
    infra/scripts/k3d-smoke.sh opens a `kubectl port-forward` to atlas-pg-rw first and tears it down
    on exit; the local tier's TEI endpoints are already host reachable (`tei.mode: external`, Task 3),
    so no port-forward is needed for those.

CHAT ENDPOINT / CHAT STREAM (always attempted, THROUGH THE INGRESS): the same /chat, /chat/stream
    calls rag_tools.smoke.py's own HTTP halves make, retargeted at the ingress base URL
    (http://localhost:<INGRESS_HTTP_PORT>/api/...) instead of a directly published backend port,
    proving Traefik routes to atlas-web (nginx), which proxies to the "backend" Service, which
    reaches the served backend, end to end. The compose backend's own default (ATLAS_MODE=replay, no
    committed cassette for this fresh question) means the SAME expected 503 cassette miss / in band
    SSE error sequence rag_tools.smoke.py documents; that miss is still evidence the full request
    path executed, not a failure of this script.

GENERATION (only if a provider key is present in the env this Taskfile target sources from the repo
    root .env): identical reasoning to rag_tools.smoke.py's own generation half (this reference
    system's agent graph never binds tools to a LIVE model, so a full live agentic turn through /chat
    is out of scope for what this codebase can do today) -- calls the live provider directly,
    grounded in the SAME retrieved passages the retrieval half above already fetched.

This script never reads .env itself; task k3d:smoke sources it into the process environment via
Task's own dotenv: loading (the same precedent task rag:smoke already sets), and this script only
checks whether a recognized provider key env var is nonempty, never printing its value.
"""
from __future__ import annotations

import json
import os

import httpx

from atlas.adapters.pgvector_retriever import PgvectorRetriever
from atlas.domain.retrieval import RetrievalConfig
from atlas.mcp_servers.knowledge_server import DEPLOYED_K

INGRESS_BASE_URL = os.environ.get("ATLAS_SMOKE_INGRESS_URL", "http://localhost:8090")
CUSTOMER_ID = "cust_legacy_term"  # Daniel, the same fixture rag_tools.smoke.py uses
QUESTION = "is my plan contract free"
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
    print("\n== retrieval half (always runs, no key needed; via kubectl port-forward to atlas-pg-rw) ==")
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
    print(f"\n== chat endpoint half (through the ingress at {INGRESS_BASE_URL}) ==")
    with httpx.Client(base_url=INGRESS_BASE_URL, timeout=30.0) as client:
        login = client.post("/api/auth/login", json={"customer_id": CUSTOMER_ID})
        login.raise_for_status()
        token = login.json()["access_token"]
        r = client.post(
            "/api/chat",
            json={"message": QUESTION, "thread_id": "k3d-smoke"},
            headers={"authorization": f"Bearer {token}"},
        )
    if r.status_code == 503:
        print("chat endpoint: 503 replay cassette miss -- EXPECTED, no committed cassette for this fresh question:")
        print(f"  {r.json().get('error', '')}")
        print("  proves the full path (Traefik -> atlas-web nginx -> the \"backend\" Service -> the served")
        print("  backend) ran end to end; retrieval is free, generation needs a cassette or a live provider.")
    elif r.status_code == 200:
        print(f"chat endpoint: 200 final_response={r.json().get('final_response')!r}")
    else:
        print(f"chat endpoint: unexpected status {r.status_code}: {r.text}")


def _chat_stream_half() -> None:
    """Prints the event TYPE sequence only (mirrors rag_tools.smoke.py's own reasoning): the shape
    (message_start first, message_end last) is the thing this half exists to demonstrate, through
    the real ingress this time, not the token text/citation ids."""
    print(f"\n== chat stream half (SSE, through the ingress at {INGRESS_BASE_URL}) ==")
    with httpx.Client(base_url=INGRESS_BASE_URL, timeout=30.0) as client:
        login = client.post("/api/auth/login", json={"customer_id": CUSTOMER_ID})
        login.raise_for_status()
        token = login.json()["access_token"]
        event_types: list[str] = []
        with client.stream(
            "POST", "/api/chat/stream",
            json={"message": QUESTION, "thread_id": "k3d-smoke-stream"},
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
        print("  message_start first, message_end last: the terminal guarantee held, through the ingress.")
    if event_types[-2:] == ["error", "message_end"]:
        print("  ends in error -> message_end: EXPECTED, no committed cassette for this fresh question")
        print("  (same boundary the chat endpoint half above hits as a 503, in band here)")


def _generation_half(passages: list[dict], provider: str) -> None:
    print(f"\n== generation half (MODEL_PROVIDER={provider}, max_tokens={MAX_TOKENS}) ==")
    from langchain_core.messages import HumanMessage, SystemMessage
    from replay.providers import DEFAULT_MODEL_IDS, build_chat_model

    model = build_chat_model(provider=provider, model_id=DEFAULT_MODEL_IDS.get(provider), max_tokens=MAX_TOKENS)
    context = "\n\n".join(f"[{p['doc_id']}] {p['text']}" for p in passages)
    prompt = [
        SystemMessage("Answer the customer's question using ONLY the passages below."),
        HumanMessage(f"Passages:\n{context}\n\nQuestion: {QUESTION}"),
    ]
    result = model.invoke(prompt)
    print(f"generated answer (grounded in the real k3d tier retrieval above): {result.content!r}")


def main() -> None:
    print(f"Atlas k3d:smoke -- the rag smoke against the ingress ({INGRESS_BASE_URL})")
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
