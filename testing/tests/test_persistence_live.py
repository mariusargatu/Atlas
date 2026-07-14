"""Postgres checkpointer persistence, live (SP4 task 2): a chat turn survives a backend container
restart. Marked `live`, excluded from `task test`; run via `task test:live` against a running
`docker compose up` stack whose backend was built from this task's code (ATLAS_CHECKPOINTER=postgres
is compose's own default; ATLAS_PG_DSN points at the in network postgres service; the checkpointer
migration is applied automatically by the `checkpointer-migrate` one shot service before backend
ever starts -- see docker-compose.yml and `task db:upgrade`).

Restarts ONLY the backend container mid test (`docker compose restart backend`), never a bare
`up --build`: rebuilding would hand the process a fresh image, which says nothing about whether
STATE survives a process restart -- the whole point of a Postgres backed checkpointer over the
hermetic InMemorySaver, which loses everything the instant the process exits.

The two turns replay against a cassette pair committed at
`testing/harness/cassettes/e2e` (seeded by `task seed-e2e`,
`testing/harness/recording/seed_e2e_cassettes.py`'s `_PERSISTENCE_TURNS`), baked into the backend
Docker image at build time (`backend/Dockerfile` COPYs the whole `testing/harness` tree). The
SECOND turn's cassette key covers the WHOLE accumulated message history: round 1's human question,
round 1's AI answer, AND round 2's question (see `replay/cassette.py`'s `build_request` and
`test_naive_variant_live.py`'s docstring for the same point made about a different endpoint). So a
200 on round 2 is itself the persistence proof: if the checkpoint had NOT survived the restart, the
fresh process would hand the graph only round 2's bare question with no prior history, the gateway
would compute a different digest, miss the cassette, and 503 -- there is no way to fake a pass here
by accident.

THREAD_ID is suffixed with a fresh uuid4 per test PROCESS (module import time, not per assertion):
Postgres, unlike the hermetic suite's InMemorySaver, keeps a thread's checkpoints across separate
`task test:live` invocations (there is no `_reset_account_state`-style teardown for a real database).
A fixed thread id would make a SECOND run of this test ask round 1's question against a thread that
already has round 1+2 baked in from the first run, missing the cassette for a reason that has
nothing to do with persistence (confirmed live: this is exactly what happened before the suffix was
added, and the failure shape -- more prior turns than the cassette expects -- is itself evidence the
checkpoint persisted correctly).
"""
from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "http://localhost:8000"
CUSTOMER_ID = "cust_current"
THREAD_ID = f"persistence-restart-check-{uuid.uuid4().hex[:8]}"
Q1 = "What is your name?"
A1 = "I'm Atlas, your broadband support assistant."
Q2 = "Can you say that again please?"
A2 = "Sure, I'm Atlas, your broadband support assistant, glad to help again."
_RESTART_TIMEOUT_SECONDS = 60.0  # a plain uvicorn boot; generous margin over the observed ~1-2s

# SP4 final fix wave carryover: also live_slow -- restarts the backend container mid test
# (container restart + health check), the OTHER dominant cost the report's own final gate
# measurement named.
pytestmark = [pytest.mark.live, pytest.mark.live_slow]


def _login(client: httpx.Client) -> str:
    r = client.post("/auth/login", json={"customer_id": CUSTOMER_ID})
    r.raise_for_status()
    return r.json()["access_token"]


def _chat(client: httpx.Client, token: str, message: str) -> httpx.Response:
    return client.post(
        "/chat",
        json={"message": message, "thread_id": THREAD_ID},
        headers={"authorization": f"Bearer {token}"},
    )


def _wait_for_healthy(deadline_seconds: float) -> None:
    """Poll /healthz until the restarted container answers again. `docker compose restart` returns
    once the container process has been asked to restart, not once uvicorn is ready to serve, so a
    test that dialed /chat immediately after would flake on a connection refused, not a real
    persistence failure."""
    deadline = time.monotonic() + deadline_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/healthz", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"backend did not become healthy again within {deadline_seconds}s: {last_error}")


def test_chat_state_survives_a_backend_container_restart() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        token = _login(client)
        r1 = _chat(client, token, Q1)
        assert r1.status_code == 200, r1.text
        assert r1.json()["final_response"] == A1

    # Restart ONLY the backend service, in place: a fresh process, the SAME Postgres volume/data.
    # Never `up --build` here (see module docstring) -- that changes the image, not the persistence
    # question this test asks.
    subprocess.run(["docker", "compose", "restart", "backend"], check=True, cwd=REPO_ROOT)
    _wait_for_healthy(_RESTART_TIMEOUT_SECONDS)

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        token = _login(client)  # a fresh login: the bearer token itself carries no conversation state
        r2 = _chat(client, token, Q2)
        assert r2.status_code == 200, r2.text
        assert r2.json()["final_response"] == A2
