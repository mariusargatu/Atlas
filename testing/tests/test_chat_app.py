"""The product API (ADR-028): auth + chat over the Atlas graph, hermetic and replayed.

Same cassette discipline as test_atlas_graph: the model is a replayed gateway, the actions backend
is in memory, the clock is frozen. Proves the SPA's integration contract: login, the cold open held
at render, and the propose -> typed CONFIRM -> idempotent execute flow over two HTTP requests.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory, fixture_kit
from replay.gateway import GatewayChatModel

from atlas.chat_app import make_chat_app
from atlas.domain import accounts
from atlas.domain.accounts import apply_write
from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph

_FALSE_ANSWER = "Your plan is contract-free, no fee, cancel any time."


@pytest.fixture(autouse=True)
def _restore_atlas_logger():
    """FIX2-G: create_app() attaches an INFO StreamHandler to the PROCESS-GLOBAL 'atlas' logger (so the
    startup line is emitted under uvicorn, where the effective level is WARNING). Snapshot and restore
    that logger's handlers and level around every test in this module, so a server test's boot does not
    leak the handler into the rest of the suite (duplicate log lines) or pin a level another test relies
    on. _configure_logging stays idempotent; this just keeps its process-global effect test-scoped."""
    import logging

    atlas_log = logging.getLogger("atlas")
    saved_handlers, saved_level = atlas_log.handlers[:], atlas_log.level
    try:
        yield
    finally:
        atlas_log.handlers[:] = saved_handlers
        atlas_log.setLevel(saved_level)


def _app(cassette_dir):
    kit = fixture_kit()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    # write through, exactly as server.py wires it, so the HTTP act path actually mutates state
    backend = ActionsBackend(IdFactory("ref"), writer=apply_write)
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer())
    return make_chat_app(kit.clock, graph)


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _login(client, customer_id):
    r = await client.post("/auth/login", json={"customer_id": customer_id})
    assert r.status_code == 200
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_login_issues_token_and_sets_refresh_cookie(tmp_path):
    async with _client(_app(tmp_path)) as client:
        r = await client.post("/auth/login", json={"customer_id": "cust_current"})
        assert r.status_code == 200 and r.json()["access_token"]
        cookie = r.headers.get("set-cookie", "")
        assert "atlas_refresh=" in cookie and "httponly" in cookie.lower()


@pytest.mark.asyncio
async def test_login_unknown_customer_404(tmp_path):
    async with _client(_app(tmp_path)) as client:
        r = await client.post("/auth/login", json={"customer_id": "cust_nope"})
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_chat_without_token_is_401(tmp_path):
    async with _client(_app(tmp_path)) as client:
        r = await client.post("/chat", json={"message": "hi"}, headers={"authorization": "Bearer garbage"})
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_chat_cold_open_held_for_legacy_customer(tmp_path, seed_cassette):
    seed_cassette(tmp_path, [HumanMessage("Is my plan contract-free?")], {"content": _FALSE_ANSWER, "tool_calls": []})
    app = _app(tmp_path)
    async with _client(app) as client:
        token = await _login(client, "cust_legacy_term")
        r = await client.post(
            "/chat", json={"message": "Is my plan contract-free?", "thread_id": "t1"},
            headers={"authorization": f"Bearer {token}"},
        )
        body = r.json()
        assert body["type"] == "final"
        assert body["final_response"].startswith("[safe handoff]")  # the false answer is NOT rendered


@pytest.mark.asyncio
async def test_chat_same_answer_renders_for_current_customer(tmp_path, seed_cassette):
    seed_cassette(tmp_path, [HumanMessage("Is my plan contract-free?")], {"content": _FALSE_ANSWER, "tool_calls": []})
    async with _client(_app(tmp_path)) as client:
        token = await _login(client, "cust_current")
        r = await client.post(
            "/chat", json={"message": "Is my plan contract-free?", "thread_id": "t1"},
            headers={"authorization": f"Bearer {token}"},
        )
        assert r.json()["final_response"] == _FALSE_ANSWER  # true for Sarah, so it renders


async def _propose(client, auth, thread_id):
    r = await client.post(
        "/chat", json={"message": "Switch me to the fast plan", "thread_id": thread_id}, headers=auth
    )
    assert r.json()["type"] == "interrupt"
    assert r.json()["pending"]["tool"] == "change_plan"


async def _step_up(client, read_auth):
    """Elevate a read session to a write token (ADR-027 step up), the turn that confirms an action."""
    r = await client.post("/auth/step-up", headers=read_auth)
    assert r.status_code == 200
    return {"authorization": f"Bearer {r.json()['access_token']}"}


@pytest.mark.asyncio
async def test_act_path_typed_confirm_executes(tmp_path, seed_cassette):
    seed_cassette(
        tmp_path, [HumanMessage("Switch me to the fast plan")],
        {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]},
    )
    async with _client(_app(tmp_path)) as client:
        auth = {"authorization": f"Bearer {await _login(client, 'cust_current')}"}
        await _propose(client, auth, "ok")
        write_auth = await _step_up(client, auth)  # elevate to write for the confirming turn
        ok = await client.post("/chat/resume", json={"thread_id": "ok", "confirmation": "CONFIRM"}, headers=write_auth)
        assert "Done" in ok.json()["final_response"]


@pytest.mark.asyncio
async def test_act_path_writes_through_account_state_over_http(tmp_path, seed_cassette):
    """The full product stack: a confirmed change_plan over HTTP mutates the account store, so the
    write the SPA triggered is real, not just an audit entry. Daniel moves off the legacy plan."""
    seed_cassette(
        tmp_path, [HumanMessage("Switch me to the fast plan")],
        {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]},
    )
    assert accounts.get_account("cust_legacy_term").plan_id == "plan_legacy_value"  # before
    async with _client(_app(tmp_path)) as client:
        auth = {"authorization": f"Bearer {await _login(client, 'cust_legacy_term')}"}
        await _propose(client, auth, "wt")
        write_auth = await _step_up(client, auth)
        ok = await client.post("/chat/resume", json={"thread_id": "wt", "confirmation": "CONFIRM"}, headers=write_auth)
        assert "Done" in ok.json()["final_response"]
    assert accounts.get_account("cust_legacy_term").plan_id == "plan_current_fast"  # after: the HTTP write landed


@pytest.mark.asyncio
async def test_act_path_bare_yes_is_rejected(tmp_path, seed_cassette):
    seed_cassette(
        tmp_path, [HumanMessage("Switch me to the fast plan")],
        {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]},
    )
    async with _client(_app(tmp_path)) as client:
        auth = {"authorization": f"Bearer {await _login(client, 'cust_current')}"}
        await _propose(client, auth, "bad")
        write_auth = await _step_up(client, auth)
        bare = await client.post("/chat/resume", json={"thread_id": "bad", "confirmation": "yes"}, headers=write_auth)
        assert bare.json()["final_response"].startswith("[safe handoff]")  # bare yes is not a typed CONFIRM


@pytest.mark.asyncio
async def test_chat_requires_read_resume_requires_write_scope(tmp_path):
    """Step up: a read scoped token can chat but cannot resume a write (ADR-027 defense in depth)."""
    from atlas.auth import issue_token

    async with _client(_app(tmp_path)) as client:
        read_only = issue_token("cust_current", ["read"], fixture_kit().clock.now())
        r = await client.post(
            "/chat/resume", json={"thread_id": "x", "confirmation": "CONFIRM"},
            headers={"authorization": f"Bearer {read_only}"},
        )
        assert r.status_code == 403  # missing write scope


@pytest.mark.asyncio
async def test_login_is_read_only_and_step_up_elevates_for_a_write(tmp_path, seed_cassette):
    """ADR-027 step up: the login token cannot confirm a write, but a stepped up token can."""
    seed_cassette(
        tmp_path, [HumanMessage("Switch me to the fast plan")],
        {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]},
    )
    async with _client(_app(tmp_path)) as client:
        login_auth = {"authorization": f"Bearer {await _login(client, 'cust_current')}"}
        await _propose(client, login_auth, "su")
        denied = await client.post("/chat/resume", json={"thread_id": "su", "confirmation": "CONFIRM"}, headers=login_auth)
        assert denied.status_code == 403  # login is read only, cannot resume a write
        write_auth = await _step_up(client, login_auth)
        ok = await client.post("/chat/resume", json={"thread_id": "su", "confirmation": "CONFIRM"}, headers=write_auth)
        assert "Done" in ok.json()["final_response"]  # elevated token confirms the action


@pytest.mark.asyncio
async def test_refresh_without_cookie_is_401(tmp_path):
    async with _client(_app(tmp_path)) as client:
        r = await client.post("/auth/refresh")
        assert r.status_code == 401


# ---- resume state conflicts are 409, never a KeyError 500 (the money path, both directions) ----

@pytest.mark.asyncio
async def test_resume_with_no_pending_confirmation_is_409(tmp_path):
    async with _client(_app(tmp_path)) as client:
        auth = {"authorization": f"Bearer {await _login(client, 'cust_current')}"}
        write_auth = await _step_up(client, auth)
        r = await client.post("/chat/resume", json={"thread_id": "never-proposed", "confirmation": "CONFIRM"}, headers=write_auth)
        assert r.status_code == 409


@pytest.mark.asyncio
async def test_second_resume_after_the_write_completed_is_409(tmp_path, seed_cassette):
    seed_cassette(
        tmp_path, [HumanMessage("Switch me to the fast plan")],
        {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]},
    )
    async with _client(_app(tmp_path)) as client:
        auth = {"authorization": f"Bearer {await _login(client, 'cust_current')}"}
        await _propose(client, auth, "dbl")
        write_auth = await _step_up(client, auth)
        first = await client.post("/chat/resume", json={"thread_id": "dbl", "confirmation": "CONFIRM"}, headers=write_auth)
        assert "Done" in first.json()["final_response"]
        again = await client.post("/chat/resume", json={"thread_id": "dbl", "confirmation": "CONFIRM"}, headers=write_auth)
        assert again.status_code == 409  # nothing pending anymore; the write is not re-runnable


# ---- graph recursion exhaustion at the HTTP edge is the typed handoff, never a raw 500 ----

class _ExplodingGraph:
    """Stands in for a graph whose invocation blows the recursion limit."""

    async def ainvoke(self, *args, **kwargs):
        from langgraph.errors import GraphRecursionError

        raise GraphRecursionError("recursion limit reached")


@pytest.mark.asyncio
async def test_graph_recursion_error_returns_the_typed_handoff_not_a_500():
    app = make_chat_app(fixture_kit().clock, _ExplodingGraph())
    async with _client(app) as client:
        token = await _login(client, "cust_current")
        r = await client.post(
            "/chat", json={"message": "hi", "thread_id": "t1"},
            headers={"authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "final"
        assert body["final_response"].startswith("[safe handoff]")
        assert body["thread_id"] == "t1"  # the one handler echoes the request's thread_id


class _ExplodingOnResumeGraph:
    """Has a pending confirmation (so /chat/resume gets past the 409 guard), then blows the limit."""

    async def aget_state(self, cfg):
        from types import SimpleNamespace

        return SimpleNamespace(tasks=(SimpleNamespace(interrupts=("pending",)),))

    async def ainvoke(self, *args, **kwargs):
        from langgraph.errors import GraphRecursionError

        raise GraphRecursionError("recursion limit reached")


@pytest.mark.asyncio
async def test_resume_route_shares_the_one_recursion_handler():
    """FIX2-J: the recursion backstop is a single app-level handler, so /chat/resume (not just /chat)
    returns the typed [safe handoff] ChatOut when the graph blows the limit, never a raw 500."""
    app = make_chat_app(fixture_kit().clock, _ExplodingOnResumeGraph())
    async with _client(app) as client:
        auth = {"authorization": f"Bearer {await _login(client, 'cust_current')}"}
        write_auth = await _step_up(client, auth)
        r = await client.post("/chat/resume", json={"thread_id": "rz", "confirmation": "CONFIRM"}, headers=write_auth)
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "final" and body["final_response"].startswith("[safe handoff]")
        assert body["thread_id"] == "rz"


# ---- both edges reject a malformed bearer header identically (one helper, no drift) ----

@pytest.mark.asyncio
async def test_both_edges_reject_a_bare_token_without_the_bearer_prefix(tmp_path):
    from atlas.auth import issue_token
    from atlas.edge_app import make_app

    kit = fixture_kit()
    token = issue_token("cust_current", ["read"], kit.clock.now())
    async with _client(_app(tmp_path)) as chat_client:
        r = await chat_client.post("/chat", json={"message": "hi"}, headers={"authorization": token})
        assert r.status_code == 401  # bare token, no "Bearer " scheme
        r = await chat_client.post("/chat", json={"message": "hi"})
        assert r.status_code == 401  # missing header entirely
    async with _client(make_app(kit.clock)) as edge_client:
        r = await edge_client.get("/account/summary", headers={"authorization": token})
        assert r.status_code == 401  # the MCP edge agrees: bare token rejected
        r = await edge_client.get("/account/summary")
        assert r.status_code == 401  # and a missing header is auth failure, not a 422


@pytest.mark.asyncio
async def test_both_edges_accept_a_lowercase_bearer_scheme(tmp_path, seed_cassette):
    """RFC 7235 auth-scheme tokens are case-insensitive: `bearer <token>` is accepted the same as
    `Bearer <token>` on both edges (the pre-branch chat edge, via FastAPI HTTPBearer, did too)."""
    from atlas.auth import issue_token
    from atlas.edge_app import make_app

    seed_cassette(tmp_path, [HumanMessage("Is my plan contract-free?")], {"content": _FALSE_ANSWER, "tool_calls": []})
    async with _client(_app(tmp_path)) as chat_client:
        token = await _login(chat_client, "cust_current")
        r = await chat_client.post(
            "/chat", json={"message": "Is my plan contract-free?", "thread_id": "lc1"},
            headers={"authorization": f"bearer {token}"},  # lowercase scheme, still valid
        )
        assert r.status_code == 200 and r.json()["final_response"] == _FALSE_ANSWER

    kit = fixture_kit()
    read_token = issue_token("cust_current", ["read"], kit.clock.now())
    async with _client(make_app(kit.clock)) as edge_client:
        r = await edge_client.get("/account/summary", headers={"authorization": f"bearer {read_token}"})
        assert r.status_code == 200 and r.json()["customer_id"] == "cust_current"  # the MCP edge agrees


# ---- the served entrypoint: /healthz contract, startup honesty, and the 503 on a cassette miss ----

def _server_app(monkeypatch, cassette_dir, mode="replay", git_sha=None):
    from atlas import server

    monkeypatch.setenv("ATLAS_MODE", mode)
    monkeypatch.setenv("ATLAS_CASSETTES", str(cassette_dir))
    if git_sha is None:
        monkeypatch.delenv("GIT_SHA", raising=False)
    else:
        monkeypatch.setenv("GIT_SHA", git_sha)
    return server.create_app()


@pytest.mark.asyncio
async def test_healthz_reports_status_mode_cassettes_and_git_sha(tmp_path, monkeypatch):
    app = _server_app(monkeypatch, tmp_path, git_sha="abc1234")
    async with _client(app) as client:
        r = await client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "mode": "replay", "cassettes": True, "git_sha": "abc1234"}


@pytest.mark.asyncio
async def test_healthz_git_sha_defaults_to_dev_when_env_is_unset(tmp_path, monkeypatch):
    app = _server_app(monkeypatch, tmp_path)
    async with _client(app) as client:
        r = await client.get("/healthz")
        assert r.json()["git_sha"] == "dev"


def test_startup_line_is_actually_emitted_with_mode_cassettes_and_git_sha(tmp_path, monkeypatch, caplog):
    # Regression: the startup INFO line was dead under uvicorn (effective level WARNING). The test does
    # NOT force the level via caplog (that would hide the bug); it relies on create_app configuring the
    # "atlas" logger to INFO. Without that configuration no record is produced and this assertion fails.
    from atlas import server

    monkeypatch.setenv("ATLAS_MODE", "replay")
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path))
    monkeypatch.setenv("GIT_SHA", "abc1234")
    caplog.clear()
    server.create_app()
    startup = [r for r in caplog.records if r.name == "atlas.server" and "startup" in r.getMessage()]
    assert startup, "startup line was not emitted: the atlas logger has no INFO handler configured"
    msg = startup[-1].getMessage()
    assert "mode=replay" in msg and f"cassettes={tmp_path}" in msg and "git_sha=abc1234" in msg


def test_unknown_atlas_mode_fails_fast_at_startup(tmp_path, monkeypatch):
    from atlas import server

    monkeypatch.setenv("ATLAS_MODE", "repaly")  # the typo must not fall through to the live branch
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path))
    with pytest.raises(RuntimeError, match="ATLAS_MODE"):
        server.create_app()


def test_replay_mode_with_a_missing_cassette_dir_fails_fast_at_startup(tmp_path, monkeypatch):
    from atlas import server

    monkeypatch.setenv("ATLAS_MODE", "replay")
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match="cassette"):
        server.create_app()


def test_live_mode_with_an_uninstalled_provider_sdk_fails_fast_with_the_group_to_install(tmp_path, monkeypatch):
    """FIX2-H: live/record mode reaches a real provider. If its SDK group was never synced (the image
    ships only the ollama group), create_app must refuse to boot with ONE actionable line naming the
    dependency group, not hang the web edge with a raw ImportError on the first turn. Hermetic: the
    import check is monkeypatched, so no SDK need actually be absent."""
    import importlib.util

    from atlas import server

    monkeypatch.setenv("ATLAS_MODE", "live")
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "langchain_anthropic" else real_find_spec(name, *a, **k),
    )
    with pytest.raises(RuntimeError) as exc:
        server.create_app()
    assert "langchain_anthropic" in str(exc.value) and "--group anthropic" in str(exc.value)


def test_live_mode_with_an_unknown_provider_fails_fast_at_startup(tmp_path, monkeypatch):
    from atlas import server

    monkeypatch.setenv("ATLAS_MODE", "live")
    monkeypatch.setenv("MODEL_PROVIDER", "nonesuch")  # not one of ollama/anthropic/openai
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path))
    with pytest.raises(RuntimeError, match="unknown MODEL_PROVIDER"):
        server.create_app()


@pytest.mark.asyncio
async def test_replay_cassette_miss_is_a_503_naming_the_miss_not_a_500(tmp_path, monkeypatch):
    app = _server_app(monkeypatch, tmp_path)  # an existing but empty cassette dir: every turn misses
    async with _client(app) as client:
        token = await _login(client, "cust_current")
        r = await client.post(
            "/chat", json={"message": "hi", "thread_id": "m1"},
            headers={"authorization": f"Bearer {token}"},
        )
        assert r.status_code == 503
        assert "cassette miss" in r.json()["error"]


# ---- /metrics (SP6 task 5): the request counting middleware + the exposition route end to end -----


@pytest.mark.asyncio
async def test_metrics_endpoint_is_plain_text_and_carries_every_registered_family(tmp_path, monkeypatch):
    app = _server_app(monkeypatch, tmp_path)
    async with _client(app) as client:
        r = await client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert "atlas_http_requests_total" in r.text
        assert "atlas_circuit_breaker_state" in r.text
        assert "atlas_corpus_staleness" in r.text
        assert "atlas_judge_pass_total 0" in r.text
        assert "atlas_judge_fail_total 0" in r.text


@pytest.mark.asyncio
async def test_metrics_endpoint_reflects_requests_the_middleware_already_counted(tmp_path, monkeypatch):
    """The middleware wraps every route (proven here against /healthz, not /metrics itself, so the
    scrape request that reads the counter is not also the one that bumps the number the assertion
    checks against)."""
    app = _server_app(monkeypatch, tmp_path)
    async with _client(app) as client:
        await client.get("/healthz")
        await client.get("/healthz")
        r = await client.get("/metrics")
        assert 'atlas_http_requests_total{status_class="2xx"} 2' in r.text


@pytest.mark.asyncio
async def test_metrics_endpoint_never_touches_pgvector_when_atlas_retriever_is_unset(tmp_path, monkeypatch):
    """The hermetic default (InMemoryRetriever, no `.breaker`): the middleware/route addition must
    not change `test_create_app_never_touches_pgvector_when_atlas_retriever_is_unset`'s own
    guarantee below, and the breaker gauge family must simply be absent, not raise."""
    monkeypatch.delenv("ATLAS_RETRIEVER", raising=False)
    app = _server_app(monkeypatch, tmp_path)
    async with _client(app) as client:
        r = await client.get("/metrics")
        assert r.status_code == 200
        assert "atlas_circuit_breaker_state{" not in r.text


@pytest.mark.asyncio
async def test_healthz_contract_is_unchanged_by_the_metrics_middleware(tmp_path, monkeypatch):
    """Regression proof: wrapping every route in the request counting middleware must not alter
    /healthz's own fixed response shape (CI readiness, the compose healthcheck, and depends_on all
    read this exact shape)."""
    app = _server_app(monkeypatch, tmp_path, git_sha="abc1234")
    async with _client(app) as client:
        r = await client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "mode": "replay", "cassettes": True, "git_sha": "abc1234"}


# ---- /version (SP6 task 6, D37): release identity, every field SURFACED, never computed here ----


@pytest.mark.asyncio
async def test_version_endpoint_shape_and_git_sha(tmp_path, monkeypatch):
    app = _server_app(monkeypatch, tmp_path, git_sha="abc1234")
    async with _client(app) as client:
        r = await client.get("/version")
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"git_sha", "contracts", "corpus_version", "index_build_id"}
        assert body["git_sha"] == "abc1234"


@pytest.mark.asyncio
async def test_version_git_sha_defaults_to_dev_when_env_is_unset_same_as_healthz(tmp_path, monkeypatch):
    """/version reads GIT_SHA the same way /healthz already does (settings.git_sha, one read): no
    second, independently drifting env lookup."""
    app = _server_app(monkeypatch, tmp_path)
    async with _client(app) as client:
        healthz = await client.get("/healthz")
        version = await client.get("/version")
        assert version.json()["git_sha"] == healthz.json()["git_sha"] == "dev"


@pytest.mark.asyncio
async def test_version_contracts_is_exactly_contract_tools_loader_contract_versions(tmp_path, monkeypatch):
    """The `contracts` field is a direct call to `contract_tools.loader.contract_versions()`
    (`{family: x-contract-version}` for all four families), never a hand rebuilt dict that could
    drift from the loader's own reading of contracts/*/schema.json."""
    from contract_tools.loader import contract_versions

    app = _server_app(monkeypatch, tmp_path)
    async with _client(app) as client:
        r = await client.get("/version")
        assert r.json()["contracts"] == contract_versions()
        assert set(r.json()["contracts"]) == {"trace", "dataset", "manifest", "sse"}


@pytest.mark.asyncio
async def test_version_corpus_version_and_index_build_id_come_from_the_settings_module(tmp_path, monkeypatch):
    """corpus_version/index_build_id are `AtlasSettings.from_env()`'s own derived fields (its own
    build_manifest.json read off ATLAS_INDEX_DIR, see test_config.py), surfaced here verbatim: this
    test proves it by pointing ATLAS_INDEX_DIR at a custom manifest and reading the SAME values back
    from /version, never a value the handler derived some other way."""
    import json

    from atlas import server

    index_dir = tmp_path / "custom-index"
    index_dir.mkdir()
    (index_dir / "build_manifest.json").write_text(
        json.dumps({"corpus_version": "corpus-9.9.9", "index_build_id": "deadbeef01234567"})
    )
    monkeypatch.setenv("ATLAS_MODE", "replay")
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path))
    monkeypatch.setenv("ATLAS_INDEX_DIR", str(index_dir))
    app = server.create_app()
    async with _client(app) as client:
        r = await client.get("/version")
        body = r.json()
        assert body["corpus_version"] == "corpus-9.9.9"
        assert body["index_build_id"] == "deadbeef01234567"


@pytest.mark.asyncio
async def test_version_corpus_version_and_index_build_id_are_empty_when_the_index_was_never_built(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ATLAS_MODE", "replay")
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path))
    monkeypatch.setenv("ATLAS_INDEX_DIR", str(tmp_path / "never-built"))
    from atlas import server

    app = server.create_app()
    async with _client(app) as client:
        r = await client.get("/version")
        body = r.json()
        assert body["corpus_version"] == ""
        assert body["index_build_id"] == ""


@pytest.mark.asyncio
async def test_version_endpoint_is_counted_by_the_request_middleware(tmp_path, monkeypatch):
    """/version is wired through the SAME `_count_requests` middleware every other route is (SP6
    task 5): a scrape right after must show it as one more 2xx request."""
    app = _server_app(monkeypatch, tmp_path)
    async with _client(app) as client:
        await client.get("/version")
        r = await client.get("/metrics")
        assert 'atlas_http_requests_total{status_class="2xx"} 1' in r.text


# ---- adapter selection + lifecycle (SP3 task 7, D36 tier 2) ----


def test_create_app_never_touches_pgvector_when_atlas_retriever_is_unset(tmp_path, monkeypatch):
    """The default (env unset) must stay InMemoryRetriever: `task serve-e2e` and every existing
    `_server_app`-based test above already exercise this path with no Postgres/TEI reachable, and
    keep passing unmodified. This test makes the "untouched" claim explicit: PgvectorRetriever is
    monkeypatched to blow up if constructed at all, so a regression that flips the default would
    fail this test even though the toy corpus happens to still answer everything else."""
    from atlas.orchestration import atlas_graph

    def _boom():
        raise AssertionError("PgvectorRetriever must not be constructed when ATLAS_RETRIEVER is unset")

    monkeypatch.delenv("ATLAS_RETRIEVER", raising=False)
    monkeypatch.setattr(atlas_graph, "PgvectorRetriever", _boom)
    app = _server_app(monkeypatch, tmp_path)  # must not raise
    assert app is not None


def test_create_app_wires_pgvector_when_atlas_retriever_is_set_and_closes_it_on_shutdown(tmp_path, monkeypatch):
    """ATLAS_RETRIEVER=pgvector picks the real adapter (D36 tier 2); the served app constructs it
    ONCE and closes it once at shutdown (the client lifecycle ride along). Hermetic: PgvectorRetriever
    itself is monkeypatched on the atlas_graph module (its real constructor needs live Postgres/TEI),
    so this proves only the SERVED APP's wiring -- select_retriever() is consulted, the constructed
    object reaches the graph, and its close() fires exactly once when the app shuts down."""
    from atlas.orchestration import atlas_graph

    closed = []

    class _FakeRetriever:
        def search_chunks(self, query, k, config):
            return []

        def close(self):
            closed.append(True)

    monkeypatch.setenv("ATLAS_RETRIEVER", "pgvector")
    monkeypatch.setattr(atlas_graph, "PgvectorRetriever", lambda: _FakeRetriever())
    app = _server_app(monkeypatch, tmp_path)

    from starlette.testclient import TestClient

    with TestClient(app):
        pass  # startup then shutdown lifespan events fire around this block
    assert closed == [True]


# ---- SP4 task 5: the MCP tool surface bind_tools binds onto the model in live/record mode ----


@pytest.mark.asyncio
async def test_create_app_never_builds_the_mcp_tool_surface_in_replay_mode(tmp_path, monkeypatch, seed_cassette):
    """The explode pattern, mirroring the pgvector proof just above: `mcp_tool_surface` is
    monkeypatched to blow up if called at all, so a regression that started building it
    unconditionally would fail this test even though a plain chat turn happens to still work.
    Replay is `_server_app`'s own default mode, exercised end to end (a full lifespan cycle AND a
    real chat turn), not just at `create_app()` construction."""
    from atlas import server
    from langchain_core.messages import HumanMessage
    from starlette.testclient import TestClient

    def _boom(*_a, **_k):
        raise AssertionError("mcp_tool_surface must not be called in replay mode")

    monkeypatch.setattr(server, "mcp_tool_surface", _boom)
    seed_cassette(tmp_path, [HumanMessage("What is your name?")], {"content": "Hi.", "tool_calls": []})
    app = _server_app(monkeypatch, tmp_path)  # must not raise

    with TestClient(app):
        pass

    async with _client(app) as client:
        token = await _login(client, "cust_current")
        r = await client.post(
            "/chat", json={"message": "What is your name?", "thread_id": "t1"},
            headers={"authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["final_response"] == "Hi."


def test_resolve_mcp_tools_is_none_for_replay_and_the_built_surface_otherwise(monkeypatch):
    """`_resolve_mcp_tools` in isolation, no app/provider SDK needed: replay always returns None
    (the explode proof above is what confirms this ALSO means "never called"); every other mode
    returns whatever `mcp_tool_surface` builds, called with the SAME retriever the caller passed."""
    from atlas import server

    assert server._resolve_mcp_tools("replay", object()) is None

    sentinel = {"search_knowledge": {"name": "search_knowledge", "description": "", "parameters": {}}}
    seen_retriever = []

    def _fake_surface(retriever):
        seen_retriever.append(retriever)
        return sentinel

    monkeypatch.setattr(server, "mcp_tool_surface", _fake_surface)
    retriever = object()
    assert server._resolve_mcp_tools("live", retriever) is sentinel
    assert server._resolve_mcp_tools("record", retriever) is sentinel
    assert seen_retriever == [retriever, retriever]


# ---- SP4 final fix wave (F2): ATLAS_FALLBACK_MODEL wires the ladder's provider_fallback rung ----


def test_atlas_fallback_model_unset_leaves_the_fallback_disabled(tmp_path, monkeypatch):
    """`_fallback_gateway` in isolation, no app/provider SDK needed (mirrors
    `_resolve_mcp_tools`'s own unit test above): the unset default keeps `fallback_model=None`,
    today's behaviour, in every mode -- `provider_fallback` still never fires without an explicit
    opt in."""
    from atlas import server

    monkeypatch.delenv("ATLAS_FALLBACK_MODEL", raising=False)
    assert server._fallback_gateway("live", tmp_path) is None
    assert server._fallback_gateway("record", tmp_path) is None
    assert server._fallback_gateway("replay", tmp_path) is None


def test_atlas_fallback_model_is_never_read_in_replay_mode_even_if_set(tmp_path, monkeypatch):
    """Replay never routes a real exception through the ladder (a cassette miss is `CassetteMiss`,
    not `ProviderError`), so `_fallback_gateway` returns None in replay WITHOUT even reading
    `ATLAS_FALLBACK_MODEL` -- proven here with a deliberately malformed value that would raise in
    any other mode, confirming the env var is genuinely unread, not merely unused."""
    from atlas import server

    monkeypatch.setenv("ATLAS_FALLBACK_MODEL", "not-a-valid-shape-at-all")
    assert server._fallback_gateway("replay", tmp_path) is None


def test_atlas_fallback_model_malformed_fails_fast_at_startup(tmp_path, monkeypatch):
    from atlas import server

    monkeypatch.setenv("ATLAS_FALLBACK_MODEL", "no-colon-here")
    with pytest.raises(RuntimeError, match="ATLAS_FALLBACK_MODEL"):
        server._fallback_gateway("live", tmp_path)


def test_atlas_fallback_model_unknown_provider_fails_fast_at_startup(tmp_path, monkeypatch):
    from atlas import server

    monkeypatch.setenv("ATLAS_FALLBACK_MODEL", "nonesuch:some-model")
    with pytest.raises(RuntimeError, match="unknown provider"):
        server._fallback_gateway("live", tmp_path)


def test_atlas_fallback_model_uninstalled_sdk_fails_fast_with_the_group_to_install(tmp_path, monkeypatch):
    """Mirrors `test_live_mode_with_an_uninstalled_provider_sdk_fails_fast_with_the_group_to_install`
    for the FALLBACK provider: an unsynced SDK group must refuse to boot with one actionable line,
    never a raw ImportError on the turn the primary model first needs the fallback."""
    import importlib.util

    from atlas import server

    monkeypatch.setenv("ATLAS_FALLBACK_MODEL", "anthropic:claude-haiku-4-5-20251001")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "langchain_anthropic" else real_find_spec(name, *a, **k),
    )
    with pytest.raises(RuntimeError) as exc:
        server._fallback_gateway("live", tmp_path)
    assert "langchain_anthropic" in str(exc.value) and "--group anthropic" in str(exc.value)


def test_atlas_fallback_model_wires_a_real_gateway_when_configured(tmp_path, monkeypatch):
    """The success path: a well formed `ATLAS_FALLBACK_MODEL` with its SDK importable builds a
    `GatewayChatModel` wired the same way `_gateway` wires the primary model (same `mode`,
    `cassette_dir`, a real `inner` from `replay.providers.build_chat_model`). `build_chat_model`
    itself is monkeypatched (constructing a real provider client needs no key at construction for
    every provider this repo supports, but this stays hermetic and independent of what happens to
    be importable in a given venv, mirroring the uninstalled SDK test's own use of a fake spec)."""
    import importlib.util

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatResult

    from atlas import server
    from replay.gateway import GatewayChatModel

    class _SentinelInner(BaseChatModel):
        """A minimal real `BaseChatModel`: `GatewayChatModel.inner` is a pydantic field typed
        `BaseChatModel`, so a bare `object()` fails validation -- this stands in for whatever
        `replay.providers.build_chat_model` would have constructed, identity checked below."""

        @property
        def _llm_type(self) -> str:
            return "sentinel-inner"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            raise NotImplementedError("never called: this test only checks wiring")

    monkeypatch.setenv("ATLAS_FALLBACK_MODEL", "anthropic:claude-haiku-4-5-20251001")
    monkeypatch.setattr(importlib.util, "find_spec", lambda *_a, **_k: object())  # every module "installed"
    sentinel_inner = _SentinelInner()
    seen = []

    def _fake_build(provider, model_id):
        seen.append((provider, model_id))
        return sentinel_inner

    monkeypatch.setattr("replay.providers.build_chat_model", _fake_build)

    gw = server._fallback_gateway("live", tmp_path)
    assert isinstance(gw, GatewayChatModel)
    assert gw.model_id == "anthropic:claude-haiku-4-5-20251001"
    assert gw.mode == "live"
    assert gw.inner is sentinel_inner
    assert seen == [("anthropic", "claude-haiku-4-5-20251001")]
