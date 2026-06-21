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
    """ADR-027 step up: the login token cannot confirm a write; a stepped up token can."""
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
