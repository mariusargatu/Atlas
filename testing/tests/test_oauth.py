"""P auth, the OAuth integration lane (ADR-027). Hermetic: in process ASGI, local key,
frozen clock. The three gating tests the call context PR lane cannot do.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from determinism.sources import FrozenClock

from atlas.auth import issue_token
from atlas.edge_app import make_app

CLOCK = FrozenClock(datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc))


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_bearer_token_validation():
    app = make_app(CLOCK)
    token = issue_token("cust_current", ["read"], CLOCK.now())
    async with _client(app) as c:
        ok = await c.get("/account/summary", headers={"Authorization": f"Bearer {token}"})
        assert ok.status_code == 200 and ok.json()["customer_id"] == "cust_current"
        bad = await c.get("/account/summary", headers={"Authorization": "Bearer garbage"})
        assert bad.status_code == 401


@pytest.mark.asyncio
async def test_step_up_scope_rejection_is_distinct_from_unreachable():
    app = make_app(CLOCK)
    read_only = issue_token("cust_current", ["read"], CLOCK.now())
    async with _client(app) as c:
        # the same token that reads fine is rejected at the write endpoint (step up)
        denied = await c.post("/actions/change_plan", headers={"Authorization": f"Bearer {read_only}"})
        assert denied.status_code == 403
        write = issue_token("cust_current", ["read", "write"], CLOCK.now())
        ok = await c.post("/actions/change_plan", headers={"Authorization": f"Bearer {write}"})
        assert ok.status_code == 200


@pytest.mark.asyncio
async def test_token_passthrough_prohibition():
    app = make_app(CLOCK)
    token = issue_token("cust_legacy_term", ["read"], CLOCK.now())
    async with _client(app) as c:
        r = await c.get("/account/summary", headers={"Authorization": f"Bearer {token}"})
        body = r.json()
        assert body["customer_id"] == "cust_legacy_term"      # identity from claims, used locally
        assert token not in str(body)                          # the raw token is not forwarded/echoed
