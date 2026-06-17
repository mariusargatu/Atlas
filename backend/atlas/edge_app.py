"""The MCP resource servers as an ASGI app, for the OAuth integration lane (ADR-027).

Bearer token identity, scope gated, run in process via ASGI (no network) so the lane is
hermetic and can gate. Identity is taken from the validated token's claims and used locally.
The raw token is never forwarded upstream.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException

from atlas.auth import TokenError, bearer_context


def make_app(clock):
    app = FastAPI()

    def require_scope(scope: str):
        # same bearer_context helper as the product API, so a missing header or a bare token
        # without the "Bearer " prefix is rejected identically (401) on both edges.
        def dependency(authorization: str | None = Header(default=None)) -> dict:
            try:
                ctx = bearer_context(authorization, clock.now())
            except TokenError:
                raise HTTPException(status_code=401, detail="missing or invalid bearer token")
            if scope not in ctx["scopes"]:
                raise HTTPException(status_code=403, detail=f"missing scope: {scope}")
            return ctx

        return dependency

    @app.get("/account/summary")
    def account_summary(ctx: dict = Depends(require_scope("read"))) -> dict:
        # identity from the validated token claims, used locally, no token passthrough
        return {"customer_id": ctx["customer_id"]}

    @app.post("/actions/change_plan")
    def change_plan(ctx: dict = Depends(require_scope("write"))) -> dict:
        return {"customer_id": ctx["customer_id"], "applied": True}

    return app
