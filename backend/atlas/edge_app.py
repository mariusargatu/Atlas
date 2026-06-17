"""The MCP resource servers as an ASGI app, for the OAuth integration lane (ADR-027).

Bearer token identity, scope gated, run in process via ASGI (no network) so the lane is
hermetic and can gate. Identity is taken from the validated token's claims and used locally.
The raw token is never forwarded upstream.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException

from atlas.auth import TokenError, validate_token


def make_app(clock):
    app = FastAPI()

    def require_scope(scope: str):
        def dependency(authorization: str = Header(...)) -> dict:
            token = authorization.removeprefix("Bearer ").strip()
            try:
                ctx = validate_token(token, clock.now())
            except TokenError:
                raise HTTPException(status_code=401, detail="invalid token")
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
