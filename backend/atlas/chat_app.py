"""The product API the Vite SPA talks to (ADR-028): auth + chat over the Atlas graph.

Thin edge over `build_atlas_graph`, no new agent logic. Identity comes from the validated bearer
token, never the request body (principle 1). `thread_id` is namespaced server side per customer so
one customer can never resume another's pending interrupt. The graph (with its checkpointer) is a
single long lived object, so a `/chat` interrupt and its `/chat/resume` share confirmation state.

Streaming is a non goal in v1: every turn is request/response. The model + graph are
injected so tests wire a replayed gateway and an in memory backend (hermetic), exactly like the
rest of the suite.
"""
from __future__ import annotations

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from atlas.auth import TokenError, issue_token, validate_token
from atlas.domain.accounts import SEED

_REFRESH_COOKIE = "atlas_refresh"
_ACCESS_TTL = 1800       # 30 min
_REFRESH_TTL = 7 * 24 * 3600

_bearer = HTTPBearer(auto_error=False)


# ---- request models ----
class LoginBody(BaseModel):
    customer_id: str


class ChatBody(BaseModel):
    message: str
    thread_id: str = "default"


class ResumeBody(BaseModel):
    thread_id: str
    confirmation: str


# ---- response models (so the generated TS client is fully typed, ADR-028 single source of truth) ----
class AuthOut(BaseModel):
    access_token: str
    customer_id: str
    name: str  # the customer's display name, so the UI never shows a raw internal id


class PendingOut(BaseModel):
    tool: str
    args: dict
    idempotency_key: str | None = None
    customer_id: str | None = None


class ResultOut(BaseModel):
    reference: str
    applied: bool


class ChatOut(BaseModel):
    type: str
    thread_id: str
    final_response: str | None = None
    pending: PendingOut | None = None
    result: ResultOut | None = None


def make_chat_app(clock, graph, *, cors_origins: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="Atlas")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_scope(scope: str):
        # Bearer as a security scheme (not a header parameter) so the generated client injects the
        # token via middleware, never as a per call argument, keeps identity out of the call site.
        def dependency(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
            if creds is None:
                raise HTTPException(status_code=401, detail="missing bearer token")
            try:
                ctx = validate_token(creds.credentials, clock.now())
            except TokenError:
                raise HTTPException(status_code=401, detail="invalid token")
            if scope not in ctx["scopes"]:
                raise HTTPException(status_code=403, detail=f"missing scope: {scope}")
            return ctx

        return dependency

    def _thread(customer_id: str, thread_id: str) -> dict:
        # namespace per customer. A client supplied thread_id can never address another's checkpoint
        return {"configurable": {"thread_id": f"{customer_id}::{thread_id}"}}

    # ---- auth ----
    @app.post("/auth/login", response_model=AuthOut)
    def login(body: LoginBody, response: Response) -> dict:
        """Demo sign in AS a seeded customer (no password). Real auth is out of scope."""
        if body.customer_id not in SEED:
            raise HTTPException(status_code=404, detail="unknown customer")
        now = clock.now()
        # least agency: a session starts read only. A write is acquired on the turn that needs it
        # via /auth/step-up (ADR-027 step up authorization), not handed out at login.
        access = issue_token(body.customer_id, ["read"], now, ttl_seconds=_ACCESS_TTL)
        refresh = issue_token(body.customer_id, ["read"], now, ttl_seconds=_REFRESH_TTL)
        response.set_cookie(
            _REFRESH_COOKIE, refresh, httponly=True, secure=True, samesite="strict", max_age=_REFRESH_TTL
        )
        return {"access_token": access, "customer_id": body.customer_id, "name": SEED[body.customer_id].name}

    @app.post("/auth/step-up", response_model=AuthOut)
    def step_up(ctx: dict = Depends(require_scope("read"))) -> dict:
        """Step up authorization: elevate an authenticated read session to a short lived write token
        for the turn that confirms an action. A production system gates this on re auth (password/MFA).
        The demo elevates on a valid read token. The write scope is never minted at login."""
        cid = ctx["customer_id"]
        access = issue_token(cid, ["read", "write"], clock.now(), ttl_seconds=_ACCESS_TTL)
        return {"access_token": access, "customer_id": cid, "name": SEED[cid].name}

    @app.post("/auth/refresh", response_model=AuthOut)
    def refresh(response: Response, atlas_refresh: str | None = Cookie(default=None)) -> dict:
        token = atlas_refresh
        if not token:
            raise HTTPException(status_code=401, detail="no refresh cookie")
        try:
            ctx = validate_token(token, clock.now())
        except TokenError:
            raise HTTPException(status_code=401, detail="invalid refresh token")
        access = issue_token(ctx["customer_id"], list(ctx["scopes"]), clock.now(), ttl_seconds=_ACCESS_TTL)
        return {"access_token": access, "customer_id": ctx["customer_id"], "name": SEED[ctx["customer_id"]].name}

    # ---- chat ----
    @app.post("/chat", response_model=ChatOut)
    async def chat(body: ChatBody, ctx: dict = Depends(require_scope("read"))) -> dict:
        customer_id = ctx["customer_id"]  # from the token claim, NEVER the body
        state = {"messages": [HumanMessage(body.message)], "session": {"customer_id": customer_id}}
        result = await graph.ainvoke(state, _thread(customer_id, body.thread_id))
        if "__interrupt__" in result:
            return {"type": "interrupt", "thread_id": body.thread_id, "pending": result.get("pending")}
        return {"type": "final", "thread_id": body.thread_id, "final_response": result["final_response"]}

    @app.post("/chat/resume", response_model=ChatOut)
    async def resume(body: ResumeBody, ctx: dict = Depends(require_scope("write"))) -> dict:
        customer_id = ctx["customer_id"]
        result = await graph.ainvoke(Command(resume=body.confirmation), _thread(customer_id, body.thread_id))
        return {
            "type": "final",
            "thread_id": body.thread_id,
            "final_response": result["final_response"],
            "result": result.get("result"),
        }

    return app
