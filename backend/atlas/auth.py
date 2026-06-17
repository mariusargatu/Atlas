"""First party OAuth 2.1 token issuer + resource server validation (ADR-027).

A local fixed signing key (zero JWKS egress). `exp`/`iat` come from the injected clock so the
lane is deterministic and hermetic. Scopes carry least agency: a read only turn gets `read`, an
action turn gets `read write` (step up). The resource server validates the token and extracts
`customer_id` locally. It never passes the token upstream (the June 2025 MCP token passthrough
prohibition).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import jwt

_KEY = "atlas-local-dev-signing-key-not-a-production-secret-v1"  # first party, local, fixed
_ALG = "HS256"
_AUDIENCE = "atlas-mcp"


class TokenError(Exception):
    pass


def issue_token(customer_id: str, scopes: list[str], now: datetime, ttl_seconds: int = 3600) -> str:
    payload = {
        "sub": customer_id,
        "scope": " ".join(scopes),
        "aud": _AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
    }
    return jwt.encode(payload, _KEY, algorithm=_ALG)


def validate_token(token: str, now: datetime) -> dict:
    """Return {customer_id, scopes}. Raise TokenError on a bad/expired/wrong audience token."""
    try:
        claims = jwt.decode(
            token, _KEY, algorithms=[_ALG], audience=_AUDIENCE, options={"verify_exp": False}
        )
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc
    if int(claims["exp"]) < int(now.timestamp()):
        raise TokenError("token expired")
    return {"customer_id": claims["sub"], "scopes": set(claims.get("scope", "").split())}
