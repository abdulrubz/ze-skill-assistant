"""Bearer-token check shared by both services.

The services verify an HS256 JWT signed with MOCK_JWT_SECRET. That stands in
for the org's identity provider: ze-ai forwards the end user's own token to
every service it calls, so these services see -- and log -- the real user,
never a shared service account. A missing or bad token is a 401, which is how
a broken pass-through shows up in a demo.
"""

from __future__ import annotations

import logging
import os
import time

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("ze.auth")

ALGORITHM = "HS256"
_bearer = HTTPBearer(auto_error=False, description="The end user's JWT, forwarded unchanged by ze-ai")


def secret() -> str:
    return os.environ.get("MOCK_JWT_SECRET", "ze-dev-secret-change-me")


def mint(sub: str, name: str | None = None, ttl_s: int = 8 * 3600) -> str:
    now = int(time.time())
    return jwt.encode({"sub": sub, "name": name or sub, "iat": now, "exp": now + ttl_s}, secret(), ALGORITHM)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    if creds is None:
        raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = jwt.decode(creds.credentials, secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(401, f"invalid token: {exc}", headers={"WWW-Authenticate": "Bearer"}) from exc
    logger.info("request by user=%s", claims.get("sub"))
    return claims
