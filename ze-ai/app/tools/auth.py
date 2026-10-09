"""The end user's token, forwarded to every service call.

Every request a tool makes carries the CALLER's own bearer token, unchanged,
so each microservice authorizes the real user exactly as it would for its own
UI -- the assistant can never see more than the person asking. There is no
service account.

The token rides in a context variable, set by the tool chokepoint
(app/tools/authz.py) from `Deps.user_token` around each tool execution, and
read by an httpx request hook on every service client. It is never put in
graph state, trace events, prompts or logs, so neither the model nor Langfuse
nor the checkpointer ever holds it.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import httpx

_USER_TOKEN: ContextVar[str | None] = ContextVar("ze_user_token", default=None)


class MissingUserToken(RuntimeError):
    pass


@contextmanager
def acting_as(token: str | None) -> Iterator[None]:
    reset = _USER_TOKEN.set(token)
    try:
        yield
    finally:
        _USER_TOKEN.reset(reset)


async def attach_user_token(request: httpx.Request) -> None:
    """httpx request hook. Refuses to send a call without a user token rather
    than letting it go out anonymous: a 401 from the service would say the
    same thing less clearly."""
    token = _USER_TOKEN.get()
    if not token:
        raise MissingUserToken(
            f"no user token for {request.method} {request.url.host}: the caller's bearer token is "
            "forwarded to every service, and this run has none"
        )
    request.headers["Authorization"] = f"Bearer {token}"


def bearer_from_header(value: str | None) -> str | None:
    """The token out of an `Authorization: Bearer ...` header, or None."""
    if not value:
        return None
    scheme, _, token = value.strip().partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None
