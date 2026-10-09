"""SSE framing."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.api.events import UIEvent

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Anything nginx-ish in front would otherwise buffer the whole stream.
    "X-Accel-Buffering": "no",
}


async def to_sse_bytes(events: AsyncIterator[UIEvent]) -> AsyncIterator[str]:
    async for event in events:
        yield event.to_sse()
