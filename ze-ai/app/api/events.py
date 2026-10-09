"""The stable event envelope the frontend consumes over SSE. One envelope
class with a typed `type` and a payload dict, so app/api/adapter.py stays a
thin, obviously-correct mapping layer."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class UIEvent(BaseModel):
    type: str
    seq: int
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict = Field(default_factory=dict)

    def to_sse(self) -> str:
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"
