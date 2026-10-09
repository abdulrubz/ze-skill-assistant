"""The final Answer -- typed blocks, not markdown-with-numbers. Every numeric
leaf is a MetricBlock built deterministically from an already-validated Fact,
never freehanded. This is what the UI's citation chips render and what
app/validation/groundedness.py checks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.skills.envelope import Fact


class TextSpan(BaseModel):
    kind: Literal["text"] = "text"
    text: str = Field(description="Prose; may contain [f_01] citation markers")


class MetricBlock(BaseModel):
    kind: Literal["metric"] = "metric"
    label: str
    ref: str
    display: str


class TableColumn(BaseModel):
    key: str
    label: str


class TableBlock(BaseModel):
    kind: Literal["table"] = "table"
    title: str | None = None
    columns: list[TableColumn]
    rows: list[dict[str, str]]


Block = TextSpan | MetricBlock | TableBlock


class Answer(BaseModel):
    blocks: list[Block]
    skills_used: list[str]
    facts: list[Fact] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    degraded: bool = Field(
        default=False, description="True when validation exhausted retries; best-effort answer"
    )
