"""The groundedness substrate. The answer is never markdown-with-numbers:
every value a skill asserts is a Fact carrying a ref id and a Provenance
pointing at the exact tool call that produced it. That is what makes "every
number traces back to a real tool call" mechanically checkable rather than a
matter of trusting the model -- see app/validation/."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    tool_call_id: str = Field(description="A real tool_call_id from the transcript")
    tool: str = Field(description="Tool name, e.g. get_order")
    source: Literal["api", "derived"] = "api"
    locator: str = Field(
        description="Where inside the result this came from, e.g. 'order:ORD-10001.total' or 'row#3.total'"
    )


class Derivation(BaseModel):
    op: Literal["sum", "sub", "ratio", "count", "pct"] = Field(
        description=(
            "How this value was computed from input_refs. Choose by the form of "
            "value_str, because it is replayed and must match exactly: "
            "'pct' when value_str is a PERCENTAGE like 4.9 (a/b*100) -- "
            "'ratio' only when value_str is a bare fraction like 0.049 (a/b); "
            "'sum' to add all inputs; 'sub' for first minus second (exactly 2 "
            "inputs); 'count' for how many inputs there are."
        )
    )
    input_refs: list[str] = Field(description="Fact refs this was computed from")


class Fact(BaseModel):
    ref: str = Field(description="Short citation id, unique within this skill result: f_01, f_02, ...")
    name: str = Field(description="snake_case name of what this is, e.g. order_total, revenue_q3")
    value_str: str = Field(description="The bare value as a string -- NO currency code appended")
    value_kind: Literal["decimal", "int", "str", "bool", "date"] = "decimal"
    currency: str | None = Field(default=None, description="USD/EUR/GBP if this is money")
    unit: str | None = Field(default=None, description="e.g. 'days', 'api-calls', 'accounts'")
    entity: dict[str, str] = Field(
        default_factory=dict, description="What it is about, e.g. {'customer_id': '...'}"
    )
    derivation: Derivation | None = None
    provenance: Provenance
    caveat: str | None = None


class SkillResult(BaseModel):
    skill_id: str
    status: Literal["ok", "partial", "error", "denied"]
    narrative: str = Field(description="Plain-language findings, citing facts as [f_01]")
    facts: list[Fact] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ToolTrace(BaseModel):
    tool_call_id: str
    tool: str
    args: dict
    ok: bool = True


class SkillExecutionResult(BaseModel):
    skill_id: str
    # Which plan step produced this. Identity for retry: a failing step is
    # re-run and REPLACES its own result, leaving every other step's work
    # intact. Keyed on the step rather than the skill id so a plan may legally
    # run the same skill twice with different inputs.
    step_index: int = 0
    result: SkillResult
    tool_trace: list[ToolTrace] = Field(default_factory=list)
    steps_used: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
