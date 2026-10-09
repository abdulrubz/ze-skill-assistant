"""Admission: deterministic, no LLM. Injection screen and scope check. A
refusal here still exits through the output skill rather than dead-ending the
stream."""

from __future__ import annotations

import re

from app.graph.state import AdmissionVerdict, PipelineState

_INJECTION = [
    re.compile(r"ignore (all|any|the) (previous|prior|above) instructions", re.I),
    re.compile(r"you are now\b", re.I),
    re.compile(r"reveal your (system prompt|instructions)", re.I),
    re.compile(r"disregard (your|the) (rules|guidelines|policy|instructions)", re.I),
    re.compile(r"print your (system )?prompt", re.I),
]
MAX_QUESTION_LEN = 4000


async def admission(state: PipelineState) -> dict:
    q = state["question"]
    if not q or not q.strip():
        return {"admission": AdmissionVerdict(allowed=False, reason="empty question")}
    if len(q) > MAX_QUESTION_LEN:
        return {"admission": AdmissionVerdict(
            allowed=False, reason=f"question exceeds {MAX_QUESTION_LEN} characters")}
    for pattern in _INJECTION:
        if pattern.search(q):
            return {"admission": AdmissionVerdict(
                allowed=False, reason="question matched a prompt-injection pattern and was refused")}
    if not state["agent"].skills:
        return {"admission": AdmissionVerdict(
            allowed=False, reason=f"agent '{state['agent'].id}' has no granted skills")}
    return {"admission": AdmissionVerdict(allowed=True, reason=None)}
