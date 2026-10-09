"""Model plane: every model call goes through an MLflow AI Gateway endpoint.

The app knows endpoint NAMES only. Which provider and model sit behind an
endpoint (a Databricks serving endpoint, OpenAI, Gemini...) and the key that
reaches it are configured in MLflow, so changing model is an MLflow change,
never a change here.

Requests use MLflow's provider-agnostic, OpenAI-shaped route
(`/gateway/mlflow/v1/chat/completions`, `model` = endpoint name). MLflow
translates to the provider's dialect, applies budgets and usage tracking, and
reports the real model id back in the response -- which is what the cost
ledger prices on (app/obs/pricing.yaml).

⚠️ Whatever the endpoint's provider is, the question text and every tool
result the model sees are sent to it. Check that against your data-handling
obligations before pointing this at production data.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.config import get_settings

logger = logging.getLogger(__name__)

ROLES = ("planner", "skill", "prose", "judge")
_SIGNATURES = "thought_signatures"


class ModelPlaneError(RuntimeError):
    pass


@lru_cache
def _mlflow_chat_class():
    from langchain_openai import ChatOpenAI

    class MlflowChat(ChatOpenAI):
        """ChatOpenAI against MLflow's unified route, plus two things it drops.

        1. Thought signatures. Some providers (Gemini 3) attach an opaque
           signature to each tool call and reject the next turn unless it comes
           back. MLflow carries it as `tool_calls[].thought_signature` both ways,
           but ChatOpenAI discards unknown fields, so every multi-turn tool loop
           failed with "Function call is missing a thought_signature". Kept on
           the AIMessage and re-attached by tool-call id. A no-op for providers
           that never send one (Databricks, OpenAI).
        2. Reasoning tokens. Providers that think may report them outside
           completion_tokens (total > prompt + completion). They bill as output,
           so they are added to output_tokens (and noted as reasoning) rather
           than silently missing from the ledger.
        """

        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            raw = response if isinstance(response, dict) else response.model_dump()
            for gen, choice in zip(result.generations, raw.get("choices") or []):
                calls = (choice.get("message") or {}).get("tool_calls") or []
                sigs = {c["id"]: c["thought_signature"] for c in calls if c.get("thought_signature")}
                if sigs:
                    gen.message.additional_kwargs[_SIGNATURES] = sigs
            usage = raw.get("usage") or {}
            hidden = (int(usage.get("total_tokens") or 0) - int(usage.get("prompt_tokens") or 0)
                      - int(usage.get("completion_tokens") or 0))
            if hidden > 0:
                for gen in result.generations:
                    meta = getattr(gen.message, "usage_metadata", None)
                    if meta:
                        details = dict(meta.get("output_token_details") or {})
                        details["reasoning"] = int(details.get("reasoning") or 0) + hidden
                        meta["output_token_details"] = details
                        meta["output_tokens"] = int(meta.get("output_tokens") or 0) + hidden
            return result

        def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            sigs: dict[str, str] = {}
            for m in self._convert_input(input_).to_messages():
                if isinstance(m, AIMessage):
                    sigs.update(m.additional_kwargs.get(_SIGNATURES) or {})
            if sigs:
                for msg in payload.get("messages", []):
                    for call in msg.get("tool_calls") or []:
                        if call.get("id") in sigs:
                            call["thought_signature"] = sigs[call["id"]]
            return payload

    return MlflowChat


@lru_cache
def _rate_limiter():
    """One token bucket for every model in the process, or None. Rides on the
    model as a field, so it survives bind_tools/with_structured_output and is
    shared (not copied) by named()."""
    settings = get_settings()
    rps = float(settings.model_requests_per_second or 0)
    if rps <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=rps,
        max_bucket_size=max(float(settings.model_rate_limit_burst or 1), 1.0),
        check_every_n_seconds=0.05,
    )


# MLflow wraps the provider's error in its own summary when a fallback is
# configured: "All 2 fallback attempts failed. Last error: ...".
_FALLBACK_WRAPPER = re.compile(r"All \d+ fallback attempts? failed\.\s*Last error:\s*", re.I)


def describe_model_error(exc: BaseException) -> str:
    """The provider's own error text, not the gateway's wrapper around it."""
    text = str(exc)
    match = _FALLBACK_WRAPPER.search(text)
    return (text[match.end():].strip() or text) if match else text


def is_budget_exceeded(exc: BaseException) -> bool:
    """MLflow answers an exhausted budget with 429 -- the same status as a
    provider rate limit -- so it is told apart by its wording."""
    return "budget limit exceeded" in str(exc).lower()


def _build(endpoint: str, *, temperature: float) -> BaseChatModel:
    settings = get_settings()
    return _mlflow_chat_class()(
        base_url=settings.gateway_chat_base,
        # The OpenAI client insists on a key; MLflow holds the real provider
        # key. It is sent as a bearer token, so it doubles as the gateway's own
        # auth token when the gateway has auth on.
        api_key=settings.mlflow_gateway_token or "unused-mlflow-holds-the-provider-key",
        model=endpoint,
        temperature=temperature,
        timeout=settings.model_timeout_s,
        max_retries=settings.model_max_retries,
        rate_limiter=_rate_limiter(),
    )


@lru_cache
def planner_llm() -> BaseChatModel:
    """Planning and fact extraction: structured output, deterministic."""
    return _build(get_settings().endpoint_for("planner"), temperature=0.0)


@lru_cache
def skill_llm() -> BaseChatModel:
    """Each skill's tool loop. Tool-argument choice should not be creative."""
    return _build(get_settings().endpoint_for("skill"), temperature=0.0)


@lru_cache
def prose_llm() -> BaseChatModel:
    """Output skills only -- phrasing, never a structured decision."""
    return _build(get_settings().endpoint_for("prose"), temperature=0.3)


@lru_cache
def judge_llm(endpoint: str | None = None) -> BaseChatModel:
    """The eval judge (scripts/langfuse_eval.py), on its own endpoint."""
    return _build(endpoint or get_settings().endpoint_for("judge"), temperature=0.0)


def named(model: BaseChatModel, name: str) -> BaseChatModel:
    """The same model under a name Langfuse can group cost by (`planner`,
    `skill:<id>`, `prose`, matching the ledger's units).

    It must be the model's own `name`: a `.with_config(run_name=...)` is dropped
    by with_structured_output, which rebuilds the chain around the bare model.
    model_copy is shallow, so the client is shared, not rebuilt."""
    copy = getattr(model, "model_copy", None)
    return copy(update={"name": name}) if copy else model


def reset_model_cache() -> None:
    for factory in (planner_llm, skill_llm, prose_llm, judge_llm):
        factory.cache_clear()
    _rate_limiter.cache_clear()


def message_text(content: Any) -> str:
    """Flatten a chat message's content to plain text. Some providers return a
    LIST of typed blocks; str() on that leaks reprs into answers."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text") and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def describe_model_plane() -> dict:
    settings = get_settings()
    return {
        "gateway": settings.gateway_chat_base,
        "endpoints": {role: settings.endpoint_for(role) for role in ROLES},
        "rate_limit_rps": settings.model_requests_per_second or None,
    }
