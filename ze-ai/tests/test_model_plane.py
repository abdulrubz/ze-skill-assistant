"""The model plane speaks only MLflow's unified route, by endpoint name."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings, get_settings
from app.llm import _mlflow_chat_class, planner_llm, reset_model_cache


def _model():
    return _mlflow_chat_class()(base_url="http://gw.test/gateway/mlflow/v1", api_key="x", model="ze-ai")


def _response(tool_calls=None, usage=None) -> dict:
    return {
        "id": "r", "object": "chat.completion", "created": 0, "model": "gemini-3.8-flash",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop",
                     "message": {"role": "assistant", "content": None if tool_calls else "ok",
                                 "tool_calls": tool_calls}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_requests_name_an_endpoint_on_the_unified_route():
    reset_model_cache()
    m = planner_llm()
    assert m.model_name == get_settings().endpoint_for("planner")
    assert str(m.openai_api_base).endswith("/gateway/mlflow/v1")


def test_thought_signatures_survive_the_round_trip():
    """Gemini 3 rejects a tool-call turn sent back without its signature; MLflow
    carries it, ChatOpenAI dropped it, and every multi-turn tool loop failed."""
    model = _model()
    call = {"id": "call_1", "type": "function", "function": {"name": "get_order", "arguments": '{"order_id": "1"}'},
            "thought_signature": "SIG-abc"}
    ai = model._create_chat_result(_response([call])).generations[0].message
    assert ai.tool_calls[0]["id"] == "call_1"

    payload = model._get_request_payload([HumanMessage("q"), ai, ToolMessage("{}", tool_call_id="call_1")])
    sent = next(m for m in payload["messages"] if m.get("tool_calls"))
    assert sent["tool_calls"][0]["thought_signature"] == "SIG-abc"


def test_no_signature_means_no_extra_field():
    model = _model()
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ai = model._create_chat_result(_response([call])).generations[0].message
    payload = model._get_request_payload([HumanMessage("q"), ai, ToolMessage("{}", tool_call_id="c")])
    sent = next(m for m in payload["messages"] if m.get("tool_calls"))
    assert "thought_signature" not in sent["tool_calls"][0]


def test_reasoning_tokens_outside_completion_tokens_are_billed_as_output():
    model = _model()
    msg = model._create_chat_result(
        _response(usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 150})
    ).generations[0].message
    assert msg.usage_metadata["output_tokens"] == 50
    assert msg.usage_metadata["output_token_details"]["reasoning"] == 30


def test_roles_fall_back_to_the_shared_endpoint():
    s = Settings(gateway_endpoint="shared", gateway_skill_endpoint="skills-only", gateway_judge_endpoint="")
    assert s.endpoint_for("planner") == "shared"
    assert s.endpoint_for("skill") == "skills-only"
    assert s.endpoint_for("judge") == "shared"


def test_ai_message_type_is_kept():
    assert isinstance(_model()._create_chat_result(_response()).generations[0].message, AIMessage)
