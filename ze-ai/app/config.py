"""Settings. Every value is overridable by an environment variable of the same
name (case-insensitive), or by ze-ai/.env. Real environment variables win.

Only URLs, credentials and runtime knobs live here. What the assistant can do
(skills, agents, tools, vocabulary) is domain/, not configuration."""

from __future__ import annotations

import os
import re
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.paths import ROOT, VAR

# Tests set ZE_AI_ENV_FILE="" so a developer's .env cannot leak into them.
ENV_FILE = os.environ.get("ZE_AI_ENV_FILE", str(ROOT / ".env")) or None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model plane: MLflow AI Gateway, and nothing else ---
    # The app knows MLflow endpoints, never providers or model ids. Which model
    # answers (Databricks, OpenAI, Gemini...) and its key live in MLflow, so a
    # model change is an MLflow change and never a deploy of this app.
    #
    # Requests go to MLflow's provider-agnostic, OpenAI-shaped route:
    #   {mlflow_gateway_url}/gateway/mlflow/v1/chat/completions  (model = endpoint name)
    mlflow_gateway_url: str = "http://127.0.0.1:5050"
    # Sent as a bearer token if the gateway has auth on. Blank = no header.
    mlflow_gateway_token: str = ""

    # One endpoint per role so budgets can be scoped per role (MLflow scopes
    # budgets per endpoint). Blank falls back to gateway_endpoint.
    gateway_endpoint: str = "ze-ai"
    gateway_planner_endpoint: str = ""
    gateway_skill_endpoint: str = ""
    gateway_prose_endpoint: str = ""
    # Its own endpoint by default: the eval judge usually runs a lighter model,
    # and an endpoint carries one model.
    gateway_judge_endpoint: str = "ze-ai-judge"

    model_timeout_s: float = 120.0
    # The gateway retries against its fallback model itself (even for a
    # permanent 400), so client-side retries multiply upstream calls. Keep low.
    model_max_retries: int = 1

    # MLflow caps SPEND but not request RATE, so the rate limit lives here, at
    # the one place every model call passes through. 0 = off.
    model_requests_per_second: float = 0.0
    model_rate_limit_burst: float = 1.0

    # --- Tools: the microservices ---
    # Tools are generated from the spec repo's OpenAPI files (see specs.lock.json
    # and scripts/tool_spec.py). Each service's base URL comes from
    # SERVICE_<NAME>_URL, e.g. SERVICE_ORDERS_URL -- see service_url().
    spec_repo_url: str = ""
    spec_repo_ref: str = "main"
    # Local checkout to sync from instead of cloning spec_repo_url (dev only).
    spec_repo_path: str = ""

    # Every service call carries the END USER's own bearer token, forwarded
    # unchanged (app/tools/auth.py). The API takes it from the request's
    # Authorization header. Scripts, evals and the CLI have no end user, so
    # they use this one. Never set it on a deployed server.
    ze_dev_token: str = ""
    service_timeout_s: float = 30.0

    # "Today" for every relative date ("last month"). Blank = the real date.
    # Pin it for demos and evals over fixed data, so answers stay stable.
    as_of_date: str = ""

    # --- Langfuse: tracing, prompt management, evals ---
    # Unset keys = tracing off: nothing is imported and no network call is made.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"
    # Serve prompts from Langfuse Prompt Management (editable in its UI without
    # a deploy). The repo files stay the fallback. See scripts/prompts.py.
    langfuse_prompts: bool = False
    langfuse_prompt_label: str = "production"
    langfuse_prompt_cache_ttl: int = 60
    # Prepended to every prompt name in Langfuse (e.g. "ze-ai/" ->
    # "ze-ai/pipeline/planner"), so several apps can share one project without
    # overwriting each other's prompts. Datasets get it too (langfuse_eval.py).
    langfuse_prompt_prefix: str = ""
    # Langfuse's environment attribute on every trace (e.g. dev, staging, prod):
    # one project, filterable per deployment.
    langfuse_environment: str = ""

    # --- Runtime ---
    checkpoint_sqlite_path: str = str(VAR / "checkpoints.sqlite")
    # The cost/quality ledger: append-only, deliberately separate from the
    # constantly rewritten checkpoints.
    obs_sqlite_path: str = str(VAR / "observability.sqlite")
    recursion_limit: int = 80

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_allow_origins: list[str] = []

    _ROLE_ENDPOINTS = {
        "planner": "gateway_planner_endpoint",
        "skill": "gateway_skill_endpoint",
        "prose": "gateway_prose_endpoint",
        "judge": "gateway_judge_endpoint",
    }

    def endpoint_for(self, role: str) -> str:
        field = self._ROLE_ENDPOINTS.get(role)
        endpoint = (getattr(self, field, "") if field else "") or self.gateway_endpoint
        if not endpoint:
            raise RuntimeError(
                f"no MLflow endpoint configured for role {role!r}: set GATEWAY_ENDPOINT "
                "(or GATEWAY_<ROLE>_ENDPOINT) to an endpoint that exists in the gateway"
            )
        return endpoint

    @property
    def gateway_chat_base(self) -> str:
        return self.mlflow_gateway_url.rstrip("/") + "/gateway/mlflow/v1"


def service_env_var(service: str) -> str:
    return "SERVICE_" + re.sub(r"[^A-Z0-9]", "_", service.upper()) + "_URL"


def service_url(service: str) -> str | None:
    """SERVICE_<NAME>_URL from the environment or .env, e.g. SERVICE_ORDERS_URL."""
    key = service_env_var(service)
    if key in os.environ:
        return os.environ[key]
    if ENV_FILE and os.path.exists(ENV_FILE):
        from dotenv import dotenv_values

        return dotenv_values(ENV_FILE).get(key)
    return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
