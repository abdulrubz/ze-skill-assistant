#!/usr/bin/env python
"""The MLflow AI Gateway endpoints this app calls.

    python scripts/mlflow_gateway.py show           # do the endpoints exist; what budgets are set
    python scripts/mlflow_gateway.py budget 50      # $50/day per endpoint, reject over
    python scripts/mlflow_gateway.py retention 2    # how to keep traces 2 days
    python scripts/mlflow_gateway.py dev-provision  # LOCAL DEV ONLY: Gemini-backed endpoints

The app knows endpoint NAMES only (GATEWAY_ENDPOINT, GATEWAY_JUDGE_ENDPOINT,
optional per-role overrides). What model serves each endpoint is configured in
MLflow. In a real environment create the endpoints in the MLflow UI (AI
Gateway -> Endpoints), e.g. pointing at a Databricks serving endpoint, using
those names. Nothing in this app changes when the model behind them does.

`dev-provision` creates them on a gateway you run yourself, backed by Gemini
(GOOGLE_API_KEY from the environment; stored in MLflow's encrypted secret
store, never printed), so the whole stack can be exercised on a laptop.

Two decisions carried over, each measured rather than assumed:

**No fallback on the structured roles.** The planner, extraction and judge use
structured output against tuned schemas. A gateway fallback silently swaps the
model at the worst moment -- and fires even for a permanent 400. Give the
`ze-ai` endpoint a fallback only if you accept that.

**Retention is tied to the budget window.** MLflow works out spend by reading
saved traces, and those traces hold full request text. Deleting them resets
the spend counter, so retention must stay LONGER than the budget window.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from app.config import get_settings  # noqa: E402
from app.llm import ROLES  # noqa: E402

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def _store():
    os.environ["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI") or get_settings().mlflow_gateway_url
    from mlflow.tracking._tracking_service.utils import _get_store

    return _get_store()


def _endpoints(settings) -> dict[str, str]:
    """Role -> endpoint name; several roles may share one."""
    return {role: settings.endpoint_for(role) for role in ROLES}


def dev_provision(model: str, judge_model: str) -> int:
    """Gemini-backed endpoints for local development."""
    from mlflow.entities.gateway_endpoint import GatewayEndpointModelConfig, GatewayModelLinkageType

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print(f"{RED}export GOOGLE_API_KEY first{RESET} -- it is stored in MLflow's secret store, never in this app")
        return 1
    store = _store()
    settings = get_settings()
    print(f"gateway: {BOLD}{os.environ['MLFLOW_TRACKING_URI']}{RESET}")
    existing_defs = {d.name: d for d in store.list_gateway_model_definitions()}
    try:
        secret_id = store.create_gateway_secret(secret_name="ze-ai-dev-gemini", secret_value={"api_key": key},
                                                provider="gemini").secret_id
        print(f"  {GREEN}secret{RESET} ze-ai-dev-gemini {DIM}(stored encrypted, not printed){RESET}")
    except Exception:  # noqa: BLE001 -- almost always "already exists"; there is no list_gateway_secrets
        secret_id = next((d.secret_id for n, d in existing_defs.items() if n.startswith("ze-ai-dev-")), None)
        if secret_id is None:
            print(f"{RED}could not create or find the dev secret{RESET}")
            return 1
        print(f"  {DIM}secret ze-ai-dev-gemini already exists, reusing it{RESET}")

    wanted = {settings.endpoint_for(r): (judge_model if r == "judge" else model) for r in ROLES}
    existing = {e.name for e in store.list_gateway_endpoints()}
    for endpoint, m in wanted.items():
        if endpoint in existing:
            print(f"  {DIM}endpoint {endpoint} already exists, left alone{RESET}")
            continue
        name = f"ze-ai-dev-{m}"
        definition = existing_defs.get(name) or store.create_gateway_model_definition(
            name=name, secret_id=secret_id, provider="gemini", model_name=m)
        existing_defs[name] = definition
        store.create_gateway_endpoint(
            name=endpoint, usage_tracking=True,
            model_configs=[GatewayEndpointModelConfig(model_definition_id=definition.model_definition_id,
                                                      linkage_type=GatewayModelLinkageType.PRIMARY)])
        print(f"  {GREEN}endpoint{RESET} {endpoint} -> gemini/{m}")
    print(f"{DIM}Verify with: python scripts/doctor.py{RESET}")
    return 0


def budget(amount: float, days: int = 1) -> int:
    from mlflow.entities.gateway_budget_policy import (
        BudgetAction, BudgetDuration, BudgetDurationUnit, BudgetTargetScope, BudgetUnit,
    )

    store = _store()
    endpoints = {e.name: e for e in store.list_gateway_endpoints()}
    settings = get_settings()
    wanted = set(_endpoints(settings).values())
    missing = wanted - set(endpoints)
    if missing:
        print(f"{RED}no such endpoint: {', '.join(sorted(missing))}{RESET} -- run provision first")
        return 1

    for name in sorted(wanted):
        store.create_budget_policy(
            budget_unit=BudgetUnit.USD,
            budget_amount=amount,
            duration=BudgetDuration(unit=BudgetDurationUnit.DAYS, value=days),
            target_scope=BudgetTargetScope.ENDPOINT,
            budget_action=BudgetAction.REJECT,
            target_value=endpoints[name].endpoint_id,
        )
        print(f"  {GREEN}budget{RESET} {name}: ${amount:.2f} per {days}d, reject over")

    print(f"\n{DIM}Set this BELOW your real cap: MLflow leaves thinking tokens out of\n"
          f"output_tokens, so its spend figure runs low. This app's ledger counts them.{RESET}")
    print(f"{DIM}A rejection arrives as HTTP 429 -- app/llm.py's is_budget_exceeded()\n"
          f"tells it apart from a provider rate limit.{RESET}")
    print(f"{DIM}The gateway refreshes its policy cache periodically, so a new budget can\n"
          f"take up to a minute to start rejecting. Calls in that gap still go through.{RESET}")
    return 0


def retention(days: int) -> int:
    """Traces hold the full request text. Keep them no longer than needed --
    but never shorter than the budget window, or spend resets to zero."""
    print(f"{BOLD}Trace retention{RESET}")
    print(f"  MLflow deletes traces on a schedule its server runs. Configure it with\n"
          f"  MLFLOW_TRACE_ARCHIVAL_CONFIG on the gateway process, or prune directly:\n")
    print(f"    {DIM}mlflow.client.MlflowClient().delete_traces("
          f"experiment_id=..., max_timestamp_millis=<now - {days}d>){RESET}\n")
    print(f"  {RED}Do not set this shorter than your budget window.{RESET} MLflow computes\n"
          f"  spend by reading saved traces, so deleting them inside an open window\n"
          f"  resets the counter and the budget stops enforcing.")
    print(f"  Daily budget -> keep {BOLD}2 days{RESET}. Monthly budget -> keep ~35 days.")
    return 0


def show() -> int:
    store = _store()
    settings = get_settings()
    print(f"gateway: {BOLD}{os.environ['MLFLOW_TRACKING_URI']}{RESET}\n")
    endpoints = {e.name: e for e in store.list_gateway_endpoints()}
    print(f"{BOLD}endpoints{RESET}")
    for role, name in _endpoints(settings).items():
        found = endpoints.get(name)
        mark = f"{GREEN}ok{RESET}" if found else f"{RED}MISSING{RESET}"
        tracking = f"tracking={found.usage_tracking}" if found else ""
        print(f"  {role:8s} -> {name:24s} {mark} {DIM}{tracking}{RESET}")
    policies = store.list_budget_policies()
    print(f"\n{BOLD}budgets{RESET}")
    if not policies:
        print(f"  {DIM}none set -- spend is uncapped{RESET}")
    for p in policies:
        target = next((n for n, e in endpoints.items() if e.endpoint_id == p.target_value), p.target_value)
        print(f"  {target}: ${p.budget_amount:.2f} per {p.duration.value} "
              f"{p.duration.unit.value.lower()} ({p.budget_action.value})")
    return 0


USAGE = __doc__


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    command = args[0]
    try:
        if command == "dev-provision":
            model = args[1] if len(args) > 1 else "gemini-3.8-flash"
            judge = args[2] if len(args) > 2 else "gemini-3.5-flash-lite"
            return dev_provision(model, judge)
        if command == "budget":
            if len(args) < 2:
                print(f"{RED}budget needs an amount{RESET}: python scripts/mlflow_gateway.py budget 50")
                return 1
            return budget(float(args[1]), int(args[2]) if len(args) > 2 else 1)
        if command == "retention":
            return retention(int(args[1]) if len(args) > 1 else 2)
        if command == "show":
            return show()
    except ModuleNotFoundError:
        print(f"{RED}mlflow is not installed{RESET} -- pip install -e '.[gateway]'")
        return 1
    print(f"{RED}unknown command {command!r}{RESET}\n\n{USAGE}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
