"""The HTTP contract: the service boots from domain/ + specs/, and /ask
refuses to run without the caller's bearer token."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    from app.config import get_settings

    monkeypatch.setenv("SERVICE_CUSTOMERS_URL", "http://customers.test")
    monkeypatch.setenv("SERVICE_ORDERS_URL", "http://orders.test")
    settings = get_settings()
    monkeypatch.setattr(settings, "checkpoint_sqlite_path", str(tmp_path / "cp.sqlite"))
    monkeypatch.setattr(settings, "obs_sqlite_path", str(tmp_path / "obs.sqlite"))
    from app.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_health_reports_what_loaded(client):
    body = client.get("/health").json()
    assert body["skills"] >= 2 and body["agents"] >= 1
    assert body["model_plane"]["gateway"].endswith("/gateway/mlflow/v1")


def test_ask_requires_a_bearer_token(client):
    r = client.post("/ask", json={"question": "who is Globex"})
    assert r.status_code == 401
    assert "bearer token" in r.json()["detail"]


def test_tools_catalogue_lists_every_service_operation(client):
    body = client.get("/tools/mcp", params={"schemas": "false"}).json()
    by_name = {t["name"]: t for t in body["tools"]}
    assert by_name["get_customer"]["service"] == "customers" and by_name["get_customer"]["callable"]
    assert by_name["orders.cancelOrder"]["access"] == "write" and not by_name["orders.cancelOrder"]["callable"]


def test_a_missing_service_url_fails_at_boot(monkeypatch, tmp_path):
    from app.app import create_app
    from app.config import get_settings
    from app.tools.openapi.server import ServiceNotConfigured

    monkeypatch.delenv("SERVICE_ORDERS_URL", raising=False)
    monkeypatch.setenv("SERVICE_CUSTOMERS_URL", "http://customers.test")
    monkeypatch.setattr(get_settings(), "checkpoint_sqlite_path", str(tmp_path / "cp.sqlite"))
    with pytest.raises(ServiceNotConfigured, match="SERVICE_ORDERS_URL"):
        with TestClient(create_app()):
            pass
