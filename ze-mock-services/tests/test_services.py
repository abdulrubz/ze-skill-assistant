from fastapi.testclient import TestClient

from common.auth import mint
from services.customers.app import app as customers
from services.orders.app import app as orders

H = {"Authorization": f"Bearer {mint('test')}"}


def test_token_required():
    assert TestClient(customers).get("/customers").status_code == 401
    assert TestClient(orders).get("/orders", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_customers_and_orders_line_up():
    cs = TestClient(customers).get("/customers", headers=H).json()
    assert len(cs) == 30
    top = TestClient(orders).get("/reports/top-customers", params={"currency": "USD", "limit": 3}, headers=H).json()
    assert [t["rank"] for t in top] == [1, 2, 3]
    ids = {c["customer_id"] for c in cs}
    assert all(t["customer_id"] in ids for t in top)


def test_every_operation_documents_returns():
    for app in (customers, orders):
        for path in app.openapi()["paths"].values():
            for op in path.values():
                assert "**Returns:**" in op.get("description", ""), op["operationId"]
