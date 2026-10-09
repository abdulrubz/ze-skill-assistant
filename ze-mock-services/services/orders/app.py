"""orders service: what customers bought.

Knows customers only by id -- names live in the customers service. A question
like "who are our top customers by spend" therefore needs both services,
which is exactly what ze-ai's skills and citations are meant to handle.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from common import data
from common.auth import current_user

logging.basicConfig(level=logging.INFO, format="%(asctime)s orders %(message)s")

app = FastAPI(
    title="orders",
    version="1.0.0",
    description="Orders and order lines, with per-customer and ranking reports.",
)
# Every data endpoint needs the end user's token; /health does not.
router = APIRouter(dependencies=[Depends(current_user)])

Status = Literal["placed", "paid", "shipped", "delivered", "cancelled", "refunded"]
# Cancelled and refunded orders are not revenue.
COUNTED = {"placed", "paid", "shipped", "delivered"}


class OrderLine(BaseModel):
    sku: str = Field(description="Product SKU")
    description: str = Field(description="Product name")
    quantity: int = Field(description="Units ordered")
    unit_price: float = Field(description="Price per unit in the order currency")
    amount: float = Field(description="quantity x unit_price")


class Order(BaseModel):
    order_id: str = Field(description="Order id, e.g. ORD-10001")
    customer_id: str = Field(description="Customer id (resolve names via the customers service)")
    created_at: str = Field(description="Order date (YYYY-MM-DD)")
    status: Status = Field(description="placed, paid, shipped, delivered, cancelled or refunded")
    currency: str = Field(description="ISO currency of the order")
    total: float = Field(description="Sum of line amounts, in the order currency")
    lines: list[OrderLine] = Field(description="The order lines")


class OrderSummary(BaseModel):
    order_id: str = Field(description="Order id")
    created_at: str = Field(description="Order date (YYYY-MM-DD)")
    status: Status = Field(description="Order status")
    currency: str = Field(description="ISO currency of the order")
    total: float = Field(description="Order total in its currency")


class CurrencyTotal(BaseModel):
    currency: str = Field(description="ISO currency")
    total: float = Field(description="Total of counted orders in this currency")
    order_count: int = Field(description="Counted orders in this currency")


class CustomerOrderSummary(BaseModel):
    customer_id: str = Field(description="Customer id")
    order_count: int = Field(description="All orders, any status")
    counted_order_count: int = Field(description="Orders that count as revenue (not cancelled or refunded)")
    totals: list[CurrencyTotal] = Field(description="Revenue per currency, counted orders only")
    open_orders: int = Field(description="Orders placed or paid but not yet shipped")
    first_order_date: str | None = Field(description="Date of the first order, null if none")
    last_order_date: str | None = Field(description="Date of the most recent order, null if none")


class TopCustomer(BaseModel):
    rank: int = Field(description="1 = highest spend")
    customer_id: str = Field(description="Customer id (names come from the customers service)")
    currency: str = Field(description="Currency of the total")
    total: float = Field(description="Revenue from counted orders in the window")
    order_count: int = Field(description="Counted orders in the window")


@app.get("/health", include_in_schema=False)
def health() -> dict:
    return {"status": "ok"}


def _window(rows, since: str | None, until: str | None):
    if since:
        rows = [o for o in rows if o["created_at"] >= since]
    if until:
        rows = [o for o in rows if o["created_at"] <= until]
    return rows


@router.get("/orders", operation_id="listOrders", response_model=list[OrderSummary],
         summary="List orders, newest first")
def list_orders(
    customer_id: str | None = Query(None, description="Only this customer's orders"),
    status: Status | None = Query(None, description="Only orders in this status"),
    since: str | None = Query(None, description="Orders on or after this date (YYYY-MM-DD)"),
    until: str | None = Query(None, description="Orders on or before this date (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=500, description="Maximum orders to return"),
) -> list[dict]:
    """Orders matching every filter, newest first, without their lines (use
    getOrder for lines).

    **Returns:** a list of orders with id, date, status, currency and total.
    """
    rows = _window(data.orders(), since, until)
    if customer_id:
        rows = [o for o in rows if o["customer_id"] == customer_id]
    if status:
        rows = [o for o in rows if o["status"] == status]
    return [{k: o[k] for k in ("order_id", "created_at", "status", "currency", "total")}
            for o in reversed(rows)][:limit]


@router.get("/orders/{order_id}", operation_id="getOrder", response_model=Order, summary="One order with its lines")
def get_order(order_id: str = Path(description="Order id, e.g. ORD-10001")) -> dict:
    """A single order including every line.

    **Returns:** the order's id, customer id, date, status, currency, total and
    lines (sku, description, quantity, unit price, amount). 404 if not found.
    """
    for o in data.orders():
        if o["order_id"] == order_id:
            return o
    raise HTTPException(404, f"no order {order_id}")


@router.get("/customers/{customer_id}/order-summary", operation_id="getCustomerOrderSummary",
         response_model=CustomerOrderSummary, summary="Order totals for one customer")
def customer_summary(customer_id: str = Path(description="Customer id, e.g. CUST-1001")) -> dict:
    """Aggregates over one customer's orders. Revenue counts only orders that
    were not cancelled or refunded, and is reported per currency.

    **Returns:** order counts, revenue per currency, open orders and the first
    and last order dates.
    """
    rows = [o for o in data.orders() if o["customer_id"] == customer_id]
    counted = [o for o in rows if o["status"] in COUNTED]
    totals: dict[str, list] = defaultdict(lambda: [0.0, 0])
    for o in counted:
        totals[o["currency"]][0] += o["total"]
        totals[o["currency"]][1] += 1
    return {
        "customer_id": customer_id,
        "order_count": len(rows),
        "counted_order_count": len(counted),
        "totals": [{"currency": c, "total": round(t, 2), "order_count": n} for c, (t, n) in sorted(totals.items())],
        "open_orders": sum(o["status"] in ("placed", "paid") for o in rows),
        "first_order_date": rows[0]["created_at"] if rows else None,
        "last_order_date": rows[-1]["created_at"] if rows else None,
    }


@router.get("/reports/top-customers", operation_id="getTopCustomers", response_model=list[TopCustomer],
         summary="Customers ranked by spend")
def top_customers(
    currency: str = Query(..., description="Rank within this currency (USD or EUR); totals are never mixed across currencies"),
    since: str | None = Query(None, description="Only orders on or after this date (YYYY-MM-DD)"),
    until: str | None = Query(None, description="Only orders on or before this date (YYYY-MM-DD)"),
    limit: int = Query(10, ge=1, le=50, description="How many customers to return"),
) -> list[dict]:
    """Customers ranked by revenue from counted orders (not cancelled or
    refunded) in one currency.

    **Returns:** rank, customer id, currency, total and order count. Customer
    ids only -- look names up in the customers service.
    """
    agg: dict[str, list] = defaultdict(lambda: [0.0, 0])
    for o in _window(data.orders(), since, until):
        if o["currency"] == currency.upper() and o["status"] in COUNTED:
            agg[o["customer_id"]][0] += o["total"]
            agg[o["customer_id"]][1] += 1
    ranked = sorted(agg.items(), key=lambda kv: (-kv[1][0], kv[0]))[:limit]
    return [{"rank": i + 1, "customer_id": cid, "currency": currency.upper(), "total": round(t, 2), "order_count": n}
            for i, (cid, (t, n)) in enumerate(ranked)]


@router.post("/orders/{order_id}/cancel", operation_id="cancelOrder", response_model=Order, summary="Cancel an order")
def cancel_order(order_id: str = Path(description="Order id to cancel")) -> dict:
    """Cancels an order that has not shipped yet. A WRITE: ze-ai lists it as
    available but never exposes it to a model (its tool layer is read-only).

    **Returns:** the cancelled order. 409 if it has already shipped.
    """
    for o in data.orders():
        if o["order_id"] == order_id:
            if o["status"] not in ("placed", "paid"):
                raise HTTPException(409, f"order {order_id} is {o['status']}")
            return {**o, "status": "cancelled"}
    raise HTTPException(404, f"no order {order_id}")


app.include_router(router)
