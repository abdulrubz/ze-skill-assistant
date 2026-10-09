"""customers service: who the customers are.

Every endpoint documents what it returns under a `**Returns:**` heading and
every field carries a description: ze-ai builds its tools from this spec, so
the spec IS the tool documentation the model reads.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from common import data
from common.auth import current_user

logging.basicConfig(level=logging.INFO, format="%(asctime)s customers %(message)s")

app = FastAPI(
    title="customers",
    version="1.0.0",
    description="Customer master data: identity, segment, region and credit terms.",
)
# Every data endpoint needs the end user's token; /health does not.
router = APIRouter(dependencies=[Depends(current_user)])

Segment = Literal["enterprise", "mid_market", "smb"]
Region = Literal["NA", "EU", "APAC"]


class Customer(BaseModel):
    customer_id: str = Field(description="Customer id, e.g. CUST-1001")
    name: str = Field(description="Legal company name")
    segment: Segment = Field(description="Sales segment: enterprise, mid_market or smb")
    region: Region = Field(description="Sales region: NA, EU or APAC")
    currency: str = Field(description="ISO currency the customer is billed in")
    email: str = Field(description="Billing contact email")
    signup_date: str = Field(description="Date the customer signed up (YYYY-MM-DD)")
    credit_limit: float = Field(description="Credit limit in the customer's currency")
    active: bool = Field(description="False once the customer has churned")


@app.get("/health", include_in_schema=False)
def health() -> dict:
    return {"status": "ok"}


@router.get("/customers", operation_id="listCustomers", response_model=list[Customer],
         summary="List customers, optionally filtered")
def list_customers(
    q: str | None = Query(None, description="Case-insensitive substring of the customer name"),
    segment: Segment | None = Query(None, description="Only this segment"),
    region: Region | None = Query(None, description="Only this region"),
    active: bool | None = Query(None, description="Only active (true) or churned (false) customers"),
    limit: int = Query(50, ge=1, le=200, description="Maximum customers to return"),
) -> list[dict]:
    """Customers matching every filter given, ordered by customer id. Use `q` to
    find a customer by (part of) its name.

    **Returns:** a list of customers with id, name, segment, region, currency,
    billing email, signup date, credit limit and whether they are active.
    """
    rows = data.customers()
    if q:
        rows = [c for c in rows if q.lower() in c["name"].lower()]
    if segment:
        rows = [c for c in rows if c["segment"] == segment]
    if region:
        rows = [c for c in rows if c["region"] == region]
    if active is not None:
        rows = [c for c in rows if c["active"] == active]
    return rows[:limit]


@router.get("/customers/{customer_id}", operation_id="getCustomer", response_model=Customer,
         summary="One customer by id")
def get_customer(customer_id: str = Path(description="Customer id, e.g. CUST-1001")) -> dict:
    """A single customer.

    **Returns:** the customer's id, name, segment, region, currency, billing
    email, signup date, credit limit and active flag. 404 if no such customer.
    """
    for c in data.customers():
        if c["customer_id"] == customer_id:
            return c
    raise HTTPException(404, f"no customer {customer_id}")


app.include_router(router)
