"""Deterministic seed data shared by both services.

Generated from a fixed seed so every run, test and eval sees the same
customers and orders: golden answers in ze-ai's eval bank depend on it.
Dates are fixed too (not relative to today), for the same reason.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from functools import lru_cache

SEED = 20261001
AS_OF = date(2026, 10, 1)

_FIRST = ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne", "Wonka", "Tyrell", "Cyberdyne", "Soylent",
          "Hooli", "Pied Piper", "Vandelay", "Dunder", "Gringotts", "Oscorp", "Aperture", "Massive", "Monarch",
          "Nakatomi", "Prestige", "Sterling", "Bluth", "Krusty", "Duff", "Gekko", "Wernham", "Benthic",
          "Yoyodyne", "Zorg"]
_SUFFIX = ["Ltd", "Inc", "GmbH", "LLC", "Corp", "Group"]
SEGMENTS = ["enterprise", "mid_market", "smb"]
REGIONS = ["NA", "EU", "APAC"]
STATUSES = ["placed", "paid", "shipped", "delivered", "cancelled", "refunded"]
PRODUCTS = [
    ("SKU-100", "Starter licence", 49.0), ("SKU-200", "Pro licence", 199.0),
    ("SKU-300", "Enterprise licence", 899.0), ("SKU-410", "Support add-on", 120.0),
    ("SKU-520", "Training day", 650.0), ("SKU-630", "Extra storage (1 TB)", 35.0),
]


@lru_cache
def customers() -> list[dict]:
    rng = random.Random(SEED)
    out = []
    for i in range(30):
        region = rng.choice(REGIONS)
        segment = rng.choices(SEGMENTS, weights=[2, 3, 5])[0]
        out.append({
            "customer_id": f"CUST-{1001 + i}",
            "name": f"{_FIRST[i]} {rng.choice(_SUFFIX)}",
            "segment": segment,
            "region": region,
            "currency": "EUR" if region == "EU" else "USD",
            "email": f"billing@{_FIRST[i].lower().replace(' ', '')}.example",
            "signup_date": (AS_OF - timedelta(days=rng.randint(30, 900))).isoformat(),
            "credit_limit": {"enterprise": 50000, "mid_market": 15000, "smb": 3000}[segment],
            "active": rng.random() > 0.1,
        })
    return out


@lru_cache
def orders() -> list[dict]:
    rng = random.Random(SEED + 1)
    out = []
    n = 0
    for c in customers():
        count = {"enterprise": 14, "mid_market": 8, "smb": 4}[c["segment"]] + rng.randint(-2, 3)
        signup = date.fromisoformat(c["signup_date"])
        for _ in range(max(count, 1)):
            n += 1
            created = signup + timedelta(days=rng.randint(0, (AS_OF - signup).days))
            lines = []
            for sku, name, price in rng.sample(PRODUCTS, rng.randint(1, 3)):
                qty = rng.randint(1, 10 if price < 100 else 3)
                lines.append({"sku": sku, "description": name, "quantity": qty, "unit_price": price,
                              "amount": round(qty * price, 2)})
            age = (AS_OF - created).days
            status = (rng.choices(STATUSES, weights=[1, 2, 3, 10, 1, 1])[0] if age > 30
                      else rng.choice(["placed", "paid", "shipped"]))
            out.append({
                "order_id": f"ORD-{10000 + n}",
                "customer_id": c["customer_id"],
                "created_at": created.isoformat(),
                "status": status,
                "currency": c["currency"],
                "total": round(sum(li["amount"] for li in lines), 2),
                "lines": lines,
            })
    return sorted(out, key=lambda o: o["created_at"])
