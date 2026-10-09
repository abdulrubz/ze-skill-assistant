# ze-mock-services

Two dummy microservices that stand in for real ones while `ze-ai` is built and
demoed. They are deliberately shaped like the real thing:

| service   | port | operations |
|-----------|------|------------|
| customers | 9101 | `listCustomers`, `getCustomer` |
| orders    | 9102 | `listOrders`, `getOrder`, `getCustomerOrderSummary`, `getTopCustomers`, `cancelOrder` (a write) |

- **Separate services, separate specs.** `orders` knows customers only by id, so
  "top customers by spend, with names" needs both, which exercises ze-ai's
  cross-service skills and citations.
- **Every call needs the end user's bearer token** (HS256, `MOCK_JWT_SECRET`).
  ze-ai forwards the caller's token unchanged; the services log `user=<sub>`
  per request, so you can see the pass-through working.
- **Deterministic data** (fixed seed, fixed dates), so eval answers are stable.
- **Specs are generated from code** and published to `ze-api-specs`, the way a
  real team's CI would.

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
cp .env.example .env
python scripts/run.py                                   # both services
python scripts/mint_token.py alice "Alice Analyst" --hours 720   # a token for ze-ai's ZE_DEV_TOKEN
python scripts/publish_specs.py ../ze-api-specs         # refresh the specs repo
pytest -q
```
