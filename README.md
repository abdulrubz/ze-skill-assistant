# ze-skill-assistant

A GenAI service that answers questions from an organisation's microservices,
with tools generated from their OpenAPI specs. Three parts:

| folder | what |
|---|---|
| [`ze-ai/`](ze-ai/) | The service: agents -> skills -> tools, MLflow gateway, Langfuse. Start with its [README](ze-ai/README.md) and the [codebase walkthrough](ze-ai/docs/codebase-walkthrough.md). |
| [`ze-api-specs/`](ze-api-specs/) | Every service's OpenAPI spec (`services/<name>/openapi.json`) and the convention check. ze-ai's tools are generated from here. |
| [`ze-mock-services/`](ze-mock-services/) | Two dummy services (customers, orders) with token auth and fixed data, for development and demos. |

CI lives in `.github/workflows/`: one workflow per folder, plus `spec-sync`,
which regenerates ze-ai's tools and opens a PR whenever a spec changes.

## Quick start

```bash
git clone git@github.com:abdulrubz/ze-skill-assistant.git && cd ze-skill-assistant

# dummy services (customers :9101, orders :9102)
cd ze-mock-services && uv venv --python 3.12 && uv pip install -e ".[dev]"
cp .env.example .env && .venv/bin/python scripts/run.py &
.venv/bin/python scripts/mint_token.py alice "Alice Analyst" --hours 720   # -> ZE_DEV_TOKEN

# the service
cd ../ze-ai && uv venv --python 3.12 && uv pip install -e ".[dev]"
cp .env.example .env            # fill in MLflow, Langfuse, ZE_DEV_TOKEN
.venv/bin/python scripts/doctor.py
```
