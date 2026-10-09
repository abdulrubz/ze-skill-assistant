# ze-api-specs

The single place every microservice publishes its OpenAPI spec. Consumers
(today: `ze-ai`, which generates its agent tools from these specs) read this
repo; they never call a service's `/openapi.json` directly.

## Layout

```
services/
  <service-name>/
    openapi.json        # the service's spec, generated from its code
scripts/validate.py     # the convention check CI runs
```

`<service-name>` is lowercase kebab-case and is the name consumers use: ze-ai
refers to an operation as `<service-name>.<operationId>` and reads the
service's base URL from `SERVICE_<NAME>_URL` (e.g. `SERVICE_ORDERS_URL`).

## Convention (enforced by `scripts/validate.py`)

| rule | why | level |
|------|-----|-------|
| OpenAPI **3.x** JSON | the tool generator reads `requestBody`/`components`; convert Swagger 2.0 first | error |
| every operation has an **`operationId`**, unique within the service | tools are keyed by it | error |
| every operation has a summary or description | it is what the model reads | error |
| description has a **`**Returns:**`** section | tells the model what comes back | warning |
| every parameter has a description | the model sets parameters from these | warning |

Optional vendor extensions consumers understand:

- `x-not-an-agent-tool: true` on an operation: never offered as a tool.
- `x-read-only: true` on a POST: treat it as a read (e.g. a search or dry run).
- `x-agent: {fixed: {...}, defaults: {...}, hidden: [...], inputs: [...]}`:
  pin or hide inputs a model should not set.
- `x-agent-hidden: true` on a parameter: plumbing the model never sees.

## How a spec gets here

A service's CI generates its spec from code and opens a PR replacing
`services/<name>/openapi.json` (see `ze-mock-services/scripts/publish_specs.py`
for the shape of it). The root workflow `api-specs.yml` validates it on the PR.
When it merges, `spec-sync.yml` regenerates ze-ai's tools and opens a PR there.

This folder lives in the `ze-skill-assistant` monorepo next to its consumer. If it
moves to a repo of its own, ze-ai switches from `SPEC_REPO_PATH` to
`SPEC_REPO_URL`, and a `repository_dispatch` from here replaces the path trigger.

## Locally

```bash
python scripts/validate.py
```
