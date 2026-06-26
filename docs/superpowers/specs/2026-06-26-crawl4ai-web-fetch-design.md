# Self-hosted Crawl4AI as the `web_fetch` provider

## Overview

Replace the failing Jina `web_fetch` provider with a self-hosted **Crawl4AI**
instance. Add a new `crawl4ai` community tool provider that calls a local
Crawl4AI Docker server's `POST /md` endpoint and returns clean markdown, then
point `config.yaml`'s `web_fetch` tool at it.

Goal (user's words): *no API costs, no auth failures.* Crawl4AI runs locally
with JWT auth disabled by default, so `web_fetch` no longer depends on an
external key.

This is a **deployment-fork customization** (see root `CLAUDE.md`). It is
designed to minimize upstream merge surface: it touches **no** upstream-tracked
`docker-compose.yaml`, `deploy.sh`, or `nginx.conf`. The provider is net-new
files; the active-provider switch lives in gitignored `config.yaml`.

## Background

- `web_fetch` in DeerFlow is a swappable **community tool provider**, not a
  "skill". `config.yaml` selects one via a `use:` import path
  (`deerflow.community.<provider>.tools:web_fetch_tool`). Currently active:
  `deerflow.community.jina_ai.tools:web_fetch_tool` (config.yaml ~line 716).
- `.env` has `JINA_API_KEY` set, so the observed "auth failing" is that key
  being expired / over-quota. Moving to self-hosted Crawl4AI removes the
  external dependency entirely.
- The closest existing analog is the **`browserless`** provider: a self-hosted
  HTTP service addressed by `base_url`, with a small async client class and the
  `"Error: …"` return convention. The new provider mirrors it.
- Crawl4AI API verified against current docs/source (`unclecode/crawl4ai`,
  `deploy/docker/server.py`):
  - Image `unclecode/crawl4ai:0.8.6`; port **11235**; `--shm-size=1g`.
  - `security.jwt_enabled: false` by default → no auth.
  - `POST /md` body `{"url": ..., "f": "fit", "q": null, "c": "0"}` where
    `f="fit"` is server-side readability-cleaned markdown. Response:
    `{"url":..., "filter":..., "markdown": "...", "success": true}`.
  - `GET /health` → `{"status":"ok","timestamp":...,"version":...}`.
  - Because Crawl4AI returns clean fit-markdown, **no** DeerFlow-side
    `ReadabilityExtractor` is needed (unlike jina/browserless, which fetch HTML
    then extract).

## Topology (decided)

**Standalone container** managed on its own lifecycle. The gateway reaches it at
`http://host.docker.internal:11235`. The gateway service already declares
`extra_hosts: host.docker.internal:host-gateway` and lists
`host.docker.internal` in `NO_PROXY`, so no compose/network edits are required.
`.env` defines no HTTP(S) proxy, so there is nothing to bypass.

## Scope of changes

New files (net-new, low merge risk):
- `backend/packages/harness/deerflow/community/crawl4ai/__init__.py`
- `backend/packages/harness/deerflow/community/crawl4ai/crawl4ai_client.py`
- `backend/packages/harness/deerflow/community/crawl4ai/tools.py`
- `backend/tests/test_crawl4ai_tools.py`
- `docker/docker-compose.crawl4ai.yaml` (standalone; NOT referenced by deploy.sh)

Edited files:
- `config.yaml` (gitignored): switch active `web_fetch` to crawl4ai; keep jina
  block commented as a documented fallback.
- root `CLAUDE.md`: document the provider, the standalone-container lifecycle,
  and a post-change verification command.

Explicitly **not** edited: `docker/docker-compose.yaml`, `scripts/deploy.sh`,
`docker/nginx/nginx.conf`, `backend/CLAUDE.md`, `frontend/CLAUDE.md`.

## Design

### Provider module

`crawl4ai_client.py`:

```python
class Crawl4AiClient:
    def __init__(self, base_url: str, token: str = "", timeout_s: float = 30.0): ...

    async def fetch_markdown(self, url: str, filter_mode: str = "fit") -> str:
        # POST {base_url}/md  body {"url": url, "f": filter_mode}
        # headers: Authorization: Bearer <token>  (only if token set)
        # non-200            -> "Error: Crawl4AI returned status <code>: <body>"
        # success is False   -> "Error: Crawl4AI reported failure for <url>"
        # empty markdown     -> "Error: Crawl4AI returned empty markdown"
        # httpx exception    -> "Error: Request to Crawl4AI failed: <type>: <e>"
        # else               -> data["markdown"]
```

- `base_url` is normalized (strip trailing `/`).
- Uses `httpx.AsyncClient` with `timeout=timeout_s`; provider is `async`, matching
  browserless (keeps blocking IO off the event loop).

`tools.py`:

```python
def _get_tool_config(tool_name: str) -> dict | None:
    # mirrors browserless: returns config.model_extra (a dict) or None
    config = get_app_config().get_tool_config(tool_name)
    if config is None:
        return None
    extras = config.model_extra
    return extras if extras is not None else {}

@tool("web_fetch", parse_docstring=True)
async def web_fetch_tool(url: str) -> str:
    # docstring identical to the other web_fetch providers (EXACT-URL guidance)
    cfg = _get_tool_config("web_fetch")          # dict of extras, or None
    base_url, timeout_s, token, filter_md = "http://localhost:11235", 30.0, "", "fit"
    if cfg is not None:
        base_url  = cfg.get("base_url", base_url)   # Docker overrides via config.yaml
        timeout_s = float(cfg.get("timeout_s", timeout_s))
        token     = cfg.get("token", token)
        filter_md = cfg.get("filter", filter_md)
    client = Crawl4AiClient(base_url=base_url, token=token, timeout_s=timeout_s)
    md = await client.fetch_markdown(url, filter_mode=filter_md)
    if md.startswith("Error:"):
        return md
    return md[:4096]   # same 4 KB cap as jina/browserless/firecrawl
```

- Config extras are read via `config.model_extra` (the `ToolConfig` API used by
  jina/browserless), with a `None` guard when `web_fetch` is unconfigured.
- In-code default `base_url` is `http://localhost:11235` (matches browserless's
  localhost default + "use the Docker host in Docker" comment). `config.yaml`
  sets the Docker-correct `http://host.docker.internal:11235` explicitly.

### config.yaml switch

Comment out the active jina `web_fetch` block; add:

```yaml
  - name: web_fetch
    group: web
    use: deerflow.community.crawl4ai.tools:web_fetch_tool
    base_url: http://host.docker.internal:11235
    timeout_s: 30
    # filter: fit        # fit (default) | raw | bm25 | llm
    # token: $CRAWL4AI_TOKEN   # only if Crawl4AI JWT auth is enabled
```

Only one `web_fetch` provider may be active at a time (enforced by convention in
config.yaml, as today).

### Standalone Crawl4AI container

`docker/docker-compose.crawl4ai.yaml` (own project name, own lifecycle):

```yaml
services:
  crawl4ai:
    image: unclecode/crawl4ai:0.8.6
    container_name: crawl4ai
    ports:
      - "11235:11235"
    shm_size: "1g"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:11235/health"]
      interval: 30s
      timeout: 5s
      retries: 3
```

Brought up independently:
`docker compose -f docker/docker-compose.crawl4ai.yaml up -d`.
(A raw `docker run -d --restart unless-stopped --name crawl4ai -p 11235:11235
--shm-size=1g unclecode/crawl4ai:0.8.6` is the equivalent imperative form.)

### Data flow

gateway container → `http://host.docker.internal:11235/md` → host port 11235 →
crawl4ai container → headless-Chromium render + fit-markdown → agent.

## Error handling

- Transport/HTTP/empty/failure cases all collapse to a single `"Error: …"`
  string (provider convention). The agent surfaces it; the user restarts the
  container if it is down.
- No automatic fallback to Jina (YAGNI, and Jina is the broken path). The
  commented jina block in `config.yaml` is the documented manual fallback.

## Testing (backend TDD — mandatory)

`backend/tests/test_crawl4ai_tools.py`, mirroring `test_browserless_client.py` /
`test_fastcrw_tools.py`, with mocked HTTP (no network):

1. Happy path: `/md` returns `{"markdown":"# Hi\n\n…","success":true}` →
   tool returns that markdown, ≤ 4096 chars.
2. Request shape: POSTs to `…/md` with body containing `url` and `f`.
3. Non-200 → `"Error:"`.
4. `success: false` → `"Error:"`.
5. Empty markdown → `"Error:"`.
6. httpx exception → `"Error:"` (no raise).
7. Config reading: `base_url` / `timeout_s` pulled from a stubbed
   `get_app_config().get_tool_config("web_fetch")`.
8. Truncation: > 4096-char markdown is capped at 4096.
9. Token header: set only when `token` is configured.

`cd backend && make test` (or the focused file) must pass.

## Rollout / verification

New provider code must be baked into the gateway image (the gateway mounts only
config/skills/data, not source), so:

1. `docker compose -f docker/docker-compose.crawl4ai.yaml up -d` and confirm
   `curl -fsS http://localhost:11235/health`.
2. `make up` — rebuilds the gateway with the new module (config still on Jina is
   harmless; the module is simply unused).
3. Flip `config.yaml`'s `web_fetch` to crawl4ai (gitignored, hot-reloads — no
   second rebuild).
4. Verify:
   - `docker logs --tail 5 deer-flow-gateway` → "Application startup complete".
   - Provider resolves:
     `docker exec deer-flow-gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python -c "from deerflow.tools.tools import get_available_tools as t; print(\"web_fetch\" in [x.name for x in t(model_name=\"minimax\", include_mcp=False)])"'`
   - From inside the gateway, `host.docker.internal:11235/health` is reachable.
   - A live `web_fetch` in the UI returns page content (use an unguessable test
     URL, per the CLAUDE.md vision-probe ethos).

## Upgrade safety

- No edits to upstream-tracked `docker-compose.yaml` / `deploy.sh` /
  `nginx.conf` / `backend/CLAUDE.md` / `frontend/CLAUDE.md`.
- The provider is net-new files; `git merge` keeps them unless upstream itself
  adds a `crawl4ai` provider at the same path (then adopt upstream's).
- `config.yaml` is gitignored and survives upgrades; the standalone compose file
  is new and unreferenced by deploy.sh.
- Documented in root `CLAUDE.md` so the customization is discoverable on the next
  upgrade.

## Out of scope (YAGNI)

- `web_search` stays on DDG (`ddg_search`) — free, not failing.
- No use of `/crawl`, screenshots, PDF, JS-exec, or multi-URL crawling.
- No automatic provider failover.
- No wiring of the Crawl4AI container into `deploy.sh` (standalone by choice).
