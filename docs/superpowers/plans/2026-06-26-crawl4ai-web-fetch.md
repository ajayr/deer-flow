# Crawl4AI `web_fetch` Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failing Jina `web_fetch` with a self-hosted Crawl4AI container, via a new `deerflow.community.crawl4ai` provider that calls Crawl4AI's `POST /md`.

**Architecture:** A small async `Crawl4AiClient` (mirrors `BrowserlessClient`) POSTs to a self-hosted Crawl4AI server's `/md` endpoint and returns server-cleaned `fit` markdown directly (no DeerFlow-side readability step). A `web_fetch_tool` reads `base_url`/`timeout_s`/`token`/`filter` from `config.yaml`. Crawl4AI runs as a **standalone** container reached at `http://host.docker.internal:11235`; the active provider is switched in gitignored `config.yaml`.

**Tech Stack:** Python 3.12, `httpx` (already a core harness dep), `langchain.tools.tool`, Crawl4AI Docker server `unclecode/crawl4ai:0.8.6`, hatchling packaging, pytest + `unittest.mock`.

## Global Constraints

- Python 3.12+; double quotes; 4-space indent; ruff line length **240**.
- **No new dependencies** — `httpx` is already a core harness dependency.
- **No packaging edits** — `[tool.hatch.build.targets.wheel] packages = ["deerflow"]` auto-includes the new subpackage.
- Provider failures return a string starting with `"Error:"` (the established community-provider convention) — never raise out of the tool.
- The tool caps returned content at **4096** chars, like jina/browserless/firecrawl.
- The provider is `async` (httpx.AsyncClient) — keeps blocking IO off the event loop.
- **Do NOT edit** upstream-tracked `docker/docker-compose.yaml`, `scripts/deploy.sh`, `docker/nginx/nginx.conf`, `backend/CLAUDE.md`, or `frontend/CLAUDE.md`.
- The web_fetch tool docstring must be byte-identical to the other providers' (the EXACT-URL guidance) so the model-facing tool description is unchanged.
- Crawl4AI API (verified against current docs/source): image `unclecode/crawl4ai:0.8.6`, port `11235`, `--shm-size=1g`, JWT auth off by default. `POST /md` body `{"url": ..., "f": "fit"}` → `{"markdown": "...", "success": true, ...}`. `GET /health` → `{"status":"ok",...}`.

## File Structure

- Create `backend/packages/harness/deerflow/community/crawl4ai/__init__.py` — package exports.
- Create `backend/packages/harness/deerflow/community/crawl4ai/crawl4ai_client.py` — `Crawl4AiClient` (HTTP, error handling).
- Create `backend/packages/harness/deerflow/community/crawl4ai/tools.py` — `web_fetch_tool` + config readers.
- Create `backend/tests/test_crawl4ai_tools.py` — client + tool unit tests (mocked HTTP).
- Create `docker/docker-compose.crawl4ai.yaml` — standalone Crawl4AI service (own lifecycle; NOT in deploy.sh).
- Modify `CLAUDE.md` (root) — document the provider, container lifecycle, verification.
- Modify `config.yaml` (root, **gitignored** — operational edit, not committed) — switch active `web_fetch`.

---

### Task 1: `Crawl4AiClient`

**Files:**
- Create: `backend/packages/harness/deerflow/community/crawl4ai/__init__.py`
- Create: `backend/packages/harness/deerflow/community/crawl4ai/crawl4ai_client.py`
- Test: `backend/tests/test_crawl4ai_tools.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `Crawl4AiClient(base_url: str, token: str = "", timeout_s: float = 30.0)` with `async def fetch_markdown(self, url: str, filter_mode: str = "fit") -> str`. Returns markdown on success, or a string starting with `"Error:"` on any failure. POSTs to `{base_url}/md` with JSON `{"url": url, "f": filter_mode}`; adds `Authorization: Bearer <token>` only when `token` is set.

- [ ] **Step 1: Write the failing client tests**

Create `backend/tests/test_crawl4ai_tools.py`:

```python
"""Tests for Crawl4AI community tools."""

from unittest.mock import MagicMock, patch

import pytest

from deerflow.community.crawl4ai.crawl4ai_client import Crawl4AiClient


class AsyncMock(MagicMock):
    """Mock that supports async call."""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


@pytest.mark.asyncio
class TestCrawl4AiClient:
    """Tests for the Crawl4AiClient class."""

    async def test_fetch_markdown_success(self):
        with patch("deerflow.community.crawl4ai.crawl4ai_client.httpx.AsyncClient") as mock_cls:
            mock_ctx = MagicMock()
            mock_cls.return_value.__aenter__.return_value = mock_ctx

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"markdown": "# Title\n\nHello", "success": True}
            mock_ctx.post = AsyncMock(return_value=mock_resp)

            client = Crawl4AiClient(base_url="http://crawl4ai:11235")
            result = await client.fetch_markdown("https://example.com")

            assert result == "# Title\n\nHello"
            call = mock_ctx.post.call_args
            assert call.args[0].endswith("/md")
            assert call.kwargs["json"]["url"] == "https://example.com"
            assert call.kwargs["json"]["f"] == "fit"

    async def test_fetch_markdown_strips_trailing_slash_in_base_url(self):
        client = Crawl4AiClient(base_url="http://crawl4ai:11235/")
        assert client.base_url == "http://crawl4ai:11235"

    async def test_fetch_markdown_http_error(self):
        with patch("deerflow.community.crawl4ai.crawl4ai_client.httpx.AsyncClient") as mock_cls:
            mock_ctx = MagicMock()
            mock_cls.return_value.__aenter__.return_value = mock_ctx

            mock_resp = MagicMock()
            mock_resp.status_code = 502
            mock_resp.text = "Bad Gateway"
            mock_ctx.post = AsyncMock(return_value=mock_resp)

            client = Crawl4AiClient(base_url="http://crawl4ai:11235")
            result = await client.fetch_markdown("https://example.com")
            assert "Error: Crawl4AI HTTP 502" in result

    async def test_fetch_markdown_success_false(self):
        with patch("deerflow.community.crawl4ai.crawl4ai_client.httpx.AsyncClient") as mock_cls:
            mock_ctx = MagicMock()
            mock_cls.return_value.__aenter__.return_value = mock_ctx

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"markdown": "", "success": False}
            mock_ctx.post = AsyncMock(return_value=mock_resp)

            client = Crawl4AiClient(base_url="http://crawl4ai:11235")
            result = await client.fetch_markdown("https://example.com")
            assert result.startswith("Error:")

    async def test_fetch_markdown_empty(self):
        with patch("deerflow.community.crawl4ai.crawl4ai_client.httpx.AsyncClient") as mock_cls:
            mock_ctx = MagicMock()
            mock_cls.return_value.__aenter__.return_value = mock_ctx

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"markdown": "   ", "success": True}
            mock_ctx.post = AsyncMock(return_value=mock_resp)

            client = Crawl4AiClient(base_url="http://crawl4ai:11235")
            result = await client.fetch_markdown("https://example.com")
            assert result == "Error: Crawl4AI returned empty markdown"

    async def test_fetch_markdown_timeout(self):
        with patch("deerflow.community.crawl4ai.crawl4ai_client.httpx.AsyncClient") as mock_cls:
            mock_ctx = MagicMock()
            mock_cls.return_value.__aenter__.return_value = mock_ctx
            import httpx

            mock_ctx.post = AsyncMock(side_effect=httpx.TimeoutException("Timed out"))

            client = Crawl4AiClient(base_url="http://crawl4ai:11235", timeout_s=10)
            result = await client.fetch_markdown("https://example.com")
            assert "timed out" in result.lower() or "timeout" in result.lower()

    async def test_fetch_markdown_with_token(self):
        with patch("deerflow.community.crawl4ai.crawl4ai_client.httpx.AsyncClient") as mock_cls:
            mock_ctx = MagicMock()
            mock_cls.return_value.__aenter__.return_value = mock_ctx

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"markdown": "ok", "success": True}
            mock_ctx.post = AsyncMock(return_value=mock_resp)

            client = Crawl4AiClient(base_url="http://crawl4ai:11235", token="secret")
            await client.fetch_markdown("https://example.com")

            headers = mock_ctx.post.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer secret"

    async def test_fetch_markdown_no_token_header_when_unset(self):
        with patch("deerflow.community.crawl4ai.crawl4ai_client.httpx.AsyncClient") as mock_cls:
            mock_ctx = MagicMock()
            mock_cls.return_value.__aenter__.return_value = mock_ctx

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"markdown": "ok", "success": True}
            mock_ctx.post = AsyncMock(return_value=mock_resp)

            client = Crawl4AiClient(base_url="http://crawl4ai:11235")
            await client.fetch_markdown("https://example.com")

            headers = mock_ctx.post.call_args.kwargs["headers"]
            assert "Authorization" not in headers
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && PYTHONPATH=. uv run pytest tests/test_crawl4ai_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deerflow.community.crawl4ai'`.

- [ ] **Step 3: Create the package `__init__.py` (client-only for now)**

Create `backend/packages/harness/deerflow/community/crawl4ai/__init__.py`:

```python
from .crawl4ai_client import Crawl4AiClient

__all__ = ["Crawl4AiClient"]
```

- [ ] **Step 4: Implement `Crawl4AiClient`**

Create `backend/packages/harness/deerflow/community/crawl4ai/crawl4ai_client.py`:

```python
import logging

import httpx

logger = logging.getLogger(__name__)


class Crawl4AiClient:
    """Client for a self-hosted Crawl4AI Docker server (POST /md)."""

    def __init__(self, base_url: str, token: str = "", timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    async def fetch_markdown(self, url: str, filter_mode: str = "fit") -> str:
        """Fetch a page's clean markdown via Crawl4AI's POST /md endpoint.

        Args:
            url: The URL to fetch.
            filter_mode: Crawl4AI markdown filter ("fit", "raw", "bm25", "llm").

        Returns:
            Markdown content, or an "Error: ..." string on failure.
        """
        payload: dict[str, object] = {"url": url, "f": filter_mode}
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        logger.debug(f"Fetching URL via Crawl4AI: {url}")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(f"{self.base_url}/md", json=payload, headers=headers)

            if resp.status_code != 200:
                return f"Error: Crawl4AI HTTP {resp.status_code}: {resp.text[:200]}"

            data = resp.json()
            if not data.get("success", False):
                return f"Error: Crawl4AI reported failure for {url}"

            markdown = data.get("markdown") or ""
            if not markdown.strip():
                return "Error: Crawl4AI returned empty markdown"

            return markdown

        except httpx.TimeoutException:
            return f"Error: Crawl4AI request timed out after {self.timeout_s}s"
        except httpx.RequestError as e:
            logger.error(f"Crawl4AI request failed: {e}")
            return f"Error: Crawl4AI request failed: {e!s}"
        except Exception as e:
            logger.error(f"Crawl4AI fetch failed: {e}")
            return f"Error: Crawl4AI fetch failed: {e!s}"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd backend && PYTHONPATH=. uv run pytest tests/test_crawl4ai_tools.py -v`
Expected: PASS (8 tests in `TestCrawl4AiClient`).

- [ ] **Step 6: Format & lint**

Run: `cd backend && make format && make lint`
Expected: no errors on the new files.

- [ ] **Step 7: Commit**

```bash
git add backend/packages/harness/deerflow/community/crawl4ai/__init__.py \
        backend/packages/harness/deerflow/community/crawl4ai/crawl4ai_client.py \
        backend/tests/test_crawl4ai_tools.py
git commit -m "feat(community): add Crawl4AiClient for self-hosted Crawl4AI /md

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `web_fetch_tool`

**Files:**
- Create: `backend/packages/harness/deerflow/community/crawl4ai/tools.py`
- Modify: `backend/packages/harness/deerflow/community/crawl4ai/__init__.py` (add tool export)
- Test: `backend/tests/test_crawl4ai_tools.py` (append tool tests)

**Interfaces:**
- Consumes: `Crawl4AiClient` from Task 1; `get_app_config().get_tool_config("web_fetch")` (returns a `ToolConfig | None` whose `.model_extra` is the extras dict).
- Produces: `web_fetch_tool` (a LangChain tool named `web_fetch`, async); module-level `_get_tool_config(tool_name) -> dict | None` and `_get_crawl4ai_client() -> Crawl4AiClient` (patched in tests). Reads `base_url` (default `http://localhost:11235`), `timeout_s` (default 30), `token` (default ""), `filter` (default "fit") from config; returns `markdown[:4096]` or an `"Error:"` string.

- [ ] **Step 1: Write the failing tool tests**

Append to `backend/tests/test_crawl4ai_tools.py`:

```python
@pytest.mark.asyncio
class TestCrawl4AiTools:
    """Tests for the Crawl4AI tool functions."""

    @patch("deerflow.community.crawl4ai.tools._get_crawl4ai_client")
    async def test_web_fetch_tool_success(self, mock_get_client):
        from deerflow.community.crawl4ai import tools

        mock_client = MagicMock()
        mock_client.fetch_markdown = AsyncMock(return_value="# Title\n\nContent")
        mock_get_client.return_value = mock_client

        with patch("deerflow.community.crawl4ai.tools._get_tool_config", return_value=None):
            result = await tools.web_fetch_tool.ainvoke("https://example.com/article")

        assert result == "# Title\n\nContent"
        assert "Error:" not in result

    @patch("deerflow.community.crawl4ai.tools._get_crawl4ai_client")
    async def test_web_fetch_tool_truncates_to_4096(self, mock_get_client):
        from deerflow.community.crawl4ai import tools

        mock_client = MagicMock()
        mock_client.fetch_markdown = AsyncMock(return_value="x" * 5000)
        mock_get_client.return_value = mock_client

        with patch("deerflow.community.crawl4ai.tools._get_tool_config", return_value=None):
            result = await tools.web_fetch_tool.ainvoke("https://example.com")

        assert len(result) == 4096

    @patch("deerflow.community.crawl4ai.tools._get_crawl4ai_client")
    async def test_web_fetch_tool_error_passthrough(self, mock_get_client):
        from deerflow.community.crawl4ai import tools

        mock_client = MagicMock()
        mock_client.fetch_markdown = AsyncMock(return_value="Error: Crawl4AI returned empty markdown")
        mock_get_client.return_value = mock_client

        with patch("deerflow.community.crawl4ai.tools._get_tool_config", return_value=None):
            result = await tools.web_fetch_tool.ainvoke("https://example.com")

        assert result.startswith("Error:")

    @patch("deerflow.community.crawl4ai.tools._get_crawl4ai_client")
    async def test_web_fetch_tool_exception(self, mock_get_client):
        from deerflow.community.crawl4ai import tools

        mock_client = MagicMock()
        mock_client.fetch_markdown = AsyncMock(side_effect=Exception("boom"))
        mock_get_client.return_value = mock_client

        with patch("deerflow.community.crawl4ai.tools._get_tool_config", return_value=None):
            result = await tools.web_fetch_tool.ainvoke("https://example.com")

        assert result.startswith("Error:")

    async def test_get_crawl4ai_client_reads_config(self):
        from deerflow.community.crawl4ai import tools

        fake_cfg = {"base_url": "http://host.docker.internal:11235", "timeout_s": 45}
        with patch("deerflow.community.crawl4ai.tools._get_tool_config", return_value=fake_cfg):
            client = tools._get_crawl4ai_client()

        assert client.base_url == "http://host.docker.internal:11235"
        assert client.timeout_s == 45.0

    async def test_get_crawl4ai_client_defaults_when_unconfigured(self):
        from deerflow.community.crawl4ai import tools

        with patch("deerflow.community.crawl4ai.tools._get_tool_config", return_value=None):
            client = tools._get_crawl4ai_client()

        assert client.base_url == "http://localhost:11235"
        assert client.timeout_s == 30.0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd backend && PYTHONPATH=. uv run pytest tests/test_crawl4ai_tools.py::TestCrawl4AiTools -v`
Expected: FAIL — `ImportError`/`AttributeError` (no `tools` module / no `web_fetch_tool`).

- [ ] **Step 3: Implement `tools.py`**

Create `backend/packages/harness/deerflow/community/crawl4ai/tools.py`:

```python
import logging

from langchain.tools import tool

from deerflow.config import get_app_config

from .crawl4ai_client import Crawl4AiClient

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11235"


def _get_tool_config(tool_name: str) -> dict | None:
    """Return the tool's config extras (model_extra) dict, or None if unconfigured."""
    config = get_app_config().get_tool_config(tool_name)
    if config is None:
        return None
    extras = config.model_extra
    return extras if extras is not None else {}


def _get_crawl4ai_client() -> Crawl4AiClient:
    cfg = _get_tool_config("web_fetch")
    base_url = DEFAULT_BASE_URL
    token = ""
    timeout_s = 30.0
    if cfg is not None:
        base_url = cfg.get("base_url", base_url)
        token = cfg.get("token", token)
        raw = cfg.get("timeout_s", timeout_s)
        timeout_s = float(raw) if not isinstance(raw, float) else raw
    return Crawl4AiClient(base_url=base_url, token=token, timeout_s=timeout_s)


@tool("web_fetch", parse_docstring=True)
async def web_fetch_tool(url: str) -> str:
    """Fetch the contents of a web page at a given URL.
    Only fetch EXACT URLs that have been provided directly by the user or have been returned in results from the web_search and web_fetch tools.
    This tool can NOT access content that requires authentication, such as private Google Docs or pages behind login walls.
    Do NOT add www. to URLs that do NOT have them.
    URLs must include the schema: https://example.com is a valid URL while example.com is an invalid URL.

    Args:
        url: The URL to fetch the contents of.
    """
    try:
        cfg = _get_tool_config("web_fetch")
        filter_mode = "fit"
        if cfg is not None:
            filter_mode = cfg.get("filter", filter_mode)

        client = _get_crawl4ai_client()
        markdown = await client.fetch_markdown(url, filter_mode=filter_mode)

        if markdown.startswith("Error:"):
            return markdown

        return markdown[:4096]

    except Exception as e:
        logger.error(f"Error in web_fetch_tool: {e}")
        return f"Error: {str(e)}"
```

- [ ] **Step 4: Update `__init__.py` to export the tool**

Replace `backend/packages/harness/deerflow/community/crawl4ai/__init__.py` with:

```python
from .crawl4ai_client import Crawl4AiClient
from .tools import web_fetch_tool

__all__ = ["Crawl4AiClient", "web_fetch_tool"]
```

- [ ] **Step 5: Run the full test file to verify all pass**

Run: `cd backend && PYTHONPATH=. uv run pytest tests/test_crawl4ai_tools.py -v`
Expected: PASS (8 client + 6 tool = 14 tests).

- [ ] **Step 6: Format & lint**

Run: `cd backend && make format && make lint`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/packages/harness/deerflow/community/crawl4ai/tools.py \
        backend/packages/harness/deerflow/community/crawl4ai/__init__.py \
        backend/tests/test_crawl4ai_tools.py
git commit -m "feat(community): add Crawl4AI web_fetch tool provider

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Standalone Crawl4AI container

**Files:**
- Create: `docker/docker-compose.crawl4ai.yaml`

**Interfaces:**
- Consumes: nothing.
- Produces: a running Crawl4AI server on host port `11235`, reachable by the gateway at `http://host.docker.internal:11235`. NOT referenced by `scripts/deploy.sh` (independent lifecycle).

- [ ] **Step 1: Create the standalone compose file**

Create `docker/docker-compose.crawl4ai.yaml`:

```yaml
# DeerFlow — Standalone Crawl4AI server for the web_fetch provider (LOCAL DEPLOYMENT)
#
# NOT part of the main stack and NOT referenced by scripts/deploy.sh. Crawl4AI
# runs on its own lifecycle so it stays decoupled from the upstream-tracked
# docker-compose.yaml / deploy.sh (minimizes merge surface — see root CLAUDE.md).
#
# Start:   docker compose -f docker/docker-compose.crawl4ai.yaml up -d
# Stop:    docker compose -f docker/docker-compose.crawl4ai.yaml down
# Health:  curl -fsS http://localhost:11235/health
#
# The gateway reaches this server at http://host.docker.internal:11235
# (config.yaml -> tools -> web_fetch -> base_url). The gateway service already
# declares extra_hosts host.docker.internal and exempts it from NO_PROXY, so no
# edits to docker-compose.yaml are required.
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

- [ ] **Step 2: Bring up the container**

Run: `docker compose -f docker/docker-compose.crawl4ai.yaml up -d`
Expected: pulls `unclecode/crawl4ai:0.8.6` and starts container `crawl4ai`.

- [ ] **Step 3: Verify health (allow ~15-30s for Chromium warmup)**

Run: `curl -fsS http://localhost:11235/health`
Expected: JSON `{"status":"ok","timestamp":...,"version":"0.8.6"}`.

If it is not up yet: `docker logs --tail 20 crawl4ai` and retry the curl.

- [ ] **Step 4: Smoke-test `/md` directly**

Run:
```bash
curl -fsS -X POST http://localhost:11235/md \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com","f":"fit"}' | head -c 300
```
Expected: JSON containing `"success": true` and a `"markdown"` field with the page content.

- [ ] **Step 5: Commit the compose file**

```bash
git add docker/docker-compose.crawl4ai.yaml
git commit -m "feat(deploy): standalone Crawl4AI compose for the web_fetch provider

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Document in root CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (root)

**Interfaces:**
- Consumes: the provider (Tasks 1-2) and the standalone container (Task 3).
- Produces: discoverable upgrade/ops notes so the customization survives future merges.

- [ ] **Step 1: Add a "Local additions" note to section 1**

In root `CLAUDE.md`, immediately after the "Local edits to TRACKED files" table in section 1, add this paragraph:

```markdown
**New local files (net-new; survive `git merge` unless upstream adds the same path):**
- `backend/packages/harness/deerflow/community/crawl4ai/` — self-hosted **Crawl4AI** `web_fetch` provider (replaces Jina; see section 4). Added by this deployment.
- `docker/docker-compose.crawl4ai.yaml` — standalone Crawl4AI container (own lifecycle; **not** wired into `deploy.sh`).
```

- [ ] **Step 2: Add a verification command to section 3**

In root `CLAUDE.md` section 3, append this block before the closing fence of the verification script:

```bash
# web_fetch provider = self-hosted Crawl4AI, end-to-end (expect markdown, not an error)
docker exec deer-flow-gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python -c "import asyncio; from deerflow.community.crawl4ai.tools import web_fetch_tool; print(asyncio.run(web_fetch_tool.ainvoke(\"https://example.com\"))[:200])"'
```

- [ ] **Step 3: Add a gotcha bullet to section 4**

In root `CLAUDE.md` section 4, add this bullet:

```markdown
- **`web_fetch` via self-hosted Crawl4AI.** `config.yaml` → `tools` → `web_fetch` uses `deerflow.community.crawl4ai.tools:web_fetch_tool` with `base_url: http://host.docker.internal:11235`. The server is the **standalone** `crawl4ai` container: start it with `docker compose -f docker/docker-compose.crawl4ai.yaml up -d` (it is NOT started by `deploy.sh`/`make up`). If `web_fetch` returns `Error: Crawl4AI ...`, check the container is up (`docker ps --filter name=crawl4ai`, `curl -fsS http://localhost:11235/health`). Jina remains as a commented fallback in `config.yaml` (its `JINA_API_KEY` in `.env` is the path that was failing). Adding the new provider module requires a **gateway rebuild** (`make up`) — config edits alone hot-reload, but new Python code does not.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(deploy): document self-hosted Crawl4AI web_fetch provider

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Rollout & end-to-end verification

**Files:**
- Modify: `config.yaml` (root, **gitignored** — operational edit, nothing to commit).

**Interfaces:**
- Consumes: committed provider code (Tasks 1-2), running container (Task 3).
- Produces: a live deployment where `web_fetch` uses Crawl4AI.

- [ ] **Step 1: Confirm the full backend test suite passes**

Run: `cd backend && make test`
Expected: PASS (including `tests/test_crawl4ai_tools.py`). Fix any failures before deploying.

- [ ] **Step 2: Rebuild & recreate the gateway so the new module is baked in**

Run (from project root): `make up`
Expected: gateway image rebuilds and containers recreate. Config is still on Jina at this point — that is harmless; the crawl4ai module is simply present-but-unused.

Verify the new module is in the running gateway:
```bash
docker exec deer-flow-gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python -c "from deerflow.community.crawl4ai.tools import web_fetch_tool; print(web_fetch_tool.name)"'
```
Expected: `web_fetch`.

- [ ] **Step 3: Verify the gateway can reach Crawl4AI over `host.docker.internal`**

```bash
docker exec deer-flow-gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python -c "import httpx; print(httpx.get(\"http://host.docker.internal:11235/health\", timeout=5).json())"'
```
Expected: `{'status': 'ok', 'timestamp': ..., 'version': '0.8.6'}`.

(Uses `httpx`, a core dep, rather than assuming `curl` exists in the gateway image.)

- [ ] **Step 4: Switch the active `web_fetch` provider in `config.yaml`**

In root `config.yaml`, comment out the active Jina `web_fetch` block (currently ~lines 715-723):

```yaml
  # Web fetch tool (uses Jina AI reader) — DISABLED: key was failing; kept as fallback
  # - name: web_fetch
  #   group: web
  #   use: deerflow.community.jina_ai.tools:web_fetch_tool
  #   timeout: 10
```

…and add the Crawl4AI block in its place:

```yaml
  # Web fetch tool (uses self-hosted Crawl4AI — see docker/docker-compose.crawl4ai.yaml)
  - name: web_fetch
    group: web
    use: deerflow.community.crawl4ai.tools:web_fetch_tool
    base_url: http://host.docker.internal:11235
    timeout_s: 30
    # filter: fit              # fit (default) | raw | bm25 | llm
    # token: $CRAWL4AI_TOKEN   # only if Crawl4AI JWT auth is enabled
```

`config.yaml` is gitignored and hot-reloads (directory mount) — **no rebuild or restart needed** for this switch.

- [ ] **Step 5: End-to-end verify the live provider**

```bash
docker exec deer-flow-gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python -c "import asyncio; from deerflow.community.crawl4ai.tools import web_fetch_tool; print(asyncio.run(web_fetch_tool.ainvoke(\"https://example.com\"))[:200])"'
```
Expected: markdown text from example.com (starts with the page heading), NOT a string starting with `Error:`. This exercises the full path: `config.yaml` base_url → `Crawl4AiClient` → Crawl4AI `/md` → markdown.

Then confirm in the UI: open the app, ask the agent to fetch a fresh, **unguessable** URL (per the CLAUDE.md probe ethos), and confirm real page content returns.

- [ ] **Step 6: (If the UI still behaves as if on Jina) clear any agent cache**

Only if Step 5's in-container check passes but the UI still errs:
Run: `docker restart deer-flow-gateway`
Then re-test in the UI. (`tools[*]` is in the hot-reload boundary, so this is rarely needed; it is the documented stopgap.)

- [ ] **Step 7: Nothing to commit**

`config.yaml` is gitignored, so there is no commit for this task. The provider code (Tasks 1-2), compose file (Task 3), and docs (Task 4) are already committed.

---

## Spec coverage (self-review)

- Provider module (client + tool) → Tasks 1, 2. ✔
- `POST /md`, `f=fit`, response `markdown`/`success`, no readability step → Task 1 client. ✔
- 4096 cap, `"Error:"` convention, async → Tasks 1-2 + Global Constraints. ✔
- Standalone container `0.8.6`, port 11235, shm 1g, restart, healthcheck, host.docker.internal → Task 3. ✔
- config.yaml switch (gitignored), Jina kept as fallback → Task 5. ✔
- Networking via existing extra_hosts/NO_PROXY, no compose/deploy/nginx edits → Global Constraints + Task 3 comments. ✔
- Tests (happy/HTTP-error/success-false/empty/timeout/token/truncate/config) → Tasks 1-2 (14 tests). ✔
- Rollout: rebuild then hot-reload flip, verification commands → Task 5. ✔
- Upgrade safety + docs in root CLAUDE.md → Task 4. ✔
- YAGNI (web_search unchanged, no /crawl, no failover, not in deploy.sh) → respected throughout. ✔
