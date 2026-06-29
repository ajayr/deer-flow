# CLAUDE.md — Local Deployment & Safe Upgrade Notes

> **Scope:** this repo is a **customized production deployment** of DeerFlow — public,
> behind a Pangolin tunnel + double auth, served via Docker (`docker/docker-compose.yaml`,
> compose project `deer-flow`: `deer-flow-gateway`, `deer-flow-nginx`, `deer-flow-frontend`).
> It carries local customizations on top of upstream. **Read this before upgrading.**
>
> Upstream's dev docs now live in `AGENTS.md` (root, `backend/`, `frontend/`); the sibling
> `CLAUDE.md` files just import them via `@AGENTS.md` (upstream #3770). **Do not put local/ops
> notes there** — they are upstream-tracked and conflict on every upgrade. Keep deployment notes
> in THIS root `CLAUDE.md`; its tail re-imports `@AGENTS.md` so shared agent guidance still loads.
> Upstream now ships a root `CLAUDE.md` too, so expect a tiny add/add conflict on this file each
> upgrade — keep our playbook and re-append the `@AGENTS.md` line.

## 1. Local state that must NOT be lost on upgrade

### Gitignored — survives `git pull`/merge automatically; never overwrite, never commit
| Path | Holds |
|---|---|
| `config.yaml` | Models (default `minimax` → `ollamacloud/minimax-m3`, vision on; `glm5stack` secondary), gateway `base_url`, sandbox config. `config_version: 15`. |
| `.env` | Secrets — gateway API key, `GATEWAY_CORS_ORIGINS` (public HTTPS origin), search API keys. |
| `extensions_config.json` | MCP servers + skills. |
| `backend/.deer-flow/` | Auth secret, internal auth token, threads, memory, runtime data. |

`scripts/deploy.sh` only **seeds** `config.yaml` / `extensions_config.json` when missing — it never overwrites an existing one. So model/vision config and all runtime data ride through an upgrade untouched.

### Local edits to TRACKED files — committed on `main`; re-apply if upstream changes them
| File | Customization | Commit |
|---|---|---|
| `docker/docker-compose.yaml` | Gateway mounts the **project root as a directory** (`..:/app/deer-flow-runtime:ro`) with `DEER_FLOW_CONFIG_PATH` / `DEER_FLOW_EXTENSIONS_CONFIG_PATH` pointing into it — so `config.yaml` edits hot-reload **without a restart** (avoids single-file bind-mount inode pinning). | `103f179e` |
| `docker/nginx/nginx.conf` | `map $http_x_forwarded_proto $forwarded_proto` + every proxy block uses `X-Forwarded-Proto $forwarded_proto` — so login works behind the Pangolin TLS proxy (else Gateway 403 "Cross-site auth request denied"). Falls back to `$scheme` when nginx is the TLS edge. | `40cbf17f` — **as of the 2026-06-29 upgrade upstream ships the identical map (#3793), so nginx.conf no longer diverges. Row kept for history; safe to drop if it stays merged.** |
| `frontend/next.config.js` | `allowedDevOrigins: ["10.20.31.204", "localhost", "127.0.0.1"]` so Next.js 16 serves `/_next/*` dev resources (HMR, RSC payloads, dev chunks) over the LAN, not just localhost. | `3279a9ae` |
| `backend/app/gateway/csrf_middleware.py` + `backend/app/gateway/routers/auth.py` (`_set_csrf_cookie`) | `csrf_token` cookie now gets `max_age` matching the `access_token` (`token_expiry_days`, HTTPS-only — mirrors `_set_session_cookie`), so the double-submit pair expires together. **Why:** upstream sets `csrf_token` as a *session* cookie while `access_token` is *persistent* over HTTPS; iOS Safari home-screen PWAs evict session cookies when iOS kills the web app, so the user stays "logged in" (persistent `access_token`) but the next state-changing request 403s with "CSRF token missing". Regression-pinned in `tests/test_auth_type_system.py::test_csrf_cookie_persistent_on_https` (+ `_session_only_on_http`, `_oidc_callback_csrf_cookie_persistent_on_https`). Upstream bug (HTTPS-only) — submitted upstream as PR #3872 (`fork` branch `fix/csrf-cookie-persistent-lifetime`); drop this row once it merges and lands via `git pull`. | `2d31d0a0` |

**New local files (net-new; survive `git merge` unless upstream adds the same path):**
- `backend/packages/harness/deerflow/community/crawl4ai/` — self-hosted **Crawl4AI** `web_fetch` provider (replaces Jina; see section 4). Added by this deployment.
- `docker/docker-compose.crawl4ai.yaml` — Crawl4AI container (own lifecycle; **not** wired into `deploy.sh`). Attaches to the existing `deer-flow_deer-flow` network with **no published host port** (unauthenticated server kept network-internal).

## 2. Safe upgrade procedure
1. **Snapshot:** `git status` — commit or stash any new local edits first; also push `main` to a dated backup branch on `fork` (e.g. `backup/pre-upgrade-YYYYMMDD`) since `fork/main` has diverged and must not be force-clobbered.
2. **Merge upstream INTO `main`** (don't `reset --hard` / `checkout` over local edits — that silently drops them):
   `git fetch origin && git merge origin/main`  *(NB: in this repo `origin` = bytedance upstream, `fork` = your GitHub fork.)*
3. **Resolve conflicts.** Expect a tiny add/add on root `CLAUDE.md` (keep our playbook + re-append `@AGENTS.md`). `docker/docker-compose.yaml` and `docker/nginx/nginx.conf` may conflict if upstream touched them — **keep the local customizations** (re-apply the directory mount; for nginx keep the `X-Forwarded-Proto` map — though as of 2026-06-29 upstream matches it, so it auto-merges). Do not accept "theirs" blindly.
4. **Config schema:** if `config.example.yaml`'s `config_version` is now higher than `config.yaml`'s, run `make config-upgrade` (merges new fields, **keeps** your `models`).
5. **Recreate:** `make up` (rebuild + recreate) or `bash scripts/deploy.sh start` (no rebuild; recreates only changed services with the correct env/secrets). Do **not** use a bare `docker compose up` — it skips deploy.sh's `${...}` interpolation + secret loading and will misconfigure the gateway.
6. **Verify** (section 3).

## 3. Post-upgrade verification
```bash
# Gateway healthy
docker ps --filter name=deer-flow-gateway --format '{{.Status}}'
docker logs --tail 5 deer-flow-gateway          # expect: "Application startup complete"

# Default model + vision intact  → expect: minimax ollamacloud/minimax-m3 True
docker exec deer-flow-gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python -c "from deerflow.config import get_app_config as g; m=g().models[0]; print(m.name, m.model, m.supports_vision)"'

# Hot-reload hardening still active → expect a /app/deer-flow-runtime dir mount, NOT a single-file /app/backend/config.yaml
docker inspect deer-flow-gateway --format '{{range .Mounts}}{{.Destination}} {{end}}'

# view_image tool wired for the default (vision) model → expect: True
docker exec deer-flow-gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python -c "from deerflow.tools.tools import get_available_tools as t; print(\"view_image\" in [x.name for x in t(model_name=\"minimax\", include_mcp=False)])"'

# Public login works (no 403)
curl -sI "$(grep -m1 '^GATEWAY_CORS_ORIGINS=' .env | cut -d= -f2)" | head -1

# CSRF cookie persists over HTTPS (iOS PWA fix) → expect a csrf_token line WITH Max-Age
curl -s -i -X POST http://localhost:2026/api/v1/auth/logout -H "X-Forwarded-Proto: https" | grep -i "set-cookie:.*csrf_token"

# web_fetch provider = self-hosted Crawl4AI, end-to-end (expect markdown, not an error)
docker exec deer-flow-gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python -c "import asyncio; from deerflow.community.crawl4ai.tools import web_fetch_tool; print(asyncio.run(web_fetch_tool.ainvoke(\"https://example.com\"))[:200])"'
```

## 4. Gotchas & fallbacks
- **Config edit not taking effect?** Only happens if the dir-mount hardening got reverted to a single-file mount. Diagnose: `docker exec deer-flow-gateway stat -c '%i' /app/deer-flow-runtime/config.yaml` (or `/app/backend/config.yaml` on the old mount) vs the host inode. Fix: re-apply the compose hardening; as a stopgap, `docker restart deer-flow-gateway` re-resolves the mount.
- **MiniMax can't see images?** Vision requires the gateway route **`ollamacloud/minimax-m3`** — the bare `minimax` alias and `ollamapro/minimax-m3` are **text-only** (silently ignore images). Ollama Cloud retires models periodically; if vision breaks, re-probe routes against the gateway with an **unguessable** test image (random digits — text-only models false-pass red/blue by guessing). Plain `langchain_openai:ChatOpenAI` is correct here (gateway returns reasoning as `reasoning_content`; do **not** switch to `PatchedChatMiniMax`).
- **Login 403 / cookies not Secure after upgrade?** The nginx `X-Forwarded-Proto` map was likely lost in the merge — re-apply it. `GATEWAY_CORS_ORIGINS` in `.env` is a partial backstop.
- **iOS home-screen PWA: "CSRF token missing" 403 after a while?** The `csrf_token` cookie lost its `max_age` (reverted to a session cookie) — re-apply the `2d31d0a0` change so it matches the `access_token` lifetime. After deploying, log out/in once on the device to shed the old session-only cookie.
- **Gateway env interpolation errors on recreate?** You ran `docker compose up` directly instead of `scripts/deploy.sh` — the `${DEER_FLOW_*}` vars and persisted secrets come from deploy.sh; always recreate through it.
- **`web_fetch` via self-hosted Crawl4AI.** `config.yaml` → `tools` → `web_fetch` uses `deerflow.community.crawl4ai.tools:web_fetch_tool` with `base_url: http://crawl4ai:11235`. The server is the `crawl4ai` container (own lifecycle, **not** started by `deploy.sh`/`make up`): start it with `docker compose -f docker/docker-compose.crawl4ai.yaml up -d`. It attaches to the `deer-flow_deer-flow` network and publishes **no host port** — Crawl4AI is unauthenticated, so it is reached only by the gateway over the internal network, never exposed on the host. If `web_fetch` returns `Error: Crawl4AI ...`: confirm the container is up (`docker ps --filter name=crawl4ai`) and healthy (`docker exec crawl4ai curl -fsS http://localhost:11235/health`), and that it shares the gateway's network. Jina remains a commented fallback in `config.yaml` (its `JINA_API_KEY` in `.env` was the failing path). Adding the provider module requires a **gateway rebuild** (`make up`) — config edits alone hot-reload, but new Python code does not.

## 5. Upstream dev guidance
The repo's shared agent guidance (architecture, commands, module guides) lives in [AGENTS.md](AGENTS.md), imported below so Claude Code loads it after the deployment notes above.

@AGENTS.md
