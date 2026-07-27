# CLAUDE.md — Local Deployment & Safe Upgrade Notes

> **Scope:** this repo is a **customized production deployment** of DeerFlow — public,
> behind a Pangolin tunnel + double auth, served via Docker (`docker/docker-compose.yaml`,
> compose project `deer-flow`: `deer-flow-gateway`, `deer-flow-nginx`, `deer-flow-frontend`,
> `deer-flow-redis` — Redis stream bridge, upstream-standard since the 2026-07-15 upgrade).
> It carries local customizations on top of upstream. **Read this before upgrading.**
>
> Upstream's dev docs now live in `AGENTS.md` (root, `backend/`, `frontend/`); the sibling
> `CLAUDE.md` files just import them via `@AGENTS.md` (upstream #3770). **Do not put local/ops
> notes there** — they are upstream-tracked and conflict on every upgrade. Keep deployment notes
> in THIS root `CLAUDE.md`; its tail re-imports `@AGENTS.md` so shared agent guidance still loads.
> Upstream ships a root `CLAUDE.md` too, so this file may conflict on upgrade — keep our
> playbook and re-append the `@AGENTS.md` line.

## 1. Local state that must NOT be lost on upgrade

### Gitignored — survives `git pull`/merge automatically; never overwrite, never commit
| Path | Holds |
|---|---|
| `config.yaml` | Models (default `minimax` → `ollamacloud/minimax-m3`, vision on; `glm5stack` secondary), gateway `base_url`, sandbox config. `config_version: 30` (2026-07-27 upgrade; new v26-30 fields kept at safe defaults — scheduler off, `database.checkpoint_channel_mode: full`, `retrieval_adapter: fts5`). NB: `make config-upgrade` rewrites the file and strips comments (verified lossless both times); pre-upgrade copies: `config.yaml.bak` (annotated, pre-v25), `config.yaml.bak-v25-20260727` (pre-v30, untracked — the exact-match `.gitignore` entry only covers `config.yaml.bak`). |
| `.env` | Secrets — gateway API key, `GATEWAY_CORS_ORIGINS` (public HTTPS origin), search API keys. |
| `extensions_config.json` | MCP servers + skills. Carries `enabled: false` for `skill-reviewer` + its 6 eval fixtures (`example-safe-skill`, `injection-example`, `vague-helper`, `partial-package-example`, `zh-output-example`, `blocked-example`). #4098 landed (2026-07-27 merge): `allowed-tools` no longer collapses the global toolset from merely-enabled skills, so the original emergency is over — but keep the FIXTURES disabled until #4164 (fixture exclusion) lands, so they don't show up as real skills (one is a prompt-injection example a user could slash-activate). |
| `frontend/.env` | Dev-only: `DEER_FLOW_DEV_ALLOWED_ORIGINS=10.20.31.204` — carries the LAN-dev customization since upstream #4471 replaced our `next.config.js` patch (see retired rows below). |
| `backend/.deer-flow/` | Auth secret, internal auth token, threads, memory, runtime data. |

`scripts/deploy.sh` only **seeds** `config.yaml` / `extensions_config.json` when missing — it never overwrites an existing one. So model/vision config and all runtime data ride through an upgrade untouched.

### Local edits to TRACKED files — committed on `main`; re-apply if upstream changes them
| File | Customization | Commit |
|---|---|---|
| `docker/docker-compose.yaml` | Gateway mounts the **project root as a directory** (`..:/app/deer-flow-runtime:ro`) with `DEER_FLOW_CONFIG_PATH` / `DEER_FLOW_EXTENSIONS_CONFIG_PATH` pointing into it — so `config.yaml` edits hot-reload **without a restart** (avoids single-file bind-mount inode pinning). Kept through the 2026-07-15 merge (`91041ab6`) and auto-merged cleanly in the 2026-07-27 merge (`4394046f` — upstream only touched build args/provisioner env, no volume conflict); the **production** compose upstream still uses single-file mounts, so this row stays. Upstream's `redis` service was adopted alongside it (2026-07-15). | `103f179e` |
*Rows retired after merging upstream (kept out of the table; see git history if needed):* the nginx `X-Forwarded-Proto` map (`40cbf17f`) — upstream ships the identical map since #3793 and it has auto-merged through two upgrades; the `csrf_token` persistent-cookie fix (`2d31d0a0`) — our PR #3872 merged upstream and landed in the 2026-07-15 merge (regression tests `tests/test_auth_type_system.py::test_csrf_cookie_persistent_on_https` et al. are now upstream-tracked); the DeerMem `from_backend_config` None-drop (`ac16a420`) — our PR #4217 merged upstream (`5c80c07d`) and landed in the 2026-07-27 merge, regression tests `test_from_backend_config_null_values_*` are now upstream-tracked (the `model: {}` guard stays in `config.yaml`, harmless either way); the `next.config.js` hardcoded `allowedDevOrigins` (`3279a9ae`) — superseded by upstream #4471 (`e17aff57`), which reads `DEER_FLOW_DEV_ALLOWED_ORIGINS` via `frontend/src/dev-origins.js`; our LAN IP moved to gitignored `frontend/.env` (see table above), took upstream's file in the 2026-07-27 merge.

**New local files (net-new; survive `git merge` unless upstream adds the same path):**
- `docker/docker-compose.crawl4ai.yaml` — Crawl4AI container (own lifecycle; **not** wired into `deploy.sh`). Attaches to the existing `deer-flow_deer-flow` network with **no published host port**, static IP `172.24.100.5`, bearer auth, and the logged-in-session mounts.
- `docker/crawl4ai/` — `cookies-to-storage-state.py` (browser `cookies.txt` → Playwright storage state) + `README.md` (session creation/rotation runbook and its blast radius). Secrets themselves live **outside the repo** in `~/.crawl4ai/` (mode 700): `session-state.json` (cookies, 600 + ACL for uid 999) and `config.yml` (server config mounted over `/app/config.yml`).
- *(Retired:)* `backend/packages/harness/deerflow/community/crawl4ai/` was net-new here until upstream merged it (#3821, hardened by #3942 SSRF guard + timeout coercion); as of the 2026-07-15 merge the module is upstream-tracked — took upstream's version in the add/add conflict.

## 2. Safe upgrade procedure
1. **Snapshot:** `git status` — commit or stash any new local edits first; also push `main` to a dated backup branch on `fork` (e.g. `backup/pre-upgrade-YYYYMMDD`) since `fork/main` has diverged and must not be force-clobbered.
2. **Merge upstream INTO `main`** (don't `reset --hard` / `checkout` over local edits — that silently drops them):
   `git fetch origin && git merge origin/main`  *(NB: in this repo `origin` = bytedance upstream, `fork` = your GitHub fork.)*
3. **Resolve conflicts.** Root `CLAUDE.md` may conflict (keep our playbook + re-append `@AGENTS.md`). `docker/docker-compose.yaml` will conflict whenever upstream touches the gateway volumes/env — **keep the dir-mount customization** while adopting upstream's other changes. Do not accept "theirs" blindly. (nginx.conf no longer diverges — upstream ships our `X-Forwarded-Proto` map since #3793.)
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

# Public login works (no 403). Unauthenticated → expect 302 to pangolin …/auth/resource… (or 200 if edge-authed)
curl -sI "$(grep -m1 '^GATEWAY_CORS_ORIGINS=' .env | cut -d= -f2)" | head -1

# Forwarded-HTTPS chain honored (Secure attr present). NB semantics changed with upstream #4255
# (2026-07-27 merge): logout now CLEARS csrf_token (Max-Age=0) and suppresses re-issue — that
# clearing cookie WITH "Secure" is the expected output. Cookie *persistence* is now set at login:
# "keep me signed in" (remember_me, default ON) → access_token+csrf_token get matching Max-Age.
curl -s -i -X POST http://localhost:2026/api/v1/auth/logout -H "X-Forwarded-Proto: https" | grep -i "set-cookie:.*csrf_token"   # expect: csrf_token=""; ... Max-Age=0; ... Secure

# web_fetch provider = self-hosted Crawl4AI, end-to-end (expect markdown, not an error)
docker exec deer-flow-gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python -c "import asyncio; from deerflow.community.crawl4ai.tools import web_fetch_tool; print(asyncio.run(web_fetch_tool.ainvoke(\"https://example.com\"))[:200])"'
```

## 4. Gotchas & fallbacks
- **Config edit not taking effect?** Only happens if the dir-mount hardening got reverted to a single-file mount. Diagnose: `docker exec deer-flow-gateway stat -c '%i' /app/deer-flow-runtime/config.yaml` (or `/app/backend/config.yaml` on the old mount) vs the host inode. Fix: re-apply the compose hardening; as a stopgap, `docker restart deer-flow-gateway` re-resolves the mount.
- **MiniMax can't see images?** Vision requires the gateway route **`ollamacloud/minimax-m3`** — the bare `minimax` alias and `ollamapro/minimax-m3` are **text-only** (silently ignore images). Ollama Cloud retires models periodically; if vision breaks, re-probe routes against the gateway with an **unguessable** test image (random digits — text-only models false-pass red/blue by guessing). Plain `langchain_openai:ChatOpenAI` is correct here (gateway returns reasoning as `reasoning_content`; do **not** switch to `PatchedChatMiniMax`).
- **Login 403 / cookies not Secure after upgrade?** The nginx `X-Forwarded-Proto` map was likely lost in the merge — re-apply it. `GATEWAY_CORS_ORIGINS` in `.env` is a partial backstop.
- **iOS home-screen PWA: "CSRF token missing" 403 after a while?** The `csrf_token` cookie became a session cookie. Originally fixed by our PR #3872; since upstream #4255 (2026-07-27 merge) persistence is decided at **login** by the "keep me signed in" checkbox (`remember_me`, **default ON**, remembered via a `deerflow_session_persistent` preference cookie; `session_cookie.py::resolve_session_cookie_policy` → `secure_persistent` on forwarded HTTPS). If the 403 returns: re-login on the device making sure "keep me signed in" is checked. Logout clearing `csrf_token` with `Max-Age=0` is normal (#4255), not the regression.
- **Gateway env interpolation errors on recreate?** You ran `docker compose up` directly instead of `scripts/deploy.sh` — the `${DEER_FLOW_*}` vars and persisted secrets come from deploy.sh; always recreate through it.
- **`web_fetch` via self-hosted Crawl4AI (0.9.2, authed since 2026-07-27).** `config.yaml` → `tools` → `web_fetch` uses `deerflow.community.crawl4ai.tools:web_fetch_tool` with `base_url: http://crawl4ai:11235` and `token: $CRAWL4AI_API_TOKEN` (crawl4ai ≥0.9 is secure-by-default: bearer auth mandatory on everything except `GET /health`; tokenless requests → 401, and a tokenless *server* would bind loopback-only and be unreachable from the gateway). The token lives in root `.env` (`CRAWL4AI_API_TOKEN`) and reaches (a) the gateway env via deploy.sh's env_file — so a fresh var needs a gateway recreate before `$`-resolution sees it — and (b) the crawl4ai container via compose interpolation. The server is the `crawl4ai` container (own lifecycle, **not** started by `deploy.sh`/`make up`): start it **from the repo root** with `docker compose --env-file .env -f docker/docker-compose.crawl4ai.yaml up -d` (`--env-file` is required; without it compose looks for `docker/.env` and the `:?` guard aborts). It attaches to `deer-flow_deer-flow` with **no host port** and a **static IP `172.24.100.5`** — the IP exists for the HOST-side consumers: the hermes skills `web/crawl4ai-web-fetch` (host curl by IP + token file `~/.hermes/secrets/crawl4ai-token`) and `research/last30days` (`scripts/lib/reddit_public.py`, `docker exec` + container env token; that skill dir symlinks to the git repo `~/last30days-skill`). **Changing the token or IP means updating those hermes files too.** Failure triage: `Error: Crawl4AI HTTP 401` → token missing/stale on the caller side; `{"error": "Internal server error", "correlation_id": ...}` → server-side crawl failure, match the id in `docker logs crawl4ai` (e.g. Reddit's IP-based anti-bot 403 — verified 2026-07-27 to hit 0.8.6 and 0.9.2 identically, i.e. Reddit-side, not an upgrade regression). Health: `docker exec crawl4ai curl -fsS http://localhost:11235/health` (auth-exempt). Jina remains a commented fallback in `config.yaml`.
- **Logged-in crawling (Reddit et al).** The server carries a browser session via `crawler.browser.kwargs.storage_state` → `/etc/crawl4ai-local/session-state.json`. It is **global**: every caller (DeerFlow agent included) crawls those domains logged in, so a prompt-injected fetch can read anything that session can — keep the state file narrowed with `--domain` and prefer a burner account. Full runbook + gotchas: [docker/crawl4ai/README.md](docker/crawl4ai/README.md). Three things that fail silently: forgetting `--grant-uid 999` after rotating (container is uid 999, file is 600/uid 1000), mounting the decoy `/tmp/project/deploy/docker/config.yml` instead of the live `/app/config.yml`, and an expired session (symptom = the site 403s again, no error). The provider module is upstream-tracked since 2026-07-15 (#3821/#3942 SSRF guard; crawl4ai 0.9 adds its own server-side SSRF/TLS validation on top). General rule: config edits hot-reload, but new/changed Python code needs a **gateway rebuild** (`make up`).

- **Every run fails with "1 validation error for DeerMemConfig … model_type"?** Fixed upstream since the 2026-07-27 merge (our PR #4217, upstream `5c80c07d`): `from_backend_config` drops explicit YAML `null` values. Can only reappear if the running image predates that merge — instant fix then: set `model: {}` in `config.yaml` (hot-reloads; the failed memory-manager singleton retries on the next run); durable fix: rebuild via `make up`. The `model: {}` guard is still in `config.yaml` and stays (harmless, and config-upgrade won't write `null` while the key exists).
- **Agent suddenly limited to `[read_file, review_skill_package]` ("X is not a valid tool, try one of …")?** Historical (2026-07-15 → 2026-07-27): skill `allowed-tools` used to be a **global union across all enabled skills** (#4095/#4191), so `skill-reviewer` + its eval fixtures (loaded as real skills by the recursive scan, incl. a prompt-injection fixture declaring `bash`) collapsed every chat's toolset. **#4098 landed in the 2026-07-27 merge:** `allowed-tools` now applies only to *actively activated* skills (slash-activated, or read into `skill_context`) — passive enabled skills no longer clamp the toolset. The fixtures stay `enabled: false` in `extensions_config.json` anyway until #4164 (fixture exclusion from the scan) lands, so they don't appear as activatable skills. Host bash was never exposed here (`sandbox.allow_host_bash: false`).
- **`extensions_config.json` skill enable/disable edits not taking effect?** Still true post-2026-07-27 (verified: `lead_agent/prompt.py` keeps the `(id(app_config), user_id)` cache in `get_enabled_skills_for_config`) — file edits to extensions_config are only picked up when the AppConfig object changes. After editing the file, make any content change to `config.yaml` (a comment tweak suffices — hot-reload mints a new AppConfig and the skills cache misses), toggle a skill via `PUT /api/skills/{name}` (invalidates in-worker), call the new admin-only `POST /api/skills/reload` (process-local invalidation, added upstream #4264), or restart the gateway.
- **Host-side `pytest tests/test_auth_type_system.py` fails one test?** `test_get_auth_config_missing_env_var_generates_ephemeral` reads `backend/.deer-flow/.jwt_secret`, which in this deployment is the **production secret written by the gateway container as `root:root` 600** — unreadable from the host, and the code intentionally refuses the ephemeral fallback then (#2933). Pre-existing environmental collision, not a regression (confirmed 2026-07-15); the rest of the file passes. Run auth tests inside the gateway container if a clean pass is needed.

## 5. Upstream dev guidance
The repo's shared agent guidance (architecture, commands, module guides) lives in [AGENTS.md](AGENTS.md), imported below so Claude Code loads it after the deployment notes above.

@AGENTS.md
