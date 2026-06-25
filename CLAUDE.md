# CLAUDE.md — Local Deployment & Safe Upgrade Notes

> **Scope:** this repo is a **customized production deployment** of DeerFlow — public,
> behind a Pangolin tunnel + double auth, served via Docker (`docker/docker-compose.yaml`,
> compose project `deer-flow`: `deer-flow-gateway`, `deer-flow-nginx`, `deer-flow-frontend`).
> It carries local customizations on top of upstream. **Read this before upgrading.**
>
> The upstream dev docs are `backend/CLAUDE.md` and `frontend/CLAUDE.md` — **do not put
> local/ops notes there**; they are upstream-tracked and will conflict on every upgrade.
> Keep deployment-specific notes in THIS file (root `CLAUDE.md`, which upstream does not have).

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
| `docker/nginx/nginx.conf` | `map $http_x_forwarded_proto $forwarded_proto` + every proxy block uses `X-Forwarded-Proto $forwarded_proto` — so login works behind the Pangolin TLS proxy (else Gateway 403 "Cross-site auth request denied"). Falls back to `$scheme` when nginx is the TLS edge. | `40cbf17f` |
| `frontend/next.config.js` | Local change — currently **uncommitted**; review and commit before upgrading or it may block/lose on merge. | — |

## 2. Safe upgrade procedure
1. **Snapshot:** `git status` — commit or stash any new local edits first (e.g. `frontend/next.config.js`).
2. **Merge upstream INTO `main`** (don't `reset --hard` / `checkout` over local edits — that silently drops them):
   `git fetch <upstream-remote> && git merge <upstream-remote>/main`
3. **Resolve conflicts.** Expect them in `docker/docker-compose.yaml` and `docker/nginx/nginx.conf` if upstream touched those files — **keep the local customizations** (re-apply the directory mount + the `X-Forwarded-Proto` map on top of upstream's version; do not accept "theirs" blindly).
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
```

## 4. Gotchas & fallbacks
- **Config edit not taking effect?** Only happens if the dir-mount hardening got reverted to a single-file mount. Diagnose: `docker exec deer-flow-gateway stat -c '%i' /app/deer-flow-runtime/config.yaml` (or `/app/backend/config.yaml` on the old mount) vs the host inode. Fix: re-apply the compose hardening; as a stopgap, `docker restart deer-flow-gateway` re-resolves the mount.
- **MiniMax can't see images?** Vision requires the gateway route **`ollamacloud/minimax-m3`** — the bare `minimax` alias and `ollamapro/minimax-m3` are **text-only** (silently ignore images). Ollama Cloud retires models periodically; if vision breaks, re-probe routes against the gateway with an **unguessable** test image (random digits — text-only models false-pass red/blue by guessing). Plain `langchain_openai:ChatOpenAI` is correct here (gateway returns reasoning as `reasoning_content`; do **not** switch to `PatchedChatMiniMax`).
- **Login 403 / cookies not Secure after upgrade?** The nginx `X-Forwarded-Proto` map was likely lost in the merge — re-apply it. `GATEWAY_CORS_ORIGINS` in `.env` is a partial backstop.
- **Gateway env interpolation errors on recreate?** You ran `docker compose up` directly instead of `scripts/deploy.sh` — the `${DEER_FLOW_*}` vars and persisted secrets come from deploy.sh; always recreate through it.
