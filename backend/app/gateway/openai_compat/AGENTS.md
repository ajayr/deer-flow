# OpenAI-compatible API

`routers/openai_compat.py` serves `/api/openai/v1` for OpenAI-style clients such as the Conduit iOS app in Direct mode. `create_app()` mounts it once at startup, and only when `openai_compat.enabled` is true (a startup-only field in `reload_boundary.py`); with the flag off the routes are not registered and are absent from OpenAPI. `default_mode` and `modes` are read on every request.

Every route carries `@require_permission("runs", "create")` and has its own entry in `auth/pat.py::_PAT_ROUTE_RULES`, so an ordinary PAT with `runs:create` is the credential. The PAT test derives its expectation from the router: a route added here fails `test_pat_openai_compat_policy_admits_exactly_the_mounted_routes` until it is allowlisted. The `/api/openai/` nginx location (in all three configs) carries the large-body, unspooled-request settings that a conversation resent on every request needs; `tests/test_nginx_openai_compat_route.py` pins them.

This package is plain Python: it imports neither FastAPI nor Starlette, and the router is the only module that does, so the logic tests without an app. `schemas.py` holds the response models.

`modes.py` builds the ids a caller can send as `model`: the bare name (`default_mode`, clamped to the model's reasoning contract) and one `<model>:<mode>` variant per mode that the contract allows and `modes` lists, in the order flash, thinking, pro, ultra. Keep these rules:

- Build the catalogue from `list_models()` output, never from `config.models`. That keeps model authorization and its fail-closed handling, and a model hidden from the caller can neither leak its name nor take a variant id from a visible model.
- A name equal to a model the caller can use always means that model, so a colliding variant id is skipped. Resolve a requested `model` the same way (`resolve_model`): it must equal an id that `GET /models` lists for the caller, compared exactly (no case folding, no trimming, no splitting at a colon), so `<name>:<mode>` for a model that offers a single mode is a 404.
- A model that offers no mode is not listed; one that offers a single mode is listed by its bare name only.
- Every entry advertises `ADVERTISED_CONTEXT_LENGTH` and `is_multimodal: false`. Conduit falls back to a 4096-token window when the length is missing and then trims the history it resends, and it treats a missing flag as multimodal. Do not substitute a model's real window.

`params.py` and `messages.py` decide what a request may contain, and `errors.py` builds the error envelope. `POLICY` in `params.py` is the only list of the parameters that are honored, ignored (reported in `X-DeerFlow-Ignored-Params`) or rejected with a 400. Keep these rules:

- Never drop a field silently: honor it, ignore and report it, or reject it with a 400 that says why. A new OpenAI field starts as ignored. `null` counts as absent.
- A message keeps only its role and text. The history starts and ends with a user message, and every user message needs text.
- Error messages stay under 400 characters (Conduit cuts at 512), quote at most 40 characters of the request and carry nothing from an exception.

`conversation.py` decides whether a resent history continues a thread or starts a new one, and `ledger.py` keeps the hashes it compares in the thread's own metadata (`openai_conv`, `openai_turns`). `ledger.py` is the only module here that touches a store, through an injected `ThreadMetaStore`. Keep these rules:

- Store hashes of whitespace-normalised text, never the text. Seeding never truncates a history; the ingest caps reject it.
- A failed ledger write is logged and the run goes ahead; the next request is then treated as a new conversation. It never fails a request.

`translator.py` is the only place that decides what a client sees of a run. It is pure (no I/O, no clock, no FastAPI): the router passes it each stream frame and the current monotonic time. Its docstrings hold the routing rules; `errors.py` holds the fixed in-band error messages, which carry nothing from the failure. Keep these invariants:

- Never forward tool arguments or outputs (the one exception, the question of an `ask_clarification` call or a sandbox network approval request, ends the answer), human or system messages (hidden memory rides that channel), AI messages from the `tools` node, or frames from nested graphs. Never emit `tool_calls`: Conduit turns it into an error.
- A stream ends with `[DONE]` or with one in-band error, never both.
- The streamed `content` should equal `stored_answer()` of the thread's messages, because clients resend it as history. Middleware that rewrites a message after it streamed breaks that, so the router must pass the stored answer to `finish(stored=...)`.
- `tests/fixtures/openai_compat/streams/` holds hermetic recordings (a scripted model, not the live stack). After a library upgrade, regenerate them with `DEERFLOW_WRITE_FIXTURES=1` and review the diff.

Config lives in `deerflow/config/openai_compat_config.py`. A new key needs `config.example.yaml`, `docs/CONFIGURATION.md` and `tests/test_openai_compat_docs.py` updated together.
