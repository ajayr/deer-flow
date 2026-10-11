"""``GET /api/openai/v1/models``: the ids an OpenAI-style client can pick from."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.auth_disabled import AUTH_SOURCE_PAT
from app.gateway.authz import AuthContext
from app.gateway.deps import get_config
from app.gateway.openai_compat.modes import ADVERTISED_CONTEXT_LENGTH
from app.gateway.routers import openai_compat as openai_router
from deerflow.authz.rbac import RbacAuthorizationProvider
from deerflow.config.app_config import AppConfig
from deerflow.config.authorization_config import AuthorizationConfig
from deerflow.config.model_config import ModelConfig
from deerflow.config.openai_compat_config import OpenAICompatConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.token_usage_config import TokenUsageConfig

URL = "/api/openai/v1/models"
ALL_VARIANTS = ["alpha", "alpha:flash", "alpha:thinking", "alpha:pro", "alpha:ultra"]


def _user():
    return SimpleNamespace(id="user-123", system_role="user", oauth_provider="github", oauth_id="oauth-456")


def _model(name: str, thinking: str = "optional", **extra) -> ModelConfig:
    return ModelConfig(name=name, model=name, use="langchain_openai:ChatOpenAI", reasoning={"thinking": thinking}, **extra)


def _app_config(models: list[ModelConfig], *, authorization: AuthorizationConfig | None = None, **openai_compat) -> AppConfig:
    return AppConfig(
        models=models,
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
        token_usage=TokenUsageConfig(enabled=False),
        authorization=authorization or AuthorizationConfig(),
        openai_compat=OpenAICompatConfig(**openai_compat),
    )


class _Served:
    """The router behind stub auth: a request is authenticated only when it names its permissions.

    An authenticated request looks the way ``AuthMiddleware`` leaves a personal-access-token call:
    ``request.state.auth`` holds the permissions, and ``request.state.user`` with ``auth_source``
    ``pat`` holds the user the Gateway's own user lookup returns, so model authorization runs on
    that lookup rather than on a stand-in for it.

    ``config`` is what ``get_config`` returns. A test may swap it between
    requests, which is what a config hot reload does.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        app = FastAPI()
        app.include_router(openai_router.router)
        app.dependency_overrides[get_config] = lambda: self.config

        @app.middleware("http")
        async def stub_auth(request, call_next):
            permissions = request.headers.get("x-test-permissions")
            granted = [p for p in (permissions or "").split(",") if p]
            request.state.auth = AuthContext(user=_user() if permissions is not None else None, permissions=granted)
            if permissions is not None:
                request.state.user = request.state.auth.user
                request.state.auth_source = AUTH_SOURCE_PAT
            return await call_next(request)

        self._client = TestClient(app)

    def models(self, *, permissions: str | None = "runs:create"):
        headers = {} if permissions is None else {"x-test-permissions": permissions}
        return self._client.get(URL, headers=headers)

    def ids(self) -> list[str]:
        response = self.models()
        assert response.status_code == 200, response.text
        return [entry["id"] for entry in response.json()["data"]]


class _FailingProvider:
    """Authorization provider whose every call raises."""

    name = "failing"

    def authorize(self, request):
        raise RuntimeError("provider down")

    async def aauthorize(self, request):
        raise RuntimeError("provider down")

    def filter_resources(self, principal, resource_type, candidates):
        raise RuntimeError("provider down")


def _enable_authorization(monkeypatch, provider, *, fail_closed: bool = True) -> None:
    config = AuthorizationConfig(enabled=True, fail_closed=fail_closed, default_role="user")
    monkeypatch.setattr("app.gateway.authz._get_route_authorization_config", lambda: config)
    monkeypatch.setattr("app.gateway.authz._get_cached_route_provider", lambda c: provider)


@pytest.fixture(autouse=True)
def _authorization_off(monkeypatch):
    """Model authorization is off unless a test enables it."""
    monkeypatch.setattr("app.gateway.authz._get_route_authorization_config", lambda: AuthorizationConfig(enabled=False))


# ── response shape ───────────────────────────────────────────────────────────


def test_the_listing_has_the_openai_shape_and_the_client_hints():
    response = _Served(_app_config([_model("alpha", display_name="Alpha", description="The first one")])).models()

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [entry["id"] for entry in body["data"]] == ALL_VARIANTS
    first = body["data"][0]
    assert first == {
        "id": "alpha",
        "object": "model",
        "created": first["created"],
        "owned_by": "deerflow",
        "name": "Alpha",
        "description": "The first one",
        "is_multimodal": False,
        "context_length": ADVERTISED_CONTEXT_LENGTH,
    }
    assert isinstance(first["created"], int)
    assert 0 < first["created"] <= int(time.time())
    assert {entry["created"] for entry in body["data"]} == {first["created"]}
    assert body["data"][3]["name"] == "Alpha (pro)"


def test_a_model_without_a_description_leaves_the_field_out():
    entry = _Served(_app_config([_model("alpha")])).models().json()["data"][0]

    assert "description" not in entry
    assert entry["name"] == "alpha"


# ── which ids are offered ────────────────────────────────────────────────────


def test_variants_follow_each_models_reasoning_contract():
    ids = _Served(_app_config([_model("alpha", "optional"), _model("fixed", "unsupported"), _model("always", "required")])).ids()

    assert ids == [*ALL_VARIANTS, "fixed", "always", "always:thinking", "always:pro", "always:ultra"]


@pytest.mark.parametrize(
    ("supports_thinking", "expected"),
    [(True, ["legacy", "legacy:flash", "legacy:thinking", "legacy:pro", "legacy:ultra"]), (False, ["legacy"])],
    ids=["thinking-flag", "no-flag"],
)
def test_a_model_with_only_the_legacy_thinking_flag_gets_the_derived_contract(supports_thinking, expected):
    legacy = ModelConfig(name="legacy", model="legacy", use="langchain_openai:ChatOpenAI", supports_thinking=supports_thinking)

    assert _Served(_app_config([legacy])).ids() == expected


def test_narrowing_modes_to_one_lists_bare_names_only():
    served = _Served(_app_config([_model("alpha"), _model("fixed", "unsupported")], modes=["pro"], default_mode="pro"))

    assert served.ids() == ["alpha"]


def test_modes_are_read_on_every_request():
    served = _Served(_app_config([_model("alpha")]))
    assert served.ids() == ALL_VARIANTS

    served.config = _app_config([_model("alpha")], modes=["flash", "pro"], default_mode="pro")

    assert served.ids() == ["alpha", "alpha:flash", "alpha:pro"]


def test_an_empty_model_list_gives_an_empty_listing():
    assert _Served(_app_config([])).models().json() == {"object": "list", "data": []}


# ── who may call ─────────────────────────────────────────────────────────────


def test_the_caller_needs_runs_create():
    response = _Served(_app_config([_model("alpha")])).models(permissions="threads:read")

    assert response.status_code == 403
    assert response.json() == {"detail": "Permission denied: runs:create"}


def test_an_unauthenticated_caller_is_rejected():
    response = _Served(_app_config([_model("alpha")])).models(permissions=None)

    assert response.status_code == 401


# ── model authorization ──────────────────────────────────────────────────────


def test_models_the_role_may_not_use_are_not_listed(monkeypatch):
    _enable_authorization(monkeypatch, RbacAuthorizationProvider(roles={"user": {"models": {"allow": ["alpha"]}}}))

    assert _Served(_app_config([_model("alpha"), _model("beta")])).ids() == ALL_VARIANTS


def test_the_listing_is_empty_when_the_role_may_use_no_configured_model(monkeypatch):
    _enable_authorization(monkeypatch, RbacAuthorizationProvider(roles={"user": {"models": {"allow": ["something-else"]}}}))

    assert _Served(_app_config([_model("alpha")])).models().json() == {"object": "list", "data": []}


@pytest.mark.parametrize(("fail_closed", "listed"), [(True, False), (False, True)], ids=["fail-closed", "fail-open"])
def test_a_provider_failure_follows_fail_closed(monkeypatch, fail_closed, listed):
    _enable_authorization(monkeypatch, _FailingProvider(), fail_closed=fail_closed)
    served = _Served(_app_config([_model("alpha")], authorization=AuthorizationConfig(fail_closed=fail_closed)))

    response = served.models()

    assert response.status_code == 200
    assert bool(response.json()["data"]) is listed


def test_a_model_hidden_from_the_caller_changes_nothing_they_see(monkeypatch):
    """A hidden model named like a variant (``alpha:pro``) neither leaks its name nor takes the id away from the visible model."""
    _enable_authorization(monkeypatch, RbacAuthorizationProvider(roles={"user": {"models": {"allow": ["alpha"]}}}))
    served = _Served(_app_config([_model("alpha", display_name="Alpha"), _model("alpha:pro", display_name="Secret Model")]))

    response = served.models()

    by_id = {entry["id"]: entry for entry in response.json()["data"]}
    assert list(by_id) == ALL_VARIANTS
    assert by_id["alpha:pro"]["name"] == "Alpha (pro)"
    assert "Secret Model" not in response.text
