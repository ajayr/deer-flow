"""``create_app()`` mounts the OpenAI-compatible router only when ``openai_compat.enabled`` is true."""

from __future__ import annotations

import warnings

import app.gateway.app as app_module
from deerflow.config.app_config import AppConfig
from deerflow.config.openai_compat_config import OpenAICompatConfig
from deerflow.config.sandbox_config import SandboxConfig

PREFIX = "/api/openai/v1"
MODELS_PATH = f"{PREFIX}/models"
CHAT_PATH = f"{PREFIX}/chat/completions"


def _config(**openai_compat) -> AppConfig:
    return AppConfig(sandbox=SandboxConfig(use="test"), openai_compat=OpenAICompatConfig(**openai_compat))


def _create_app(monkeypatch, config: AppConfig):
    monkeypatch.setattr(app_module, "get_app_config", lambda: config)
    return app_module.create_app()


def _route_paths(app) -> set[str]:
    return {route.path for route in app.routes if hasattr(route, "path")}


def test_the_api_is_not_mounted_by_default(monkeypatch):
    app = _create_app(monkeypatch, _config())

    assert not any(path.startswith(PREFIX) for path in _route_paths(app))
    assert not any(path.startswith(PREFIX) for path in app.openapi()["paths"])
    # The rest of the Gateway is untouched.
    assert "/api/models" in _route_paths(app)


def test_the_api_is_mounted_when_enabled(monkeypatch):
    app = _create_app(monkeypatch, _config(enabled=True))

    assert {MODELS_PATH, CHAT_PATH} <= _route_paths(app)
    paths = app.openapi()["paths"]
    assert paths[MODELS_PATH]["get"]["tags"] == ["openai-compat"]
    assert paths[CHAT_PATH]["post"]["tags"] == ["openai-compat"]
    assert "/api/models" in _route_paths(app)


def test_the_chat_operation_documents_every_way_it_can_end(monkeypatch):
    app = _create_app(monkeypatch, _config(enabled=True))

    responses = app.openapi()["paths"][CHAT_PATH]["post"]["responses"]

    assert set(responses) == {"200", "400", "401", "403", "404", "409", "500", "502"}
    assert set(responses["200"]["content"]) == {"application/json", "text/event-stream"}
    assert responses["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/ChatCompletion")
    for status in ("400", "404", "409", "500", "502"):
        assert responses[status]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse"), status
    for status in ("401", "403"):
        assert "detail" in responses[status]["description"], status


def test_narrowing_modes_does_not_mount_the_api(monkeypatch):
    """Only ``enabled`` decides whether the routes exist; ``modes`` and ``default_mode`` shape the listing."""
    app = _create_app(monkeypatch, _config(modes=["flash"], default_mode="flash"))

    assert not {MODELS_PATH, CHAT_PATH} & _route_paths(app)


def test_an_absent_config_file_leaves_the_api_unmounted(monkeypatch):
    """create_app() runs at import time, where config.yaml may not exist (CI runners, fresh checkouts)."""

    def missing_config():
        raise FileNotFoundError("config.yaml")

    monkeypatch.setattr(app_module, "get_app_config", missing_config)

    assert not {MODELS_PATH, CHAT_PATH} & _route_paths(app_module.create_app())


def test_the_mounted_api_keeps_operation_ids_unique(monkeypatch):
    """test_openapi_operation_ids.py only sees the module-level app, which is built without a config.yaml and so leaves the API off."""
    app = _create_app(monkeypatch, _config(enabled=True))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = app.openapi()["paths"]

    assert [str(item.message) for item in caught if "Duplicate Operation ID" in str(item.message)] == []
    operation_ids = [operation["operationId"] for path_item in paths.values() for operation in path_item.values() if isinstance(operation, dict) and "operationId" in operation]
    assert len(operation_ids) == len(set(operation_ids))
    assert paths[MODELS_PATH]["get"]["operationId"] in operation_ids
    assert paths[CHAT_PATH]["post"]["operationId"] in operation_ids
