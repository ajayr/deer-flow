"""nginx routing for the OpenAI-compatible API (``/api/openai/v1``).

Without its own location the API would fall into the ``/api/`` catch-all, whose
default 1m ``client_max_body_size`` rejects a long conversation: an OpenAI-style
client resends the whole history on every request. The location carries the
large-body settings of ``/api/langgraph/`` in all three maintained configs.
"""

from __future__ import annotations

import re

import pytest
from support.nginx_conf import NGINX_CONFIGS
from support.nginx_conf import extract_location_block as _extract_location_block
from support.nginx_conf import parse_body_size_bytes as _parse_body_size_bytes
from support.nginx_conf import parse_read_timeout_seconds as _parse_read_timeout_seconds
from support.nginx_conf import read_config as _read

_LOCATION = "/api/openai/"
_API_PATHS = ["/api/openai/v1/models", "/api/openai/v1/chat/completions"]

# Same bounds as the /api/langgraph/ text-prompt route: well above nginx's 1m
# default, well below the 100M a file upload may carry.
_MIN_BODY_SIZE_BYTES = 5 * 1024 * 1024
_MAX_BODY_SIZE_BYTES = 100 * 1024 * 1024
_MIN_READ_TIMEOUT_SECONDS = 600


@pytest.mark.parametrize("path", NGINX_CONFIGS)
def test_openai_route_accepts_a_long_conversation(path):
    block = _extract_location_block(_read(path), _LOCATION)

    size_bytes = _parse_body_size_bytes(block)

    assert _MIN_BODY_SIZE_BYTES <= size_bytes <= _MAX_BODY_SIZE_BYTES, f"{path}: {_LOCATION} client_max_body_size is {size_bytes} bytes, expected between {_MIN_BODY_SIZE_BYTES} and {_MAX_BODY_SIZE_BYTES}"
    assert "proxy_request_buffering off;" in block, f"{path}: {_LOCATION} spools request bodies to a temp file, which fails on a non-root nginx (issue #3952)"


@pytest.mark.parametrize("path", NGINX_CONFIGS)
def test_openai_route_streams_replies_unbuffered(path):
    block = _extract_location_block(_read(path), _LOCATION)

    assert "proxy_buffering off;" in block, f"{path}: {_LOCATION} would hold a streamed reply back until the run ends"
    assert "proxy_cache off;" in block


@pytest.mark.parametrize("path", NGINX_CONFIGS)
def test_openai_route_outlasts_a_whole_run(path):
    block = _extract_location_block(_read(path), _LOCATION)

    assert _parse_read_timeout_seconds(block) >= _MIN_READ_TIMEOUT_SECONDS


@pytest.mark.parametrize("path", NGINX_CONFIGS)
def test_openai_route_proxies_to_the_gateway_with_the_caller_address(path):
    block = _extract_location_block(_read(path), _LOCATION)

    # No URI after the upstream, so /api/openai/v1/... reaches the Gateway unchanged.
    assert re.search(r"proxy_pass\s+http://[^\s/;]+;", block), f"{path}: proxy_pass must not rewrite the request path"
    assert "proxy_set_header X-Real-IP $remote_addr;" in block
    assert "proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;" in block
    assert "proxy_set_header Connection '';" in block


@pytest.mark.parametrize("path", NGINX_CONFIGS)
def test_openai_settings_stay_out_of_the_api_catch_all(path):
    block = _extract_location_block(_read(path), "/api/")

    assert "client_max_body_size" not in block
    assert "proxy_request_buffering" not in block


@pytest.mark.parametrize("api_path", _API_PATHS)
@pytest.mark.parametrize("path", NGINX_CONFIGS)
def test_no_regex_location_takes_the_openai_paths(path, api_path):
    """nginx prefers a matching regex location over any plain prefix location, so one that matched would bypass ``/api/openai/``."""
    regex_locations = re.findall(r"location\s+~\*?\s+(\S+)\s*\{", _read(path))

    assert regex_locations, "the config has regex locations, so this guard is looking at the right syntax"
    assert [pattern for pattern in regex_locations if re.search(pattern, api_path)] == []
