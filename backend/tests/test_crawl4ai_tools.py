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
