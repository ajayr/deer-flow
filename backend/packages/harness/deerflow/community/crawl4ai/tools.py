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
