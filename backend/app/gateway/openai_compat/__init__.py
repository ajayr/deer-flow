"""Support code for the OpenAI-compatible API (``/api/openai/v1``).

``app.gateway.routers.openai_compat`` is the only module that imports FastAPI;
the modules in this package stay pure so they can be tested without an app.
See ``AGENTS.md`` in this directory.
"""
