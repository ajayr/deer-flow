"""Response models for the OpenAI-compatible API: the subset of the OpenAI schema this API serves."""

from typing import Literal

from pydantic import BaseModel, Field


class ModelObject(BaseModel):
    """One entry of ``GET /models``: an OpenAI ``model`` object plus the fields Open WebUI-style clients read."""

    id: str = Field(..., description="Id to send as `model` in a request: `<model>` or `<model>:<mode>`")
    object: Literal["model"] = "model"
    created: int = Field(..., description="Unix time the Gateway process started")
    owned_by: str = "deerflow"
    name: str = Field(..., description="Display name; a mode variant carries its mode in parentheses")
    description: str | None = Field(default=None, description="The model's description; omitted when it has none")
    is_multimodal: bool = Field(..., description="Whether the model accepts images; false until image input is supported")
    context_length: int = Field(..., description="Advertised context window; the same constant for every entry")


class ModelList(BaseModel):
    """Response of ``GET /models``."""

    object: Literal["list"] = "list"
    data: list[ModelObject]
