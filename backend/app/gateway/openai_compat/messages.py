"""The ``messages`` of a chat-completions request: what is accepted and what the router keeps of it.

Pure: no FastAPI, no I/O. The router keeps only the role and the text of each message. Roles other
than ``user`` and ``assistant``, content other than text, and every field that would make the
agent behave differently (tool calls, audio) are refused with a 400 that says why, so a client
never believes it sent something the agent did not receive. Fields that do not change the answer
(``name``, the reasoning an earlier reply carried) are accepted, ignored and reported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from app.gateway.openai_compat.errors import invalid_message, invalid_request, quote
from app.gateway.openai_compat.params import MESSAGE_REJECTED, safe_name

Role = Literal["user", "assistant"]

_ROLES: tuple[Role, ...] = ("user", "assistant")
_KEPT_KEYS = frozenset({"role", "content"})

_SYSTEM_PROMPT_HINT = "Clear the system prompt in your client; DeerFlow uses its own."
_NO_TOOLS_HINT = "DeerFlow runs its own tools on the server."
_NOT_YET = {
    "image_url": "Image input is not supported yet.",
    "input_image": "Image input is not supported yet.",
    "input_audio": "Audio input is not supported yet.",
    "file": "File input is not supported yet.",
}


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message of the history, reduced to what the router uses."""

    role: Role
    text: str


@dataclass(frozen=True, slots=True)
class ParsedMessages:
    """The history and the message-level fields that were accepted and ignored."""

    messages: tuple[ChatMessage, ...]
    #: Names such as ``messages[].name``, each once, in order of first appearance.
    ignored: tuple[str, ...]


def _text_of(content: Any, param: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise invalid_message("Message content must be a string or a list of text parts.", param=f"{param}.content")
    texts: list[str] = []
    for index, part in enumerate(content):
        part_param = f"{param}.content[{index}]"
        if not isinstance(part, Mapping):
            raise invalid_message("Each content part must be an object.", param=part_param)
        kind = part.get("type")
        if kind != "text":
            reason = _NOT_YET.get(kind) if isinstance(kind, str) else None
            raise invalid_message(reason or f"Content part type {quote(kind)} is not supported; only text is.", param=f"{part_param}.type")
        text = part.get("text")
        if not isinstance(text, str):
            raise invalid_message("A text part needs a string 'text'.", param=f"{part_param}.text")
        texts.append(text)
    return "\n".join(texts)


def _role_of(message: Mapping[str, Any], param: str) -> Role:
    role = message.get("role")
    if role in _ROLES:
        return role
    if role in ("system", "developer"):
        raise invalid_message(f"{str(role).capitalize()} messages are not supported. {_SYSTEM_PROMPT_HINT}", param=f"{param}.role")
    if role in ("tool", "function"):
        raise invalid_message(f"{str(role).capitalize()} messages are not supported. {_NO_TOOLS_HINT}", param=f"{param}.role")
    raise invalid_message("Each message needs a role of 'user' or 'assistant'.", param=f"{param}.role")


def parse_messages(raw: Any) -> ParsedMessages:
    """Check ``messages`` and reduce it to roles and text.

    Raises ``OpenAIError`` (a 400): ``invalid_request`` when the list itself is malformed,
    ``invalid_message`` when a message breaks a message rule, ``unsupported_parameter`` for a
    field that would change what the agent does. The history must start and end with a user
    message, and every user message needs some text.
    """
    if not isinstance(raw, list) or not raw:
        raise invalid_request("'messages' is required and must be a non-empty list.", param="messages")
    messages: list[ChatMessage] = []
    ignored: list[str] = []
    for index, item in enumerate(raw):
        param = f"messages[{index}]"
        if not isinstance(item, Mapping):
            raise invalid_message("Each message must be an object.", param=param)
        role = _role_of(item, param)
        for key, rule in MESSAGE_REJECTED.items():
            value = item.get(key)
            if value is not None and not (rule.accepts and rule.accepts(value)):
                raise rule.rejection(f"{param}.{key}")
        for key, value in item.items():
            if value is not None and key not in _KEPT_KEYS and key not in MESSAGE_REJECTED:
                name = f"messages[].{safe_name(key)}"
                if name not in ignored:
                    ignored.append(name)
        text = _text_of(item.get("content"), param)
        if role == "user" and not text.split():
            raise invalid_message("A user message needs some text.", param=param)
        messages.append(ChatMessage(role=role, text=text))
    if messages[0].role != "user":
        raise invalid_message("The conversation must start with a user message.", param="messages[0].role")
    last = len(messages) - 1
    if messages[last].role != "user":
        raise invalid_message("The last message must be a user message.", param=f"messages[{last}]")
    return ParsedMessages(messages=tuple(messages), ignored=tuple(ignored))
