from __future__ import annotations

import collections.abc
import dataclasses
import enum
import os
import typing

from openai import OpenAI
import pydantic


_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _no_api_key() -> str:
    return ""


class RequestError(RuntimeError):
    """Raised when an OpenAI-compatible chat completion request fails."""


def _empty_headers() -> dict[str, str]:
    return {}


def _empty_body() -> dict[str, object]:
    return {}


class OpenAIChatCompletionsResource(typing.Protocol):
    """OpenAI SDK chat completions resource used by this provider."""

    def create(self, **kwargs: object) -> object:
        """Create a chat completion with SDK-compatible keyword arguments."""
        ...


class OpenAIChatResource(typing.Protocol):
    """OpenAI SDK chat resource used by this provider."""

    completions: OpenAIChatCompletionsResource


class OpenAIClient(typing.Protocol):
    """OpenAI SDK client shape used by this provider."""

    chat: OpenAIChatResource


class ChatCompletionClient(typing.Protocol):
    """Client interface used by stateforward.bot OpenAI-compatible text generators."""

    def create_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Create a chat completion response using provider-neutral JSON values."""
        ...


def jsonable(value: object) -> object:
    """Convert Pydantic, enum, dataclass, mapping, and sequence values to JSON-compatible values."""

    if isinstance(value, pydantic.BaseModel):
        return typing.cast(object, value.model_dump(mode="json"))
    if isinstance(value, enum.Enum):
        return typing.cast(object, value.value)
    if isinstance(value, (bytes, bytearray)):
        # Prefer UTF-8 text for speech/transcript bytes; fall back to base64 for binary.
        raw = bytes(value)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            import base64

            return base64.b64encode(raw).decode("ascii")
    # HSM events are dataclasses whose ``schema`` field holds a TypeAdapter; project fields only.
    try:
        import hsm
    except ImportError:  # pragma: no cover - provider always runs with hsm installed
        hsm = None  # type: ignore[assignment]
    if hsm is not None and isinstance(value, hsm.Event):
        event = typing.cast(hsm.Event[object], value)
        return {
            "name": event.name,
            "data": jsonable(event.data),
            "kind": event.kind,
            "id": event.id or "",
            "source": event.source or "",
            "target": event.target or "",
            "metadata": jsonable(dict(event.metadata)) if event.metadata else {},
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        return {str(key): jsonable(item) for key, item in mapping.items()}
    if isinstance(value, collections.abc.Sequence) and not isinstance(value, str | bytes | bytearray):
        return [jsonable(item) for item in value]
    return value


def jsonable_mapping(value: collections.abc.Mapping[str, object]) -> dict[str, object]:
    """Convert a string-keyed mapping to a JSON-compatible dictionary."""

    converted = jsonable(value)
    if not isinstance(converted, collections.abc.Mapping):
        raise TypeError("JSON-compatible mapping conversion must produce an object.")
    converted_mapping = typing.cast(collections.abc.Mapping[object, object], converted)
    return {str(key): item for key, item in converted_mapping.items()}


def _normalized_base_url(value: str) -> str:
    return value.rstrip("/")


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChatClient(ChatCompletionClient):
    """OpenAI SDK-backed client for providers that expose the Chat Completions API shape."""

    model: str
    base_url: str = _DEFAULT_OPENAI_BASE_URL
    api_key: str | None = None
    timeout_seconds: float = 240.0
    default_headers: collections.abc.Mapping[str, str] = dataclasses.field(default_factory=_empty_headers)
    default_body: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_body)
    client: OpenAIClient | None = None
    _sdk_client: OpenAIClient | None = dataclasses.field(default=None, init=False, repr=False, compare=False)

    @typing.override
    def create_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        body: dict[str, object] = dict(self.default_body)
        if extra_body is not None:
            body.update(jsonable_mapping(dict(extra_body)))

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": typing.cast(list[object], jsonable(messages)),
        }
        if tools:
            kwargs["tools"] = typing.cast(list[object], jsonable(tools))
        if response_format is not None:
            kwargs["response_format"] = jsonable(response_format)
        if body:
            kwargs["extra_body"] = body

        client = self._client()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as error:
            message = "OpenAI-compatible chat completion request failed."
            raise RequestError(message) from error
        json_response = jsonable(response)
        if not isinstance(json_response, collections.abc.Mapping):
            raise TypeError("OpenAI-compatible chat completion response must be a JSON object.")
        response_mapping = typing.cast(collections.abc.Mapping[object, object], json_response)
        return {str(key): item for key, item in response_mapping.items()}

    def _client(self) -> OpenAIClient:
        if self.client is not None:
            return self.client
        if self._sdk_client is not None:
            return self._sdk_client
        client = OpenAI(
            api_key=self._api_key(),
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            default_headers=dict(self.default_headers) or None,
        )
        sdk_client = typing.cast(OpenAIClient, typing.cast(object, client))
        object.__setattr__(self, "_sdk_client", sdk_client)
        return sdk_client

    def _api_key(self) -> str | collections.abc.Callable[[], str]:
        if self.api_key is not None:
            return self.api_key
        if _normalized_base_url(self.base_url) == _normalized_base_url(_DEFAULT_OPENAI_BASE_URL):
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                return api_key
            raise ValueError("api_key is required when base_url is the default OpenAI API endpoint.")
        return _no_api_key


__all__ = [
    "ChatCompletionClient",
    "OpenAI",
    "ChatClient",
    "RequestError",
    "OpenAIClient",
    "jsonable",
    "jsonable_mapping",
]
