from __future__ import annotations

import base64
import collections.abc
import dataclasses
import enum
import os
import typing

import openai
from openai import OpenAI
from opentelemetry import trace
import pydantic

import mosfet.media as media
from mosfet.abilities import embedding

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_STATUS_CLASS_ATTR = "bot.http.status_class"


def _no_api_key() -> str:
    return ""


class RequestError(RuntimeError):
    """Raised when an OpenAI-compatible chat completion or Responses API request fails."""


def _empty_headers() -> dict[str, str]:
    return {}


def _empty_body() -> dict[str, object]:
    return {}


class OpenAIChatCompletionsResource(typing.Protocol):
    """OpenAI SDK chat completions resource used by this provider."""

    def create(self, **kwargs: object) -> object:
        """Create a chat completion with SDK-compatible keyword arguments."""
        ...


class OpenAIResponsesResource(typing.Protocol):
    """OpenAI SDK responses resource used by this provider."""

    def create(self, **kwargs: object) -> object:
        """Create a response with SDK-compatible keyword arguments."""
        ...


class OpenAIChatResource(typing.Protocol):
    """OpenAI SDK chat resource used by this provider."""

    completions: OpenAIChatCompletionsResource


class OpenAIClient(typing.Protocol):
    """OpenAI SDK client shape used by this provider."""

    chat: OpenAIChatResource
    responses: OpenAIResponsesResource


class OpenAIEmbeddingsResource(typing.Protocol):
    """OpenAI SDK embeddings resource used by this provider."""

    def create(self, **kwargs: object) -> object:
        """Create embeddings with SDK-compatible keyword arguments."""
        ...


class OpenAIEmbeddingsSDK(typing.Protocol):
    """OpenAI SDK client shape the embeddings client uses."""

    embeddings: OpenAIEmbeddingsResource


class ReasoningEffort(enum.StrEnum):
    """Reasoning effort accepted by OpenAI reasoning models."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ChatCompletionClient(typing.Protocol):
    """Client interface used by stateforward.mosfet OpenAI-compatible text generators."""

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


class ResponsesClient(typing.Protocol):
    """Client interface used by stateforward.mosfet OpenAI Responses API text generators."""

    def create_response(
        self,
        *,
        input: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        tool_choice: str | None = None,
        text_format: collections.abc.Mapping[str, object] | None = None,
        reasoning: collections.abc.Mapping[str, object] | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Create a Responses API response using provider-neutral JSON values."""
        ...


class EmbeddingsClient(typing.Protocol):
    """Client interface used by the OpenAI-compatible ``Embedder``."""

    def create_embeddings(
        self,
        *,
        input: collections.abc.Sequence[object],
        dimensions: int | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Create one embedding per ``input`` entry and return the JSON response object.

        Raises:
            embedding.Error: With the ``FailureKind`` an operator should act on.
        """
        ...


def data_url(image: media.Image) -> str:
    """Encode ``image`` as an RFC 2397 base64 ``data:`` URL, the form OpenAI-compatible APIs accept.

    Shared by the chat and embeddings paths of this package; not part of the exported ``client`` contract.
    """

    return f"data:{image.media_type};base64,{base64.b64encode(image.content).decode('ascii')}"


def _json_object(value: object, message: str) -> dict[str, object]:
    json_value = jsonable(value)
    if not isinstance(json_value, collections.abc.Mapping):
        raise TypeError(message)
    mapping = typing.cast(collections.abc.Mapping[object, object], json_value)
    return {str(key): item for key, item in mapping.items()}


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
        return {
            "name": value.name,
            "data": jsonable(value.data),
            "kind": value.kind,
            "id": value.id or "",
            "source": value.source or "",
            "target": value.target or "",
            "metadata": jsonable(dict(value.metadata)) if value.metadata else {},
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


def _sdk(
    *, base_url: str, api_key: str | None, timeout_seconds: float, default_headers: collections.abc.Mapping[str, str]
) -> object:
    """Build the OpenAI SDK client both clients use; never sends ``OPENAI_API_KEY`` to another endpoint."""

    resolved: str | collections.abc.Callable[[], str] = _no_api_key
    if api_key is not None:
        resolved = api_key
    elif _normalized_base_url(base_url) == _normalized_base_url(_DEFAULT_OPENAI_BASE_URL):
        environment_key = os.environ.get("OPENAI_API_KEY")
        if not environment_key:
            raise ValueError("api_key is required when base_url is the default OpenAI API endpoint.")
        resolved = environment_key
    return OpenAI(
        api_key=resolved,
        base_url=base_url,
        timeout=timeout_seconds,
        default_headers=dict(default_headers) or None,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChatClient(ChatCompletionClient, ResponsesClient):
    """OpenAI SDK-backed client for the Chat Completions API shape and, where served, the Responses API."""

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
        return _json_object(response, "OpenAI-compatible chat completion response must be a JSON object.")

    @typing.override
    def create_response(
        self,
        *,
        input: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        tool_choice: str | None = None,
        text_format: collections.abc.Mapping[str, object] | None = None,
        reasoning: collections.abc.Mapping[str, object] | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        body: dict[str, object] = dict(self.default_body)
        if extra_body is not None:
            body.update(jsonable_mapping(dict(extra_body)))

        kwargs: dict[str, object] = {
            "model": self.model,
            "input": typing.cast(list[object], jsonable(input)),
        }
        if tools:
            kwargs["tools"] = typing.cast(list[object], jsonable(tools))
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if text_format is not None:
            kwargs["text"] = {"format": jsonable_mapping(dict(text_format))}
        if reasoning is not None:
            kwargs["reasoning"] = jsonable_mapping(dict(reasoning))
        if body:
            kwargs["extra_body"] = body

        client = self._client()
        try:
            response = client.responses.create(**kwargs)
        except Exception as error:
            message = "OpenAI Responses API request failed."
            raise RequestError(message) from error
        return _json_object(response, "OpenAI Responses API response must be a JSON object.")

    def _client(self) -> OpenAIClient:
        if self.client is not None:
            return self.client
        if self._sdk_client is not None:
            return self._sdk_client
        sdk_client = typing.cast(
            OpenAIClient,
            _sdk(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
                default_headers=self.default_headers,
            ),
        )
        object.__setattr__(self, "_sdk_client", sdk_client)
        return sdk_client


@dataclasses.dataclass(frozen=True, kw_only=True)
class EmbeddingClient(EmbeddingsClient):
    """OpenAI SDK-backed client for the ``/v1/embeddings`` API shape.

    It has its own connection fields because an embedding model is served separately from a chat
    model. Every request asks for float vectors. Failures raise ``embedding.Error`` with a closed
    kind, and the active span is annotated with ``bot.http.status_class`` (2xx, 4xx, 5xx, or none).
    """

    model: str
    base_url: str = _DEFAULT_OPENAI_BASE_URL
    api_key: str | None = None
    timeout_seconds: float = 60.0
    default_headers: collections.abc.Mapping[str, str] = dataclasses.field(default_factory=_empty_headers)
    default_body: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_body)
    client: OpenAIEmbeddingsSDK | None = None
    _sdk_client: OpenAIEmbeddingsSDK | None = dataclasses.field(default=None, init=False, repr=False, compare=False)

    @typing.override
    def create_embeddings(
        self,
        *,
        input: collections.abc.Sequence[object],
        dimensions: int | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        body: dict[str, object] = dict(self.default_body)
        if extra_body is not None:
            body.update(jsonable_mapping(dict(extra_body)))
        kwargs: dict[str, object] = {
            "model": self.model,
            "input": typing.cast(list[object], jsonable(input)),
            "encoding_format": "float",
        }
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        if body:
            kwargs["extra_body"] = body

        sdk_client = self.client if self.client is not None else self._sdk_client
        if sdk_client is None:
            sdk_client = typing.cast(
                OpenAIEmbeddingsSDK,
                _sdk(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout_seconds=self.timeout_seconds,
                    default_headers=self.default_headers,
                ),
            )
            object.__setattr__(self, "_sdk_client", sdk_client)
        active = trace.get_current_span()
        try:
            response = sdk_client.embeddings.create(**kwargs)
        except Exception as error:
            status = error.status_code if isinstance(error, openai.APIStatusError) else None
            status_class = "none" if status is None or status < 400 else "4xx" if status < 500 else "5xx"
            active.set_attribute(_STATUS_CLASS_ATTR, status_class)
            if isinstance(error, openai.APITimeoutError):
                kind = embedding.FailureKind.TIMEOUT
            elif isinstance(error, openai.APIConnectionError) or (status is not None and status >= 500):
                kind = embedding.FailureKind.UNAVAILABLE
            elif status in (401, 403):
                kind = embedding.FailureKind.UNAUTHORIZED
            elif status == 429:
                kind = embedding.FailureKind.OVERLOADED
            elif status in (400, 413, 422):
                kind = embedding.FailureKind.INVALID_INPUT
            else:
                kind = embedding.FailureKind.EMBED_FAILED
            suffix = "" if status is None else f" (HTTP {status})"
            message = f"OpenAI-compatible embeddings request failed: {type(error).__name__}{suffix}."
            raise embedding.Error(kind, message) from error
        active.set_attribute(_STATUS_CLASS_ATTR, "2xx")
        try:
            return _json_object(response, "OpenAI-compatible embeddings response must be a JSON object.")
        except TypeError as error:
            raise embedding.Error(embedding.FailureKind.INVALID_RESPONSE, str(error)) from error


__all__ = [
    "ChatCompletionClient",
    "OpenAI",
    "ChatClient",
    "EmbeddingClient",
    "EmbeddingsClient",
    "OpenAIEmbeddingsResource",
    "OpenAIEmbeddingsSDK",
    "ReasoningEffort",
    "RequestError",
    "OpenAIClient",
    "ResponsesClient",
    "jsonable",
    "jsonable_mapping",
]
