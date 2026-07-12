from __future__ import annotations

import collections.abc
import dataclasses
import enum
import os
import typing

from google.genai import Client as GenaiClient
import pydantic


_DEFAULT_MODEL = "gemini-3.5-flash"


class RequestError(RuntimeError):
    """Raised when a Gemini SDK request fails."""


def _empty_config() -> dict[str, object]:
    return {}


class GeminiModelsResource(typing.Protocol):
    """google-genai models resource used by this provider."""

    def generate_content(self, **kwargs: object) -> object:
        """Generate content with SDK-compatible keyword arguments."""
        ...


class GeminiInteractionsResource(typing.Protocol):
    """google-genai interactions resource used by this provider."""

    def create(self, **kwargs: object) -> object:
        """Create an interaction with SDK-compatible keyword arguments."""
        ...


class GeminiSdkClient(typing.Protocol):
    """google-genai Client shape used by this provider."""

    models: GeminiModelsResource
    interactions: GeminiInteractionsResource


class ContentClient(typing.Protocol):
    """Client interface used by stateforward.bot Gemini text generators and speech adapters."""

    def generate_content(
        self,
        *,
        contents: collections.abc.Sequence[object],
        model: str | None = None,
        system_instruction: str | None = None,
        tools: collections.abc.Sequence[object] = (),
        tool_config: collections.abc.Mapping[str, object] | None = None,
        response_mime_type: str | None = None,
        response_json_schema: collections.abc.Mapping[str, object] | None = None,
        config: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Generate content via the models.generate_content path."""
        ...

    def create_interaction(
        self,
        *,
        model: str | None = None,
        input: object | None = None,
        response_format: object | None = None,
        generation_config: collections.abc.Mapping[str, object] | None = None,
        system_instruction: str | None = None,
        store: bool = False,
        extra: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Create an interaction via the Interactions API (recommended SDK path)."""
        ...


def _bytes_jsonable(value: bytes | bytearray) -> object:
    """Prefer UTF-8 text for speech/transcript bytes; fall back to base64 for binary."""

    raw = bytes(value)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        import base64

        return base64.b64encode(raw).decode("ascii")


def jsonable(value: object) -> object:
    """Convert Pydantic, enum, dataclass, mapping, and sequence values to JSON-compatible values."""

    if isinstance(value, pydantic.BaseModel):
        return typing.cast(object, value.model_dump(mode="json"))
    if isinstance(value, enum.Enum):
        return typing.cast(object, value.value)
    if isinstance(value, (bytes, bytearray)):
        return _bytes_jsonable(value)
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


def _response_object(response: object, *, kind: str) -> dict[str, object]:
    json_response = jsonable(response)
    if not isinstance(json_response, collections.abc.Mapping):
        raise TypeError(f"Gemini {kind} response must be a JSON object.")
    response_mapping = typing.cast(collections.abc.Mapping[object, object], json_response)
    return {str(key): item for key, item in response_mapping.items()}


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChatClient(ContentClient):
    """google-genai SDK-backed client for Gemini models and interactions."""

    model: str = _DEFAULT_MODEL
    api_key: str | None = None
    vertexai: bool = False
    project: str | None = None
    location: str | None = None
    default_config: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_config)
    client: GeminiSdkClient | None = None
    _sdk_client: GeminiSdkClient | None = dataclasses.field(default=None, init=False, repr=False, compare=False)

    @typing.override
    def generate_content(
        self,
        *,
        contents: collections.abc.Sequence[object],
        model: str | None = None,
        system_instruction: str | None = None,
        tools: collections.abc.Sequence[object] = (),
        tool_config: collections.abc.Mapping[str, object] | None = None,
        response_mime_type: str | None = None,
        response_json_schema: collections.abc.Mapping[str, object] | None = None,
        config: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        request_config: dict[str, object] = dict(self.default_config)
        if config is not None:
            request_config.update(jsonable_mapping(dict(config)))
        if system_instruction is not None:
            request_config["system_instruction"] = system_instruction
        if tools:
            request_config["tools"] = typing.cast(list[object], jsonable(tools))
            # stateforward.bot owns tool execution; never let the SDK auto-invoke Python callables.
            if "automatic_function_calling" not in request_config:
                request_config["automatic_function_calling"] = {"disable": True}
        if tool_config is not None:
            request_config["tool_config"] = jsonable_mapping(dict(tool_config))
        if response_mime_type is not None:
            request_config["response_mime_type"] = response_mime_type
        if response_json_schema is not None:
            request_config["response_json_schema"] = jsonable_mapping(dict(response_json_schema))

        kwargs: dict[str, object] = {
            "model": model if model is not None else self.model,
            "contents": typing.cast(list[object], jsonable(contents)),
        }
        if request_config:
            kwargs["config"] = request_config

        sdk_client = self._client()
        try:
            response = sdk_client.models.generate_content(**kwargs)
        except Exception as error:
            message = "Gemini generate_content request failed."
            raise RequestError(message) from error
        return _response_object(response, kind="generate_content")

    @typing.override
    def create_interaction(
        self,
        *,
        model: str | None = None,
        input: object | None = None,
        response_format: object | None = None,
        generation_config: collections.abc.Mapping[str, object] | None = None,
        system_instruction: str | None = None,
        store: bool = False,
        extra: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Create an interaction using the GA Interactions API.

        Matches ``client.interactions.create(...)`` from the google-genai SDK.
        Defaults ``store=False`` so library calls do not persist server-side state.
        """

        body: dict[str, object] = {
            "model": model if model is not None else self.model,
            "store": store,
        }
        if input is not None:
            body["input"] = jsonable(input)
        if response_format is not None:
            body["response_format"] = jsonable(response_format)
        if generation_config is not None:
            body["generation_config"] = jsonable_mapping(dict(generation_config))
        if system_instruction is not None:
            body["system_instruction"] = system_instruction
        if extra is not None:
            body.update(jsonable_mapping(dict(extra)))

        sdk_client = self._client()
        try:
            response = sdk_client.interactions.create(**body)
        except Exception as error:
            message = "Gemini interactions.create request failed."
            raise RequestError(message) from error
        return _response_object(response, kind="interactions.create")

    def _client(self) -> GeminiSdkClient:
        if self.client is not None:
            return self.client
        if self._sdk_client is not None:
            return self._sdk_client
        if self.vertexai:
            constructed = GenaiClient(
                vertexai=True,
                project=self.project,
                location=self.location,
            )
        else:
            constructed = GenaiClient(api_key=self._api_key())
        sdk_client = typing.cast(GeminiSdkClient, typing.cast(object, constructed))
        object.__setattr__(self, "_sdk_client", sdk_client)
        return sdk_client

    def _api_key(self) -> str:
        if self.api_key is not None:
            return self.api_key
        for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            api_key = os.environ.get(env_name)
            if api_key:
                return api_key
        raise ValueError("api_key is required (set GEMINI_API_KEY or GOOGLE_API_KEY, or pass api_key=).")


# Public alias matching other provider sugar names.
Gemini = ChatClient

__all__ = [
    "ChatClient",
    "ContentClient",
    "Gemini",
    "GeminiInteractionsResource",
    "GeminiModelsResource",
    "GeminiSdkClient",
    "RequestError",
    "jsonable",
    "jsonable_mapping",
]
