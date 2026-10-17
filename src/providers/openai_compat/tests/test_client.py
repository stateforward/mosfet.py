from __future__ import annotations

import dataclasses

import httpx
import openai
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mosfet.abilities import embedding
from mosfet.providers.openai_compat import ChatClient, EmbeddingClient
from mosfet.providers.openai_compat import client as chat_client
from mosfet.telemetry import span


@pytest.fixture
def finished_spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)
    return exporter


@dataclasses.dataclass
class ChatCompletionCall:
    kwargs: dict[str, object]


@dataclasses.dataclass
class FakeChatCompletions:
    response: object
    calls: list[ChatCompletionCall] = dataclasses.field(default_factory=list)

    def create(self, **kwargs: object) -> object:
        self.calls.append(ChatCompletionCall(kwargs=dict(kwargs)))
        return self.response


@dataclasses.dataclass
class FakeChat:
    completions: chat_client.OpenAIChatCompletionsResource


@dataclasses.dataclass
class FakeOpenAIClient:
    chat: chat_client.OpenAIChatResource
    responses: chat_client.OpenAIResponsesResource = dataclasses.field(
        default_factory=lambda: FakeChatCompletions(response={})
    )


def test_chat_client_uses_injected_sdk_client() -> None:
    completions = FakeChatCompletions(
        response={
            "model": "compatible-model",
            "choices": [{"message": {"content": "done"}}],
        }
    )
    sdk_client = FakeOpenAIClient(chat=FakeChat(completions=completions))
    client = ChatClient(
        model="compatible-model",
        base_url="https://llm.example.test/v1/",
        api_key="test-api-key",
        timeout_seconds=7.5,
        default_headers={"x-provider-scope": "voice"},
        default_body={"store": False},
        client=sdk_client,
    )

    response = client.create_chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        response_format={"type": "json_object"},
        extra_body={"temperature": 0.2},
    )

    assert response == {
        "model": "compatible-model",
        "choices": [{"message": {"content": "done"}}],
    }
    assert completions.calls == [
        ChatCompletionCall(
            kwargs={
                "model": "compatible-model",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "response_format": {"type": "json_object"},
                "extra_body": {"store": False, "temperature": 0.2},
            },
        )
    ]


def test_chat_client_builds_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = FakeChatCompletions(response={"choices": [{"message": {"content": "done"}}]})
    constructed: list[dict[str, object]] = []

    def fake_openai(**kwargs: object) -> FakeOpenAIClient:
        if "_enforce_credentials" in kwargs:
            raise TypeError("OpenAI.__init__() got an unexpected keyword argument '_enforce_credentials'")
        constructed.append(dict(kwargs))
        return FakeOpenAIClient(chat=FakeChat(completions=completions))

    monkeypatch.setattr(chat_client, "OpenAI", fake_openai)
    client = ChatClient(
        model="compatible-model",
        base_url="https://llm.example.test/v1/",
        api_key=None,
        timeout_seconds=7.5,
        default_headers={"x-provider-scope": "voice"},
    )

    _ = client.create_chat_completion(messages=[{"role": "user", "content": "hello"}])
    _ = client.create_chat_completion(messages=[{"role": "user", "content": "again"}])

    assert constructed == [
        {
            "api_key": constructed[0]["api_key"],
            "base_url": "https://llm.example.test/v1/",
            "timeout": 7.5,
            "default_headers": {"x-provider-scope": "voice"},
        }
    ]
    api_key = constructed[0]["api_key"]
    assert callable(api_key)
    assert api_key() == ""
    assert len(completions.calls) == 2


def test_chat_client_does_not_send_openai_key_to_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = FakeChatCompletions(response={"choices": [{"message": {"content": "done"}}]})
    constructed: list[dict[str, object]] = []

    def fake_openai(**kwargs: object) -> FakeOpenAIClient:
        constructed.append(dict(kwargs))
        return FakeOpenAIClient(chat=FakeChat(completions=completions))

    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    monkeypatch.setattr(chat_client, "OpenAI", fake_openai)
    client = ChatClient(
        model="compatible-model",
        base_url="https://llm.example.test/v1/",
        api_key=None,
    )

    _ = client.create_chat_completion(messages=[{"role": "user", "content": "hello"}])

    api_key = constructed[0]["api_key"]
    assert callable(api_key)
    assert api_key() == ""


def test_chat_client_defaults_to_the_loopback_endpoint_and_sends_it_no_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DSL-28e: local first; llama.cpp's server listens on 127.0.0.1:8080 by default."""

    completions = FakeChatCompletions(response={"choices": [{"message": {"content": "done"}}]})
    constructed: list[dict[str, object]] = []

    def fake_openai(**kwargs: object) -> FakeOpenAIClient:
        constructed.append(dict(kwargs))
        return FakeOpenAIClient(chat=FakeChat(completions=completions))

    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    monkeypatch.setattr(chat_client, "OpenAI", fake_openai)
    client = ChatClient(model="local-model")

    _ = client.create_chat_completion(messages=[{"role": "user", "content": "hello"}])

    assert client.base_url == "http://127.0.0.1:8080/v1"
    assert constructed[0]["base_url"] == "http://127.0.0.1:8080/v1"
    api_key = constructed[0]["api_key"]
    assert callable(api_key)
    assert api_key() == ""


def test_chat_client_rejects_non_object_responses() -> None:
    completions = FakeChatCompletions(response=["not", "an", "object"])
    client = ChatClient(
        model="local-model",
        client=FakeOpenAIClient(chat=FakeChat(completions=completions)),
    )

    try:
        _ = client.create_chat_completion(messages=[{"role": "user", "content": "hello"}])
    except TypeError as error:
        assert str(error) == "OpenAI-compatible chat completion response must be a JSON object."
    else:
        raise AssertionError("Expected TypeError.")


def test_chat_client_creates_responses_api_request() -> None:
    responses = FakeChatCompletions(response={"model": "luna", "status": "completed", "output": []})
    client = ChatClient(
        model="luna",
        api_key="test-api-key",
        default_body={"store": False},
        client=FakeOpenAIClient(chat=FakeChat(completions=FakeChatCompletions(response={})), responses=responses),
    )

    response = client.create_response(
        input=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
        tool_choice="required",
        text_format={"type": "json_object"},
        reasoning={"effort": "high"},
        extra_body={"temperature": 0.2},
    )

    assert response == {"model": "luna", "status": "completed", "output": []}
    assert responses.calls == [
        ChatCompletionCall(
            kwargs={
                "model": "luna",
                "input": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
                "tool_choice": "required",
                "text": {"format": {"type": "json_object"}},
                "reasoning": {"effort": "high"},
                "extra_body": {"store": False, "temperature": 0.2},
            }
        )
    ]


def test_chat_client_wraps_responses_api_failure() -> None:
    @dataclasses.dataclass
    class FailingResponses:
        def create(self, **kwargs: object) -> object:
            raise ValueError(f"boom {len(kwargs)}")

    client = ChatClient(
        model="luna",
        api_key="test-api-key",
        client=FakeOpenAIClient(
            chat=FakeChat(completions=FakeChatCompletions(response={})), responses=FailingResponses()
        ),
    )

    with pytest.raises(chat_client.RequestError, match="OpenAI Responses API request failed."):
        _ = client.create_response(input=[{"role": "user", "content": "hello"}])


@dataclasses.dataclass
class FakeEmbeddings:
    response: object
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@dataclasses.dataclass
class FakeEmbeddingsSDK:
    embeddings: chat_client.OpenAIEmbeddingsResource


_EMBEDDINGS_RESPONSE = {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.6, 0.8]}]}


def test_client_exports_only_the_chat_and_approved_embeddings_contract() -> None:
    assert set(chat_client.__all__) == {
        "AsyncChatStream",
        "AsyncOpenAIClient",
        "ChatClient",
        "ChatCompletionClient",
        "EmbeddingClient",
        "EmbeddingsClient",
        "OpenAI",
        "OpenAIClient",
        "OpenAIEmbeddingsResource",
        "OpenAIEmbeddingsSDK",
        "ReasoningEffort",
        "RequestError",
        "ResponsesClient",
        "jsonable",
        "jsonable_mapping",
    }


def test_embedding_client_always_requests_float_vectors() -> None:
    embeddings = FakeEmbeddings(response=_EMBEDDINGS_RESPONSE)
    client = EmbeddingClient(
        model="embeddinggemma-2",
        base_url="https://spark.example.test/v1",
        default_body={"truncate": False},
        client=FakeEmbeddingsSDK(embeddings=embeddings),
    )

    response = client.create_embeddings(input=["a red circle"], extra_body={"priority": 1})

    assert response == _EMBEDDINGS_RESPONSE
    assert embeddings.calls == [
        {
            "model": "embeddinggemma-2",
            "input": ["a red circle"],
            "encoding_format": "float",
            "extra_body": {"truncate": False, "priority": 1},
        }
    ]


def test_embedding_client_sends_optional_dimensions() -> None:
    embeddings = FakeEmbeddings(response=_EMBEDDINGS_RESPONSE)
    client = EmbeddingClient(model="text-embedding-3-small", client=FakeEmbeddingsSDK(embeddings=embeddings))

    _ = client.create_embeddings(input=[{"content": [{"type": "text", "text": "a"}]}], dimensions=256)

    assert embeddings.calls == [
        {
            "model": "text-embedding-3-small",
            "input": [{"content": [{"type": "text", "text": "a"}]}],
            "encoding_format": "float",
            "dimensions": 256,
        }
    ]


def test_embedding_client_rejects_non_object_responses() -> None:
    client = EmbeddingClient(
        model="embeddinggemma-2", client=FakeEmbeddingsSDK(embeddings=FakeEmbeddings(response=["not", "an", "object"]))
    )

    with pytest.raises(embedding.Error) as caught:
        _ = client.create_embeddings(input=["a"])

    assert caught.value.kind is embedding.FailureKind.INVALID_RESPONSE


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://spark.example.test/v1/embeddings")


def _status_error(status: int) -> openai.APIStatusError:
    return openai.APIStatusError("rejected", response=httpx.Response(status, request=_request()), body=None)


@pytest.mark.parametrize(
    ("error", "kind", "status_class"),
    [
        (openai.APITimeoutError(request=_request()), embedding.FailureKind.TIMEOUT, "none"),
        (openai.APIConnectionError(request=_request()), embedding.FailureKind.UNAVAILABLE, "none"),
        (_status_error(401), embedding.FailureKind.UNAUTHORIZED, "4xx"),
        (_status_error(403), embedding.FailureKind.UNAUTHORIZED, "4xx"),
        (_status_error(429), embedding.FailureKind.OVERLOADED, "4xx"),
        (_status_error(400), embedding.FailureKind.INVALID_INPUT, "4xx"),
        (_status_error(413), embedding.FailureKind.INVALID_INPUT, "4xx"),
        (_status_error(422), embedding.FailureKind.INVALID_INPUT, "4xx"),
        (_status_error(500), embedding.FailureKind.UNAVAILABLE, "5xx"),
        (_status_error(503), embedding.FailureKind.UNAVAILABLE, "5xx"),
        (_status_error(404), embedding.FailureKind.EMBED_FAILED, "4xx"),
        (ValueError("unexpected"), embedding.FailureKind.EMBED_FAILED, "none"),
    ],
)
def test_embedding_client_maps_sdk_failures_to_kinds(
    error: Exception,
    kind: embedding.FailureKind,
    status_class: str,
    finished_spans: InMemorySpanExporter,
) -> None:
    client = EmbeddingClient(
        model="embeddinggemma-2", client=FakeEmbeddingsSDK(embeddings=FakeEmbeddings(response=error))
    )

    with span.operation("test.embed", scope="tests", component="test", stage="request"):
        with pytest.raises(embedding.Error) as caught:
            _ = client.create_embeddings(input=["a"])

    assert caught.value.kind is kind
    assert caught.value.__cause__ is error
    [recorded] = finished_spans.get_finished_spans()
    assert dict(recorded.attributes or {})["bot.http.status_class"] == status_class


def test_embedding_client_marks_a_served_request_2xx(finished_spans: InMemorySpanExporter) -> None:
    client = EmbeddingClient(
        model="embeddinggemma-2", client=FakeEmbeddingsSDK(embeddings=FakeEmbeddings(response=_EMBEDDINGS_RESPONSE))
    )

    with span.operation("test.embed", scope="tests", component="test", stage="request"):
        _ = client.create_embeddings(input=["a"])

    [recorded] = finished_spans.get_finished_spans()
    assert dict(recorded.attributes or {})["bot.http.status_class"] == "2xx"


def test_embedding_client_builds_its_sdk_client_once(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings = FakeEmbeddings(response=_EMBEDDINGS_RESPONSE)
    constructed: list[dict[str, object]] = []

    def fake_openai(**kwargs: object) -> FakeEmbeddingsSDK:
        constructed.append(dict(kwargs))
        return FakeEmbeddingsSDK(embeddings=embeddings)

    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    monkeypatch.setattr(chat_client, "OpenAI", fake_openai)
    client = EmbeddingClient(
        model="embeddinggemma-2",
        base_url="https://spark.example.test/v1/",
        default_headers={"x-provider-scope": "memory"},
    )

    _ = client.create_embeddings(input=["a"])
    _ = client.create_embeddings(input=["b"])

    assert constructed == [
        {
            "api_key": constructed[0]["api_key"],
            "base_url": "https://spark.example.test/v1/",
            "timeout": 60.0,
            "default_headers": {"x-provider-scope": "memory"},
        }
    ]
    api_key = constructed[0]["api_key"]
    assert callable(api_key)
    assert api_key() == ""
    assert len(embeddings.calls) == 2


def test_embedding_client_requires_a_key_for_the_default_openai_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake_openai(**kwargs: object) -> FakeEmbeddingsSDK:
        return FakeEmbeddingsSDK(embeddings=FakeEmbeddings(response=dict(kwargs)))

    monkeypatch.setattr(chat_client, "OpenAI", fake_openai)

    with pytest.raises(ValueError, match="api_key is required"):
        _ = EmbeddingClient(model="text-embedding-3-small").create_embeddings(input=["a"])
