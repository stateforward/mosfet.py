from __future__ import annotations

import asyncio
import dataclasses
import inspect
import typing

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

    async def create(self, **kwargs: object) -> object:
        self.calls.append(ChatCompletionCall(kwargs=dict(kwargs)))
        return self.response


@dataclasses.dataclass
class FakeResponses:
    response: object
    calls: list[ChatCompletionCall] = dataclasses.field(default_factory=list)

    def create(self, **kwargs: object) -> object:
        self.calls.append(ChatCompletionCall(kwargs=dict(kwargs)))
        return self.response


@dataclasses.dataclass
class FakeAsyncChat:
    completions: chat_client.AsyncOpenAIChatCompletionsResource


@dataclasses.dataclass
class FakeAsyncOpenAIClient:
    chat: FakeAsyncChat


def _async_sdk(completions: chat_client.AsyncOpenAIChatCompletionsResource) -> chat_client.AsyncOpenAIClient:
    # The fake serves chat completions only; the protocol also names the audio resources this client never calls.
    return typing.cast(
        chat_client.AsyncOpenAIClient, typing.cast(object, FakeAsyncOpenAIClient(chat=FakeAsyncChat(completions)))
    )


@dataclasses.dataclass
class FakeOpenAIClient:
    responses: chat_client.OpenAIResponsesResource


def test_chat_client_uses_injected_sdk_client() -> None:
    completions = FakeChatCompletions(
        response={
            "model": "compatible-model",
            "choices": [{"message": {"content": "done"}}],
        }
    )
    client = ChatClient(
        model="compatible-model",
        base_url="https://llm.example.test/v1/",
        api_key="test-api-key",
        timeout_seconds=7.5,
        default_headers={"x-provider-scope": "voice"},
        default_body={"store": False},
        async_client=_async_sdk(completions),
    )

    response = asyncio.run(
        client.create_chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            response_format={"type": "json_object"},
            extra_body={"temperature": 0.2},
        )
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

    def fake_async_openai(**kwargs: object) -> FakeAsyncOpenAIClient:
        if "_enforce_credentials" in kwargs:
            raise TypeError("AsyncOpenAI.__init__() got an unexpected keyword argument '_enforce_credentials'")
        constructed.append(dict(kwargs))
        return FakeAsyncOpenAIClient(chat=FakeAsyncChat(completions))

    monkeypatch.setattr(chat_client, "KeylessAsyncOpenAI", fake_async_openai)
    client = ChatClient(
        model="compatible-model",
        base_url="https://llm.example.test/v1/",
        api_key=None,
        timeout_seconds=7.5,
        default_headers={"x-provider-scope": "voice"},
    )

    _ = asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hello"}]))
    _ = asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "again"}]))

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
    key = api_key()
    assert inspect.iscoroutine(key)
    assert asyncio.run(key) == ""
    assert len(completions.calls) == 2


def test_chat_client_does_not_send_openai_key_to_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = FakeChatCompletions(response={"choices": [{"message": {"content": "done"}}]})
    constructed: list[dict[str, object]] = []

    def fake_async_openai(**kwargs: object) -> FakeAsyncOpenAIClient:
        constructed.append(dict(kwargs))
        return FakeAsyncOpenAIClient(chat=FakeAsyncChat(completions))

    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    monkeypatch.setattr(chat_client, "KeylessAsyncOpenAI", fake_async_openai)
    client = ChatClient(
        model="compatible-model",
        base_url="https://llm.example.test/v1/",
        api_key=None,
    )

    _ = asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hello"}]))

    api_key = constructed[0]["api_key"]
    assert callable(api_key)
    key = api_key()
    assert inspect.iscoroutine(key)
    assert asyncio.run(key) == ""


def test_chat_client_defaults_to_the_loopback_endpoint_and_sends_it_no_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DSL-28e: local first; llama.cpp's server listens on 127.0.0.1:8080 by default."""

    completions = FakeChatCompletions(response={"choices": [{"message": {"content": "done"}}]})
    constructed: list[dict[str, object]] = []

    def fake_async_openai(**kwargs: object) -> FakeAsyncOpenAIClient:
        constructed.append(dict(kwargs))
        return FakeAsyncOpenAIClient(chat=FakeAsyncChat(completions))

    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    monkeypatch.setattr(chat_client, "KeylessAsyncOpenAI", fake_async_openai)
    client = ChatClient(model="local-model")

    _ = asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hello"}]))

    assert client.base_url == "http://127.0.0.1:8080/v1"
    assert constructed[0]["base_url"] == "http://127.0.0.1:8080/v1"
    api_key = constructed[0]["api_key"]
    assert callable(api_key)
    key = api_key()
    assert inspect.iscoroutine(key)
    assert asyncio.run(key) == ""


def test_chat_client_rejects_non_object_responses() -> None:
    completions = FakeChatCompletions(response=["not", "an", "object"])
    client = ChatClient(
        model="local-model",
        async_client=_async_sdk(completions),
    )

    try:
        _ = asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hello"}]))
    except TypeError as error:
        assert str(error) == "OpenAI-compatible chat completion response must be a JSON object."
    else:
        raise AssertionError("Expected TypeError.")


def test_chat_client_creates_responses_api_request() -> None:
    responses = FakeResponses(response={"model": "luna", "status": "completed", "output": []})
    client = ChatClient(
        model="luna",
        api_key="test-api-key",
        default_body={"store": False},
        client=FakeOpenAIClient(responses=responses),
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
        client=FakeOpenAIClient(responses=FailingResponses()),
    )

    with pytest.raises(chat_client.RequestError, match=r"OpenAI Responses API request failed \(ValueError\)\."):
        _ = client.create_response(input=[{"role": "user", "content": "hello"}])


_SECRET_BODY = "upstream said: prompt was 'call me at +15555550100'"


def _status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")
    response = httpx.Response(status, request=request, text=_SECRET_BODY)
    return openai.APIStatusError(_SECRET_BODY, response=response, body={"error": _SECRET_BODY})


@dataclasses.dataclass
class FailingCompletions:
    error: Exception

    async def create(self, **kwargs: object) -> object:
        del kwargs
        raise self.error


def test_a_failed_chat_completion_names_the_sdk_error_type_and_http_status_never_the_body() -> None:
    # Live (builder check round 3 on the cpa proxy, G3): 12 of 466 calls failed with only "RequestError:
    # OpenAI-compatible chat completion request failed.", so a 429, a 5xx and a timeout looked the same.
    client = ChatClient(
        model="luna",
        api_key="test-api-key",
        async_client=_async_sdk(FailingCompletions(_status_error(503))),
    )

    with pytest.raises(chat_client.RequestError) as raised:
        _ = asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hello"}]))

    assert str(raised.value) == "OpenAI-compatible chat completion request failed (APIStatusError, HTTP 503)."
    assert isinstance(raised.value.__cause__, openai.APIStatusError)


def test_a_failed_request_without_a_status_names_the_sdk_error_type_alone() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")
    client = ChatClient(
        model="luna",
        api_key="test-api-key",
        async_client=_async_sdk(FailingCompletions(openai.APITimeoutError(request))),
    )

    with pytest.raises(chat_client.RequestError) as raised:
        _ = asyncio.run(client.create_chat_completion(messages=[{"role": "user", "content": "hello"}]))

    assert str(raised.value) == "OpenAI-compatible chat completion request failed (APITimeoutError)."


def test_a_failed_streamed_chat_completion_names_the_sdk_error_type_and_http_status_never_the_body() -> None:
    client = ChatClient(
        model="luna",
        api_key="test-api-key",
        async_client=_async_sdk(FailingCompletions(_status_error(429))),
    )

    async def run() -> None:
        async for _ in client.stream_chat_completion(messages=[{"role": "user", "content": "hello"}]):
            pass

    with pytest.raises(chat_client.RequestError) as raised:
        asyncio.run(run())

    message = str(raised.value)
    assert message == "OpenAI-compatible streamed chat completion request failed (APIStatusError, HTTP 429)."
    assert _SECRET_BODY not in message


def test_a_cancelled_chat_completion_cancels_its_http_request() -> None:
    # Live (builder check round 4, F1): a cancelled call's request ran on to completion in a worker thread.
    async def run() -> None:
        sent = asyncio.Event()
        cancelled = asyncio.Event()

        async def handle(request: httpx.Request) -> httpx.Response:
            del request
            sent.set()
            try:
                _ = await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("the server never answers")

        sdk = openai.AsyncOpenAI(
            api_key="test-api-key",
            base_url="http://127.0.0.1:9/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )
        client = ChatClient(
            model="luna",
            base_url="http://127.0.0.1:9/v1",
            api_key="test-api-key",
            async_client=typing.cast(chat_client.AsyncOpenAIClient, typing.cast(object, sdk)),
        )
        call = asyncio.create_task(client.create_chat_completion(messages=[{"role": "user", "content": "hello"}]))
        _ = await asyncio.wait_for(sent.wait(), 1.0)

        _ = call.cancel()

        _ = await asyncio.wait_for(cancelled.wait(), 1.0)
        with pytest.raises(asyncio.CancelledError):
            _ = await call

    asyncio.run(run())


def test_a_keyless_sdk_client_asks_without_an_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Companion live check (D-W8-21 D3, run3): on openai 2.54 a keyless client raised "Could not resolve
    authentication method" before sending anything, so the keyless speech servers were never asked. A client
    built without a key sends its requests, whole and streamed, sync and async, with no Authorization header."""

    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    seen: list[httpx.Headers] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json=_EMBEDDINGS_RESPONSE)

    async def ask() -> None:
        built = chat_client.sdk(
            base_url="http://127.0.0.1:9/v1", api_key=None, timeout_seconds=5.0, default_headers={}, asynchronous=True
        )
        assert isinstance(built, openai.AsyncOpenAI)
        sdk = built.with_options(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        _ = await sdk.embeddings.create(model="local", input=["a"])

    built = chat_client.sdk(base_url="http://127.0.0.1:9/v1", api_key=None, timeout_seconds=5.0, default_headers={})
    assert isinstance(built, openai.OpenAI)
    sdk = built.with_options(http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    _ = sdk.embeddings.create(model="local", input=["a"])
    asyncio.run(ask())

    assert len(seen) == 2
    assert all("authorization" not in headers for headers in seen)


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
    monkeypatch.setattr(chat_client, "KeylessOpenAI", fake_openai)
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
