from __future__ import annotations

import dataclasses

import pytest

from mosfet.providers.openai_compat import ChatClient
from mosfet.providers.openai_compat import client as chat_client


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
