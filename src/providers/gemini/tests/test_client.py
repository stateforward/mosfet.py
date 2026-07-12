from __future__ import annotations

import dataclasses

import pytest

import hsm

from bot.providers.gemini import ChatClient
from bot.providers.gemini import client as gemini_client


@dataclasses.dataclass
class GenerateCall:
    kwargs: dict[str, object]


@dataclasses.dataclass
class FakeModels:
    response: object
    calls: list[GenerateCall] = dataclasses.field(default_factory=list)

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(GenerateCall(kwargs=dict(kwargs)))
        return self.response


@dataclasses.dataclass
class FakeInteractions:
    response: object
    calls: list[GenerateCall] = dataclasses.field(default_factory=list)

    def create(self, **kwargs: object) -> object:
        self.calls.append(GenerateCall(kwargs=dict(kwargs)))
        return self.response


@dataclasses.dataclass
class FakeSdkClient:
    models: gemini_client.GeminiModelsResource
    interactions: gemini_client.GeminiInteractionsResource = dataclasses.field(
        default_factory=lambda: FakeInteractions(response={})
    )


def test_jsonable_projects_hsm_event_and_utf8_bytes() -> None:
    event = hsm.Event[bytes](
        name="bot.ability.hearing.speech.decoding.output",
        data=b"Hey I'm Gabe",
        kind=hsm.CallEventKind,
    )
    projected = gemini_client.jsonable(event)
    assert isinstance(projected, dict)
    assert projected["name"] == "bot.ability.hearing.speech.decoding.output"
    assert projected["data"] == "Hey I'm Gabe"
    assert "schema" not in projected


def test_chat_client_uses_injected_sdk_client() -> None:
    models = FakeModels(
        response={
            "model_version": "gemini-3.5-flash",
            "candidates": [{"content": {"parts": [{"text": "done"}]}}],
        }
    )
    client = ChatClient(
        model="gemini-3.5-flash",
        api_key="test-api-key",
        default_config={"temperature": 0.1},
        client=FakeSdkClient(models=models),
    )

    response = client.generate_content(
        contents=[{"role": "user", "parts": [{"text": "hello"}]}],
        system_instruction="Be concise.",
        tools=[{"function_declarations": [{"name": "lookup"}]}],
        tool_config={"function_calling_config": {"mode": "AUTO"}},
        response_mime_type="application/json",
        response_json_schema={"type": "object", "properties": {}},
        config={"max_output_tokens": 64},
    )

    assert response == {
        "model_version": "gemini-3.5-flash",
        "candidates": [{"content": {"parts": [{"text": "done"}]}}],
    }
    assert models.calls == [
        GenerateCall(
            kwargs={
                "model": "gemini-3.5-flash",
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "config": {
                    "temperature": 0.1,
                    "max_output_tokens": 64,
                    "system_instruction": "Be concise.",
                    "tools": [{"function_declarations": [{"name": "lookup"}]}],
                    "automatic_function_calling": {"disable": True},
                    "tool_config": {"function_calling_config": {"mode": "AUTO"}},
                    "response_mime_type": "application/json",
                    "response_json_schema": {"type": "object", "properties": {}},
                },
            }
        )
    ]


def test_chat_client_create_interaction_uses_interactions_api() -> None:
    interactions = FakeInteractions(
        response={
            "output_audio": {"type": "audio", "data": "AAAA"},
            "id": "interaction-1",
        }
    )
    client = ChatClient(
        model="gemini-3.5-flash",
        client=FakeSdkClient(models=FakeModels(response={}), interactions=interactions),
    )

    response = client.create_interaction(
        model="gemini-3.1-flash-tts-preview",
        input="Say hello",
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": "Kore"}]},
        store=False,
    )

    assert response == {
        "output_audio": {"type": "audio", "data": "AAAA"},
        "id": "interaction-1",
    }
    assert interactions.calls == [
        GenerateCall(
            kwargs={
                "model": "gemini-3.1-flash-tts-preview",
                "store": False,
                "input": "Say hello",
                "response_format": {"type": "audio"},
                "generation_config": {"speech_config": [{"voice": "Kore"}]},
            }
        )
    ]


def test_chat_client_builds_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    models = FakeModels(response={"candidates": [{"content": {"parts": [{"text": "done"}]}}]})
    constructed: list[dict[str, object]] = []

    def fake_client(**kwargs: object) -> FakeSdkClient:
        constructed.append(dict(kwargs))
        return FakeSdkClient(models=models)

    monkeypatch.setattr(gemini_client, "GenaiClient", fake_client)
    client = ChatClient(model="gemini-3.5-flash", api_key="from-arg")

    _ = client.generate_content(contents=[{"role": "user", "parts": [{"text": "hello"}]}])
    _ = client.generate_content(contents=[{"role": "user", "parts": [{"text": "again"}]}])

    assert constructed == [{"api_key": "from-arg"}]
    assert len(models.calls) == 2


def test_chat_client_reads_gemini_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    models = FakeModels(response={"candidates": [{"content": {"parts": [{"text": "done"}]}}]})
    constructed: list[dict[str, object]] = []

    def fake_client(**kwargs: object) -> FakeSdkClient:
        constructed.append(dict(kwargs))
        return FakeSdkClient(models=models)

    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(gemini_client, "GenaiClient", fake_client)
    client = ChatClient(model="gemini-3.5-flash", api_key=None)

    _ = client.generate_content(contents=[{"role": "user", "parts": [{"text": "hello"}]}])

    assert constructed == [{"api_key": "env-key"}]


def test_chat_client_vertex_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    models = FakeModels(response={"candidates": [{"content": {"parts": [{"text": "done"}]}}]})
    constructed: list[dict[str, object]] = []

    def fake_client(**kwargs: object) -> FakeSdkClient:
        constructed.append(dict(kwargs))
        return FakeSdkClient(models=models)

    monkeypatch.setattr(gemini_client, "GenaiClient", fake_client)
    client = ChatClient(
        model="gemini-3.5-flash",
        vertexai=True,
        project="demo-project",
        location="us-central1",
    )

    _ = client.generate_content(contents=[{"role": "user", "parts": [{"text": "hello"}]}])

    assert constructed == [
        {
            "vertexai": True,
            "project": "demo-project",
            "location": "us-central1",
        }
    ]


def test_chat_client_rejects_non_object_responses() -> None:
    models = FakeModels(response=["not", "an", "object"])
    client = ChatClient(model="gemini-3.5-flash", client=FakeSdkClient(models=models))

    try:
        _ = client.generate_content(contents=[{"role": "user", "parts": [{"text": "hello"}]}])
    except TypeError as error:
        assert str(error) == "Gemini generate_content response must be a JSON object."
    else:
        raise AssertionError("Expected TypeError.")
