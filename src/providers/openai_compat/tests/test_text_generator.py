from __future__ import annotations

import mosfet.media as media
from mosfet.abilities.language import text

import asyncio
import base64
import collections.abc
import dataclasses
import json
import pathlib
import typing

import httpx
import openai
import pytest

from mosfet.providers.openai_compat import TextGenerationError, TextGenerator
import mosfet.providers.openai_compat as openai_compat


@dataclasses.dataclass
class ChatCompletionCall:
    messages: list[dict[str, object]]
    tools: list[object]
    response_format: object | None
    extra_body: dict[str, object]


@dataclasses.dataclass
class FakeChatClient:
    response: dict[str, object]
    calls: list[ChatCompletionCall] = dataclasses.field(default_factory=list)

    async def create_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            ChatCompletionCall(
                messages=[dict(message) for message in messages],
                tools=list(tools),
                response_format=response_format,
                extra_body=dict(extra_body or {}),
            )
        )
        return self.response

    def stream_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.AsyncIterator[dict[str, object]]:
        del messages, tools, response_format, extra_body
        raise AssertionError("this fake serves whole responses only")


def test_text_generator_maps_bot_text_generation_contract() -> None:
    client = FakeChatClient(
        response={
            "model": "compatible-model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "9am is available.",
                        "reasoning": "The tool returned one slot.",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "book_appointment",
                                    "arguments": '{"slot": "9am"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )
    tool_call = text.TextToolCall(id="call_1", name="get_appointments", args={"date": "2026-03-19"})
    generator = TextGenerator(
        client=client,
        provider="litellm",
        response_format={"type": "json_object"},
        extra_body={"temperature": 0.1},
    )

    output = asyncio.run(
        generator.generate(
            text.InputData(
                messages=(
                    text.TextMessage(role=text.TextRole.SYSTEM, content="Be concise."),
                    text.TextMessage(role=text.TextRole.USER, content="Check availability."),
                    text.TextMessage(role=text.TextRole.ASSISTANT, content="", tool_calls=(tool_call,)),
                    text.TextMessage(role=text.TextRole.TOOL, content='{"slots": ["9am"]}', tool_call_id="call_1"),
                ),
                tools=({"type": "function", "function": {"name": "get_appointments"}},),
            )
        )
    )

    assert output == text.OutputData(
        content="9am is available.",
        reasoning="The tool returned one slot.",
        provider="litellm",
        model="compatible-model",
        tool_calls=(text.TextToolCall(id="call_2", name="book_appointment", args={"slot": "9am"}),),
    )
    assert client.calls == [
        ChatCompletionCall(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Check availability."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_appointments",
                                "arguments": '{"date":"2026-03-19"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"slots": ["9am"]}', "tool_call_id": "call_1"},
            ],
            tools=[{"type": "function", "function": {"name": "get_appointments"}}],
            response_format={"type": "json_object"},
            extra_body={"temperature": 0.1, "tool_choice": "auto", "reasoning_effort": "none"},
        )
    ]


def test_text_generator_is_awaitable() -> None:
    client = FakeChatClient(response={"choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]})
    generator = TextGenerator(client=client)

    generated = generator.generate(
        text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),))
    )

    assert isinstance(generated, collections.abc.Coroutine)
    assert asyncio.run(generated) == text.OutputData(content="hello", provider="openai_compat")


def test_text_generator_maps_string_tools_to_openai_function_tools() -> None:
    client = FakeChatClient(response={"choices": [{"finish_reason": "stop", "message": {"content": "available"}}]})
    generator = TextGenerator(client=client)

    output = asyncio.run(
        generator.generate(
            text.InputData(
                messages=(text.TextMessage(role=text.TextRole.USER, content="Check availability."),),
                tools=("get_appointments",),
            )
        )
    )

    assert output == text.OutputData(content="available", provider="openai_compat")
    assert client.calls == [
        ChatCompletionCall(
            messages=[{"role": "user", "content": "Check availability."}],
            tools=[{"type": "function", "function": {"name": "get_appointments"}}],
            response_format=None,
            extra_body={"tool_choice": "auto", "reasoning_effort": "none"},
        )
    ]


def test_text_generator_maps_tool_selection_policy_to_openai_tool_choice() -> None:
    client = FakeChatClient(
        response={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_appointments", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ]
        }
    )
    generator = TextGenerator(client=client)

    _ = asyncio.run(
        generator.generate(
            text.InputData(
                messages=(text.TextMessage(role=text.TextRole.USER, content="Check availability."),),
                tools=("get_appointments",),
                tool_selection=text.ToolSelectionPolicy.REQUIRED,
            )
        )
    )

    assert client.calls[0].extra_body == {"tool_choice": "required", "reasoning_effort": "none"}


def test_text_generator_omits_tool_choice_without_tools() -> None:
    client = FakeChatClient(response={"choices": [{"finish_reason": "stop", "message": {"content": "available"}}]})
    generator = TextGenerator(client=client)

    _ = asyncio.run(
        generator.generate(
            text.InputData(
                messages=(text.TextMessage(role=text.TextRole.USER, content="Check availability."),),
                tool_selection=text.ToolSelectionPolicy.REQUIRED,
            )
        )
    )

    assert client.calls[0].extra_body == {}


def test_text_generator_rejects_missing_required_tool_call() -> None:
    client = FakeChatClient(response={"choices": [{"finish_reason": "stop", "message": {"content": "available"}}]})
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(
                text.InputData(
                    messages=(text.TextMessage(role=text.TextRole.USER, content="Check availability."),),
                    tools=("get_appointments",),
                    tool_selection=text.ToolSelectionPolicy.REQUIRED,
                )
            )
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible chat completion did not include a required tool call."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_tool_calls_when_selection_is_none() -> None:
    client = FakeChatClient(
        response={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "book_appointment", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ]
        }
    )
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(
                text.InputData(
                    messages=(text.TextMessage(role=text.TextRole.USER, content="Check availability."),),
                    tools=("get_appointments",),
                    tool_selection=text.ToolSelectionPolicy.NONE,
                )
            )
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible chat completion included tool calls when tool selection is none."
    else:
        raise AssertionError("Expected TextGenerationError.")
    assert client.calls[0].extra_body == {"tool_choice": "none", "reasoning_effort": "none"}


def test_text_generator_rejects_malformed_tools() -> None:
    client = FakeChatClient(response={"choices": [{"finish_reason": "stop", "message": {"content": "available"}}]})
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(
                text.InputData(
                    messages=(text.TextMessage(role=text.TextRole.USER, content="Check availability."),),
                    tools=(123,),
                )
            )
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible tools must be function tool objects or non-blank tool names."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_malformed_tool_arguments() -> None:
    client = FakeChatClient(
        response={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "book_appointment",
                                    "arguments": "not-json",
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == (
            "OpenAI-compatible tool call arguments must be a JSON object string "
            "(JSONDecodeError: Expecting value: line 1 column 1 (char 0))."
        )
        assert isinstance(error.__cause__, json.JSONDecodeError)
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_decodes_tool_arguments_sent_as_a_json_string_once() -> None:
    def generated(arguments: str) -> text.OutputData:
        client = FakeChatClient(
            response={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "book_appointment", "arguments": arguments},
                                }
                            ],
                        },
                    }
                ]
            }
        )
        message = text.TextMessage(role=text.TextRole.USER, content="hello")
        return asyncio.run(TextGenerator(client=client).generate(text.InputData(messages=(message,))))

    encoded = json.dumps({"slot": "9am"})

    assert generated(json.dumps(encoded)).tool_calls[0].args == {"slot": "9am"}
    with pytest.raises(TextGenerationError, match="must be a JSON object string"):
        _ = generated(json.dumps(json.dumps(encoded)))


def test_text_generator_rejects_mapping_tool_arguments() -> None:
    client = FakeChatClient(
        response={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "book_appointment",
                                    "arguments": {"slot": "9am"},
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible tool call arguments must be a JSON object string."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_tool_calls_without_function_type() -> None:
    client = FakeChatClient(
        response={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "book_appointment",
                                    "arguments": '{"slot": "9am"}',
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible tool calls must use function tools."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_blank_tool_call_identity() -> None:
    client = FakeChatClient(
        response={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": '{"slot": "9am"}',
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible function tool calls must include a non-blank string name."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_blank_tool_call_id() -> None:
    client = FakeChatClient(
        response={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "book_appointment",
                                    "arguments": '{"slot": "9am"}',
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible tool calls must include a non-blank string id."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_legacy_function_call() -> None:
    client = FakeChatClient(
        response={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": None,
                        "function_call": {
                            "name": "book_appointment",
                            "arguments": '{"slot": "9am"}',
                        },
                    },
                }
            ]
        }
    )
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible legacy function_call responses are not supported."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_refusals() -> None:
    client = FakeChatClient(
        response={"choices": [{"finish_reason": "stop", "message": {"content": None, "refusal": "policy refusal"}}]}
    )
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible chat completion refused the request."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_missing_content_without_tool_calls() -> None:
    client = FakeChatClient(response={"choices": [{"finish_reason": "stop", "message": {"content": None}}]})
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible chat completion message did not include string content."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_missing_finish_reason() -> None:
    client = FakeChatClient(response={"choices": [{"message": {"content": "hello"}}]})
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible chat completion included an invalid finish reason."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_tool_finish_without_tool_calls() -> None:
    client = FakeChatClient(response={"choices": [{"finish_reason": "tool_calls", "message": {"content": ""}}]})
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible chat completion finished with no tool calls."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_tool_calls_without_tool_finish_reason() -> None:
    client = FakeChatClient(
        response={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "book_appointment",
                                    "arguments": '{"slot": "9am"}',
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible chat completion included tool calls without a tool_calls finish reason."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_unsuccessful_finish_reason() -> None:
    client = FakeChatClient(response={"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})
    generator = TextGenerator(client=client)

    try:
        _ = asyncio.run(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))
        )
    except TextGenerationError as error:
        assert str(error) == "OpenAI-compatible chat completion finished unsuccessfully: length."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_records_otel_request_when_configured(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live TextGenerator.generate records OTEL; covers generator requests used by processing."""

    import json

    import mosfet.telemetry
    from opentelemetry import _logs

    from mosfet.telemetry.configure import logger_provider

    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    monkeypatch.delenv("BOT_OTEL_DISABLED", raising=False)
    monkeypatch.setenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", "1")
    monkeypatch.setattr(_logs, "set_logger_provider", lambda _provider: None)

    log_path = pathlib.Path("openai-compat-generator.jsonl")
    assert mosfet.telemetry.configure(log_file=log_path) is True

    @dataclasses.dataclass
    class ClientWithModel:
        model: str = "compat-model"
        response: dict[str, object] = dataclasses.field(
            default_factory=lambda: {
                "model": "compat-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            }
        )
        calls: list[ChatCompletionCall] = dataclasses.field(default_factory=list)

        async def create_chat_completion(
            self,
            *,
            messages: collections.abc.Sequence[dict[str, object]],
            tools: collections.abc.Sequence[object] = (),
            response_format: object | None = None,
            extra_body: collections.abc.Mapping[str, object] | None = None,
        ) -> dict[str, object]:
            self.calls.append(
                ChatCompletionCall(
                    messages=[dict(message) for message in messages],
                    tools=list(tools),
                    response_format=response_format,
                    extra_body=dict(extra_body or {}),
                )
            )
            return self.response

        def stream_chat_completion(
            self,
            *,
            messages: collections.abc.Sequence[dict[str, object]],
            tools: collections.abc.Sequence[object] = (),
            response_format: object | None = None,
            extra_body: collections.abc.Mapping[str, object] | None = None,
        ) -> collections.abc.AsyncIterator[dict[str, object]]:
            del messages, tools, response_format, extra_body
            raise AssertionError("this fake serves whole responses only")

    client = ClientWithModel()
    generator = TextGenerator(client=client, provider="openai_compat")
    _ = asyncio.run(
        generator.generate(
            text.InputData(
                messages=(
                    text.TextMessage(role=text.TextRole.SYSTEM, content="Be brief."),
                    text.TextMessage(role=text.TextRole.USER, content="Alice greets Bob"),
                ),
            )
        )
    )

    otel_provider = logger_provider()
    assert otel_provider is not None
    _ = otel_provider.force_flush()

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [json.loads(line)["attributes"]["stage"] for line in lines] == ["request", "response"]
    response_record = json.loads(lines[1])
    assert response_record["body"]["response"]["choices"]
    assert response_record["body"]["error"] is None
    assert isinstance(response_record["body"]["latency_ms"], float)
    payload = json.loads(lines[0])
    assert payload["body"]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Alice greets Bob"},
    ]
    assert payload["attributes"]["provider"] == "openai_compat"
    assert payload["attributes"]["model"] == "compat-model"
    assert payload["attributes"]["stage"] == "request"
    assert "Alice greets Bob" not in json.dumps(payload["attributes"])


_TOOL = {
    "type": "function",
    "function": {"name": "dispatch", "parameters": {"type": "object"}, "strict": True},
}


def _tool_input(selection: text.ToolSelectionPolicy = text.ToolSelectionPolicy.REQUIRED) -> text.InputData:
    return text.InputData(
        messages=(
            text.TextMessage(role=text.TextRole.SYSTEM, content="be brief"),
            text.TextMessage(role=text.TextRole.USER, content="hi"),
        ),
        tools=(_TOOL,),
        tool_selection=selection,
    )


def test_chat_text_generator_rejects_reasoning_effort_with_tools() -> None:
    client = FakeChatClient(response={})
    generator = TextGenerator(client=client, reasoning_effort=openai_compat.ReasoningEffort.HIGH)

    with pytest.raises(openai_compat.UnsupportedReasoningEffortError, match="reasoning_effort='high'"):
        _ = asyncio.run(generator.generate(_tool_input()))
    assert client.calls == []


def test_chat_text_generator_sends_requested_reasoning_effort_without_tools() -> None:
    client = FakeChatClient(
        response={"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
    )
    generator = TextGenerator(client=client, reasoning_effort=openai_compat.ReasoningEffort.LOW)

    output = asyncio.run(
        generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hi"),)))
    )

    assert output.content == "ok"
    assert client.calls[0].extra_body == {"reasoning_effort": "low"}


@dataclasses.dataclass
class ResponsesCall:
    input: list[dict[str, object]]
    tools: list[object]
    tool_choice: str | None
    text_format: collections.abc.Mapping[str, object] | None
    reasoning: collections.abc.Mapping[str, object] | None
    extra_body: dict[str, object]


@dataclasses.dataclass
class FakeResponsesClient:
    response: dict[str, object]
    model: str = "gpt-luna"
    calls: list[ResponsesCall] = dataclasses.field(default_factory=list)

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
        self.calls.append(
            ResponsesCall(
                input=[dict(item) for item in input],
                tools=list(tools),
                tool_choice=tool_choice,
                text_format=text_format,
                reasoning=reasoning,
                extra_body=dict(extra_body or {}),
            )
        )
        return self.response


def test_responses_text_generator_maps_reasoning_and_function_calls() -> None:
    client = FakeResponsesClient(
        response={
            "model": "gpt-luna-2026",
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Pick dispatch."}]},
                {"type": "function_call", "call_id": "call_9", "name": "dispatch", "arguments": '{"events": []}'},
            ],
            "usage": {"input_tokens": 3, "output_tokens": 9, "output_tokens_details": {"reasoning_tokens": 8}},
        }
    )
    generator = openai_compat.ResponsesTextGenerator(
        client=client,
        provider="openai_luna",
        reasoning_effort=openai_compat.ReasoningEffort.HIGH,
    )

    output = asyncio.run(generator.generate(_tool_input()))

    assert output == text.OutputData(
        content="",
        reasoning="Pick dispatch.",
        provider="openai_luna",
        model="gpt-luna-2026",
        tool_calls=(text.TextToolCall(id="call_9", name="dispatch", args={"events": []}),),
    )
    assert client.calls == [
        ResponsesCall(
            input=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
            tools=[{"type": "function", "name": "dispatch", "parameters": {"type": "object"}, "strict": True}],
            tool_choice="required",
            text_format=None,
            reasoning={"effort": "high"},
            extra_body={},
        )
    ]


def test_responses_text_generator_maps_tool_history_and_text_output() -> None:
    client = FakeResponsesClient(
        response={
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "9am works."}]}],
        }
    )
    generator = openai_compat.ResponsesTextGenerator(
        client=client,
        response_format={"type": "json_schema", "json_schema": {"name": "reply", "schema": {"type": "object"}}},
    )
    tool_call = text.TextToolCall(id="call_1", name="slots", args={"day": "mon"})

    output = asyncio.run(
        generator.generate(
            text.InputData(
                messages=(
                    text.TextMessage(role=text.TextRole.ASSISTANT, content="", tool_calls=(tool_call,)),
                    text.TextMessage(role=text.TextRole.TOOL, content="9am", tool_call_id="call_1"),
                )
            )
        )
    )

    assert output.content == "9am works."
    assert client.calls[0].input == [
        {"type": "function_call", "call_id": "call_1", "name": "slots", "arguments": '{"day":"mon"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "9am"},
    ]
    assert client.calls[0].tool_choice is None
    assert client.calls[0].reasoning is None
    assert client.calls[0].text_format == {"type": "json_schema", "name": "reply", "schema": {"type": "object"}}


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, "finished unsuccessfully"),
        ({"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]}, "refused"),
        ({"status": "completed", "output": []}, "did not include a required tool call"),
    ],
)
def test_responses_text_generator_rejects_unusable_responses(response: dict[str, object], message: str) -> None:
    generator = openai_compat.ResponsesTextGenerator(client=FakeResponsesClient(response=response))

    with pytest.raises(TextGenerationError, match=message):
        _ = asyncio.run(generator.generate(_tool_input()))


@dataclasses.dataclass
class _ModelChatClient(FakeChatClient):
    model: str = "compat-model"


@pytest.mark.parametrize("path", ["chat", "responses"])
def test_generator_records_join_the_callers_trace(
    path: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chat awaits its request and Responses hops to a worker thread; both records join the calling span."""

    import json

    import mosfet.telemetry
    from opentelemetry import _logs

    from mosfet.telemetry.configure import logger_provider

    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    monkeypatch.delenv("BOT_OTEL_DISABLED", raising=False)
    monkeypatch.setenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", "1")
    monkeypatch.setattr(_logs, "set_logger_provider", lambda _provider: None)
    log_path = pathlib.Path("generator-trace.jsonl")
    assert mosfet.telemetry.configure(log_file=log_path) is True

    generator: text.TextGenerator
    if path == "chat":
        generator = TextGenerator(
            client=_ModelChatClient(
                response={"model": "compat-model", "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
            ),
            provider="openai_compat",
        )
    else:
        generator = openai_compat.ResponsesTextGenerator(
            client=FakeResponsesClient(
                response={
                    "model": "gpt-luna",
                    "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                }
            ),
            provider="openai_luna",
        )

    async def run() -> tuple[str, str]:
        with mosfet.telemetry.span.operation("test.turn", scope="tests", component="test", stage="turn") as active:
            _ = await generator.generate(
                text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hi"),))
            )
            context = active.get_span_context()
            return format(context.trace_id, "032x"), format(context.span_id, "016x")

    trace_id, span_id = asyncio.run(run())
    otel_provider = logger_provider()
    assert otel_provider is not None
    _ = otel_provider.force_flush()
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records
    assert {(record["trace_id"], record["span_id"]) for record in records} == {(trace_id, span_id)}


_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
_PNG = media.Image(content=_PNG_BYTES, media_type="image/png")
_PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode("ascii")
_VISION = media.ImageCapacity(max_images=1, max_total_bytes=len(_PNG_BYTES), media_types=frozenset({"image/png"}))
_OK_CHAT: dict[str, object] = {"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": "a cat"}}]}
_OK_RESPONSES: dict[str, object] = {
    "status": "completed",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "a cat"}]}],
}


def _image_input(*images: media.Image) -> text.InputData:
    parts: tuple[text.ContentPart, ...] = (text.TextPart(text="What is this?"),)
    parts += tuple(text.ImagePart(image=image) for image in images)
    return text.InputData(
        messages=(
            text.TextMessage(role=text.TextRole.SYSTEM, content="Be brief."),
            text.TextMessage(role=text.TextRole.USER, content=parts),
        )
    )


def test_chat_text_generator_sends_image_parts_as_standard_base64_data_urls() -> None:
    client = FakeChatClient(response=_OK_CHAT)
    generator = TextGenerator(client=client, image_capacity=_VISION)

    output = asyncio.run(generator.generate(_image_input(_PNG)))

    assert output.content == "a cat"
    assert "+" in _PNG_DATA_URL and "/" in _PNG_DATA_URL.removeprefix("data:image/png;base64,")
    assert client.calls[0].messages == [
        {"role": "system", "content": "Be brief."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": _PNG_DATA_URL}},
            ],
        },
    ]


def test_responses_text_generator_sends_input_text_and_input_image_parts() -> None:
    client = FakeResponsesClient(response=_OK_RESPONSES)
    generator = openai_compat.ResponsesTextGenerator(client=client, image_capacity=_VISION)

    output = asyncio.run(generator.generate(_image_input(_PNG)))

    assert output.content == "a cat"
    assert client.calls[0].input == [
        {"role": "system", "content": "Be brief."},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is this?"},
                {"type": "input_image", "image_url": _PNG_DATA_URL},
            ],
        },
    ]


@pytest.mark.parametrize("path", ["chat", "responses"])
def test_generators_reject_requests_over_their_image_capacity(path: str) -> None:
    chat = FakeChatClient(response=_OK_CHAT)
    responses = FakeResponsesClient(response=_OK_RESPONSES)
    generator: text.TextGenerator = (
        TextGenerator(client=chat, image_capacity=_VISION)
        if path == "chat"
        else openai_compat.ResponsesTextGenerator(client=responses, image_capacity=_VISION)
    )

    with pytest.raises(TextGenerationError, match="exceeds the generator's image capacity"):
        _ = asyncio.run(generator.generate(_image_input(_PNG, _PNG)))
    assert chat.calls == []
    assert responses.calls == []


def test_text_only_default_capacity_rejects_images() -> None:
    client = FakeChatClient(response=_OK_CHAT)
    generator = TextGenerator(client=client)

    assert generator.image_capacity == media.ImageCapacity()
    with pytest.raises(TextGenerationError, match="exceeds the generator's image capacity"):
        _ = asyncio.run(generator.generate(_image_input(_PNG)))
    assert client.calls == []


def test_chat_text_generator_records_image_count_without_payload_capture(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import mosfet.telemetry
    from opentelemetry import _logs

    from mosfet.telemetry.configure import logger_provider

    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    monkeypatch.delenv("BOT_OTEL_DISABLED", raising=False)
    monkeypatch.delenv("BOT_OTEL_CAPTURE", raising=False)
    monkeypatch.delenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", raising=False)

    def keep_global_logger_provider(_provider: object) -> None:
        return None

    monkeypatch.setattr(_logs, "set_logger_provider", keep_global_logger_provider)
    log_path = pathlib.Path("image-count.jsonl")
    assert mosfet.telemetry.configure(log_file=log_path) is True

    _ = asyncio.run(
        TextGenerator(client=_ModelChatClient(response=_OK_CHAT), image_capacity=_VISION).generate(_image_input(_PNG))
    )

    otel_provider = logger_provider()
    assert otel_provider is not None
    _ = otel_provider.force_flush()
    records: list[dict[str, object]] = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    (request,) = records
    assert request["body"] == {"capture": "disabled", "message_count": 2, "tool_count": 0, "image_count": 1}
    mosfet.telemetry.reset()


def _sse(*chunks: object) -> bytes:
    """A Chat Completions stream as served: one ``data:`` event per chunk, then ``[DONE]``."""

    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _chunk(delta: dict[str, object], finish_reason: str | None = None) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "streamer",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


_STREAMED_ARGUMENTS = ('{"events": [{"event": "say", ', '"data": {"text": "Hi ', 'there."}}]}')


def _streaming_client(requests: list[httpx.Request]) -> openai_compat.ChatClient:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = _sse(
            _chunk({"role": "assistant", "content": None, "reasoning_content": "Greet them."}),
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "dispatch", "arguments": ""},
                        }
                    ]
                }
            ),
            *(
                _chunk({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]})
                for piece in _STREAMED_ARGUMENTS
            ),
            _chunk({}, "tool_calls"),
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "streamer",
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
            },
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    sdk = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="http://compat.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    # The SDK's create takes its own keywords; the protocol is the narrow shape this client calls.
    return openai_compat.ChatClient(
        model="streamer",
        base_url="http://compat.test/v1",
        async_client=typing.cast(openai_compat.AsyncOpenAIClient, typing.cast(object, sdk)),
    )


def test_text_generator_streams_deltas_over_sse_and_returns_the_same_whole_output() -> None:
    requests: list[httpx.Request] = []
    deltas: list[text.Delta] = []
    generator = TextGenerator(client=_streaming_client(requests))
    tool = {"type": "function", "function": {"name": "dispatch", "parameters": {"type": "object"}}}

    output = asyncio.run(
        generator.generate(
            text.InputData(
                messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),),
                tools=(tool,),
                tool_selection=text.ToolSelectionPolicy.REQUIRED,
            ),
            deltas=deltas.append,
        )
    )

    assert [request.url.path for request in requests] == ["/v1/chat/completions"]
    sent = json.loads(requests[0].content)
    assert sent["stream"] is True
    assert sent["model"] == "streamer"
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["tool_choice"] == "required"
    assert [delta.arguments for delta in deltas if delta.tool_call == 0 and delta.arguments] == list(
        _STREAMED_ARGUMENTS
    )
    assert output == text.OutputData(
        content="",
        reasoning="Greet them.",
        provider="openai_compat",
        model="streamer",
        tool_calls=(
            text.TextToolCall(
                id="call_1",
                name="dispatch",
                args={"events": [{"event": "say", "data": {"text": "Hi there."}}]},
            ),
        ),
    )


def test_text_generator_without_deltas_keeps_the_whole_response_path() -> None:
    requests: list[httpx.Request] = []
    client = FakeChatClient(
        response={"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
    )
    generator = TextGenerator(client=client)

    output = asyncio.run(
        generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hi"),)))
    )

    assert output.content == "ok"
    assert len(client.calls) == 1
    assert requests == []


@dataclasses.dataclass
class _HangingChatClient:
    """A whole-response client whose create never returns, so a test can cancel the call while it is in flight."""

    started: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

    async def create_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        del messages, tools, response_format, extra_body
        self.started.set()
        try:
            _ = await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("the call never returns")

    def stream_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.AsyncIterator[dict[str, object]]:
        del messages, tools, response_format, extra_body
        raise AssertionError("this fake serves whole responses only")


def test_text_generator_cancelled_without_deltas_cancels_the_in_flight_call() -> None:
    # Live (builder check round 4, F1): a reflection call cancelled at its step bound ran on in a worker
    # thread for 344 s and generated 4512 tokens nobody read, on the shared model server.
    async def run() -> None:
        client = _HangingChatClient()
        generator = TextGenerator(client=client)
        call = asyncio.create_task(
            generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hi"),)))
        )
        _ = await asyncio.wait_for(client.started.wait(), 1.0)

        _ = call.cancel()

        _ = await asyncio.wait_for(client.cancelled.wait(), 1.0)
        with pytest.raises(asyncio.CancelledError):
            _ = await call

    asyncio.run(run())
