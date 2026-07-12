from __future__ import annotations

from bot.abilities.language import text

import asyncio
import collections.abc
import dataclasses

from bot.providers.openai_compat import TextGenerationError, TextGenerator

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

    def create_chat_completion(
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
            extra_body={"temperature": 0.1, "tool_choice": "auto"},
        )
    ]

def test_text_generator_is_awaitable() -> None:
    client = FakeChatClient(response={"choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]})
    generator = TextGenerator(client=client)

    generated = generator.generate(text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)))

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
            extra_body={"tool_choice": "auto"},
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

    assert client.calls[0].extra_body == {"tool_choice": "required"}

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
    assert client.calls[0].extra_body == {"tool_choice": "none"}

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
        assert str(error) == "OpenAI-compatible tool call arguments must be a JSON object string."
    else:
        raise AssertionError("Expected TextGenerationError.")

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
