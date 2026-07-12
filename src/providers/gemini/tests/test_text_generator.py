from __future__ import annotations

from bot.abilities.language import text

import asyncio
import collections.abc
import dataclasses

from bot.providers.gemini import TextGenerationError, TextGenerator


@dataclasses.dataclass
class GenerateCall:
    contents: list[object]
    system_instruction: str | None
    tools: list[object]
    tool_config: dict[str, object] | None
    response_mime_type: str | None
    response_json_schema: dict[str, object] | None
    config: dict[str, object]


@dataclasses.dataclass
class FakeContentClient:
    response: dict[str, object]
    calls: list[GenerateCall] = dataclasses.field(default_factory=list)

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
        del model
        self.calls.append(
            GenerateCall(
                contents=list(contents),
                system_instruction=system_instruction,
                tools=list(tools),
                tool_config=dict(tool_config) if tool_config is not None else None,
                response_mime_type=response_mime_type,
                response_json_schema=dict(response_json_schema) if response_json_schema is not None else None,
                config=dict(config or {}),
            )
        )
        return self.response

    def create_interaction(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        raise AssertionError("TextGenerator uses generate_content, not create_interaction.")


def test_text_generator_maps_bot_text_generation_contract() -> None:
    client = FakeContentClient(
        response={
            "model_version": "gemini-3.5-flash",
            "candidates": [
                {
                    "finish_reason": "STOP",
                    "content": {
                        "parts": [
                            {"text": "9am is available."},
                            {
                                "function_call": {
                                    "id": "call_2",
                                    "name": "book_appointment",
                                    "args": {"slot": "9am"},
                                }
                            },
                        ]
                    },
                }
            ],
        }
    )
    tool_call = text.TextToolCall(id="call_1", name="get_appointments", args={"date": "2026-03-19"})
    generator = TextGenerator(
        client=client,
        provider="gemini",
        response_mime_type="application/json",
        response_json_schema={"type": "object"},
        config={"temperature": 0.1},
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
        reasoning="",
        provider="gemini",
        model="gemini-3.5-flash",
        tool_calls=(text.TextToolCall(id="call_2", name="book_appointment", args={"slot": "9am"}),),
    )
    assert client.calls == [
        GenerateCall(
            contents=[
                {"role": "user", "parts": [{"text": "Check availability."}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "name": "get_appointments",
                                "args": {"date": "2026-03-19"},
                                "id": "call_1",
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": "call_1",
                                "response": {"slots": ["9am"]},
                            }
                        }
                    ],
                },
            ],
            system_instruction="Be concise.",
            tools=[
                {
                    "function_declarations": [
                        {
                            "name": "get_appointments",
                        }
                    ]
                }
            ],
            tool_config={"function_calling_config": {"mode": "AUTO"}},
            response_mime_type="application/json",
            response_json_schema={"type": "object"},
            config={"temperature": 0.1},
        )
    ]


def test_text_generator_is_awaitable() -> None:
    client = FakeContentClient(
        response={"candidates": [{"finish_reason": "STOP", "content": {"parts": [{"text": "hello"}]}}]}
    )
    generator = TextGenerator(client=client)

    generated = generator.generate(
        text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),))
    )

    assert isinstance(generated, collections.abc.Coroutine)
    assert asyncio.run(generated) == text.OutputData(content="hello", provider="gemini")


def test_text_generator_maps_string_tools() -> None:
    client = FakeContentClient(
        response={"candidates": [{"finish_reason": "STOP", "content": {"parts": [{"text": "available"}]}}]}
    )
    generator = TextGenerator(client=client)

    output = asyncio.run(
        generator.generate(
            text.InputData(
                messages=(text.TextMessage(role=text.TextRole.USER, content="Check availability."),),
                tools=("get_appointments",),
            )
        )
    )

    assert output == text.OutputData(content="available", provider="gemini")
    assert client.calls[0].tools == [
        {
            "function_declarations": [
                {
                    "name": "get_appointments",
                    "parameters_json_schema": {"type": "object", "properties": {}},
                }
            ]
        }
    ]
    assert client.calls[0].tool_config == {"function_calling_config": {"mode": "AUTO"}}


def test_text_generator_maps_required_tool_selection() -> None:
    client = FakeContentClient(
        response={
            "candidates": [
                {
                    "finish_reason": "STOP",
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "id": "call_1",
                                    "name": "get_appointments",
                                    "args": {},
                                }
                            }
                        ]
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

    assert client.calls[0].tool_config == {"function_calling_config": {"mode": "ANY"}}


def test_text_generator_rejects_missing_required_tool_call() -> None:
    client = FakeContentClient(
        response={"candidates": [{"finish_reason": "STOP", "content": {"parts": [{"text": "available"}]}}]}
    )
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
        assert str(error) == "Gemini generate_content did not include a required tool call."
    else:
        raise AssertionError("Expected TextGenerationError.")


def test_text_generator_rejects_tool_calls_when_selection_is_none() -> None:
    client = FakeContentClient(
        response={
            "candidates": [
                {
                    "finish_reason": "STOP",
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "id": "call_1",
                                    "name": "book_appointment",
                                    "args": {},
                                }
                            }
                        ]
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
        assert str(error) == "Gemini generate_content included tool calls when tool selection is none."
    else:
        raise AssertionError("Expected TextGenerationError.")
    assert client.calls[0].tool_config == {"function_calling_config": {"mode": "NONE"}}


def test_text_generator_rejects_safety_finish() -> None:
    client = FakeContentClient(
        response={
            "candidates": [
                {
                    "finish_reason": "SAFETY",
                    "content": {"parts": [{"text": "blocked"}]},
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
        assert str(error) == "Gemini generate_content finished unsuccessfully: SAFETY."
    else:
        raise AssertionError("Expected TextGenerationError.")
