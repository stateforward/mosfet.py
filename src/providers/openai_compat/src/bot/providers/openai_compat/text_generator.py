from __future__ import annotations

from bot.abilities.language import text

import asyncio
import collections.abc
import dataclasses
import json
import typing

from .client import ChatCompletionClient, jsonable

class TextGenerationError(RuntimeError):
    """Raised when an OpenAI-compatible text generation response cannot be mapped to stateforward.bot output."""

_SUCCESS_FINISH_REASONS = frozenset({"stop", "tool_calls"})

def _empty_body() -> dict[str, object]:
    return {}

def _tool_call_to_openai(tool_call: text.TextToolCall) -> dict[str, object]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.args, separators=(",", ":")),
        },
    }

def _message_to_openai(message: text.TextMessage) -> dict[str, object]:
    output: dict[str, object] = {
        "role": str(message.role),
        "content": message.content,
    }
    if message.tool_call_id is not None:
        output["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        output["tool_calls"] = [_tool_call_to_openai(tool_call) for tool_call in message.tool_calls]
    return output

def _tool_to_openai(tool: object) -> dict[str, object]:
    if isinstance(tool, str):
        name = tool.strip()
        if not name:
            raise TextGenerationError("OpenAI-compatible tools must be function tool objects or non-blank tool names.")
        return {"type": "function", "function": {"name": name}}

    json_tool = jsonable(tool)
    if not isinstance(json_tool, collections.abc.Mapping):
        raise TextGenerationError("OpenAI-compatible tools must be function tool objects or non-blank tool names.")
    tool_mapping = {
        str(key): item for key, item in typing.cast(collections.abc.Mapping[object, object], json_tool).items()
    }
    if tool_mapping.get("type") != "function":
        raise TextGenerationError("OpenAI-compatible tools must be function tool objects or non-blank tool names.")
    function = tool_mapping.get("function")
    if not isinstance(function, collections.abc.Mapping):
        raise TextGenerationError("OpenAI-compatible tools must be function tool objects or non-blank tool names.")
    function_mapping = typing.cast(collections.abc.Mapping[object, object], function)
    name = function_mapping.get("name")
    if not isinstance(name, str) or not name.strip():
        raise TextGenerationError("OpenAI-compatible tools must be function tool objects or non-blank tool names.")
    return tool_mapping

def _tools_to_openai(tools: collections.abc.Sequence[object]) -> tuple[dict[str, object], ...]:
    return tuple(_tool_to_openai(tool) for tool in tools)

def _extra_body_for_input(
    input: text.InputData,
    extra_body: collections.abc.Mapping[str, object],
) -> dict[str, object]:
    body = dict(extra_body)
    if input.tools:
        body["tool_choice"] = str(input.tool_selection)
    return body

def _mapping(value: object, message: str) -> collections.abc.Mapping[str, object]:
    if not isinstance(value, collections.abc.Mapping):
        raise TextGenerationError(message)
    return typing.cast(collections.abc.Mapping[str, object], value)

def _first_choice(response: collections.abc.Mapping[str, object]) -> tuple[collections.abc.Mapping[str, object], str]:
    choices = response.get("choices")
    if not isinstance(choices, collections.abc.Sequence) or isinstance(choices, str | bytes | bytearray) or not choices:
        raise TextGenerationError("OpenAI-compatible response did not include a chat completion choice.")
    choice = _mapping(choices[0], "OpenAI-compatible chat completion choice must be an object.")
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        raise TextGenerationError("OpenAI-compatible chat completion included an invalid finish reason.")
    if finish_reason not in _SUCCESS_FINISH_REASONS:
        raise TextGenerationError(f"OpenAI-compatible chat completion finished unsuccessfully: {finish_reason}.")
    return (
        _mapping(choice.get("message"), "OpenAI-compatible chat completion choice did not include a message object."),
        finish_reason,
    )

def _content_from_message(message: collections.abc.Mapping[str, object], *, has_tool_calls: bool) -> str:
    if message.get("refusal") is not None:
        raise TextGenerationError("OpenAI-compatible chat completion refused the request.")
    content = message.get("content")
    if content is None and has_tool_calls:
        return ""
    if isinstance(content, str):
        return content
    raise TextGenerationError("OpenAI-compatible chat completion message did not include string content.")

def _reasoning_from_message(message: collections.abc.Mapping[str, object]) -> str:
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        return reasoning_content
    return ""

def _tool_args(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        raise TextGenerationError("OpenAI-compatible tool call arguments must be a JSON object string.")
    try:
        parsed = typing.cast(object, json.loads(value))
    except json.JSONDecodeError:
        raise TextGenerationError("OpenAI-compatible tool call arguments must be a JSON object string.") from None
    if isinstance(parsed, collections.abc.Mapping):
        parsed_mapping = typing.cast(collections.abc.Mapping[object, object], parsed)
        return {str(key): item for key, item in parsed_mapping.items()}
    raise TextGenerationError("OpenAI-compatible tool call arguments must be a JSON object string.")

def _tool_calls_from_message(message: collections.abc.Mapping[str, object]) -> tuple[text.TextToolCall, ...]:
    if message.get("function_call") is not None:
        raise TextGenerationError("OpenAI-compatible legacy function_call responses are not supported.")
    if "tool_calls" not in message or message.get("tool_calls") is None:
        return ()
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, collections.abc.Sequence) or isinstance(raw_tool_calls, str | bytes | bytearray):
        raise TextGenerationError("OpenAI-compatible tool calls must be a list of objects.")

    tool_calls: list[text.TextToolCall] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, collections.abc.Mapping):
            raise TextGenerationError("OpenAI-compatible tool calls must be objects.")
        tool_call = typing.cast(collections.abc.Mapping[str, object], raw_tool_call)
        type_value = tool_call.get("type")
        if type_value != "function":
            raise TextGenerationError("OpenAI-compatible tool calls must use function tools.")
        function = tool_call.get("function")
        if not isinstance(function, collections.abc.Mapping):
            raise TextGenerationError("OpenAI-compatible tool calls must include a function object.")
        function_mapping = typing.cast(collections.abc.Mapping[str, object], function)
        name = function_mapping.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TextGenerationError("OpenAI-compatible function tool calls must include a non-blank string name.")
        id_value = tool_call.get("id")
        if not isinstance(id_value, str) or not id_value.strip():
            raise TextGenerationError("OpenAI-compatible tool calls must include a non-blank string id.")
        tool_calls.append(
            text.TextToolCall(
                id=id_value,
                name=name,
                args=_tool_args(function_mapping.get("arguments")),
            )
        )
    return tuple(tool_calls)

def _extra_body_for_input(
    input: text.InputData,
    extra_body: collections.abc.Mapping[str, object],
) -> dict[str, object]:
    body = dict(extra_body)
    if input.tools:
        body["tool_choice"] = str(input.tool_selection)
    return body

def _validate_tool_selection(input: text.InputData, tool_calls: tuple[text.TextToolCall, ...]) -> None:
    if input.tools and input.tool_selection == text.ToolSelectionPolicy.REQUIRED and not tool_calls:
        raise TextGenerationError("OpenAI-compatible chat completion did not include a required tool call.")
    if input.tool_selection == text.ToolSelectionPolicy.NONE and tool_calls:
        raise TextGenerationError("OpenAI-compatible chat completion included tool calls when tool selection is none.")

@dataclasses.dataclass(frozen=True, kw_only=True)
class TextGenerator(text.TextGenerator):
    """Text generator backed by any provider with an OpenAI-compatible Chat Completions endpoint."""

    client: ChatCompletionClient
    provider: str | None = "openai_compat"
    response_format: object | None = None
    extra_body: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_body)

    @typing.override
    async def generate(self, input: text.InputData) -> text.OutputData:
        return await asyncio.to_thread(self._generate_blocking, input)

    def _generate_blocking(self, input: text.InputData) -> text.OutputData:
        response = self.client.create_chat_completion(
            messages=[_message_to_openai(message) for message in input.messages],
            tools=_tools_to_openai(input.tools),
            response_format=self.response_format,
            extra_body=_extra_body_for_input(input, self.extra_body),
        )
        message, finish_reason = _first_choice(response)
        tool_calls = _tool_calls_from_message(message)
        if finish_reason == "tool_calls" and not tool_calls:
            raise TextGenerationError("OpenAI-compatible chat completion finished with no tool calls.")
        if finish_reason != "tool_calls" and tool_calls:
            raise TextGenerationError(
                "OpenAI-compatible chat completion included tool calls without a tool_calls finish reason."
            )
        _validate_tool_selection(input, tool_calls)
        model = response.get("model")
        return text.OutputData(
            content=_content_from_message(message, has_tool_calls=bool(tool_calls)),
            reasoning=_reasoning_from_message(message),
            provider=self.provider,
            model=model if isinstance(model, str) else None,
            tool_calls=tool_calls,
        )

__all__ = ["TextGenerationError", "TextGenerator"]
