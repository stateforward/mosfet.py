from __future__ import annotations

from mosfet.abilities.language import text
import mosfet.telemetry

import asyncio
import collections.abc
import dataclasses
import json
import typing

from .client import ChatCompletionClient, jsonable
import time
from .client import ReasoningEffort, ResponsesClient


class TextGenerationError(RuntimeError):
    """Raised when an OpenAI-compatible text generation response cannot be mapped to stateforward.mosfet output."""


class UnsupportedReasoningEffortError(TextGenerationError):
    """Raised when a requested reasoning effort cannot be sent on the selected OpenAI API."""


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
    reasoning_effort: ReasoningEffort | None,
) -> dict[str, object]:
    body = dict(extra_body)
    if reasoning_effort is not None:
        if input.tools and reasoning_effort != ReasoningEffort.NONE:
            # OpenAI reasoning models reject function tools on /v1/chat/completions unless effort is "none".
            message = (
                f"Chat Completions cannot send function tools with reasoning_effort={reasoning_effort.value!r}; "
                + "use ResponsesTextGenerator or reasoning_effort='none'."
            )
            raise UnsupportedReasoningEffortError(message)
        body["reasoning_effort"] = reasoning_effort.value
    if input.tools:
        body["tool_choice"] = str(input.tool_selection)
        # Unrequested effort defaults to "none" so reasoning models accept function tools here.
        _ = body.setdefault("reasoning_effort", ReasoningEffort.NONE.value)
    return body


def _usage_counts(
    usage: object,
    fields: collections.abc.Mapping[str, tuple[str, ...]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, path in fields.items():
        value = usage
        for key in path:
            value = (
                typing.cast(collections.abc.Mapping[str, object], value).get(key)
                if isinstance(value, collections.abc.Mapping)
                else None
            )
        if isinstance(value, int) and not isinstance(value, bool):
            counts[name] = value
    return counts


def _record_usage(
    provider: str | None,
    response: collections.abc.Mapping[str, object],
    fields: collections.abc.Mapping[str, tuple[str, ...]],
) -> None:
    counts = _usage_counts(response.get("usage"), fields)
    model = response.get("model")
    if counts:
        mosfet.telemetry.record_generator_usage(
            provider=provider or "",
            model=model if isinstance(model, str) else None,
            usage=counts,
        )


def _record_response(
    provider: str | None,
    model: object,
    response: collections.abc.Mapping[str, object] | None,
    started: float,
    error: BaseException | None,
) -> None:
    mosfet.telemetry.record_generator_response(
        provider=provider or "",
        model=model if isinstance(model, str) else None,
        response=response,
        latency_s=time.monotonic() - started,
        error=error,
    )


_CHAT_USAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "input_tokens": ("prompt_tokens",),
    "output_tokens": ("completion_tokens",),
    "reasoning_tokens": ("completion_tokens_details", "reasoning_tokens"),
    "total_tokens": ("total_tokens",),
}
_RESPONSES_USAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens",),
    "cached_input_tokens": ("input_tokens_details", "cached_tokens"),
    "output_tokens": ("output_tokens",),
    "reasoning_tokens": ("output_tokens_details", "reasoning_tokens"),
    "total_tokens": ("total_tokens",),
}


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
    reasoning_effort: ReasoningEffort | None = None

    @typing.override
    async def generate(self, input: text.InputData) -> text.OutputData:
        return await asyncio.to_thread(self._generate_blocking, input)

    def _generate_blocking(self, input: text.InputData) -> text.OutputData:
        messages = [_message_to_openai(message) for message in input.messages]
        tools = _tools_to_openai(input.tools)
        model = getattr(self.client, "model", None)
        # Processing (and other callers) reach the provider via TextGenerator;
        # this records the wire request used on that path.
        mosfet.telemetry.record_generator_request(
            provider=self.provider or "",
            model=model if isinstance(model, str) else None,
            messages=messages,
            tools=tools,
        )
        extra_body = _extra_body_for_input(input, self.extra_body, self.reasoning_effort)
        started = time.monotonic()
        response: collections.abc.Mapping[str, object] | None = None
        try:
            response = self.client.create_chat_completion(
                messages=messages,
                tools=tools,
                response_format=self.response_format,
                extra_body=extra_body,
            )
            _record_usage(self.provider, response, _CHAT_USAGE_FIELDS)
            message, finish_reason = _first_choice(response)
            tool_calls = _tool_calls_from_message(message)
            if finish_reason == "tool_calls" and not tool_calls:
                raise TextGenerationError("OpenAI-compatible chat completion finished with no tool calls.")
            if finish_reason != "tool_calls" and tool_calls:
                raise TextGenerationError(
                    "OpenAI-compatible chat completion included tool calls without a tool_calls finish reason."
                )
            _validate_tool_selection(input, tool_calls)
            response_model = response.get("model")
            output = text.OutputData(
                content=_content_from_message(message, has_tool_calls=bool(tool_calls)),
                reasoning=_reasoning_from_message(message),
                provider=self.provider,
                model=response_model if isinstance(response_model, str) else None,
                tool_calls=tool_calls,
            )
        except Exception as error:
            _record_response(self.provider, model, response, started, error)
            raise
        _record_response(self.provider, model, response, started, None)
        return output


def _responses_input(messages: collections.abc.Sequence[text.TextMessage]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for message in messages:
        if message.role == text.TextRole.TOOL:
            if message.tool_call_id is None:
                raise TextGenerationError("OpenAI Responses tool messages must include a tool_call_id.")
            items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": message.content})
            continue
        if message.content or not message.tool_calls:
            items.append({"role": str(message.role), "content": message.content})
        items.extend(
            {
                "type": "function_call",
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": json.dumps(tool_call.args, separators=(",", ":")),
            }
            for tool_call in message.tool_calls
        )
    return items


def _responses_tool(tool: dict[str, object]) -> dict[str, object]:
    function = typing.cast(collections.abc.Mapping[str, object], tool["function"])
    return {"type": "function", **function}


def _responses_text_format(response_format: object | None) -> dict[str, object] | None:
    if response_format is None:
        return None
    json_format = _mapping(jsonable(response_format), "OpenAI Responses response_format must be an object.")
    if json_format.get("type") == "json_schema":
        schema = _mapping(
            json_format.get("json_schema"), "OpenAI json_schema response_format must include json_schema."
        )
        return {"type": "json_schema", **schema}
    return dict(json_format)


def _sequence(value: object, message: str) -> collections.abc.Sequence[object]:
    if not isinstance(value, collections.abc.Sequence) or isinstance(value, str | bytes | bytearray):
        raise TextGenerationError(message)
    return value


def _responses_output(
    response: collections.abc.Mapping[str, object],
) -> tuple[str, str, tuple[text.TextToolCall, ...]]:
    status = response.get("status")
    if status != "completed":
        details = response.get("incomplete_details")
        raise TextGenerationError(f"OpenAI Responses API response finished unsuccessfully: {status} {details}.")
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[text.TextToolCall] = []
    for raw_item in _sequence(response.get("output"), "OpenAI Responses API response output must be a list."):
        item = _mapping(raw_item, "OpenAI Responses API output items must be objects.")
        item_type = item.get("type")
        if item_type == "message":
            for raw_part in _sequence(item.get("content"), "OpenAI Responses message content must be a list."):
                part = _mapping(raw_part, "OpenAI Responses message content parts must be objects.")
                if part.get("type") == "refusal":
                    raise TextGenerationError("OpenAI Responses API refused the request.")
                part_text = part.get("text")
                if part.get("type") == "output_text" and isinstance(part_text, str):
                    content.append(part_text)
        elif item_type == "reasoning":
            for raw_summary in _sequence(item.get("summary", ()), "OpenAI Responses reasoning summary must be a list."):
                summary_text = _mapping(raw_summary, "OpenAI Responses reasoning summary parts must be objects.").get(
                    "text"
                )
                if isinstance(summary_text, str):
                    reasoning.append(summary_text)
        elif item_type == "function_call":
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise TextGenerationError("OpenAI Responses function calls must include a non-blank string name.")
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise TextGenerationError("OpenAI Responses function calls must include a non-blank string call_id.")
            tool_calls.append(text.TextToolCall(id=call_id, name=name, args=_tool_args(item.get("arguments"))))
    return "".join(content), "\n\n".join(reasoning), tuple(tool_calls)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResponsesTextGenerator(text.TextGenerator):
    """Text generator backed by the OpenAI Responses API (reasoning effort with function tools)."""

    client: ResponsesClient
    provider: str | None = "openai_compat"
    response_format: object | None = None
    reasoning_effort: ReasoningEffort | None = None
    extra_body: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_body)

    @typing.override
    async def generate(self, input: text.InputData) -> text.OutputData:
        return await asyncio.to_thread(self._generate_blocking, input)

    def _generate_blocking(self, input: text.InputData) -> text.OutputData:
        items = _responses_input(input.messages)
        tools = tuple(_responses_tool(tool) for tool in _tools_to_openai(input.tools))
        model = getattr(self.client, "model", None)
        mosfet.telemetry.record_generator_request(
            provider=self.provider or "",
            model=model if isinstance(model, str) else None,
            messages=items,
            tools=tools,
        )
        started = time.monotonic()
        response: collections.abc.Mapping[str, object] | None = None
        try:
            response = self.client.create_response(
                input=items,
                tools=tools,
                tool_choice=str(input.tool_selection) if tools else None,
                text_format=_responses_text_format(self.response_format),
                reasoning=None if self.reasoning_effort is None else {"effort": self.reasoning_effort.value},
                extra_body=self.extra_body,
            )
            _record_usage(self.provider, response, _RESPONSES_USAGE_FIELDS)
            content, reasoning, tool_calls = _responses_output(response)
            _validate_tool_selection(input, tool_calls)
            response_model = response.get("model")
            output = text.OutputData(
                content=content,
                reasoning=reasoning,
                provider=self.provider,
                model=response_model if isinstance(response_model, str) else None,
                tool_calls=tool_calls,
            )
        except Exception as error:
            _record_response(self.provider, model, response, started, error)
            raise
        _record_response(self.provider, model, response, started, None)
        return output


__all__ = ["ResponsesTextGenerator", "TextGenerationError", "TextGenerator", "UnsupportedReasoningEffortError"]
