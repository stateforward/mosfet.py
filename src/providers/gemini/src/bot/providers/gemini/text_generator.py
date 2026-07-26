from __future__ import annotations

from bot.abilities.language import text

import asyncio
import collections.abc
import dataclasses
import enum
import json
import typing
import uuid

from .client import ContentClient, jsonable


class TextGenerationError(RuntimeError):
    """Raised when a Gemini text generation response cannot be mapped to stateforward.bot output."""


def _empty_config() -> dict[str, object]:
    return {}


def _mapping(value: object, message: str) -> collections.abc.Mapping[str, object]:
    if not isinstance(value, collections.abc.Mapping):
        raise TextGenerationError(message)
    return typing.cast(collections.abc.Mapping[str, object], value)


def _sequence(value: object) -> list[object]:
    if not isinstance(value, collections.abc.Sequence) or isinstance(value, str | bytes | bytearray):
        return []
    return list(value)


def _tool_to_gemini(tool: object) -> dict[str, object]:
    if isinstance(tool, str):
        name = tool.strip()
        if not name:
            raise TextGenerationError("Gemini tools must be function tool objects or non-blank tool names.")
        return {
            "function_declarations": [
                {
                    "name": name,
                    "parameters_json_schema": {
                        "type": "object",
                        "properties": {},
                    },
                }
            ]
        }

    json_tool = jsonable(tool)
    if not isinstance(json_tool, collections.abc.Mapping):
        raise TextGenerationError("Gemini tools must be function tool objects or non-blank tool names.")
    tool_mapping = {
        str(key): item for key, item in typing.cast(collections.abc.Mapping[object, object], json_tool).items()
    }

    # Already a Gemini Tool shape.
    if "function_declarations" in tool_mapping:
        return tool_mapping

    # OpenAI-compatible intermediate shape used by Processing.
    if tool_mapping.get("type") == "function":
        function = tool_mapping.get("function")
        if not isinstance(function, collections.abc.Mapping):
            raise TextGenerationError("Gemini tools must be function tool objects or non-blank tool names.")
        function_mapping = typing.cast(collections.abc.Mapping[object, object], function)
        name = function_mapping.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TextGenerationError("Gemini tools must be function tool objects or non-blank tool names.")
        declaration: dict[str, object] = {"name": name}
        description = function_mapping.get("description")
        if isinstance(description, str) and description.strip():
            declaration["description"] = description
        parameters = function_mapping.get("parameters")
        if isinstance(parameters, collections.abc.Mapping):
            declaration["parameters_json_schema"] = {
                str(key): item for key, item in typing.cast(collections.abc.Mapping[object, object], parameters).items()
            }
        return {"function_declarations": [declaration]}

    raise TextGenerationError("Gemini tools must be function tool objects or non-blank tool names.")


def _tools_to_gemini(tools: collections.abc.Sequence[object]) -> tuple[dict[str, object], ...]:
    if not tools:
        return ()
    declarations: list[dict[str, object]] = []
    for tool in tools:
        gemini_tool = _tool_to_gemini(tool)
        raw_declarations = gemini_tool.get("function_declarations")
        for declaration in _sequence(raw_declarations):
            if isinstance(declaration, collections.abc.Mapping):
                declarations.append(
                    {
                        str(key): item
                        for key, item in typing.cast(collections.abc.Mapping[object, object], declaration).items()
                    }
                )
    if not declarations:
        return ()
    return ({"function_declarations": declarations},)


def _tool_selection_mode(policy: text.ToolSelectionPolicy) -> str:
    if policy == text.ToolSelectionPolicy.REQUIRED:
        return "ANY"
    if policy == text.ToolSelectionPolicy.NONE:
        return "NONE"
    return "AUTO"


def _message_parts_for_assistant(message: text.TextMessage) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    if message.content:
        parts.append({"text": message.content})
    for tool_call in message.tool_calls:
        parts.append(
            {
                "function_call": {
                    "name": tool_call.name,
                    "args": dict(tool_call.args),
                    "id": tool_call.id,
                }
            }
        )
    if not parts:
        parts.append({"text": ""})
    return parts


def _messages_to_gemini(
    messages: collections.abc.Sequence[text.TextMessage],
) -> tuple[str | None, list[dict[str, object]]]:
    system_chunks: list[str] = []
    contents: list[dict[str, object]] = []

    for message in messages:
        if message.role == text.TextRole.SYSTEM:
            if message.content.strip():
                system_chunks.append(message.content)
            continue
        if message.role == text.TextRole.USER:
            contents.append({"role": "user", "parts": [{"text": message.content}]})
            continue
        if message.role == text.TextRole.ASSISTANT:
            contents.append({"role": "model", "parts": _message_parts_for_assistant(message)})
            continue
        if message.role == text.TextRole.TOOL:
            name = message.tool_call_id or "tool"
            try:
                response_payload: object = json.loads(message.content) if message.content else {}
            except json.JSONDecodeError:
                response_payload = {"result": message.content}
            if not isinstance(response_payload, collections.abc.Mapping):
                response_payload = {"result": response_payload}
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": name,
                                "response": {
                                    str(key): item
                                    for key, item in typing.cast(
                                        collections.abc.Mapping[object, object], response_payload
                                    ).items()
                                },
                            }
                        }
                    ],
                }
            )
            continue
        raise TextGenerationError(f"Unsupported text role for Gemini: {message.role}.")

    system_instruction = "\n\n".join(system_chunks) if system_chunks else None
    if not contents:
        raise TextGenerationError("Gemini text generation requires at least one non-system message.")
    return system_instruction, contents


def _parts_from_candidate(
    candidate: collections.abc.Mapping[str, object],
) -> list[collections.abc.Mapping[str, object]]:
    content = candidate.get("content")
    if content is None:
        return []
    content_mapping = _mapping(content, "Gemini candidate content must be an object.")
    parts: list[collections.abc.Mapping[str, object]] = []
    for part in _sequence(content_mapping.get("parts")):
        if isinstance(part, collections.abc.Mapping):
            parts.append(typing.cast(collections.abc.Mapping[str, object], part))
    return parts


def _text_from_parts(parts: collections.abc.Sequence[collections.abc.Mapping[str, object]]) -> str:
    chunks: list[str] = []
    for part in parts:
        text_value = part.get("text")
        if isinstance(text_value, str):
            chunks.append(text_value)
    return "".join(chunks)


def _tool_calls_from_parts(
    parts: collections.abc.Sequence[collections.abc.Mapping[str, object]],
) -> tuple[text.TextToolCall, ...]:
    tool_calls: list[text.TextToolCall] = []
    for index, part in enumerate(parts):
        function_call = part.get("function_call")
        if function_call is None:
            continue
        call = _mapping(function_call, "Gemini function_call parts must be objects.")
        name = call.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TextGenerationError("Gemini function calls must include a non-blank string name.")
        raw_args = call.get("args")
        if raw_args is None:
            args: dict[str, object] = {}
        elif isinstance(raw_args, collections.abc.Mapping):
            args = {
                str(key): item for key, item in typing.cast(collections.abc.Mapping[object, object], raw_args).items()
            }
        else:
            raise TextGenerationError("Gemini function call args must be an object.")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"call_{index + 1}_{uuid.uuid4().hex[:8]}"
        tool_calls.append(text.TextToolCall(id=call_id, name=name, args=args))
    return tuple(tool_calls)


def _finish_reason(candidate: collections.abc.Mapping[str, object]) -> str | None:
    reason = candidate.get("finish_reason")
    if isinstance(reason, str):
        return reason
    if isinstance(reason, enum.Enum):
        value = typing.cast(object, reason.value)
        return str(value)
    return None


def _validate_tool_selection(input: text.InputData, tool_calls: tuple[text.TextToolCall, ...]) -> None:
    if input.tools and input.tool_selection == text.ToolSelectionPolicy.REQUIRED and not tool_calls:
        raise TextGenerationError("Gemini generate_content did not include a required tool call.")
    if input.tool_selection == text.ToolSelectionPolicy.NONE and tool_calls:
        raise TextGenerationError("Gemini generate_content included tool calls when tool selection is none.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class TextGenerator(text.TextGenerator):
    """Text generator backed by Google Gemini via the google-genai SDK."""

    client: ContentClient
    provider: str | None = "gemini"
    response_mime_type: str | None = None
    response_json_schema: collections.abc.Mapping[str, object] | None = None
    config: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_config)

    @typing.override
    async def generate(self, input: text.InputData) -> text.OutputData:
        return await asyncio.to_thread(self._generate_blocking, input)

    def _generate_blocking(self, input: text.InputData) -> text.OutputData:
        system_instruction, contents = _messages_to_gemini(input.messages)
        tools = _tools_to_gemini(input.tools)
        tool_config: dict[str, object] | None = None
        if tools:
            tool_config = {
                "function_calling_config": {
                    "mode": _tool_selection_mode(input.tool_selection),
                }
            }

        response = self.client.generate_content(
            contents=contents,
            system_instruction=system_instruction,
            tools=tools,
            tool_config=tool_config,
            response_mime_type=self.response_mime_type,
            response_json_schema=self.response_json_schema,
            config=self.config,
        )

        candidates = response.get("candidates")
        if not isinstance(candidates, collections.abc.Sequence) or isinstance(candidates, str | bytes | bytearray):
            raise TextGenerationError("Gemini response did not include candidates.")
        if not candidates:
            raise TextGenerationError("Gemini response did not include candidates.")
        candidate = _mapping(candidates[0], "Gemini candidate must be an object.")
        finish_reason = _finish_reason(candidate)
        if finish_reason is not None and finish_reason.upper() in {
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
        }:
            raise TextGenerationError(f"Gemini generate_content finished unsuccessfully: {finish_reason}.")

        parts = _parts_from_candidate(candidate)
        tool_calls = _tool_calls_from_parts(parts)
        content = _text_from_parts(parts)
        # AUTO tool selection may produce neither tools nor text (decline / empty selection).
        if (
            not tool_calls
            and not content
            and not (input.tools and input.tool_selection == text.ToolSelectionPolicy.AUTO)
        ):
            raise TextGenerationError("Gemini generate_content message did not include text content.")
        _validate_tool_selection(input, tool_calls)

        model = response.get("model_version")
        if not isinstance(model, str):
            model = response.get("model")
        return text.OutputData(
            content=content or "",
            reasoning="",
            provider=self.provider,
            model=model if isinstance(model, str) else None,
            tool_calls=tool_calls,
        )


__all__ = ["TextGenerationError", "TextGenerator"]
