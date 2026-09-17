from ... import ability
from ... import generative

import abc
import dataclasses
import enum
import typing

import hsm
import mosfet
import pydantic

from mosfet.telemetry import observer


def _empty_text_tool_args() -> dict[str, object]:
    return {}


class TextRole(enum.StrEnum):
    """Provider-neutral role for text-generation messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    AGENT = "assistant"
    TOOL = "tool"


class ToolSelectionPolicy(enum.StrEnum):
    """Provider-neutral policy for tool selection during text generation."""

    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


class TextToolCall(pydantic.BaseModel):
    """A provider-neutral tool call requested by generated text."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"id": "call_1", "name": "get_appointments", "args": {"date": "2026-03-19"}}],
        },
    )

    id: str = pydantic.Field(
        description="Provider-neutral identifier for this tool call.",
        examples=["call_1"],
    )
    name: str = pydantic.Field(
        description="Name of the requested tool operation.",
        examples=["get_appointments"],
    )
    args: dict[str, object] = pydantic.Field(
        default_factory=_empty_text_tool_args,
        description="JSON-serializable arguments for the requested tool operation.",
        examples=[{"date": "2026-03-19"}],
    )


class TextMessage(pydantic.BaseModel):
    """A single provider-neutral message in a text-generation input."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"role": "user", "content": "Check availability"}],
        },
    )

    role: TextRole = pydantic.Field(
        description="Provider-neutral conversation role for this message.",
        examples=[TextRole.USER],
    )
    content: str = pydantic.Field(
        description="Text content for this message. Assistant tool-call messages may use an empty string.",
        examples=["Check availability"],
    )
    tool_call_id: str | None = pydantic.Field(
        default=None,
        description="Identifier of the tool call this message responds to, when the role is tool.",
        examples=["call_1"],
    )
    tool_calls: tuple[TextToolCall, ...] = pydantic.Field(
        default=(),
        description="Tool calls requested by an assistant message.",
    )


class InputData(pydantic.BaseModel):
    """Canonical input for text generation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"messages": [{"role": "user", "content": "hello"}], "tools": []}],
        },
    )

    messages: tuple[TextMessage, ...] = pydantic.Field(
        description="Ordered provider-neutral messages that make up the text-generation prompt.",
        examples=[[{"role": "user", "content": "hello"}]],
    )
    tools: tuple[object, ...] = pydantic.Field(
        default=(),
        description="Tool definitions available to the text generator.",
    )
    tool_selection: ToolSelectionPolicy = pydantic.Field(
        default=ToolSelectionPolicy.AUTO,
        description=(
            "Provider-neutral policy for tool selection when tools are available. AUTO allows tool calls or message "
            "content, REQUIRED requires a tool call when tools are present, and NONE disables tool calls."
        ),
        examples=[ToolSelectionPolicy.AUTO],
    )


class OutputData(pydantic.BaseModel):
    """Canonical output for text generation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"content": "HELLO", "reasoning": "", "provider": "openai", "model": "gpt-5.4"}],
        },
    )

    content: str = pydantic.Field(
        description="Generated text content.",
        examples=["HELLO"],
    )
    reasoning: str = pydantic.Field(
        default="",
        description="Optional provider-neutral reasoning summary associated with the generated content.",
        examples=["The tool returned an available slot."],
    )
    provider: str | None = pydantic.Field(
        default=None,
        description="Provider that produced the output, when known.",
        examples=["openai"],
    )
    model: str | None = pydantic.Field(
        default=None,
        description="Provider model that produced the output, when known.",
        examples=["gpt-5.4"],
    )
    tool_calls: tuple[TextToolCall, ...] = pydantic.Field(
        default=(),
        description="Tool calls requested by the generated text.",
    )


class TextGenerator(generative.Generator[InputData, OutputData], abc.ABC):
    """Generator that produces text from text-generation input."""


_TextGenerationApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.language.text.generation.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_TextGenerationApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.language.text.generation.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _has_text_generation_input(
    ctx: hsm.Context,
    instance: "TextGeneration",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)


def _has_text_generation_output(
    ctx: hsm.Context,
    instance: "TextGeneration",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, OutputData)


def _has_invalid_text_generation_output(
    ctx: hsm.Context,
    instance: "TextGeneration",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, OutputData)


def _has_text_generation_failure(
    ctx: hsm.Context,
    instance: "TextGeneration",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


class TextGeneration(generative.Generative[InputData, OutputData]):
    """Ability to generate text."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.language.text.generation.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.language.text.generation.output",
        schema=OutputData,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _TextGenerationApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _TextGenerationApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_text_generation_output_failure(
        ctx: hsm.Context,
        instance: "TextGeneration",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(
            message="Ability operation produced output that does not match its output schema."
        )
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "TextGeneration",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_text_generation_input),
                hsm.target("../applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(generative.Generative._run_behavior_activity),
            hsm.transition(
                hsm.on(_TextGenerationApplyCompletedEvent),
                hsm.guard(_has_text_generation_output),
                hsm.effect(generative.Generative._dispatch_generative_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_TextGenerationApplyCompletedEvent),
                hsm.guard(_has_invalid_text_generation_output),
                hsm.effect(_dispatch_invalid_text_generation_output_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_TextGenerationApplyFailedEvent),
                hsm.guard(_has_text_generation_failure),
                hsm.effect(generative.Generative._dispatch_generative_failure),
                hsm.target("../idle"),
            ),
        ),
        hsm.observe(observer),
    )


InputEvent = TextGeneration.input_event
OutputEvent = TextGeneration.output_event
