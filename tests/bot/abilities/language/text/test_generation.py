from mosfet import abilities, media
from mosfet.abilities.language import text

import asyncio
import collections.abc
import typing
from typing import override

import hsm
import pydantic
import pytest

from tests.bot.abilities.support import dispatch_ability_for_test, start_abilities_for_test
from tests.type_helpers import object_dict


_PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"body", media_type="image/png")


class EchoTextGenerator(text.TextGenerator):
    @typing.override
    async def generate(
        self, input: text.InputData, *, deltas: collections.abc.Callable[[text.Delta], None] | None = None
    ) -> text.OutputData:
        content = input.messages[-1].content
        assert isinstance(content, str)
        if deltas is not None:
            for word in content.upper().split():
                deltas(text.Delta(content=word))
        return text.OutputData(content=content.upper())


class RecordingTextGeneration(text.TextGeneration):
    outputs: list[text.OutputData]

    def __init__(self, *, generator: text.TextGenerator) -> None:
        super().__init__(generator=generator)
        self.outputs = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, text.OutputData)
            self.outputs.append(output)
        return super().dispatch(ctx, event)


async def await_text_output(output: collections.abc.Awaitable[text.OutputData]) -> text.OutputData:
    return await output


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


async def start_ability_tree(ctx: hsm.Context | None, ability: abilities.Ability[typing.Any, typing.Any]) -> None:
    await start_abilities_for_test(hsm.Context() if ctx is None else ctx, ability)


def test_text_generation_uses_message_sequence_input() -> None:
    async def run() -> list[text.OutputData]:
        generation = RecordingTextGeneration(generator=EchoTextGenerator())
        await start_ability_tree(None, generation)
        input = text.InputData(
            messages=(
                text.TextMessage(role=text.TextRole.SYSTEM, content="Reply in uppercase."),
                text.TextMessage(role=text.TextRole.USER, content="hello"),
            )
        )

        _ = await generation.apply(input)
        for _ in range(100):
            if generation.outputs:
                break
            await asyncio.sleep(0)

        return generation.outputs

    outputs = asyncio.run(run())

    assert outputs == [text.OutputData(content="HELLO")]


def test_text_generation_apply_dispatches_input_event() -> None:
    async def run() -> tuple[bool, list[text.OutputData]]:
        generation = RecordingTextGeneration(generator=EchoTextGenerator())
        await start_ability_tree(None, generation)
        input = text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),))

        result = await generation.apply(input)
        for _ in range(100):
            if generation.outputs:
                break
            await asyncio.sleep(0)

        return result, generation.outputs

    result, outputs = asyncio.run(run())

    assert result is True
    assert outputs == [text.OutputData(content="HELLO")]


def test_text_generation_completion_correlates_via_event_id_without_instance_stash() -> None:
    """HSM-COMPLETION-001: leaf apply correlation rides event id/metadata, not instance stash."""

    async def run() -> text.OutputData:
        generation = RecordingTextGeneration(generator=EchoTextGenerator())
        await start_ability_tree(None, generation)
        input = text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),))
        # dispatch_ability_for_test keys the result future on the input event id and
        # resolves it from the terminal output's carried id (no ability instance stash).
        return await dispatch_ability_for_test(generation, None, input)

    output = asyncio.run(run())

    assert output == text.OutputData(content="HELLO")


def test_text_generation_defines_specific_input_and_output_events() -> None:
    input_schema = object_dict(text.InputEvent.schema)
    output_schema = object_dict(text.OutputEvent.schema)
    text_generation_input_schema = object_dict(text.TextGeneration.input_event.schema)
    text_generation_output_schema = object_dict(text.TextGeneration.output_event.schema)

    assert text.InputEvent.name == "bot.ability.language.text.generation.input"
    assert input_schema == text_generation_input_schema
    assert input_schema == text.InputData.model_json_schema()
    assert input_schema["description"]
    assert text.OutputEvent.name == "bot.ability.language.text.generation.output"
    assert output_schema == text_generation_output_schema
    assert output_schema == text.OutputData.model_json_schema()
    assert output_schema["description"]


def test_deltas_are_handed_out_while_the_final_output_stays_authoritative() -> None:
    received: list[text.Delta] = []
    output = asyncio.run(
        EchoTextGenerator().generate(
            text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="say this"),)),
            deltas=received.append,
        )
    )

    assert received == [text.Delta(content="SAY"), text.Delta(content="THIS")]
    assert output == text.OutputData(content="SAY THIS")
    delta = text.Delta(tool_call=0, arguments='{"events":')
    assert (delta.content, delta.tool_call, delta.arguments) == ("", 0, '{"events":')
    with pytest.raises(pydantic.ValidationError):
        _ = text.Delta(tool_call=-1)


def test_text_generation_inherits_generator_field_from_generative() -> None:
    assert issubclass(text.TextGenerator, abilities.Generator)
    assert "generator" not in text.TextGeneration.__annotations__


def test_text_message_roles_are_provider_neutral() -> None:
    assert text.TextRole.SYSTEM == "system"
    assert text.TextRole.USER == "user"
    assert text.TextRole.ASSISTANT == "assistant"
    assert text.TextRole.AGENT == text.TextRole.ASSISTANT
    assert text.TextRole.TOOL == "tool"


def test_text_generation_shape_supports_voice_agent_react_messages() -> None:
    tool_call = text.TextToolCall(
        id="call_1",
        name="get_appointments",
        args={"date": "2026-03-19"},
    )

    input = text.InputData(
        messages=(
            text.TextMessage(role=text.TextRole.SYSTEM, content="system prompt"),
            text.TextMessage(role=text.TextRole.USER, content="Check availability"),
            text.TextMessage(role=text.TextRole.ASSISTANT, content="", tool_calls=(tool_call,)),
            text.TextMessage(role=text.TextRole.TOOL, content='{"slots": ["9am"]}', tool_call_id="call_1"),
        ),
        tools=("get_appointments",),
    )
    output = text.OutputData(
        content="9am is available.",
        reasoning="The tool returned an available slot.",
        provider="openai",
        model="gpt-5.4",
        tool_calls=(tool_call,),
    )

    assert input.messages[2].role == "assistant"
    assert input.messages[2].tool_calls[0].id == "call_1"
    assert input.messages[2].tool_calls[0].name == "get_appointments"
    assert input.messages[2].tool_calls[0].args == {"date": "2026-03-19"}
    assert input.messages[3].tool_call_id == "call_1"
    assert input.tools == ("get_appointments",)
    assert input.tool_selection == text.ToolSelectionPolicy.AUTO
    assert output.reasoning == "The tool returned an available slot."
    assert output.provider == "openai"
    assert output.model == "gpt-5.4"
    assert output.tool_calls == (tool_call,)


def test_user_message_accepts_text_and_image_parts() -> None:
    parts = (text.TextPart(text="What is this?"), text.ImagePart(image=_PNG))

    message = text.TextMessage(role=text.TextRole.USER, content=parts)

    assert message.content == parts
    assert text.TextMessage.model_validate_json(message.model_dump_json()) == message


def test_string_content_stays_valid_on_every_role() -> None:
    for role in text.TextRole:
        assert text.TextMessage(role=role, content="plain").content == "plain"


@pytest.mark.parametrize("role", [text.TextRole.SYSTEM, text.TextRole.ASSISTANT, text.TextRole.TOOL])
def test_content_parts_are_rejected_off_the_user_role(role: text.TextRole) -> None:
    with pytest.raises(pydantic.ValidationError, match="content parts are allowed only on user messages"):
        _ = text.TextMessage(role=role, content=(text.TextPart(text="hi"),))


def test_empty_content_parts_are_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match="content parts must not be empty"):
        _ = text.TextMessage(role=text.TextRole.USER, content=())


def test_text_part_rejects_empty_text() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = text.TextPart(text="")


def test_input_schema_exposes_kind_discriminated_content_parts() -> None:
    schema = text.InputData.model_json_schema()
    definitions = object_dict(typing.cast(object, schema["$defs"]))

    assert set(definitions) >= {"TextPart", "ImagePart", "Image"}
    encoded = str(object_dict(definitions["TextMessage"])["properties"])
    assert "'propertyName': 'kind'" in encoded
    assert "'text': '#/$defs/TextPart'" in encoded
    assert "'image': '#/$defs/ImagePart'" in encoded


def test_content_parts_are_reexported_on_language_packages() -> None:
    from mosfet.abilities import language

    assert language.TextPart is text.TextPart is text.generation.TextPart
    assert language.ImagePart is text.ImagePart is text.generation.ImagePart


def test_text_generator_defaults_to_text_only_image_capacity() -> None:
    assert EchoTextGenerator().image_capacity == media.ImageCapacity()
