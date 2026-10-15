"""``vision.description``: the ``Description`` wrapper and ``GenerativeDescriber`` over any text generator."""

import asyncio
import collections.abc
import datetime
import typing

import hsm
import pytest

from mosfet import abilities, media
from mosfet import event as event_contract
from mosfet.abilities import vision
from mosfet.abilities.language import text
from mosfet.abilities.vision import description

from tests.bot.abilities.support import start_abilities_for_test
from tests.type_helpers import invalid_value, model_view

_JPEG = media.Image(content=b"\xff\xd8\xff" + b"room", media_type="image/jpeg")
_PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"room", media_type="image/png")
_SEES_JPEG = media.ImageCapacity(max_images=1, max_total_bytes=1024, media_types=frozenset({"image/jpeg"}))


class FakeGenerator(text.TextGenerator):
    inputs: list[text.InputData]
    image_capacity: media.ImageCapacity
    reply: str

    def __init__(self, reply: str, *, capacity: media.ImageCapacity = _SEES_JPEG) -> None:
        self.reply = reply
        self.image_capacity = capacity
        self.inputs = []

    @typing.override
    async def generate(
        self, input: text.InputData, *, deltas: collections.abc.Callable[[text.Delta], None] | None = None
    ) -> text.OutputData:
        self.inputs.append(input)
        return text.OutputData(content=self.reply)


class FixedDescriber(description.Describer):
    inputs: list[media.Image]
    said: str

    def __init__(self, said: str) -> None:
        self.said = said
        self.inputs = []

    @typing.override
    async def decode(self, input: media.Image) -> str:
        self.inputs.append(input)
        return self.said


class FailingDescriber(description.Describer):
    @typing.override
    async def decode(self, input: media.Image) -> str:
        del input
        raise RuntimeError("describer unavailable")


class WrongDescriber(description.Describer):
    @typing.override
    async def decode(self, input: media.Image) -> str:
        del input
        return invalid_value(str, b"not text")


def _describe(describer: description.Describer) -> hsm.Event[typing.Any]:
    async def run() -> hsm.Event[typing.Any]:
        child = description.Description(decoder=describer)
        ctx = hsm.Context()
        await start_abilities_for_test(ctx, child)
        return await abilities.run_terminal_operation(
            ctx,
            child=child,
            request=child.input_event.with_data_and_id(_JPEG, "describe-1"),
            terminals=(child.output_event, child.failed_event),
            timeout=datetime.timedelta(seconds=2),
        )

    return asyncio.run(run())


def test_vision_exports_description() -> None:
    assert vision.description is description
    assert "description" in vision.__all__
    assert description.MAX_DESCRIPTION_CHARS == 4000
    assert set(description.__all__) == {
        "DESCRIBE_INSTRUCTIONS",
        "MAX_DESCRIPTION_CHARS",
        "Describer",
        "Description",
        "Error",
        "GenerativeDescriber",
    }
    assert issubclass(description.Error, RuntimeError)


def test_description_answers_with_what_the_describer_said() -> None:
    describer = FixedDescriber("A cup on a desk.")

    terminal = _describe(describer)

    assert terminal.name == "bot.ability.vision.description.output"
    assert terminal.id == "describe-1"
    assert terminal.data == "A cup on a desk."
    assert describer.inputs == [_JPEG]


def test_description_answers_a_describer_failure_with_its_failed_event() -> None:
    terminal = _describe(FailingDescriber())

    assert terminal.name == abilities.FailedEvent.name
    assert isinstance(terminal.data, abilities.FailureData)
    assert terminal.data.message == "describer unavailable"


@pytest.mark.parametrize(
    "said",
    ["", "   ", "x" * (description.MAX_DESCRIPTION_CHARS + 1)],
    ids=["empty", "blank", "too-long"],
)
def test_description_answers_text_outside_its_contract_with_a_failure(said: str) -> None:
    terminal = _describe(FixedDescriber(said))

    assert terminal.name == abilities.FailedEvent.name
    assert isinstance(terminal.data, abilities.FailureData)
    assert "output schema" in terminal.data.message


def test_description_answers_output_that_is_not_text_with_a_failure() -> None:
    terminal = _describe(WrongDescriber())

    assert terminal.name == abilities.FailedEvent.name
    assert isinstance(terminal.data, abilities.FailureData)
    assert "output schema" in terminal.data.message


def test_description_events_are_named_in_the_vision_domain() -> None:
    assert description.Description.input_event.name == "bot.ability.vision.description.input"
    assert description.Description.output_event.name == "bot.ability.vision.description.output"
    assert description.Description.input_data_type is media.Image
    assert description.Description.output_data_type is str
    model = model_view(description.Description.model)
    behavior = "/DescriptionLifecycle/attached/behavior"
    assert "bot.ability.vision.description.input" in model.transition_map[f"{behavior}/idle"]
    assert "bot.ability.vision.description.apply.completed" in model.transition_map[f"{behavior}/applying"]
    assert "bot.ability.vision.description.apply.failed" in model.transition_map[f"{behavior}/applying"]
    assert not hsm.kind.Is(description.Description.input_event.kind, event_contract.EventKind)


def test_generative_describer_asks_once_with_the_instructions_then_the_image() -> None:
    generator = FakeGenerator("  A cup on a desk.\n")
    describer = description.GenerativeDescriber(generator=generator)

    said = asyncio.run(describer.decode(_JPEG))

    assert said == "A cup on a desk."
    (asked,) = generator.inputs
    (message,) = asked.messages
    assert message.role == text.TextRole.USER
    assert message.content == (
        text.TextPart(text=description.DESCRIBE_INSTRUCTIONS),
        text.ImagePart(image=_JPEG),
    )
    assert asked.tools == ()


def test_generative_describer_instructions_are_an_overridable_default() -> None:
    assert "guess" in description.DESCRIBE_INSTRUCTIONS
    generator = FakeGenerator("Two people talking.")

    _ = asyncio.run(
        description.GenerativeDescriber(generator=generator, instructions="Say who is there.").decode(_JPEG)
    )

    content = generator.inputs[0].messages[0].content
    assert isinstance(content, tuple) and content[0] == text.TextPart(text="Say who is there.")
    with pytest.raises(ValueError, match="instructions"):
        _ = description.GenerativeDescriber(generator=generator, instructions=" ")


def test_generative_describer_refuses_an_image_its_model_cannot_take_before_asking() -> None:
    for generator, image in (
        (FakeGenerator("unused", capacity=media.ImageCapacity()), _JPEG),
        (FakeGenerator("unused"), _PNG),
    ):
        with pytest.raises(description.Error, match="cannot take"):
            _ = asyncio.run(description.GenerativeDescriber(generator=generator).decode(image))
        assert generator.inputs == []


def test_generative_describer_refuses_an_empty_reply() -> None:
    with pytest.raises(description.Error, match="empty"):
        _ = asyncio.run(description.GenerativeDescriber(generator=FakeGenerator(" \n")).decode(_JPEG))
