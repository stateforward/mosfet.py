"""Description: what an image shows, in a few plain sentences, through one injected describer.

``Describer`` is the plain, non-HSM capacity a provider implements: any
``decoding.Decoder[media.Image, str]``, such as a local VLM decoder, or ``GenerativeDescriber`` over
any image-capable ``text.TextGenerator`` (a remote VLM endpoint, an omni-modal model).
``Description`` is the HSM wrapper an owning ability attaches and addresses with typed events.
A description is model output: never a metric attribute, a routing key or anything the bot
branches on.
"""

import dataclasses
import typing

import hsm
import mosfet

from mosfet import media

from .. import ability
from .. import decoding
from ..language import text

MAX_DESCRIPTION_CHARS: typing.Final = 4000
"""The longest description one ``Description`` answers with."""

DESCRIBE_INSTRUCTIONS: typing.Final = (
    "Describe what this image shows in two or three plain sentences: the people, objects, text and activity "
    "in view. Do not guess who anyone is; describe people only by what can be seen."
)
"""The default request ``GenerativeDescriber`` sends with the image; a composition may override it."""

Describer: typing.TypeAlias = decoding.Decoder[media.Image, str]
"""A provider-agnostic, non-HSM capacity to describe an image in plain text.

``decode`` raises on failure. A call that cannot be cancelled (a worker thread, a blocking SDK)
keeps running after its owner gives up on it, so a provider bounds its own work.
"""


class Error(RuntimeError):
    """A describer could not describe an image."""


class GenerativeDescriber(Describer):
    """Describes an image by asking any image-capable text generator, once per image.

    Not an actor: a plain collaborator over a ``text.TextGenerator``, called directly.
    """

    _generator: text.TextGenerator
    _instructions: str

    def __init__(self, *, generator: text.TextGenerator, instructions: str = DESCRIBE_INSTRUCTIONS) -> None:
        """Describe through ``generator``, asking with ``instructions`` placed before the image.

        Raises:
            ValueError: If ``instructions`` is blank.
        """

        if not instructions.strip():
            raise ValueError("GenerativeDescriber instructions must not be blank.")
        self._generator = generator
        self._instructions = instructions

    @typing.override
    async def decode(self, input: media.Image) -> str:
        """Ask the generator to describe ``input``; return its stripped answer.

        Raises:
            Error: If the generator's model cannot take the image (checked before asking), or it
                answered with no text.
        """

        if self._generator.image_capacity.admit((input,)) != (None,):
            raise Error("the generator's model cannot take this image")
        reply = await self._generator.generate(
            text.InputData(
                messages=(
                    text.TextMessage(
                        role=text.TextRole.USER,
                        content=(text.TextPart(text=self._instructions), text.ImagePart(image=input)),
                    ),
                ),
                tool_selection=text.ToolSelectionPolicy.NONE,
            )
        )
        said = reply.content.strip()
        if not said:
            raise Error("the generator answered with an empty description")
        return said


_ApplyCompletedEvent = hsm.Event[str](
    name="bot.ability.vision.description.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=str,
)
_ApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.vision.description.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _has_input(ctx: hsm.Context, instance: "Description", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, media.Image)


def _has_output(ctx: hsm.Context, instance: "Description", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    said = event.data
    return isinstance(said, str) and bool(said.strip()) and len(said) <= MAX_DESCRIPTION_CHARS


def _has_invalid_output(ctx: hsm.Context, instance: "Description", event: hsm.Event[typing.Any]) -> bool:
    return not _has_output(ctx, instance, event)


def _has_failure(ctx: hsm.Context, instance: "Description", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


class Description(decoding.Decoding[media.Image, str]):
    """Ability that describes one image at a time through its injected ``Describer``.

    It answers with non-blank text of at most ``MAX_DESCRIPTION_CHARS`` characters; anything
    else the describer produces is answered as a failure.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = media.Image
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = str
    input_event: typing.ClassVar[hsm.Event[media.Image]] = hsm.Event[media.Image](
        name="bot.ability.vision.description.input",
        schema=media.Image,
    )
    output_event: typing.ClassVar[hsm.Event[str]] = hsm.Event[str](
        name="bot.ability.vision.description.output",
        schema=str,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _ApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _ApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_output_failure(
        ctx: hsm.Context,
        instance: "Description",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(message="Description produced output that does not match its output schema.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Description",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(hsm.on(input_event), hsm.guard(_has_input), hsm.target("../applying")),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(decoding.Decoding._run_behavior_activity),
            hsm.transition(
                hsm.on(_ApplyCompletedEvent),
                hsm.guard(_has_output),
                hsm.effect(decoding.Decoding._dispatch_decoding_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_ApplyCompletedEvent),
                hsm.guard(_has_invalid_output),
                hsm.effect(_dispatch_invalid_output_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(decoding.Decoding._dispatch_decoding_failure),
                hsm.target("../idle"),
            ),
        ),
    )


__all__ = [
    "DESCRIBE_INSTRUCTIONS",
    "MAX_DESCRIPTION_CHARS",
    "Describer",
    "Description",
    "Error",
    "GenerativeDescriber",
]
