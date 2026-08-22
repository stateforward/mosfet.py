"""Identity: a bot's sense of who it is.

A bot is born nameless. Nothing constructs it with a name, no environment variable supplies one,
and no code reads a name out of what it hears. A name arrives the way it does for a person:
someone in the environment says one, and the bot may take it — :data:`AdoptEvent` is the modeled,
model-callable act of taking it. Hearing words is perception; deciding those words name you is not,
and that decision lives on the other side of that event.

Once a bot has a name it can notice being addressed. That noticing is perception again — a lookup
for a token it already has, in :mod:`bot.abilities.identity.recognition` — and its product is a
stimulus, not an action: "I was addressed." What being addressed *means* is cognition's, exactly as
an interrupt may request attention while its meaning stays in cognition.

A bot that is told a name and does not adopt it is working correctly, and so is a bot that hears
its name and does nothing. Both are decisions this ability makes available and never makes.
"""

from __future__ import annotations

from .. import ability
from .. import memory
from . import recognition
from .name import Name, adopted_name, name_insert_input, name_select_input

import dataclasses
import typing
import uuid

import hsm
import bot
import pydantic

from bot import event
from bot.abilities import cognition
from bot.environment import SoundData, SoundEvent
from bot.telemetry import observer


class AdoptData(pydantic.BaseModel):
    """Take a name as your own.

    No example name, deliberately — the same reason a phone number has none. A name here is a
    complete valid answer in the one field that decides it, so a bot unsure what it just heard
    adopts the schema's name and then answers to it for the rest of its life.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Adopt a name as your own. Select this when a name has been offered to you and you choose "
                "to answer to it. You are not required to: hearing a name offered is not the same as "
                "taking it, and declining leaves you nameless, which is a normal state. Adopting replaces "
                "any name you already answer to. Once adopted, you can tell when you are being addressed "
                "by that name; until then you only hear words. This does not speak, answer, or act on "
                "anything else."
            ),
        },
    )

    name: str = pydantic.Field(
        min_length=1,
        max_length=64,
        description=(
            "The name to answer to, exactly as it was given: the name itself and nothing around it — not "
            "the sentence it arrived in, not a greeting or title attached to it, and not a description of "
            "it. It can only come from what was actually said to you, because a name you were not given "
            "names nobody and answering to it makes you answer to the wrong thing."
        ),
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional short reason this name was adopted.",
        examples=["The caller addressed me by it and asked me to use it."],
    )


AdoptEvent = hsm.Event[AdoptData](
    name="bot.ability.identity.adopt",
    kind=event.EventKind,
    schema=AdoptData,
)


class AddressedData(pydantic.BaseModel):
    """The bot's own name was heard: it was being addressed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Stimulus product: the bot heard the name it answers to. It carries no words and no "
                "audio — what was said arrives separately from listening, and what being addressed "
                "means is the bot's to decide."
            ),
        },
    )

    # Carries no example name either. This product is read rather than filled in, but it is read on
    # the same turns AdoptData is offered, and a name shown here is a name available to adopt.
    name: str = pydantic.Field(
        min_length=1,
        description="The adopted name that was recognized in the heard sound.",
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional recognizer confidence for the recognition, normalized from 0.0 to 1.0.",
        examples=[0.82],
    )


AddressedEvent = hsm.Event[AddressedData](
    name="bot.ability.identity.addressed",
    schema=AddressedData,
)


class _RecalledData(pydantic.BaseModel):
    """Private bring-up result: the name this bot woke up with, if any."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    name: Name | None = None


class _AdoptedData(pydantic.BaseModel):
    """Private completion: the name was taken and (when there is memory) written down."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    name: Name


class _RecognizedData(pydantic.BaseModel):
    """Private completion carrying one recognition decision about one heard chunk."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    name: str = pydantic.Field(min_length=1)
    recognized: recognition.OutputData


_RecalledEvent = hsm.Event[_RecalledData](
    name="bot.ability.identity.recalled",
    kind=hsm.CompletionEventKind,
    schema=_RecalledData,
)
_AdoptedEvent = hsm.Event[_AdoptedData](
    name="bot.ability.identity.adopted",
    kind=hsm.CompletionEventKind,
    schema=_AdoptedData,
)
_AdoptFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.identity.adopt.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)
_RecognizedEvent = hsm.Event[_RecognizedData](
    name="bot.ability.identity.recognized",
    kind=hsm.CompletionEventKind,
    schema=_RecognizedData,
)
_RecognitionFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.identity.recognition.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _with_context(event: hsm.Event[typing.Any], source: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
    """Copy operation id, acoustic provenance, and metadata along the private event chain."""

    operation_id = source.id if source.id else uuid.uuid4().hex
    carried = event.with_data_and_id(event.data, operation_id)
    return dataclasses.replace(
        carried,
        source=source.source or event.source,
        metadata=dict(source.metadata),
    )


def _dispatch_terminal_failure(
    ctx: hsm.Context,
    instance: "Identity",
    source: hsm.Event[typing.Any],
    message: str,
) -> None:
    terminal = _with_context(instance.failed_event.with_data(ability.FailureData(message=message)), source)
    terminal = dataclasses.replace(terminal, source=hsm.id(instance))
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


def _has_recall(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, _RecalledData)


def _is_adoption(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, AdoptData)


def _has_adoption(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, _AdoptedData)


def _has_failure(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


def _was_addressed(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, _RecognizedData) and data.recognized.addressed


def _was_not_addressed(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, _RecognizedData) and not data.recognized.addressed


class Identity(ability.Ability[SoundData, cognition.InputData]):
    """Input ability that knows the bot's own name and notices it being addressed.

    Public input is ``environment.sound``, the same stimulus every other input ability is fanned.
    The success terminal is a cognitive input whose stimulus is :data:`AddressedEvent`. Sound that
    does not carry the bot's name produces nothing, and a bot that has adopted no name produces
    nothing at all — it simply hears words. That asymmetry is the whole model: recognition needs
    something to recognize, and only adoption supplies it.
    """

    input_event: typing.ClassVar[hsm.Event[SoundData]] = SoundEvent
    output_event: typing.ClassVar[hsm.Event[cognition.InputData]] = cognition.InputEvent

    _recognizer: recognition.NameRecognizer
    _memory: memory.Memory | None
    _name: Name | None

    def __init__(
        self,
        *,
        recognizer: recognition.NameRecognizer,
        memory: memory.Memory | None = None,
    ) -> None:
        super().__init__()
        self._recognizer = recognizer
        # Not owned: the bot's memory is attached elsewhere (Ability attachment is exclusive), so
        # identity uses it as a store the way Reasoning does rather than taking its lifecycle.
        self._memory = memory
        self._name = None

    @staticmethod
    async def _recall_name(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> None:
        """Wake up remembering the name, when there is a memory to have remembered it in.

        A bot with no memory ability is nameless on every bring-up. That is not an error: it is
        what having no memory means.
        """

        store = instance._memory
        recalled: Name | None = None
        if store is not None:
            try:
                output = store.execute(name_select_input())
            except Exception as error:
                _dispatch_terminal_failure(
                    ctx,
                    instance,
                    event,
                    f"Identity could not recall a name from memory: {error}",
                )
            else:
                recalled = adopted_name(output)
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_context(_RecalledEvent.with_data(_RecalledData(name=recalled)), event),
        )

    @staticmethod
    async def _adopt_name(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> None:
        """Take the selected name and write it down before answering to it.

        Fail-closed: a name that could not be written down is a name the bot did not take, and the
        failure says so. Cognition can select adoption again.
        """

        request = event.data
        if not isinstance(request, AdoptData):
            # Unreachable: paired with hsm.guard(_is_adoption), which narrows this payload.
            return
        adopting = Name(text=request.name)
        store = instance._memory
        if store is not None:
            try:
                _ = store.execute(name_insert_input(adopting))
            except Exception as error:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    _with_context(
                        _AdoptFailedEvent.with_data(
                            ability.FailureData(message=f"Identity could not remember the name it took: {error}")
                        ),
                        event,
                    ),
                )
                return
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_context(_AdoptedEvent.with_data(_AdoptedData(name=adopting)), event),
        )

    @staticmethod
    async def _run_recognition(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> None:
        heard = event.data
        known = instance._name
        if not isinstance(heard, SoundData) or known is None:
            # Unreachable: paired with hsm.guard(_can_recognize), which narrows both.
            return
        try:
            recognized = await instance._recognizer.classify(recognition.InputData(heard=heard, name=known.text))
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                _with_context(
                    _RecognitionFailedEvent.with_data(ability.FailureData(message=str(error))),
                    event,
                ),
            )
            return
        if not isinstance(recognized, recognition.OutputData):
            _ = hsm.dispatch(
                ctx,
                instance,
                _with_context(
                    _RecognitionFailedEvent.with_data(
                        ability.FailureData(
                            message="Identity recognizer produced output that does not match its output schema."
                        )
                    ),
                    event,
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_context(
                _RecognizedEvent.with_data(_RecognizedData(name=known.text, recognized=recognized)),
                event,
            ),
        )

    @staticmethod
    def _can_recognize(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> bool:
        """Only a bot that has a name can be looking for it."""

        del ctx
        return isinstance(event.data, SoundData) and instance._name is not None

    @staticmethod
    def _remember_recalled(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        if not isinstance(data, _RecalledData):
            # Unreachable: paired with hsm.guard(_has_recall), which narrows this payload.
            return
        instance._name = data.name

    @staticmethod
    def _remember_adopted(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        if not isinstance(data, _AdoptedData):
            # Unreachable: paired with hsm.guard(_has_adoption), which narrows this payload.
            return
        instance._name = data.name

    @staticmethod
    def _dispatch_addressed(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> None:
        """Hand the body a cognition input whose stimulus is "I was addressed"."""

        data = event.data
        if not isinstance(data, _RecognizedData):
            # Unreachable: paired with hsm.guard(_was_addressed), which narrows this payload.
            return
        stimulus = _with_context(
            AddressedEvent.with_data(AddressedData(name=data.name, confidence=data.recognized.confidence)),
            event,
        )
        handoff = _with_context(cognition.InputEvent.with_data(cognition.InputData(stimulus=stimulus)), event)
        handoff = dataclasses.replace(handoff, source=hsm.id(instance))
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(handoff))

    @staticmethod
    def _fail_from_stage(ctx: hsm.Context, instance: "Identity", event: hsm.Event[typing.Any]) -> None:
        failure = event.data
        if not isinstance(failure, ability.FailureData):
            # Unreachable: paired with hsm.guard(_has_failure), which narrows this payload.
            return
        _dispatch_terminal_failure(ctx, instance, event, failure.message)

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "Identity",
        hsm.initial(hsm.target("/Identity/recalling")),
        hsm.state(
            "recalling",
            hsm.defer(input_event, AdoptEvent),
            hsm.activity(_recall_name),
            hsm.transition(
                hsm.on(_RecalledEvent),
                hsm.guard(_has_recall),
                hsm.effect(_remember_recalled),
                hsm.target("/Identity/knowing"),
            ),
        ),
        hsm.state(
            "knowing",
            hsm.initial(hsm.target("/Identity/knowing/listening")),
            # Adoption lives on the composite so it is offered for the whole time the ability is
            # working, not only between recognitions — a bot can be given a name mid-sentence.
            hsm.transition(
                hsm.on(AdoptEvent),
                hsm.guard(_is_adoption),
                hsm.target("/Identity/adopting"),
            ),
            hsm.state(
                "listening",
                # Sound with no name to look for has no transition here at all: a nameless bot
                # hears words and nothing follows from them.
                hsm.transition(
                    hsm.on(input_event),
                    hsm.guard(_can_recognize),
                    hsm.target("/Identity/knowing/recognizing"),
                ),
            ),
            hsm.state(
                "recognizing",
                hsm.defer(input_event),
                hsm.activity(_run_recognition),
                hsm.transition(
                    hsm.on(_RecognizedEvent),
                    hsm.guard(_was_addressed),
                    hsm.effect(_dispatch_addressed),
                    hsm.target("/Identity/knowing/listening"),
                ),
                # Not addressed is a real outcome and produces nothing. Cognition is never given a
                # turn about every sound that failed to be the bot's name.
                hsm.transition(
                    hsm.on(_RecognizedEvent),
                    hsm.guard(_was_not_addressed),
                    hsm.target("/Identity/knowing/listening"),
                ),
                hsm.transition(
                    hsm.on(_RecognitionFailedEvent),
                    hsm.guard(_has_failure),
                    hsm.effect(_fail_from_stage),
                    hsm.target("/Identity/knowing/listening"),
                ),
            ),
        ),
        hsm.state(
            "adopting",
            hsm.defer(input_event, AdoptEvent),
            hsm.activity(_adopt_name),
            hsm.transition(
                hsm.on(_AdoptedEvent),
                hsm.guard(_has_adoption),
                hsm.effect(_remember_adopted),
                hsm.target("/Identity/knowing"),
            ),
            hsm.transition(
                hsm.on(_AdoptFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Identity/knowing"),
            ),
        ),
        hsm.observe(observer),
    )


__all__ = [
    "AddressedData",
    "AddressedEvent",
    "AdoptData",
    "AdoptEvent",
    "Identity",
]
