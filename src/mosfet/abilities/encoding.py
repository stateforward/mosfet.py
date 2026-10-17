from . import ability

import abc
import collections.abc
import dataclasses
import typing

import hsm
import mosfet


class Encoder(abc.ABC, typing.Generic[ability.TInput, ability.TOutput]):
    """Encoder interface."""

    @abc.abstractmethod
    def encode(self, input: ability.TInput) -> collections.abc.Awaitable[ability.TOutput]:
        """EncodeData the input into the output."""
        ...


class StreamEncoder(abc.ABC, typing.Generic[ability.TInput, ability.TOutput]):
    """Streaming encoder interface: a stream of input in, a stream of output out as it is produced.

    A sibling of :class:`Encoder`, not a form of it: one value in and one out is a different contract from a
    stream both ways.
    """

    @abc.abstractmethod
    def encode(
        self, input: collections.abc.AsyncIterable[ability.TInput]
    ) -> collections.abc.AsyncIterator[ability.TOutput]:
        """Encode ``input`` as it is read, yielding output as it is produced.

        The encoder reads ``input`` at its own pace; ``input`` ending means the input is final. Closing the
        returned iterator before it is exhausted (``aclose``) cancels the work and stops reading ``input``.

        Raises:
            Error: The encoding failed; ``kind`` says why.
        """
        ...


type FailureKind = typing.Literal["invalid_input", "unavailable", "timeout", "invalid_response", "encode_failed"]
"""Why an encoding failed, in existing ``FAILURE_KINDS.md`` strings.

``invalid_input`` the input was refused; ``unavailable`` the encoder's service is not serving; ``timeout`` it did
not answer in time; ``invalid_response`` it returned output outside its contract; ``encode_failed`` the encoding
failed for no narrower reason.
"""


class Error(Exception):
    """An encoder failure with a closed, low-cardinality kind. Any encoder may raise it."""

    kind: FailureKind
    failure_kind: str
    """The same string as ``kind``, under the name ``telemetry.span.failure_kind`` reads."""

    def __init__(self, kind: FailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.failure_kind = kind


# Generic envelope (see the intentional-exception note on ability.InputEvent): this base is
# generic over TInput/TOutput, so subclasses narrow input/output schemas and validate payloads
# through input_data_type/output_data_type.
_EncodingApplyCompletedEvent = hsm.Event[object](
    name="bot.ability.encoding.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=object,
)
_EncodingApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.encoding.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class Encoding(ability.Ability[ability.TInput, ability.TOutput]):
    """Encoding ability."""

    encoder: Encoder[ability.TInput, ability.TOutput]
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = object
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = object
    input_event: typing.ClassVar[hsm.Event[typing.Any]] = hsm.Event[object](
        name="bot.ability.encoding.input",
        schema=object,
    )
    output_event: typing.ClassVar[hsm.Event[typing.Any]] = hsm.Event[object](
        name="bot.ability.encoding.output",
        schema=object,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _EncodingApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _EncodingApplyFailedEvent

    @staticmethod
    def _has_encoding_failure(
        ctx: hsm.Context,
        instance: "Encoding[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, ability.FailureData)

    @staticmethod
    def _dispatch_encoding_output(
        ctx: hsm.Context,
        instance: "Encoding[object, object]",
        event: hsm.Event[object],
    ) -> None:
        output = event.data
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _dispatch_encoding_failure(
        ctx: hsm.Context,
        instance: "Encoding[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    async def _run_behavior_activity(
        ctx: hsm.Context,
        instance: "Encoding[object, object]",
        event: hsm.Event[object],
    ) -> None:
        try:
            output = await instance._apply(ctx, event.data)
        except Exception as error:
            _ = instance.dispatch(
                ctx,
                dataclasses.replace(
                    instance._apply_failed_event.with_data(ability.FailureData(message=str(error))),
                    id=event.id or None,
                    source=event.source,
                    target=event.target,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = instance.dispatch(
            ctx,
            dataclasses.replace(
                instance._apply_completed_event.with_data(output),
                id=event.id or None,
                source=event.source,
                target=event.target,
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Encoding",
        hsm.initial(hsm.target("/Encoding/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.target("/Encoding/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(_run_behavior_activity),
            hsm.transition(
                hsm.on(_EncodingApplyCompletedEvent),
                hsm.effect(_dispatch_encoding_output),
                hsm.target("/Encoding/idle"),
            ),
            hsm.transition(
                hsm.on(_EncodingApplyFailedEvent),
                hsm.guard(_has_encoding_failure),
                hsm.effect(_dispatch_encoding_failure),
                hsm.target("/Encoding/idle"),
            ),
        ),
    )

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        encoder: Encoder[ability.TInput, ability.TOutput],
    ) -> None:
        super().__init__(id=id, clock=clock)
        self.encoder = encoder

    def _apply(self, ctx: hsm.Context, input: ability.TInput) -> collections.abc.Awaitable[ability.TOutput]:
        del ctx
        return self.encoder.encode(input)
