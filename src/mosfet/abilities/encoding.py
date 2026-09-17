from . import ability

import abc
import collections.abc
import dataclasses
import typing

import hsm
import mosfet

from mosfet.telemetry import observer


class Encoder(abc.ABC, typing.Generic[ability.TInput, ability.TOutput]):
    """Encoder interface."""

    @abc.abstractmethod
    def encode(self, input: ability.TInput) -> collections.abc.Awaitable[ability.TOutput]:
        """EncodeData the input into the output."""
        ...


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
        hsm.observe(observer),
    )

    def __init__(
        self,
        *,
        encoder: Encoder[ability.TInput, ability.TOutput],
    ) -> None:
        super().__init__()
        self.encoder = encoder

    def _apply(self, ctx: hsm.Context, input: ability.TInput) -> collections.abc.Awaitable[ability.TOutput]:
        del ctx
        return self.encoder.encode(input)
