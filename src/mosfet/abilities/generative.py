from . import ability

import abc
import collections.abc
import dataclasses
import typing

import hsm
import mosfet

from mosfet.telemetry import observer


class Generator(abc.ABC, typing.Generic[ability.TInput, ability.TOutput]):
    """Generator interface."""

    @abc.abstractmethod
    def generate(self, input: ability.TInput) -> collections.abc.Awaitable[ability.TOutput]:
        """Generate output from the input."""
        ...


# Generic envelope (see the intentional-exception note on ability.InputEvent): this base is
# generic over TInput/TOutput, so subclasses narrow input/output schemas and validate payloads
# through input_data_type/output_data_type.
_GenerativeApplyCompletedEvent = hsm.Event[object](
    name="bot.ability.generative.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=object,
)
_GenerativeApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.generative.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class Generative(ability.Ability[ability.TInput, ability.TOutput]):
    """Generative ability."""

    generator: Generator[ability.TInput, ability.TOutput]
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = object
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = object
    input_event: typing.ClassVar[hsm.Event[typing.Any]] = hsm.Event[object](
        name="bot.ability.generative.input",
        schema=object,
    )
    output_event: typing.ClassVar[hsm.Event[typing.Any]] = hsm.Event[object](
        name="bot.ability.generative.output",
        schema=object,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _GenerativeApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _GenerativeApplyFailedEvent

    @staticmethod
    def _has_generative_failure(
        ctx: hsm.Context,
        instance: "Generative[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, ability.FailureData)

    @staticmethod
    def _dispatch_generative_output(
        ctx: hsm.Context,
        instance: "Generative[object, object]",
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
    def _dispatch_generative_failure(
        ctx: hsm.Context,
        instance: "Generative[typing.Any, typing.Any]",
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
        instance: "Generative[object, object]",
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
        "Generative",
        hsm.initial(hsm.target("/Generative/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.target("/Generative/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(_run_behavior_activity),
            hsm.transition(
                hsm.on(_GenerativeApplyCompletedEvent),
                hsm.effect(_dispatch_generative_output),
                hsm.target("/Generative/idle"),
            ),
            hsm.transition(
                hsm.on(_GenerativeApplyFailedEvent),
                hsm.guard(_has_generative_failure),
                hsm.effect(_dispatch_generative_failure),
                hsm.target("/Generative/idle"),
            ),
        ),
        hsm.observe(observer),
    )

    def __init__(
        self,
        *,
        generator: Generator[ability.TInput, ability.TOutput],
    ) -> None:
        super().__init__()
        self.generator = generator

    def _apply(self, ctx: hsm.Context, input: ability.TInput) -> collections.abc.Awaitable[ability.TOutput]:
        del ctx
        return self.generator.generate(input)
