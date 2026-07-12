from . import ability

import abc
import collections.abc
import dataclasses
import typing

import hsm

from bot.telemetry import observer


class Classifier(abc.ABC, typing.Generic[ability.TInput, ability.TOutput]):
    """Classifier interface."""

    @abc.abstractmethod
    def classify(self, input: ability.TInput) -> collections.abc.Awaitable[ability.TOutput]:
        """Classify the input into the output."""
        ...


_ClassifyingApplyCompletedEvent = hsm.Event[object](
    name="bot.ability.classifying.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=object,
)
_ClassifyingApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.classifying.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class Classifying(ability.Ability[ability.TInput, ability.TOutput]):
    """Classifying ability."""

    classifier: Classifier[ability.TInput, ability.TOutput]
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = object
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = object
    input_event: typing.ClassVar[hsm.Event[typing.Any]] = ability.ability_input_event(
        name="bot.ability.classifying.input",
        data_type=object,
        description="InputData event data to classify.",
        examples=["classification input"],
    )
    output_event: typing.ClassVar[hsm.Event[typing.Any]] = ability.ability_output_event(
        name="bot.ability.classifying.output",
        data_type=object,
        description="OutputData event data produced by classification.",
        examples=["classification output"],
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _ClassifyingApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _ClassifyingApplyFailedEvent

    @staticmethod
    def _has_classifying_failure(
        ctx: hsm.Context,
        instance: "Classifying[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, ability.FailureData)

    @staticmethod
    def _dispatch_classifying_output(
        ctx: hsm.Context,
        instance: "Classifying[object, object]",
        event: hsm.Event[object],
    ) -> None:
        output = event.data
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _dispatch_classifying_failure(
        ctx: hsm.Context,
        instance: "Classifying[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    async def _run_behavior_activity(
        ctx: hsm.Context,
        instance: "Classifying[object, object]",
        event: hsm.Event[object],
    ) -> None:
        try:
            output = await instance._apply(ctx, event.data)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    instance._apply_failed_event.with_data(ability.FailureData(message=str(error))),
                    id=event.id or None,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                instance._apply_completed_event.with_data(output),
                id=event.id or None,
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Classifying",
        hsm.initial(hsm.target("/Classifying/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.target("/Classifying/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(_run_behavior_activity),
            hsm.transition(
                hsm.on(_ClassifyingApplyCompletedEvent),
                hsm.effect(_dispatch_classifying_output),
                hsm.target("/Classifying/idle"),
            ),
            hsm.transition(
                hsm.on(_ClassifyingApplyFailedEvent),
                hsm.guard(_has_classifying_failure),
                hsm.effect(_dispatch_classifying_failure),
                hsm.target("/Classifying/idle"),
            ),
        ),
        hsm.observe(observer),
    )

    def __init__(
        self,
        *,
        classifier: Classifier[ability.TInput, ability.TOutput],
    ) -> None:
        super().__init__()
        self.classifier = classifier

    def _apply(self, ctx: hsm.Context, input: ability.TInput) -> collections.abc.Awaitable[ability.TOutput]:
        del ctx
        return self.classifier.classify(input)
