"""Executable HSM behavior for compiled behaviors.

Behaviors compile to event-only HSM machines: Starlark callbacks may dispatch and process
declared events, never bound abilities.
"""

from . import source
from . import schema
from . import runtime
from bot.abilities import ability

import collections.abc
import asyncio
import dataclasses
import datetime
import hashlib
import re
import typing
import uuid

import hsm
import bot
import pydantic

from bot import abilities

from bot.telemetry import observer


class _OperationalEventData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    name: str
    data: object
    kind: int
    id: str
    source: str
    target: str
    metadata: dict[str, object] = pydantic.Field(default_factory=dict)


class _GuardEvaluationRequestedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    original_event: _OperationalEventData
    candidate_id: str
    callback: str


class _GuardEvaluationResultData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    request: _GuardEvaluationRequestedData
    message: str = ""


class _CallbackExecutionRequestedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    original_event: _OperationalEventData
    callbacks: tuple[str, ...]
    settle: bool = False


class _CallbackExecutionResultData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    request: _CallbackExecutionRequestedData
    message: str = ""


class _BehaviorApplySettledData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str


_GuardEvaluationRequestedEvent = hsm.Event[_GuardEvaluationRequestedData](
    name="bot.behavior.guard.evaluation.requested",
    schema=_GuardEvaluationRequestedData,
)
_GuardEvaluationAcceptedEvent = hsm.Event[_GuardEvaluationResultData](
    name="bot.behavior.guard.evaluation.accepted",
    schema=_GuardEvaluationResultData,
)
_GuardEvaluationRejectedEvent = hsm.Event[_GuardEvaluationResultData](
    name="bot.behavior.guard.evaluation.rejected",
    schema=_GuardEvaluationResultData,
)
_GuardEvaluationFailedEvent = hsm.Event[_GuardEvaluationResultData](
    name="bot.behavior.guard.evaluation.failed",
    schema=_GuardEvaluationResultData,
)
_BehaviorApplySettledEvent = hsm.Event[_BehaviorApplySettledData](
    name="bot.behavior.apply.settled",
    schema=_BehaviorApplySettledData,
)
_CallbackExecutionRequestedEvent = hsm.Event[_CallbackExecutionRequestedData](
    name="bot.behavior.callback.execution.requested",
    schema=_CallbackExecutionRequestedData,
)
_CallbackExecutionCompletedEvent = hsm.Event[_CallbackExecutionResultData](
    name="bot.behavior.callback.execution.completed",
    schema=_CallbackExecutionResultData,
)
_CallbackExecutionFailedEvent = hsm.Event[_CallbackExecutionResultData](
    name="bot.behavior.callback.execution.failed",
    schema=_CallbackExecutionResultData,
)
_CallbackCompletedEvent = hsm.Event[_CallbackExecutionResultData](
    name="bot.behavior.callback.completed",
    schema=_CallbackExecutionResultData,
)
_CallbackFailedEvent = hsm.Event[_CallbackExecutionResultData](
    name="bot.behavior.callback.failed",
    schema=_CallbackExecutionResultData,
)


class _ApplyOperation(hsm.Instance):
    """One-shot apply settlement address; the awaitable is closure-bound."""


class _GuardEvaluator(hsm.Instance):
    _callback_runtime: runtime.CallbackRuntime
    _owner: "Behavior"

    def __init__(self, *, callback_runtime: runtime.CallbackRuntime, owner: "Behavior") -> None:
        super().__init__()
        self._callback_runtime = callback_runtime
        self._owner = owner

    @staticmethod
    async def evaluate(
        ctx: hsm.Context,
        instance: "_GuardEvaluator",
        event: hsm.Event[_GuardEvaluationRequestedData],
    ) -> None:
        request = event.data
        assert isinstance(request, _GuardEvaluationRequestedData)
        try:
            admitted = await instance._callback_runtime.evaluate_guard(
                callback=request.callback,
                event=_operational_event(request.original_event),
                behavior_id=hsm.id(instance._owner),
            )
        except runtime.CallbackError as error:
            completion = _GuardEvaluationFailedEvent.with_data(
                _GuardEvaluationResultData(request=request, message=str(error))
            )
        else:
            completion_event = _GuardEvaluationAcceptedEvent if admitted else _GuardEvaluationRejectedEvent
            completion = completion_event.with_data(_GuardEvaluationResultData(request=request))
        await hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                completion,
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def forward(
        outcome: hsm.Event[_GuardEvaluationResultData],
    ) -> collections.abc.Callable[[hsm.Context, "_GuardEvaluator", hsm.Event[_GuardEvaluationResultData]], None]:
        def forward_outcome(
            ctx: hsm.Context,
            instance: "_GuardEvaluator",
            event: hsm.Event[_GuardEvaluationResultData],
        ) -> None:
            result = event.data
            assert isinstance(result, _GuardEvaluationResultData)
            owner_outcome = _guard_outcome_event(result.request.candidate_id, outcome.name)
            _ = hsm.dispatch_to(
                ctx,
                dataclasses.replace(
                    owner_outcome.with_data(result),
                    id=event.id,
                    source=hsm.id(instance),
                    target=hsm.id(instance._owner),
                    metadata=dict(event.metadata),
                ),
                hsm.id(instance._owner),
            )

        return forward_outcome

    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "BehaviorGuardEvaluator",
        hsm.initial(hsm.target("/BehaviorGuardEvaluator/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(_GuardEvaluationRequestedEvent),
                hsm.target("/BehaviorGuardEvaluator/evaluating"),
            ),
        ),
        hsm.state(
            "evaluating",
            hsm.activity(evaluate.__get__(None, object)),
            hsm.defer(_GuardEvaluationRequestedEvent),
            hsm.transition(
                hsm.on(_GuardEvaluationAcceptedEvent),
                hsm.effect(forward(_GuardEvaluationAcceptedEvent)),
                hsm.target("/BehaviorGuardEvaluator/idle"),
            ),
            hsm.transition(
                hsm.on(_GuardEvaluationRejectedEvent),
                hsm.effect(forward(_GuardEvaluationRejectedEvent)),
                hsm.target("/BehaviorGuardEvaluator/idle"),
            ),
            hsm.transition(
                hsm.on(_GuardEvaluationFailedEvent),
                hsm.effect(forward(_GuardEvaluationFailedEvent)),
                hsm.target("/BehaviorGuardEvaluator/idle"),
            ),
        ),
    )


class _CallbackExecutor(hsm.Instance):
    _callback_runtime: runtime.CallbackRuntime
    _owner: "Behavior"

    def __init__(self, *, callback_runtime: runtime.CallbackRuntime, owner: "Behavior") -> None:
        super().__init__()
        self._callback_runtime = callback_runtime
        self._owner = owner

    @staticmethod
    async def execute(
        ctx: hsm.Context,
        instance: "_CallbackExecutor",
        event: hsm.Event[_CallbackExecutionRequestedData],
    ) -> None:
        request = event.data
        assert isinstance(request, _CallbackExecutionRequestedData)
        original_event = _operational_event(request.original_event)
        try:
            for callback in request.callbacks:
                await instance._callback_runtime.effect(callback, ctx, instance._owner, original_event)
        except runtime.CallbackError as error:
            completion = _CallbackExecutionFailedEvent.with_data(
                _CallbackExecutionResultData(request=request, message=str(error))
            )
        else:
            completion = _CallbackExecutionCompletedEvent.with_data(_CallbackExecutionResultData(request=request))
        await hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                completion,
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def forward(
        outcome: hsm.Event[_CallbackExecutionResultData],
    ) -> collections.abc.Callable[[hsm.Context, "_CallbackExecutor", hsm.Event[_CallbackExecutionResultData]], None]:
        def forward_outcome(
            ctx: hsm.Context,
            instance: "_CallbackExecutor",
            event: hsm.Event[_CallbackExecutionResultData],
        ) -> None:
            _ = hsm.dispatch_to(
                ctx,
                dataclasses.replace(
                    outcome.with_data(event.data),
                    id=event.id,
                    source=hsm.id(instance),
                    target=hsm.id(instance._owner),
                    metadata=dict(event.metadata),
                ),
                hsm.id(instance._owner),
            )

        return forward_outcome

    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "BehaviorCallbackExecutor",
        hsm.initial(hsm.target("/BehaviorCallbackExecutor/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(_CallbackExecutionRequestedEvent),
                hsm.target("/BehaviorCallbackExecutor/executing"),
            ),
        ),
        hsm.state(
            "executing",
            hsm.activity(execute.__get__(None, object)),
            hsm.defer(_CallbackExecutionRequestedEvent),
            hsm.transition(
                hsm.on(_CallbackExecutionCompletedEvent),
                hsm.effect(forward(_CallbackCompletedEvent)),
                hsm.target("/BehaviorCallbackExecutor/idle"),
            ),
            hsm.transition(
                hsm.on(_CallbackExecutionFailedEvent),
                hsm.effect(forward(_CallbackFailedEvent)),
                hsm.target("/BehaviorCallbackExecutor/idle"),
            ),
        ),
    )


class Behavior(abilities.Ability[object, object]):
    """Executable behavior compiled from Starlark source into HSM-visible behavior."""

    input_event: typing.ClassVar[hsm.Event[object]]
    output_event: typing.ClassVar[hsm.Event[object]]
    failed_event: typing.ClassVar[hsm.Event[ability.FailureData]]
    _spec: source.Source
    _callback_runtime: runtime.CallbackRuntime
    _guard_evaluator: _GuardEvaluator
    _callback_executor: _CallbackExecutor
    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "Behavior",
        hsm.initial(hsm.target("/Behavior/idle")),
        hsm.state("idle"),
        hsm.observe(observer),
    )

    @staticmethod
    def request_guard(
        candidate_id: str,
        callback: str,
    ) -> collections.abc.Callable[[hsm.Context, "Behavior", hsm.Event[typing.Any]], None]:
        def request(
            ctx: hsm.Context,
            instance: Behavior,
            event: hsm.Event[typing.Any],
        ) -> None:
            original_event = _callback_event(event)
            evaluator = instance._guard_evaluator
            request_event = dataclasses.replace(
                _GuardEvaluationRequestedEvent.with_data(
                    _GuardEvaluationRequestedData(
                        original_event=_operational_event_data(original_event),
                        candidate_id=candidate_id,
                        callback=callback,
                    )
                ),
                id=original_event.id,
                source=hsm.id(instance),
                target=hsm.id(evaluator),
                metadata=dict(original_event.metadata),
            )
            _ = hsm.dispatch(ctx, evaluator, request_event)

        return request

    @staticmethod
    def reject_guard_failure(
        ctx: hsm.Context,
        instance: "Behavior",
        event: hsm.Event[_GuardEvaluationResultData],
    ) -> None:
        result = event.data
        assert isinstance(result, _GuardEvaluationResultData)
        _dispatch_callback_failure(ctx, instance, _operational_event(result.request.original_event), result.message)

    @staticmethod
    def ignore_guard_rejection(
        ctx: hsm.Context,
        instance: "Behavior",
        event: hsm.Event[_GuardEvaluationResultData],
    ) -> None:
        del ctx, instance, event

    @staticmethod
    def queue_apply_settlement(
        ctx: hsm.Context,
        instance: "Behavior",
        event: hsm.Event[typing.Any],
    ) -> None:
        original_event = _callback_event(event)
        if not original_event.id:
            return
        request = dataclasses.replace(
            _CallbackExecutionRequestedEvent.with_data(
                _CallbackExecutionRequestedData(
                    original_event=_operational_event_data(original_event),
                    callbacks=(),
                    settle=True,
                )
            ),
            id=original_event.id,
            source=hsm.id(instance),
            target=hsm.id(instance._callback_executor),
            metadata=dict(original_event.metadata),
        )
        _ = hsm.dispatch(ctx, instance._callback_executor, request)

    @staticmethod
    def request_callbacks(
        callbacks: tuple[str, ...],
    ) -> collections.abc.Callable[[hsm.Context, "Behavior", hsm.Event[typing.Any]], None]:
        def request(
            ctx: hsm.Context,
            instance: Behavior,
            event: hsm.Event[typing.Any],
        ) -> None:
            original_event = _callback_event(event)
            request_event = dataclasses.replace(
                _CallbackExecutionRequestedEvent.with_data(
                    _CallbackExecutionRequestedData(
                        original_event=_operational_event_data(original_event),
                        callbacks=callbacks,
                    )
                ),
                id=original_event.id,
                source=hsm.id(instance),
                target=hsm.id(instance._callback_executor),
                metadata=dict(original_event.metadata),
            )
            _ = hsm.dispatch(ctx, instance._callback_executor, request_event)

        return request

    @staticmethod
    def complete_callback_execution(
        ctx: hsm.Context,
        instance: "Behavior",
        event: hsm.Event[_CallbackExecutionResultData],
    ) -> None:
        result = event.data
        assert isinstance(result, _CallbackExecutionResultData)
        if not result.request.settle:
            return
        original_event = _operational_event(result.request.original_event)
        if not original_event.source:
            return
        settlement = dataclasses.replace(
            _BehaviorApplySettledEvent.with_data(_BehaviorApplySettledData(operation_id=original_event.id)),
            id=original_event.id,
            source=hsm.id(instance),
            target=original_event.source,
            metadata=dict(original_event.metadata),
        )
        _ = hsm.dispatch_to(ctx, settlement, original_event.source)

    @staticmethod
    def fail_callback_execution(
        ctx: hsm.Context,
        instance: "Behavior",
        event: hsm.Event[_CallbackExecutionResultData],
    ) -> None:
        result = event.data
        assert isinstance(result, _CallbackExecutionResultData)
        original_event = _operational_event(result.request.original_event)
        _dispatch_callback_failure(ctx, instance, original_event, result.message)
        if result.request.settle:
            Behavior.complete_callback_execution(ctx, instance, event)

    @staticmethod
    def input_valid(ctx: hsm.Context, instance: "Behavior", event: hsm.Event[object]) -> bool:
        del ctx
        return schema.matches_json_schema(event.data, instance._spec.input_event.payload_schema)

    @staticmethod
    def input_invalid(ctx: hsm.Context, instance: "Behavior", event: hsm.Event[object]) -> bool:
        return not Behavior.input_valid(ctx, instance, event)

    @staticmethod
    def dispatch_input_schema_failure(
        ctx: hsm.Context,
        instance: "Behavior",
        event: hsm.Event[object],
    ) -> None:
        _dispatch_callback_failure(
            ctx, instance, event, f"{instance._spec.name} input does not match its input schema."
        )

    @staticmethod
    def starlark_effect(
        *callbacks: str,
    ) -> collections.abc.Callable[[hsm.Context, "Behavior", hsm.Event[object]], None]:
        effect = Behavior.request_callbacks(tuple(callbacks))
        effect.__name__ = f"_behavior_effect_{'_'.join(callbacks)}"
        return effect

    @staticmethod
    def starlark_entry(
        *callbacks: str,
    ) -> collections.abc.Callable[[hsm.Context, "Behavior", hsm.Event[object]], None]:
        return Behavior._starlark_lifecycle(*callbacks, stage="entry")

    @staticmethod
    def starlark_exit(
        *callbacks: str,
    ) -> collections.abc.Callable[[hsm.Context, "Behavior", hsm.Event[object]], None]:
        return Behavior._starlark_lifecycle(*callbacks, stage="exit")

    @staticmethod
    def _starlark_lifecycle(
        *callbacks: str,
        stage: typing.Literal["entry", "exit"],
    ) -> collections.abc.Callable[[hsm.Context, "Behavior", hsm.Event[object]], None]:
        def lifecycle(
            ctx: hsm.Context,
            instance: Behavior,
            event: hsm.Event[object],
        ) -> None:
            original_event = _callback_event(event)
            try:
                for callback in callbacks:
                    instance._callback_runtime.lifecycle(
                        callback,
                        ctx,
                        instance,
                        original_event,
                        stage=stage,
                    )
            except runtime.CallbackError as error:
                _dispatch_callback_failure(ctx, instance, original_event, str(error))

        lifecycle.__name__ = f"_behavior_{stage}_{'_'.join(callbacks)}"
        return lifecycle

    @staticmethod
    def starlark_activity(
        callback: str,
    ) -> collections.abc.Callable[[hsm.Context, "Behavior", hsm.Event[object]], collections.abc.Awaitable[None]]:
        async def activity(ctx: hsm.Context, instance: "Behavior", event: hsm.Event[object]) -> None:
            try:
                await instance._callback_runtime.activity(callback, ctx, instance, _callback_event(event))
            except runtime.CallbackError as error:
                _dispatch_callback_failure(ctx, instance, event, str(error))

        activity.__name__ = f"_behavior_activity_{callback}"
        return activity

    def __init__(self, *, spec: source.Source) -> None:
        super().__init__()
        callback_runtime = runtime.CallbackRuntime(
            source=spec.source,
            declared_events=spec.declared_event_specs(failed_event=typing.cast(hsm.Event[object], self.failed_event)),
        )
        self._spec = spec
        self._callback_runtime = callback_runtime
        self._guard_evaluator = _GuardEvaluator(callback_runtime=callback_runtime, owner=self)
        self._callback_executor = _CallbackExecutor(callback_runtime=callback_runtime, owner=self)

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        await self._callback_runtime.warm()
        instance = await super().start(ctx, data)
        evaluator_model = self._guard_evaluator.model
        if evaluator_model is None:
            raise RuntimeError("Behavior guard evaluator has no model.")
        executor_model = self._callback_executor.model
        if executor_model is None:
            raise RuntimeError("Behavior callback executor has no model.")
        _ = await bot.started(self.context(), self._guard_evaluator, evaluator_model)
        _ = await bot.started(self.context(), self._callback_executor, executor_model)
        initial_event = hsm.Event[object](name="bot.behavior.initializing", target=hsm.id(self))
        for callback in _root_initial_callbacks(self._spec.model):
            await self._callback_runtime.effect(callback, self.context(), self, initial_event)
        return instance

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        await self._callback_executor.stop(ctx)
        await self._guard_evaluator.stop(ctx)
        await super().stop(ctx)

    @typing.override
    def apply(self, input: object, *, ctx: hsm.Context | None = None) -> collections.abc.Awaitable[None]:
        """Dispatch one behavior operation and wait for its correlated settlement.

        Cancellation only stops this caller's settlement wait. Once dispatch commits,
        the HSM and its ordered callback executor continue the operation and may still
        emit modeled outputs or failures. The operation id isolates late settlement
        cleanup so it cannot satisfy a later ``apply`` call.
        """

        operation_id = uuid.uuid4().hex
        dispatch_context = self.context() if ctx is None else ctx

        async def dispatch_and_wait() -> None:
            settled: asyncio.Future[None] = asyncio.get_running_loop().create_future()

            def is_correlated(
                operation_ctx: hsm.Context,
                instance: _ApplyOperation,
                event: hsm.Event[_BehaviorApplySettledData],
            ) -> bool:
                del operation_ctx
                return (
                    isinstance(event.data, _BehaviorApplySettledData)
                    and event.data.operation_id == operation_id
                    and event.id == operation_id
                    and event.target == hsm.id(instance)
                )

            def mark_settled(
                operation_ctx: hsm.Context,
                instance: _ApplyOperation,
                event: hsm.Event[_BehaviorApplySettledData],
            ) -> None:
                del operation_ctx, instance, event
                if not settled.done():
                    settled.set_result(None)

            operation_model = bot.define(
                "BehaviorApplyOperation",
                hsm.initial(hsm.target("/BehaviorApplyOperation/waiting")),
                hsm.state(
                    "waiting",
                    hsm.transition(
                        hsm.on(_BehaviorApplySettledEvent),
                        hsm.guard(is_correlated),
                        hsm.effect(mark_settled),
                        hsm.target("/BehaviorApplyOperation/completed"),
                    ),
                ),
                hsm.final("completed"),
            )
            operation = _ApplyOperation()
            try:
                _ = await bot.started(self.context(), operation, operation_model)
                await hsm.dispatch(
                    dispatch_context,
                    self,
                    dataclasses.replace(
                        self.input_event.with_data_and_id(input, operation_id),
                        source=hsm.id(operation),
                        target=hsm.id(self),
                    ),
                )
                await settled
            finally:
                await asyncio.shield(operation.stop(self.context()))

        return dispatch_and_wait()


def _snake_case_model_name(name: str) -> str:
    with_boundaries = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return with_boundaries.lower()


def _callback_event(event: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
    data = event.data
    if isinstance(data, _GuardEvaluationResultData):
        return _operational_event(data.request.original_event)
    if isinstance(data, _CallbackExecutionResultData):
        return _operational_event(data.request.original_event)
    return event


def _operational_event_data(event: hsm.Event[typing.Any]) -> _OperationalEventData:
    return _OperationalEventData(
        name=event.name,
        data=event.data,
        kind=event.kind,
        id=event.id,
        source=event.source,
        target=event.target,
        metadata=dict(event.metadata),
    )


def _operational_event(data: _OperationalEventData) -> hsm.Event[object]:
    return hsm.Event[object](
        name=data.name,
        data=data.data,
        kind=data.kind,
        id=data.id,
        source=data.source,
        target=data.target,
        metadata=dict(data.metadata),
    )


def _guard_outcome_name(candidate_id: str, outcome: str) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:16]
    suffix = outcome.rsplit(".", maxsplit=1)[-1]
    return f"bot.behavior.guard.candidate.{digest}.{suffix}"


def _guard_outcome_event(
    candidate_id: str,
    outcome: str,
) -> hsm.Event[_GuardEvaluationResultData]:
    return hsm.Event[_GuardEvaluationResultData](
        name=_guard_outcome_name(candidate_id, outcome),
        schema=_GuardEvaluationResultData,
    )


def generated_event_names(spec: source.Source) -> tuple[str, ...]:
    """Return generated failure event names for a behavior spec."""

    event_prefix = f"bot.behavior.{_snake_case_model_name(spec.name)}"
    return (f"{event_prefix}.failed",)


def validate_event_names(spec: source.Source) -> None:
    """Reject public behavior event names that collide with each other or generated events."""

    source_names = (spec.input_event.name, spec.output_event.name)
    if len(set(source_names)) != len(source_names):
        raise ValueError("Behavior source event names must be unique.")
    generated_names = generated_event_names(spec)
    collisions = tuple(sorted(set(spec.declared_event_specs()).intersection(generated_names)))
    if collisions:
        raise ValueError(
            f"Behavior source event names cannot use generated behavior event names: {', '.join(collisions)}."
        )


def define_model(
    spec: source.Source,
    *,
    input_event: hsm.Event[object],
    output_event: hsm.Event[object],
    failed_event: hsm.Event[ability.FailureData],
) -> hsm.Model:
    """Lower the source-authored Starlark HSM model into a static HSM topology."""

    event_specs = spec.declared_event_specs(failed_event=typing.cast(hsm.Event[object], failed_event))
    event_objects: dict[str, hsm.Event[object] | str] = {
        name: eventspec.hsm_event() for name, eventspec in event_specs.items()
    }
    event_objects[input_event.name] = input_event
    event_objects[output_event.name] = output_event
    event_objects[failed_event.name] = typing.cast(hsm.Event[object], failed_event)
    input_validation_failure = hsm.transition(
        hsm.on(input_event),
        hsm.guard(Behavior.input_invalid),
        hsm.effect(Behavior.dispatch_input_schema_failure, Behavior.queue_apply_settlement),
    )
    unmatched_input_settlement = hsm.transition(
        hsm.on(input_event),
        hsm.effect(Behavior.queue_apply_settlement),
    )
    callback_completion = hsm.transition(
        hsm.on(_CallbackCompletedEvent),
        hsm.effect(Behavior.complete_callback_execution),
    )
    callback_failure = hsm.transition(
        hsm.on(_CallbackFailedEvent),
        hsm.effect(Behavior.fail_callback_execution),
    )
    return typing.cast(
        hsm.Model,
        _lower_element(
            spec.model,
            event_objects=event_objects,
            input_event_name=input_event.name,
            generated_failed_event=typing.cast(hsm.Event[object], failed_event),
            generated_failure_elements=_failure_recovery_elements(spec.model),
            extra_root_elements=(
                input_validation_failure,
                unmatched_input_settlement,
                callback_completion,
                callback_failure,
            ),
        ),
    )


def _lower_element(
    element: dict[str, object],
    *,
    event_objects: collections.abc.Mapping[str, hsm.Event[object] | str],
    input_event_name: str,
    generated_failed_event: hsm.Event[object] | None = None,
    generated_failure_elements: tuple[object, ...] | None = None,
    extra_root_elements: tuple[hsm.Element, ...] = (),
    parent_path: str = "",
) -> hsm.Element:
    kind = typing.cast(str, element["kind"])
    if kind == "define":
        current_root_name = typing.cast(str, element["name"])
        children = tuple(
            _lower_element(
                _without_initial_callbacks(typing.cast(dict[str, object], child)),
                event_objects=event_objects,
                input_event_name=input_event_name,
                parent_path=f"/{current_root_name}",
            )
            for child in typing.cast(tuple[object, ...], element["elements"])
        )
        generated_failure = ()
        if generated_failed_event is not None and generated_failure_elements is not None:
            generated_failure = (
                hsm.transition(
                    hsm.on(generated_failed_event),
                    *(
                        _lower_element(
                            typing.cast(dict[str, object], child),
                            event_objects=event_objects,
                            input_event_name=input_event_name,
                        )
                        for child in generated_failure_elements
                    ),
                    hsm.effect(Behavior.queue_apply_settlement),
                ),
            )
        return bot.define(
            current_root_name,
            *extra_root_elements,
            *children,
            *generated_failure,
            hsm.observe(observer),
        )
    if kind == "state":
        state_name = typing.cast(str, element["name"])
        state_path = f"{parent_path}/{state_name}"
        return hsm.state(
            state_name,
            *_lower_state_elements(
                typing.cast(tuple[object, ...], element["elements"]),
                event_objects=event_objects,
                input_event_name=input_event_name,
                state_path=state_path,
            ),
        )
    if kind == "transition":
        source_elements = typing.cast(tuple[object, ...], element["elements"])
        children = tuple(
            _lower_element(
                typing.cast(dict[str, object], child),
                event_objects=event_objects,
                input_event_name=input_event_name,
                parent_path=parent_path,
            )
            for child in source_elements
        )
        if _transition_handles_event(source_elements, input_event_name):
            children = (
                *children,
                hsm.guard(Behavior.input_valid),
                hsm.effect(Behavior.queue_apply_settlement),
            )
        name = typing.cast(str | None, element.get("name"))
        if name is None:
            return hsm.transition(*children)
        return hsm.transition(name, *children)
    if kind == "initial":
        return hsm.initial(
            *(
                _lower_element(
                    typing.cast(dict[str, object], child),
                    event_objects=event_objects,
                    input_event_name=input_event_name,
                    parent_path=parent_path,
                )
                for child in typing.cast(tuple[object, ...], element["elements"])
            )
        )
    if kind == "on":
        return hsm.on(*(_lower_event_ref(event, event_objects=event_objects) for event in _event_refs(element)))
    if kind == "defer":
        return hsm.defer(*(_lower_event_ref(event, event_objects=event_objects) for event in _event_refs(element)))
    if kind == "target":
        return hsm.target(typing.cast(str, element["path"]))
    if kind == "guard":
        raise ValueError("Starlark guards must be lowered to typed evaluator outcomes before HSM construction.")
    if kind == "effect":
        return hsm.effect(Behavior.starlark_effect(*_callback_names(element)))
    if kind == "entry":
        return hsm.entry(Behavior.starlark_entry(*_callback_names(element)))
    if kind == "exit":
        return hsm.exit(Behavior.starlark_exit(*_callback_names(element)))
    if kind == "activity":
        return hsm.activity(*(Behavior.starlark_activity(callback) for callback in _callback_names(element)))
    if kind == "after":
        return hsm.after(_static_after(typing.cast(float | int, element["seconds"])))
    if kind == "final":
        return hsm.final(typing.cast(str, element["name"]))
    raise ValueError(f"Unsupported behavior hsm element kind: {kind}.")


def _lower_state_elements(
    elements: tuple[object, ...],
    *,
    event_objects: collections.abc.Mapping[str, hsm.Event[object] | str],
    input_event_name: str,
    state_path: str,
) -> tuple[hsm.Element, ...]:
    guarded_indices = {
        index
        for index, raw_element in enumerate(elements)
        if isinstance(raw_element, dict)
        and raw_element.get("kind") == "transition"
        and _guard_callback(typing.cast(tuple[object, ...], raw_element["elements"])) is not None
    }
    if not guarded_indices:
        return tuple(
            _lower_element(
                typing.cast(dict[str, object], element),
                event_objects=event_objects,
                input_event_name=input_event_name,
                parent_path=state_path,
            )
            for element in elements
            if isinstance(element, dict)
        )

    lifecycle_kinds = {"entry", "exit", "activity", "defer"}
    lifecycle: list[hsm.Element] = []
    ready_elements: list[hsm.Element] = []
    structural_elements: list[hsm.Element] = []
    consumed: set[int] = set()
    has_nested_state = any(isinstance(item, dict) and item.get("kind") == "state" for item in elements)
    initial_target = _initial_target(elements)
    ready_path = f"{state_path}/guard_ready"
    history_path = f"{state_path}/guard_history"
    fallback_path = initial_target or ready_path

    for index, raw_element in enumerate(elements):
        if index in consumed or not isinstance(raw_element, dict):
            continue
        element = typing.cast(dict[str, object], raw_element)
        kind = typing.cast(str, element.get("kind"))
        if kind in lifecycle_kinds:
            lifecycle.append(
                _lower_element(
                    element,
                    event_objects=event_objects,
                    input_event_name=input_event_name,
                    parent_path=state_path,
                )
            )
            continue
        if kind == "transition":
            source_elements = typing.cast(tuple[object, ...], element["elements"])
            callback = _guard_callback(source_elements)
            trigger_key = _transition_trigger_key(source_elements)
            if callback is None or trigger_key is None:
                destination = structural_elements if has_nested_state else ready_elements
                destination.append(
                    _lower_element(
                        element,
                        event_objects=event_objects,
                        input_event_name=input_event_name,
                        parent_path=state_path,
                    )
                )
                continue

            candidates: list[tuple[int, dict[str, object], str, str]] = []
            fallback: tuple[int, dict[str, object]] | None = None
            for candidate_index in range(index, len(elements)):
                candidate_raw = elements[candidate_index]
                if candidate_index in consumed or not isinstance(candidate_raw, dict):
                    continue
                candidate = typing.cast(dict[str, object], candidate_raw)
                if candidate.get("kind") != "transition":
                    continue
                candidate_elements = typing.cast(tuple[object, ...], candidate["elements"])
                if _transition_trigger_key(candidate_elements) != trigger_key:
                    continue
                candidate_callback = _guard_callback(candidate_elements)
                if candidate_callback is None:
                    fallback = (candidate_index, candidate)
                    break
                candidate_id = f"{state_path}:{candidate_index}"
                candidates.append((candidate_index, candidate, candidate_callback, candidate_id))
            if not candidates or candidates[0][0] != index:
                continue
            consumed.update(candidate[0] for candidate in candidates)
            if fallback is not None:
                consumed.add(fallback[0])

            first_elements = typing.cast(tuple[object, ...], candidates[0][1]["elements"])
            ingress_children = tuple(
                _lower_element(
                    item_spec,
                    event_objects=event_objects,
                    input_event_name=input_event_name,
                    parent_path=state_path,
                )
                for item in first_elements
                if isinstance(item, dict)
                and (item_spec := typing.cast(dict[str, object], item)).get("kind") in {"on", "after"}
            )
            first_candidate_path = f"{state_path}/guard_candidate_{candidates[0][0]}"
            ingress: tuple[hsm.Element, ...] = (*ingress_children, hsm.target(first_candidate_path))
            if _transition_handles_event(first_elements, input_event_name):
                ingress = (*ingress, hsm.guard(Behavior.input_valid))
            (structural_elements if has_nested_state else ready_elements).append(hsm.transition(*ingress))

            for candidate_offset, (candidate_index, candidate, candidate_callback, candidate_id) in enumerate(
                candidates
            ):
                candidate_elements = typing.cast(tuple[object, ...], candidate["elements"])
                handles_input = _transition_handles_event(candidate_elements, input_event_name)
                accepted_body = _lower_transition_body(
                    candidate_elements,
                    event_objects=event_objects,
                    input_event_name=input_event_name,
                    parent_path=state_path,
                    default_target=history_path,
                )
                if handles_input:
                    accepted_body = (*accepted_body, hsm.effect(Behavior.queue_apply_settlement))

                if candidate_offset + 1 < len(candidates):
                    rejected_body: tuple[hsm.Element, ...] = (
                        hsm.target(f"{state_path}/guard_candidate_{candidates[candidate_offset + 1][0]}"),
                    )
                elif fallback is not None:
                    fallback_elements = typing.cast(tuple[object, ...], fallback[1]["elements"])
                    rejected_body = _lower_transition_body(
                        fallback_elements,
                        event_objects=event_objects,
                        input_event_name=input_event_name,
                        parent_path=state_path,
                        default_target=history_path,
                    )
                    if _transition_handles_event(fallback_elements, input_event_name):
                        rejected_body = (*rejected_body, hsm.effect(Behavior.queue_apply_settlement))
                else:
                    rejected_body = (hsm.target(history_path),)
                    if handles_input:
                        rejected_body = (*rejected_body, hsm.effect(Behavior.queue_apply_settlement))

                failed_body: tuple[hsm.Element, ...] = (
                    hsm.effect(Behavior.reject_guard_failure),
                    hsm.target(history_path),
                )
                if handles_input:
                    failed_body = (*failed_body, hsm.effect(Behavior.queue_apply_settlement))
                structural_elements.append(
                    hsm.state(
                        f"guard_candidate_{candidate_index}",
                        hsm.entry(Behavior.request_guard(candidate_id, candidate_callback)),
                        *(
                            (
                                hsm.defer(
                                    *(
                                        _lower_event_ref(event_ref, event_objects=event_objects)
                                        for trigger in candidate_elements
                                        if isinstance(trigger, dict) and trigger.get("kind") == "on"
                                        for event_ref in _event_refs(typing.cast(dict[str, object], trigger))
                                    )
                                ),
                            )
                            if any(
                                isinstance(trigger, dict) and trigger.get("kind") == "on"
                                for trigger in candidate_elements
                            )
                            else ()
                        ),
                        hsm.transition(
                            hsm.on(_guard_outcome_event(candidate_id, _GuardEvaluationAcceptedEvent.name)),
                            *accepted_body,
                        ),
                        hsm.transition(
                            hsm.on(_guard_outcome_event(candidate_id, _GuardEvaluationRejectedEvent.name)),
                            *rejected_body,
                        ),
                        hsm.transition(
                            hsm.on(_guard_outcome_event(candidate_id, _GuardEvaluationFailedEvent.name)),
                            *failed_body,
                        ),
                    )
                )
            continue

        structural_elements.append(
            _lower_element(
                element,
                event_objects=event_objects,
                input_event_name=input_event_name,
                parent_path=state_path,
            )
        )

    if has_nested_state:
        return (*lifecycle, *structural_elements)
    return (
        *lifecycle,
        hsm.initial(hsm.target(ready_path)),
        hsm.shallow_history("guard_history", hsm.transition(hsm.target(fallback_path))),
        hsm.state("guard_ready", *ready_elements),
        *structural_elements,
    )


def _lower_transition_body(
    elements: tuple[object, ...],
    *,
    event_objects: collections.abc.Mapping[str, hsm.Event[object] | str],
    input_event_name: str,
    parent_path: str,
    default_target: str,
) -> tuple[hsm.Element, ...]:
    body = tuple(
        _lower_element(
            item_spec,
            event_objects=event_objects,
            input_event_name=input_event_name,
            parent_path=parent_path,
        )
        for item in elements
        if isinstance(item, dict)
        and (item_spec := typing.cast(dict[str, object], item)).get("kind") not in {"on", "after", "guard"}
    )
    if any(isinstance(item, dict) and item.get("kind") == "target" for item in elements):
        return body
    return (*body, hsm.target(default_target))


def _initial_target(elements: tuple[object, ...]) -> str | None:
    for raw_element in elements:
        if not isinstance(raw_element, dict) or raw_element.get("kind") != "initial":
            continue
        for child in typing.cast(tuple[object, ...], raw_element.get("elements", ())):
            if isinstance(child, dict) and child.get("kind") == "target":
                return typing.cast(str, child.get("path"))
    return None


def _guard_callback(elements: tuple[object, ...]) -> str | None:
    callback_names: list[str] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        element_spec = typing.cast(dict[str, object], element)
        if element_spec.get("kind") == "guard":
            callback_names.append(typing.cast(str, element_spec["callback"]))
    callbacks = tuple(callback_names)
    if len(callbacks) > 1:
        raise ValueError("Behavior transitions may declare at most one Starlark guard.")
    return callbacks[0] if callbacks else None


def _transition_trigger_key(elements: tuple[object, ...]) -> tuple[object, ...] | None:
    triggers: list[object] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        element_spec = typing.cast(dict[str, object], element)
        if element_spec.get("kind") == "on":
            triggers.append(("on", tuple(_event_ref_name(event) for event in _event_refs(element_spec))))
        elif element_spec.get("kind") == "after":
            triggers.append(("after", element_spec.get("seconds")))
    return tuple(triggers) if triggers else (("completion",),)


def _event_refs(element: dict[str, object]) -> tuple[object, ...]:
    return typing.cast(tuple[object, ...], element["events"])


def _transition_handles_event(elements: tuple[object, ...], event_name: str) -> bool:
    for child in elements:
        if not isinstance(child, dict):
            continue
        childspec = typing.cast(dict[str, object], child)
        if childspec.get("kind") != "on":
            continue
        for event_ref in _event_refs(childspec):
            if _event_ref_name(event_ref) == event_name:
                return True
    return False


def _event_ref_name(event_ref: object) -> str | None:
    if isinstance(event_ref, str):
        return event_ref
    if isinstance(event_ref, dict):
        event_ref_data = typing.cast(dict[str, object], event_ref)
        name = event_ref_data.get("name")
        if isinstance(name, str):
            return name
    return None


def _callback_names(element: dict[str, object]) -> tuple[str, ...]:
    return typing.cast(tuple[str, ...], element["callbacks"])


def _root_initial_transition_elements(model: dict[str, object]) -> tuple[object, ...]:
    for child in typing.cast(tuple[object, ...], model.get("elements", ())):
        if not isinstance(child, dict):
            continue
        childspec = typing.cast(dict[str, object], child)
        if childspec.get("kind") != "initial":
            continue
        return typing.cast(tuple[object, ...], childspec.get("elements", ()))
    raise ValueError("Behavior model must declare a root initial target for generated failure recovery.")


def _root_initial_callbacks(model: dict[str, object]) -> tuple[str, ...]:
    callbacks: list[str] = []
    for element in _root_initial_transition_elements(model):
        if isinstance(element, dict) and element.get("kind") == "effect":
            callbacks.extend(_callback_names(typing.cast(dict[str, object], element)))
    return tuple(callbacks)


def _failure_recovery_elements(model: dict[str, object]) -> tuple[object, ...]:
    elements = _root_initial_transition_elements(model)
    guarded_paths = {
        f"/{typing.cast(str, model['name'])}/{typing.cast(str, child['name'])}"
        for child in typing.cast(tuple[object, ...], model.get("elements", ()))
        if isinstance(child, dict)
        and child.get("kind") == "state"
        and any(
            isinstance(state_element, dict)
            and state_element.get("kind") == "transition"
            and _guard_callback(typing.cast(tuple[object, ...], state_element["elements"])) is not None
            for state_element in typing.cast(tuple[object, ...], child.get("elements", ()))
        )
    }
    return tuple(
        {**element, "path": f"{element['path']}/guard_ready"}
        if isinstance(element, dict) and element.get("kind") == "target" and element.get("path") in guarded_paths
        else element
        for element in elements
    )


def _without_initial_callbacks(element: dict[str, object]) -> dict[str, object]:
    if element.get("kind") != "initial":
        return element
    return {
        **element,
        "elements": tuple(
            child
            for child in typing.cast(tuple[object, ...], element.get("elements", ()))
            if not isinstance(child, dict) or child.get("kind") != "effect"
        ),
    }


def _lower_event_ref(
    event: object,
    *,
    event_objects: collections.abc.Mapping[str, hsm.Event[object] | str],
) -> hsm.Event[object] | str:
    if isinstance(event, str):
        event_object = event_objects.get(event)
        if event_object is None:
            raise ValueError(f"Behavior event reference {event!r} is not declared with a schema.")
        return event_object
    eventspec = source.EventContract.model_validate(event)
    return event_objects.get(eventspec.name, eventspec.hsm_event())


def _dispatch_callback_failure(
    ctx: hsm.Context,
    instance: Behavior,
    event: hsm.Event[object],
    message: str,
) -> None:
    failure = ability.FailureData(message=message)
    failed_event = dataclasses.replace(
        instance.failed_event.with_data(failure),
        id=event.id or None,
        metadata=dict(event.metadata),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, failed_event)
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(failed_event))


def _static_after(
    seconds: float | int,
) -> collections.abc.Callable[[hsm.Context, Behavior, hsm.Event[object]], datetime.timedelta]:
    def after(ctx: hsm.Context, instance: Behavior, event: hsm.Event[object]) -> datetime.timedelta:
        del ctx, instance, event
        return datetime.timedelta(seconds=float(seconds))

    after.__name__ = f"_behavior_after_{seconds:g}_seconds"
    return after


__all__ = [
    "Behavior",
    "define_model",
    "generated_event_names",
    "validate_event_names",
]
