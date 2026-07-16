import bot
from bot.abilities import ability
from bot.abilities import processing
from bot.abilities.cognition import operations
from bot.protocols import attachment
from bot.device import Device

import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm
import pytest


class _UncooperativeChild(ability.Ability[dict[str, object], dict[str, object]]):
    input_event = ability.ability_input_event("test.operation.child.input", dict[str, object])
    output_event = hsm.Event[dict[str, object]](
        name="test.operation.child.output",
        schema=typing.cast(typing.Any, dict[str, object]),
    )
    submodel = hsm.define(
        "UncooperativeChild",
        hsm.initial(hsm.target("/UncooperativeChild/waiting")),
        hsm.state(
            "waiting",
            hsm.transition(hsm.on(input_event), hsm.effect(lambda ctx, instance, event: None)),
        ),
    )


class _OperationOwner(hsm.Instance):
    child: _UncooperativeChild
    terminals: list[operations.TerminalData]
    resolutions: list[operations.CancelResolvedData]
    unresolved: list[operations.ResolveCancelData]
    terminal_events: list[hsm.Event[typing.Any]]
    model: typing.ClassVar[hsm.Model]

    def __init__(self, child: _UncooperativeChild) -> None:
        super().__init__()
        self.child = child
        self.terminals = []
        self.resolutions = []
        self.unresolved = []
        self.terminal_events = []


def _forward(ctx: hsm.Context, instance: _OperationOwner, event: hsm.Event[typing.Any]) -> None:
    operations.forward_terminal(ctx, instance, event)


def _record(ctx: hsm.Context, instance: _OperationOwner, event: hsm.Event[typing.Any]) -> None:
    del ctx
    assert isinstance(event.data, operations.TerminalData)
    instances = instance.context().value(hsm.Keys.Instances)
    assert isinstance(instances, collections.abc.Mapping)
    assert isinstance(instances.get(event.source), operations.Operation)
    instance.terminals.append(event.data)
    instance.terminal_events.append(event)


def _record_resolution(ctx: hsm.Context, instance: _OperationOwner, event: hsm.Event[typing.Any]) -> None:
    del ctx
    assert isinstance(event.data, operations.CancelResolvedData)
    instance.resolutions.append(event.data)


def _record_unresolved(ctx: hsm.Context, instance: _OperationOwner, event: hsm.Event[typing.Any]) -> None:
    del ctx
    assert isinstance(event.data, operations.ResolveCancelData)
    instance.unresolved.append(event.data)


_OperationOwner.model = hsm.define(
    "OperationOwner",
    hsm.initial(hsm.target("/OperationOwner/active")),
    hsm.state(
        "active",
        hsm.transition(hsm.on(processing.CancelledEvent), hsm.effect(_forward)),
        hsm.transition(hsm.on(operations.CancelResolvedEvent), hsm.effect(_record_resolution)),
        hsm.transition(hsm.on(operations.CancelUnresolvedEvent), hsm.effect(_record_unresolved)),
        hsm.transition(
            hsm.on(operations.CancelTeardownTimedOutEvent),
            hsm.guard(operations.matches_teardown_timeout),
            hsm.effect(operations.force_cancel_timeout),
        ),
        hsm.transition(
            hsm.on(operations.TerminalEvent),
            hsm.effect(_record, operations.retire_resolution, operations.retire_operation),
        ),
    ),
)


def test_focus_metadata_cannot_add_unconfigured_device_candidate() -> None:
    async def run() -> None:
        source = hsm.Instance()
        input = processing.InputData(
            input=bot.InputEventData(target_device="phone", priority=0),
            actors={"phone": Device()},
        )
        selection = (
            processing.SelectedEvent(
                event=bot.FocusDeviceEvent.name,
                data=bot.FocusDeviceEventData(device="ghost").model_dump(mode="json"),
            ),
        )
        with pytest.raises(RuntimeError, match="outside available device candidates"):
            await operations.dispatch_selected_events(
                hsm.Context(),
                input,
                selection,
                operation_id="focus-operation",
                source=source,
                metadata={"bot.focus_candidates": ("ghost",)},
            )

    asyncio.run(run())


def test_child_timeout_waits_for_bounded_cancellation_before_truthful_terminal() -> None:
    async def run() -> tuple[list[operations.TerminalData], list[operations.TerminalData], int]:
        ctx = hsm.Context()
        child = _UncooperativeChild()
        owner = _OperationOwner(child)
        _ = await hsm.started(ctx, owner, owner.model)
        await child.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await operations.Operation.begin(
            owner=owner,
            child=child,
            request=child.input_event.with_data_and_id({}, "held-child"),
            operation_id="held-parent",
            phase="test",
            timeout=datetime.timedelta(milliseconds=10),
        )
        await asyncio.sleep(0.015)
        before_cancel_timeout = list(owner.terminals)
        await asyncio.sleep(0.02)
        instances = owner.context().value(hsm.Keys.Instances)
        operation_count = (
            sum(isinstance(actor, operations.Operation) for actor in instances.values())
            if isinstance(instances, collections.abc.Mapping)
            else 0
        )
        return before_cancel_timeout, owner.terminals, operation_count

    before_cancel_timeout, terminals, operation_count = asyncio.run(run())

    assert before_cancel_timeout == []
    assert len(terminals) == 1
    assert terminals[0].outcome == "cancel_timeout"
    assert terminals[0].failure is not None
    assert "cancellation timed out" in terminals[0].failure.message.lower()
    assert operation_count == 0


def test_child_terminal_payload_schema_is_complete_and_json_safe() -> None:
    schema = operations.TerminalData.model_json_schema()

    assert "terminal" not in schema.get("properties", {})
    assert {"operation", "outcome", "terminal_name", "output", "failure"} <= set(schema.get("properties", {}))


def test_cancel_resolution_reports_no_exact_match_with_unrelated_mediator() -> None:
    async def run() -> tuple[list[operations.ResolveCancelData], str]:
        ctx = hsm.Context()
        child = _UncooperativeChild()
        owner = _OperationOwner(child)
        _ = await hsm.started(ctx, owner, owner.model)
        await child.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        unrelated = await operations.Operation.begin(
            owner=owner,
            child=child,
            request=child.input_event.with_data_and_id({}, "unrelated-request"),
            operation_id="unrelated-operation",
            phase="unrelated",
            timeout=datetime.timedelta(seconds=1),
        )
        await operations.CancelResolution.begin(
            owner=owner,
            operation_id="missing-operation",
            token="missing-token",
            metadata={},
            teardown_timeout=datetime.timedelta(seconds=1),
        )
        await asyncio.sleep(0.02)
        instances = owner.context().value(hsm.Keys.Instances)
        assert isinstance(instances, collections.abc.Mapping)
        actor = instances[unrelated.actor_id]
        assert isinstance(actor, operations.Operation)
        return owner.unresolved, actor.state()

    unresolved, unrelated_state = asyncio.run(run())

    assert len(unresolved) == 1
    assert unresolved[0].operation_id == "missing-operation"
    assert unrelated_state.endswith("/waiting")


def test_host_cancel_timeout_preserves_exact_resolution_provenance() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], int]:
        ctx = hsm.Context()
        child = _UncooperativeChild()
        owner = _OperationOwner(child)
        _ = await hsm.started(ctx, owner, owner.model)
        await child.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        operation = await operations.Operation.begin(
            owner=owner,
            child=child,
            request=child.input_event.with_data_and_id({}, "stubborn-request"),
            operation_id="stubborn-operation",
            phase="stubborn",
            timeout=datetime.timedelta(milliseconds=10),
        )
        request = await operations.CancelResolution.begin(
            owner=owner,
            operation_id=operation.operation_id,
            token=operation.token,
            metadata={},
            teardown_timeout=datetime.timedelta(milliseconds=10),
        )
        for _ in range(100):
            if owner.resolutions:
                break
            await asyncio.sleep(0)
        assert owner.resolutions
        resolved = owner.resolutions[0]
        resolved_event = dataclasses.replace(
            operations.CancelResolvedEvent.with_data(resolved),
            id=operation.operation_id,
            source=operation.actor_id,
            target=operation.owner_id,
            metadata={
                operations.OPERATION_METADATA_KEY: operation,
                operations.RESOLVE_CANCEL_METADATA_KEY: request,
            },
        )
        operations.complete_resolution(ctx, owner, resolved_event)
        operations.cancel_resolved_operation(ctx, owner, resolved_event)
        await asyncio.sleep(0.03)
        instances = owner.context().value(hsm.Keys.Instances)
        actor_count = (
            sum(isinstance(actor, operations.Operation | operations.CancelResolution) for actor in instances.values())
            if isinstance(instances, collections.abc.Mapping)
            else 0
        )
        return owner.terminal_events[0], actor_count

    terminal, actor_count = asyncio.run(run())

    assert isinstance(terminal.data, operations.TerminalData)
    assert terminal.data.outcome == "cancel_timeout"
    assert isinstance(terminal.metadata.get(operations.CANCEL_METADATA_KEY), operations.CancelData)
    assert isinstance(terminal.metadata.get(operations.RESOLVE_CANCEL_METADATA_KEY), operations.ResolveCancelData)
    assert actor_count == 0


def test_operation_rejects_host_cancel_with_wrong_capability_token() -> None:
    async def run() -> tuple[str, list[operations.TerminalData]]:
        ctx = hsm.Context()
        child = _UncooperativeChild()
        owner = _OperationOwner(child)
        _ = await hsm.started(ctx, owner, owner.model)
        await child.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        operation = await operations.Operation.begin(
            owner=owner,
            child=child,
            request=child.input_event.with_data_and_id({}, "reused-request"),
            operation_id="current-operation",
            phase="test",
            timeout=datetime.timedelta(seconds=1),
        )
        instances = owner.context().value(hsm.Keys.Instances)
        assert isinstance(instances, collections.abc.Mapping)
        actor = instances[operation.actor_id]
        assert isinstance(actor, operations.Operation)
        wrong = operations.CancelData(
            owner_id=operation.owner_id,
            child_id=operation.child_id,
            request_id=operation.request_id,
            operation_id=operation.operation_id,
            token="wrong-token",
            resolver_id="wrong-resolver",
        )
        _ = await hsm.dispatch(
            ctx,
            actor,
            dataclasses.replace(
                operations.CancelEvent.with_data(wrong),
                id=operation.operation_id,
                source=operation.owner_id,
                target=operation.actor_id,
                metadata={operations.CANCEL_METADATA_KEY: wrong},
            ),
        )
        await asyncio.sleep(0)
        return actor.state(), owner.terminals

    state, terminals = asyncio.run(run())

    assert state.endswith("/waiting")
    assert terminals == []


def test_operation_rejects_cancel_resolution_with_wrong_capability_token() -> None:
    async def run() -> list[operations.CancelResolvedData]:
        ctx = hsm.Context()
        child = _UncooperativeChild()
        owner = _OperationOwner(child)
        _ = await hsm.started(ctx, owner, owner.model)
        await child.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        operation = await operations.Operation.begin(
            owner=owner,
            child=child,
            request=child.input_event.with_data_and_id({}, "reused-request"),
            operation_id="current-operation",
            phase="test",
            timeout=datetime.timedelta(seconds=1),
        )
        instances = owner.context().value(hsm.Keys.Instances)
        assert isinstance(instances, collections.abc.Mapping)
        actor = instances[operation.actor_id]
        assert isinstance(actor, operations.Operation)
        wrong = operations.ResolveCancelData(
            owner_id=operation.owner_id,
            operation_id=operation.operation_id,
            token="wrong-token",
            resolver_id="wrong-resolver",
        )
        _ = await hsm.dispatch(
            ctx,
            actor,
            dataclasses.replace(
                operations.ResolveCancelEvent.with_data(wrong),
                id=operation.operation_id,
                source=operation.owner_id,
                target=operation.actor_id,
                metadata={operations.RESOLVE_CANCEL_METADATA_KEY: wrong},
            ),
        )
        await asyncio.sleep(0)
        return owner.resolutions

    assert asyncio.run(run()) == []


def test_operation_prefers_explicit_cancel_token_over_inherited_metadata() -> None:
    async def run() -> str:
        ctx = hsm.Context()
        child = _UncooperativeChild()
        owner = _OperationOwner(child)
        _ = await hsm.started(ctx, owner, owner.model)
        await child.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        forged = operations.OperationData(
            operation_id="current-operation",
            token="forged-token",
            owner_id="forged-owner",
            child_id="forged-child",
            request_id="forged-request",
            phase="forged-phase",
            actor_id="forged-actor",
        )
        operation = await operations.Operation.begin(
            owner=owner,
            child=child,
            request=dataclasses.replace(
                child.input_event.with_data_and_id({}, "reused-request"),
                metadata={
                    operations.OPERATION_METADATA_KEY: forged,
                    operations.CANCEL_TOKEN_METADATA_KEY: "legitimate-token",
                },
            ),
            operation_id="current-operation",
            phase="test",
            timeout=datetime.timedelta(seconds=1),
        )
        return operation.token

    assert asyncio.run(run()) == "legitimate-token"


def test_cancel_resolution_startup_cancellation_removes_registered_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> int:
        ctx = hsm.Context()
        owner = _OperationOwner(_UncooperativeChild())
        _ = await hsm.started(ctx, owner, owner.model)
        start_entered = asyncio.Event()
        release_start = asyncio.Event()
        original_dispatch = hsm.dispatch

        async def held_dispatch(
            dispatch_ctx: hsm.Context,
            instance: hsm.Instance,
            event: hsm.Event[typing.Any],
        ) -> None:
            if event.name == "bot.ability.cognition.child_operation.cancel.resolution.start":
                start_entered.set()
                await release_start.wait()
            await original_dispatch(dispatch_ctx, instance, event)

        monkeypatch.setattr(hsm, "dispatch", held_dispatch)
        task = asyncio.create_task(
            operations.CancelResolution.begin(
                owner=owner,
                operation_id="cancelled-resolution",
                token="exact-token",
                metadata={},
                teardown_timeout=datetime.timedelta(seconds=1),
            )
        )
        await start_entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        instances = owner.context().value(hsm.Keys.Instances)
        return (
            sum(isinstance(actor, operations.CancelResolution) for actor in instances.values())
            if isinstance(instances, collections.abc.Mapping)
            else 0
        )

    assert asyncio.run(run()) == 0
