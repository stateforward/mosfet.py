from bot import abilities, environment
from bot.abilities.language import text

import asyncio
import ast
import collections.abc
import dataclasses
import datetime
import importlib.util
import pathlib
import typing
import uuid
from typing import override

import hsm
import pydantic
import pytest
import bot.abilities as abilities_module
import bot.abilities.ability as ability_module
import bot.lifecycle
from bot.protocols import attachment

from tests.type_helpers import invalid_value, object_dict


class SlowTextGenerator(text.TextGenerator):
    release: asyncio.Event

    def __init__(self, release: asyncio.Event) -> None:
        self.release = release

    @override
    async def generate(self, input: text.InputData) -> text.OutputData:
        _ = await self.release.wait()
        return text.OutputData(content=input.messages[-1].content.upper())


class RecordingDelayedTextGenerator(text.TextGenerator):
    calls: list[str]
    release_first: asyncio.Event

    def __init__(self, calls: list[str], release_first: asyncio.Event) -> None:
        self.calls = calls
        self.release_first = release_first

    @override
    async def generate(self, input: text.InputData) -> text.OutputData:
        content = input.messages[-1].content
        self.calls.append(content)
        if content == "first":
            _ = await self.release_first.wait()
        return text.OutputData(content=content.upper())


class EchoTextGenerator(text.TextGenerator):
    @override
    async def generate(self, input: text.InputData) -> text.OutputData:
        return text.OutputData(content=input.messages[-1].content.upper())


class WrongTextGenerator(text.TextGenerator):
    @override
    async def generate(self, input: text.InputData) -> text.OutputData:
        del input
        return invalid_value(text.OutputData, "not text generation output")


class RecordingTextGeneration(text.TextGeneration):
    outputs: list[text.OutputData]
    failures: list[abilities.FailureData]

    def __init__(self, *, generator: text.TextGenerator) -> None:
        super().__init__(generator=generator)
        self.outputs = []
        self.failures = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, text.OutputData)
            self.outputs.append(output)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, abilities.FailureData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


def _record_ability_terminal_owner_event(
    ctx: hsm.Context,
    instance: "AbilityTerminalOwner",
    event: hsm.Event[typing.Any],
) -> None:
    del ctx
    instance.record(event)


class AbilityTerminalOwner(hsm.Instance):
    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "AbilityTerminalOwner",
        hsm.initial(hsm.target("/AbilityTerminalOwner/recording")),
        hsm.state(
            "recording",
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.effect(_record_ability_terminal_owner_event),
            ),
        ),
    )
    outputs: list[object]
    failures: list[object]
    lifecycle: list[hsm.Event[typing.Any]]
    events: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.outputs = []
        self.failures = []
        self.lifecycle = []
        self.events = []

    def record(self, event: hsm.Event[typing.Any]) -> None:
        self.events.append(event)
        if event.name.endswith(".output"):
            self.outputs.append(event.data)
        if event.name.endswith(".failed"):
            self.failures.append(event.data)
        if event.name in {
            attachment.AttachCompleteEvent.name,
            attachment.AttachFailedEvent.name,
            attachment.DetachedEvent.name,
            attachment.DetachFailedEvent.name,
        }:
            self.lifecycle.append(event)


class TerminalOperationInputData(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    value: str


class TerminalOperationOutputData(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    value: str


TerminalOperationInputEvent = hsm.Event[TerminalOperationInputData](
    name="test.ability.terminal.operation.input",
    schema=TerminalOperationInputData,
)
TerminalOperationOutputEvent = hsm.Event[TerminalOperationOutputData](
    name="test.ability.terminal.operation.output",
    schema=TerminalOperationOutputData,
)
TerminalOperationFailedEvent = hsm.Event[abilities.FailureData](
    name="test.ability.terminal.operation.failed",
    schema=abilities.FailureData,
)


class TerminalOperationAbility(abilities.Ability[TerminalOperationInputData, TerminalOperationOutputData]):
    input_event = TerminalOperationInputEvent
    output_event = TerminalOperationOutputEvent
    failed_event = TerminalOperationFailedEvent
    requests: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.requests = []

    @staticmethod
    def record_request(
        ctx: hsm.Context,
        instance: "TerminalOperationAbility",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        instance.requests.append(event)

    async def emit_terminal(
        self,
        ctx: hsm.Context,
        terminal: hsm.Event[typing.Any],
    ) -> None:
        wrapper = (
            ability_module.TerminalErrorEvent
            if terminal.name == self.failed_event.name
            else ability_module.TerminalOutputEvent
        )
        await hsm.dispatch(ctx, self, wrapper.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "TerminalOperationAbility",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(TerminalOperationInputEvent),
                hsm.effect(record_request),
            ),
        ),
    )


class CompositeAbility(abilities.Ability[object, object]):
    _composite_attachment_lifecycle = True

    def __init__(self) -> None:
        super().__init__()
        self.held_terminals: list[hsm.Event[typing.Any]] = []
        self.hold_terminal = False

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if self.hold_terminal and event.name == self._composite_attachment_terminal_event.name:
            self.held_terminals.append(event)
            held = asyncio.get_running_loop().create_future()
            held.set_result(None)
            return held
        return super().dispatch(ctx, event)

    async def dispatch_terminal(
        self,
        ctx: hsm.Context,
        request: attachment.AttachData | attachment.DetachData,
        terminal: hsm.Event[typing.Any],
        operation_id: str,
    ) -> None:
        operation_event = dataclasses.replace(terminal, id=operation_id)
        reply = await self._start_composite_attachment_reply(
            self,
            request,
            operation_event,
        )
        await hsm.dispatch(
            ctx,
            reply,
            dataclasses.replace(
                terminal,
                id=operation_id,
                source=hsm.id(self),
                target=hsm.id(reply),
                metadata=dict(operation_event.metadata),
            ),
        )

    async def dispatch_uncorrelated_terminal(self, ctx: hsm.Context) -> None:
        terminal = typing.cast(hsm.Event[typing.Any], self._composite_attachment_terminal_event)
        await hsm.dispatch(
            ctx,
            self,
            dataclasses.replace(
                terminal,
                data=None,
                source="untrusted",
                target=hsm.id(self),
            ),
        )

    async def dispatch_substituted_reply_terminal(self, ctx: hsm.Context) -> None:
        terminal = self.held_terminals[-1]
        data = terminal.data
        assert isinstance(data, pydantic.BaseModel)
        substitute = hsm.Instance()
        setattr(substitute, "id", hsm.id(getattr(data, "reply")))
        self.hold_terminal = False
        await super().dispatch(
            ctx,
            dataclasses.replace(
                terminal,
                data=data.model_copy(update={"reply": substitute}),
            ),
        )

    async def dispatch_held_terminal(self, ctx: hsm.Context) -> None:
        self.hold_terminal = False
        await super().dispatch(ctx, self.held_terminals[-1])

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "CompositeAbility",
        hsm.initial(hsm.target("initializing")),
        hsm.state(
            "initializing",
            hsm.transition(
                hsm.on(abilities.Ability._composite_attachment_terminal_event),
                hsm.guard(abilities.Ability._is_composite_attach_complete),
                hsm.effect(abilities.Ability._deliver_composite_attachment_terminal),
                hsm.target("/CompositeAbility/operational"),
            ),
        ),
        hsm.state(
            "operational",
        ),
        hsm.state("detaching"),
    )


class _DirectApplyReferenceVisitor(ast.NodeVisitor):
    relative_path: str
    references: set[str]
    _scope: list[str]

    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.references = set()
        self._scope = []

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        _ = self._scope.pop()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_callable(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_apply":
            self.references.add(f"{self.relative_path}:{self._qualname()}:attribute")
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "_apply"
        ):
            self.references.add(f"{self.relative_path}:{self._qualname()}:getattr")
        self.generic_visit(node)

    def _visit_callable(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        _ = self._scope.pop()

    def _qualname(self) -> str:
        if self._scope:
            return ".".join(self._scope)
        return "<module>"


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


async def start_terminal_operation_ability(
    ctx: hsm.Context,
) -> tuple[TerminalOperationAbility, AbilityTerminalOwner]:
    owner = AbilityTerminalOwner()
    child = TerminalOperationAbility()
    _ = await bot.started(ctx, owner, require_model(owner.model))
    _ = await child.attach(
        ctx,
        attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
    )
    return child, owner


def terminal_operation_request(operation_id: str, value: str) -> hsm.Event[TerminalOperationInputData]:
    return TerminalOperationInputEvent.with_data_and_id(
        TerminalOperationInputData(value=value),
        operation_id,
    )


def terminal_operation_output(
    child: TerminalOperationAbility,
    request: hsm.Event[typing.Any],
    value: str,
    *,
    operation_id: str | None = None,
    source: str | None = None,
    target: str | None = None,
) -> hsm.Event[TerminalOperationOutputData]:
    return dataclasses.replace(
        TerminalOperationOutputEvent.with_data(
            TerminalOperationOutputData(value=value),
        ),
        id=request.id if operation_id is None else operation_id,
        source=hsm.id(child) if source is None else source,
        target=request.source if target is None else target,
        metadata=dict(request.metadata),
    )


async def start_recorded_generation(
    generation: RecordingTextGeneration,
) -> tuple[hsm.Context, AbilityTerminalOwner]:
    ctx = hsm.Context()
    owner = AbilityTerminalOwner()
    _ = await bot.started(ctx, owner, require_model(owner.model))
    _ = await generation.attach(
        ctx,
        attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
    )
    return ctx, owner


def test_ability() -> None:
    ability = abilities.Ability[object, object]()

    input_schema = object_dict(abilities.InputEvent.schema)
    output_schema = object_dict(abilities.OutputEvent.schema)

    assert isinstance(ability, hsm.Instance)
    assert abilities.Ability.__annotations__["model"] == typing.ClassVar[hsm.Model | None]
    assert require_model(abilities.Ability.model).qualified_name == "/Ability"
    assert not hasattr(abilities_module, "ability_model")
    assert not hasattr(ability_module, "ability_model")
    assert not hasattr(abilities_module, "ability_use_model")
    assert importlib.util.find_spec("bot.abilities._event_driven") is None
    assert not hasattr(ability, "submit")
    assert not hasattr(ability, "perform")
    assert not hasattr(ability, "use")
    assert hasattr(ability, "apply")
    assert not hasattr(ability, "publish")
    assert not hasattr(ability, "target_ids")
    assert not hasattr(ability, "operation_timeout")
    assert not hasattr(abilities.Ability, "operation_completed_event")
    assert not hasattr(abilities.Ability, "operation_failed_event")
    assert not hasattr(abilities.Ability, "_apply_completed_event")
    assert not hasattr(abilities.Ability, "_apply_failed_event")
    assert not hasattr(abilities.Ability, "_apply")
    assert not hasattr(abilities_module, "ability_operation_model")
    assert not hasattr(abilities.Ability, "run_apply_activity")
    assert not hasattr(ability_module, "ABILITY_APPLY_ACTIVITY")
    assert not hasattr(abilities_module, "ABILITY_APPLY_ACTIVITY")
    assert "ability_operation_model" not in ability_module.__all__
    assert "ABILITY_APPLY_ACTIVITY" not in ability_module.__all__
    assert abilities.InputEvent.name == "bot.ability.input"
    # Generic base events are bare object schemas — concrete abilities replace them.
    assert "input" not in object_dict(input_schema.get("properties", {}))
    assert abilities.OutputEvent.name == "bot.ability.output"
    assert "output" not in object_dict(output_schema.get("properties", {}))


def test_terminal_waiter_registry_api_is_removed() -> None:
    legacy = {
        "_terminal_waiters",
        "_child_terminal_waiters",
        "_terminal_reply_owners",
        "register_terminal_waiter",
        "clear_terminal_waiter",
        "prepare_child_terminal_wait",
        "clear_child_terminal_wait",
        "await_child_terminal",
    }

    assert legacy.isdisjoint(vars(abilities.Ability))
    assert "weakref" not in pathlib.Path(ability_module.__file__).read_text()
    assert abilities.run_terminal_operation is ability_module.run_terminal_operation


def test_terminal_operation_rejects_invalid_configuration() -> None:
    async def run(
        *,
        request: hsm.Event[typing.Any],
        terminals: tuple[hsm.Event[typing.Any], ...],
        timeout: datetime.timedelta,
    ) -> None:
        await ability_module.run_terminal_operation(
            hsm.Context(),
            child=TerminalOperationAbility(),
            request=request,
            terminals=terminals,
            timeout=timeout,
        )

    with pytest.raises(ValueError, match="request.id"):
        asyncio.run(
            run(
                request=TerminalOperationInputEvent.with_data(TerminalOperationInputData(value="input")),
                terminals=(TerminalOperationOutputEvent,),
                timeout=datetime.timedelta(seconds=1),
            )
        )
    with pytest.raises(ValueError, match="at least one terminal"):
        asyncio.run(
            run(
                request=terminal_operation_request("empty-terminals", "input"),
                terminals=(),
                timeout=datetime.timedelta(seconds=1),
            )
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        asyncio.run(
            run(
                request=terminal_operation_request("negative-timeout", "input"),
                terminals=(TerminalOperationOutputEvent,),
                timeout=datetime.timedelta(microseconds=-1),
            )
        )
    with pytest.raises(RuntimeError, match="must be started"):
        asyncio.run(
            run(
                request=terminal_operation_request("unstarted", "input"),
                terminals=(TerminalOperationOutputEvent,),
                timeout=datetime.timedelta(seconds=1),
            )
        )


def test_terminal_operations_isolate_concurrent_results() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any]]:
        ctx = environment.Environment()
        child, _owner = await start_terminal_operation_ability(ctx)
        first = asyncio.create_task(
            ability_module.run_terminal_operation(
                ctx,
                child=child,
                request=terminal_operation_request("first", "one"),
                terminals=(child.output_event, child.failed_event),
                timeout=datetime.timedelta(seconds=1),
            )
        )
        second = asyncio.create_task(
            ability_module.run_terminal_operation(
                ctx,
                child=child,
                request=terminal_operation_request("second", "two"),
                terminals=(child.output_event, child.failed_event),
                timeout=datetime.timedelta(seconds=1),
            )
        )
        while len(child.requests) != 2:
            await asyncio.sleep(0)
        requests = {event.id: event for event in child.requests}
        await child.emit_terminal(ctx, terminal_operation_output(child, requests["second"], "TWO"))
        await child.emit_terminal(ctx, terminal_operation_output(child, requests["first"], "ONE"))
        return await first, await second

    first, second = asyncio.run(run())

    assert first.data == TerminalOperationOutputData(value="ONE")
    assert second.data == TerminalOperationOutputData(value="TWO")


def test_terminal_operation_rejects_wrong_envelopes_and_duplicate() -> None:
    async def run() -> hsm.Event[typing.Any]:
        ctx = environment.Environment()
        instances = typing.cast(
            collections.abc.Mapping[str, hsm.Instance],
            ctx.value(hsm.Keys.Instances),
        )
        child, _owner = await start_terminal_operation_ability(ctx)
        task = asyncio.create_task(
            ability_module.run_terminal_operation(
                ctx,
                child=child,
                request=terminal_operation_request("correlated", "input"),
                terminals=(child.output_event, child.failed_event),
                timeout=datetime.timedelta(seconds=1),
            )
        )
        while not child.requests:
            await asyncio.sleep(0)
        request = child.requests[-1]
        operation = instances[request.source]
        wrong = (
            terminal_operation_output(child, request, "wrong-id", operation_id="other"),
            terminal_operation_output(child, request, "wrong-source", source="other"),
            terminal_operation_output(child, request, "wrong-target", target="other"),
        )
        for terminal in wrong:
            await hsm.dispatch(ctx, operation, terminal)
            assert not task.done()
        accepted = terminal_operation_output(child, request, "accepted")
        await hsm.dispatch(ctx, operation, accepted)
        result = await task
        with pytest.raises(RuntimeError, match="started HSM"):
            await hsm.dispatch(ctx, operation, dataclasses.replace(accepted))
        return result

    result = asyncio.run(run())

    assert result.data == TerminalOperationOutputData(value="accepted")


def test_terminal_operation_rejects_stale_terminal_from_prior_operation() -> None:
    async def run() -> hsm.Event[typing.Any]:
        ctx = environment.Environment()
        instances = typing.cast(
            collections.abc.Mapping[str, hsm.Instance],
            ctx.value(hsm.Keys.Instances),
        )
        child, _owner = await start_terminal_operation_ability(ctx)

        first = asyncio.create_task(
            ability_module.run_terminal_operation(
                ctx,
                child=child,
                request=terminal_operation_request("old", "old"),
                terminals=(child.output_event, child.failed_event),
                timeout=datetime.timedelta(seconds=1),
            )
        )
        while len(child.requests) != 1:
            await asyncio.sleep(0)
        old_request = child.requests[-1]
        await child.emit_terminal(ctx, terminal_operation_output(child, old_request, "old"))
        _ = await first

        second = asyncio.create_task(
            ability_module.run_terminal_operation(
                ctx,
                child=child,
                request=terminal_operation_request("new", "new"),
                terminals=(child.output_event, child.failed_event),
                timeout=datetime.timedelta(seconds=1),
            )
        )
        while len(child.requests) != 2:
            await asyncio.sleep(0)
        new_request = child.requests[-1]
        operation = instances[new_request.source]
        await hsm.dispatch(
            ctx,
            operation,
            terminal_operation_output(
                child,
                new_request,
                "stale",
                operation_id=old_request.id,
            ),
        )
        assert not second.done()
        await child.emit_terminal(ctx, terminal_operation_output(child, new_request, "new"))
        return await second

    result = asyncio.run(run())

    assert result.data == TerminalOperationOutputData(value="new")


def test_terminal_operation_models_timeout() -> None:
    async def run() -> None:
        ctx = environment.Environment()
        child, _owner = await start_terminal_operation_ability(ctx)
        await ability_module.run_terminal_operation(
            ctx,
            child=child,
            request=terminal_operation_request("timeout", "input"),
            terminals=(child.output_event, child.failed_event),
            timeout=datetime.timedelta(0),
        )

    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(run())


def test_terminal_operation_cancellation_stops_actor_and_late_terminal_isolated() -> None:
    async def run() -> hsm.Event[typing.Any]:
        ctx = environment.Environment()
        instances = typing.cast(
            collections.abc.Mapping[str, hsm.Instance],
            ctx.value(hsm.Keys.Instances),
        )
        child, _owner = await start_terminal_operation_ability(ctx)
        canceled = asyncio.create_task(
            ability_module.run_terminal_operation(
                ctx,
                child=child,
                request=terminal_operation_request("canceled", "input"),
                terminals=(child.output_event, child.failed_event),
                timeout=datetime.timedelta(seconds=1),
            )
        )
        while len(child.requests) != 1:
            await asyncio.sleep(0)
        canceled_request = child.requests[-1]
        canceled_actor = instances[canceled_request.source]
        _ = canceled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await canceled
        assert not bot.lifecycle.is_started(canceled_actor)

        active = asyncio.create_task(
            ability_module.run_terminal_operation(
                ctx,
                child=child,
                request=terminal_operation_request("active", "input"),
                terminals=(child.output_event, child.failed_event),
                timeout=datetime.timedelta(seconds=1),
            )
        )
        while len(child.requests) != 2:
            await asyncio.sleep(0)
        active_request = child.requests[-1]
        active_actor = instances[active_request.source]
        await hsm.dispatch(
            ctx,
            active_actor,
            terminal_operation_output(
                child,
                active_request,
                "late",
                operation_id=canceled_request.id,
            ),
        )
        assert not active.done()
        await child.emit_terminal(ctx, terminal_operation_output(child, active_request, "active"))
        return await active

    result = asyncio.run(run())

    assert result.data == TerminalOperationOutputData(value="active")


def test_terminal_operation_returns_correlated_typed_failure() -> None:
    async def run() -> hsm.Event[typing.Any]:
        ctx = environment.Environment()
        child, _owner = await start_terminal_operation_ability(ctx)
        task = asyncio.create_task(
            ability_module.run_terminal_operation(
                ctx,
                child=child,
                request=terminal_operation_request("failed", "input"),
                terminals=(child.output_event, child.failed_event),
                timeout=datetime.timedelta(seconds=1),
            )
        )
        while not child.requests:
            await asyncio.sleep(0)
        request = child.requests[-1]
        await child.emit_terminal(
            ctx,
            dataclasses.replace(
                child.failed_event.with_data(abilities.FailureData(message="failed")),
                id=request.id,
                source=hsm.id(child),
                target=request.source,
            ),
        )
        return await task

    result = asyncio.run(run())

    assert result.name == TerminalOperationFailedEvent.name
    assert result.data == abilities.FailureData(message="failed")


def test_targeted_terminal_is_delivered_once_without_owner_fallback() -> None:
    """Ability terminal emission: a set target is delivered once and never also to the owner."""

    async def run() -> tuple[AbilityTerminalOwner, AbilityTerminalOwner, TerminalOperationAbility]:
        ctx = environment.Environment()
        child, owner = await start_terminal_operation_ability(ctx)
        recipient = AbilityTerminalOwner()
        _ = await bot.started(ctx, recipient, require_model(recipient.model))
        terminal = dataclasses.replace(
            child.output_event.with_data(TerminalOperationOutputData(value="targeted")),
            id="targeted",
            source="spoofed",
            target=hsm.id(recipient),
        )
        await child.emit_terminal(ctx, terminal)
        return recipient, owner, child

    recipient, owner, child = asyncio.run(run())

    routed = [event for event in recipient.events if event.name == TerminalOperationOutputEvent.name]
    assert len(routed) == 1
    assert routed[0].source == hsm.id(child)
    assert routed[0].target == hsm.id(recipient)
    assert owner.outputs == []


def test_targetless_terminal_falls_back_to_attachment_owner_once() -> None:
    """Ability terminal emission: an absent or empty target is delivered once to the owner."""

    async def run() -> tuple[AbilityTerminalOwner, TerminalOperationAbility]:
        ctx = environment.Environment()
        child, owner = await start_terminal_operation_ability(ctx)
        terminal = dataclasses.replace(
            child.output_event.with_data(TerminalOperationOutputData(value="ambient")),
            id="ambient",
            source="spoofed",
            target=None,
        )
        await child.emit_terminal(ctx, terminal)
        return owner, child

    owner, child = asyncio.run(run())

    routed = [event for event in owner.events if event.name == TerminalOperationOutputEvent.name]
    assert len(routed) == 1
    assert routed[0].source == hsm.id(child)
    assert routed[0].target == hsm.id(owner)


def test_ability_operation_model_helper_is_removed_from_ability_sources() -> None:
    ability_sources = pathlib.Path("src/bot/abilities").rglob("*.py")

    for source_path in ability_sources:
        source = source_path.read_text()
        assert "ability_operation_model" not in source, source_path


def test_ability_does_not_cache_transient_output_or_failure() -> None:
    ability = abilities.Ability[object, object]()

    assert not hasattr(ability, "last_output")
    assert not hasattr(ability, "last_failure")


def test_result_bridge_helper_is_removed() -> None:
    assert not hasattr(ability_module, "apply" + "_ability")
    assert not hasattr(abilities_module, "apply" + "_ability")
    assert "apply" + "_ability" not in ability_module.__all__


def test_ability_owner_is_claimed_and_cleared_by_lifecycle_events() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        ctx = hsm.Context()
        child = abilities.Ability[object, object]()
        owner = AbilityTerminalOwner()

        _ = await bot.started(ctx, owner, require_model(owner.model))
        _ = await child.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        _ = await child.detach(
            ctx,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )

        return owner.lifecycle

    lifecycle = asyncio.run(run())

    assert [event.name for event in lifecycle] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachedEvent.name,
    ]
    created = lifecycle[0].data
    removed = lifecycle[1].data
    assert created is not None and created.created
    assert removed is not None and removed.removed


def test_ordinary_ability_ignores_composite_terminal_events() -> None:
    ordinary = require_model(text.TextGeneration.model)

    assert "bot.ability.attachment.terminal" not in ordinary.events


def test_composite_ability_waits_for_submodel_readiness_before_reporting_attached() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ctx = hsm.Context()
        owner = AbilityTerminalOwner()
        composite = CompositeAbility()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        request = attachment.AttachData(actor=owner)

        _ = await composite.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(request, "composite-attach"),
        )
        assert owner.lifecycle == []
        assert composite.state().endswith("/attached/behavior/initializing")

        await composite.dispatch_terminal(
            ctx,
            request,
            attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=composite, created=True)),
            "composite-attach",
        )
        return owner.lifecycle, composite.state()

    lifecycle, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.AttachCompleteEvent.name]
    assert lifecycle[0].id == "composite-attach"
    assert state.endswith("/attached/behavior/operational")


def test_composite_ability_rejects_uncorrelated_private_terminal() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ctx = hsm.Context()
        owner = AbilityTerminalOwner()
        composite = CompositeAbility()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        _ = await composite.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        await composite.dispatch_uncorrelated_terminal(ctx)
        return owner.lifecycle, composite.state()

    lifecycle, state = asyncio.run(run())

    assert lifecycle == []
    assert state.endswith("/attached/behavior/initializing")


def test_composite_ability_rejects_substituted_reply_identity() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, str]:
        ctx = hsm.Context()
        owner = AbilityTerminalOwner()
        composite = CompositeAbility()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        request = attachment.AttachData(actor=owner)
        _ = await composite.attach(ctx, attachment.AttachEvent.with_data(request))
        composite.hold_terminal = True
        await composite.dispatch_terminal(
            ctx,
            request,
            attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=composite, created=True)),
            "identity-bound",
        )
        await composite.dispatch_substituted_reply_terminal(ctx)
        rejected_state = composite.state()
        await composite.dispatch_held_terminal(ctx)
        await composite.dispatch_held_terminal(ctx)
        return owner.lifecycle, rejected_state, composite.state()

    lifecycle, rejected_state, accepted_state = asyncio.run(run())

    assert rejected_state.endswith("/attached/behavior/initializing")
    assert [event.name for event in lifecycle] == [attachment.AttachCompleteEvent.name]
    assert accepted_state.endswith("/attached/behavior/operational")


def test_composite_ability_initialization_failure_releases_owner_for_retry() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ctx = hsm.Context()
        owner = AbilityTerminalOwner()
        composite = CompositeAbility()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        request = attachment.AttachData(actor=owner)

        _ = await composite.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(request, "composite-failed"),
        )
        await composite.dispatch_terminal(
            ctx,
            request,
            attachment.AttachFailedEvent.with_data(
                attachment.FailedData(
                    actor=composite,
                    kind=attachment.FailureKind.INITIALIZATION,
                    message="children failed",
                )
            ),
            "composite-failed",
        )
        _ = await composite.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(request, "composite-retry"),
        )
        return owner.lifecycle, composite.state()

    lifecycle, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.AttachFailedEvent.name]
    assert lifecycle[0].id == "composite-failed"
    assert state.endswith("/attached/behavior/initializing")


def test_composite_ability_waits_for_private_detach_terminal() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, str]:
        ctx = hsm.Context()
        owner = AbilityTerminalOwner()
        composite = CompositeAbility()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        attach_request = attachment.AttachData(actor=owner)
        detach_request = attachment.DetachData(actor=owner)
        _ = await composite.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(attach_request, "composite-attach"),
        )
        await composite.dispatch_terminal(
            ctx,
            attach_request,
            attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=composite, created=True)),
            "composite-attach",
        )
        owner.lifecycle.clear()

        _ = await composite.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(detach_request, "composite-detach"),
        )
        waiting_state = composite.state()
        assert owner.lifecycle == []
        await composite.dispatch_terminal(
            ctx,
            detach_request,
            attachment.DetachedEvent.with_data(attachment.DetachedData(actor=composite, removed=True)),
            "composite-detach",
        )
        return owner.lifecycle, waiting_state, composite.state()

    lifecycle, waiting_state, final_state = asyncio.run(run())

    assert waiting_state.endswith("/attached/behavior/detaching")
    assert [event.name for event in lifecycle] == [attachment.DetachedEvent.name]
    assert lifecycle[0].id == "composite-detach"
    assert final_state.endswith("/detached")


def test_ability_reports_correlated_attachment_success_and_conflict() -> None:
    async def run() -> tuple[
        AbilityTerminalOwner,
        AbilityTerminalOwner,
        list[hsm.Event[typing.Any]],
        list[hsm.Event[typing.Any]],
    ]:
        ctx = hsm.Context()
        child = abilities.Ability[object, object]()
        owner = AbilityTerminalOwner()
        other_owner = AbilityTerminalOwner()
        _ = await bot.started(ctx, owner, require_model(owner.model))
        _ = await bot.started(ctx, other_owner, require_model(other_owner.model))
        _ = await bot.started(ctx, child, require_model(child.model))

        await child.dispatch(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "attach-owner",
            ),
        )
        await child.dispatch(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=other_owner),
                "attach-conflict",
            ),
        )

        return owner, other_owner, owner.lifecycle, other_owner.lifecycle

    owner, other_owner, owner_lifecycle, other_lifecycle = asyncio.run(run())

    assert [event.name for event in owner_lifecycle] == [attachment.AttachCompleteEvent.name]
    assert owner_lifecycle[0].id == "attach-owner"
    attached = owner_lifecycle[0].data
    assert isinstance(attached, attachment.AttachCompleteData)
    assert attached.actor is owner
    assert attached.created
    assert [event.name for event in other_lifecycle] == [attachment.AttachFailedEvent.name]
    assert other_lifecycle[0].id == "attach-conflict"
    failure = other_lifecycle[0].data
    assert isinstance(failure, attachment.FailedData)
    assert failure.actor is other_owner
    assert failure.kind is attachment.FailureKind.CONFLICT


def test_ability_owner_public_mutators_are_removed() -> None:
    ability = abilities.Ability[object, object]()

    assert not hasattr(ability, "owner")
    assert not hasattr(ability, "claim_owner")
    assert not hasattr(ability, "clear_owner")
    assert not hasattr(ability_module, "claim_ability_owner")


def test_ability_attach_does_not_poll_active_state(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_sleep(delay: float) -> None:
        del delay
        raise AssertionError("Ability attach must not poll active state.")

    async def run() -> str:
        monkeypatch.setattr(asyncio, "sleep", fail_sleep)
        generation = RecordingTextGeneration(generator=EchoTextGenerator())
        ctx = hsm.Context()
        owner = AbilityTerminalOwner()
        _ = await bot.started(ctx, owner, require_model(owner.model))

        _ = await generation.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )

        return generation.state()

    state = asyncio.run(run())

    assert state.endswith("/attached/behavior/idle")


def test_ability_attach_source_does_not_poll_active_state() -> None:
    source = pathlib.Path(ability_module.__file__).read_text()

    assert ".state()" not in source
    assert "_await_ability_initialized" not in source
    assert "asyncio.sleep(0)" not in source


def test_production_code_does_not_call_private_apply_hook_directly() -> None:
    root = pathlib.Path(__file__).resolve().parents[3]
    references: set[str] = set()
    for root_name in ("src", "examples/phone_bot/src"):
        for source_path in (root / root_name).rglob("*.py"):
            relative_path = source_path.relative_to(root).as_posix()
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            visitor = _DirectApplyReferenceVisitor(relative_path)
            visitor.visit(tree)
            references.update(visitor.references)

    disallowed = {reference for reference in references if not _is_modeled_ability_apply_hook_reference(reference)}
    assert disallowed == set()


def _is_modeled_ability_apply_hook_reference(reference: str) -> bool:
    path, qualname, kind = reference.split(":")
    del path
    return kind == "attribute" and qualname.endswith(
        (
            "._run_behavior_activity",
            ".run_behavior_activity",
        )
    )


def test_apply_dispatches_input_event_without_result_bridge() -> None:
    async def run() -> tuple[object, list[text.OutputData]]:
        generation = RecordingTextGeneration(generator=EchoTextGenerator())
        ctx, owner = await start_recorded_generation(generation)

        result = await hsm.dispatch(
            ctx,
            generation,
            generation.input_event.with_data_and_id(
                text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)),
                uuid.uuid4().hex,
            ),
        )
        for _ in range(100):
            if owner.outputs:
                break
            await asyncio.sleep(0)

        return result, typing.cast(list[text.OutputData], owner.outputs)

    result, outputs = asyncio.run(run())

    assert result is None
    assert outputs == [text.OutputData(content="HELLO")]


def test_public_output_event_does_not_complete_in_flight_ability() -> None:
    async def run() -> None:
        release = asyncio.Event()
        generation = RecordingTextGeneration(generator=SlowTextGenerator(release))
        ctx, owner = await start_recorded_generation(generation)

        await hsm.dispatch(
            ctx,
            generation,
            generation.input_event.with_data_and_id(
                text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="real"),)),
                uuid.uuid4().hex,
            ),
        )
        await hsm.dispatch(
            ctx,
            generation,
            generation.output_event.with_data(text.OutputData(content="spoofed")),
        )

        assert owner.outputs == []
        assert generation.state() == "/RecordingTextGenerationLifecycle/attached/behavior/applying"

        release.set()
        for _ in range(100):
            if owner.outputs[-1:] == [text.OutputData(content="REAL")]:
                break
            await asyncio.sleep(0)

        assert owner.outputs == [text.OutputData(content="REAL")]
        assert generation.state() == "/RecordingTextGenerationLifecycle/attached/behavior/idle"

    asyncio.run(run())


def test_repeated_input_is_deferred_while_ability_is_applying() -> None:
    async def run() -> None:
        release_first = asyncio.Event()
        calls: list[str] = []
        generation = RecordingTextGeneration(generator=RecordingDelayedTextGenerator(calls, release_first))
        ctx, owner = await start_recorded_generation(generation)

        await hsm.dispatch(
            ctx,
            generation,
            generation.input_event.with_data_and_id(
                text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="first"),)),
                uuid.uuid4().hex,
            ),
        )
        await hsm.dispatch(
            ctx,
            generation,
            generation.input_event.with_data_and_id(
                text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="second"),)),
                uuid.uuid4().hex,
            ),
        )

        assert calls == ["first"]
        release_first.set()
        for _ in range(100):
            if calls == ["first", "second"] and owner.outputs[-1:] == [text.OutputData(content="SECOND")]:
                break
            await asyncio.sleep(0)

        assert calls == ["first", "second"]
        assert owner.outputs == [text.OutputData(content="FIRST"), text.OutputData(content="SECOND")]
        assert generation.state() == "/RecordingTextGenerationLifecycle/attached/behavior/idle"

    asyncio.run(run())


def test_text_generation_rejects_input_event_with_wrong_payload_type() -> None:
    async def run() -> None:
        generation = RecordingTextGeneration(generator=EchoTextGenerator())
        ctx, owner = await start_recorded_generation(generation)

        await hsm.dispatch(
            ctx,
            generation,
            abilities.InputEvent.with_data("not text generation input"),
        )

        assert generation.state() == "/RecordingTextGenerationLifecycle/attached/behavior/idle"
        assert owner.outputs == []

        await hsm.dispatch(
            ctx,
            generation,
            generation.input_event.with_data_and_id(
                text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)),
                uuid.uuid4().hex,
            ),
        )
        for _ in range(100):
            if owner.outputs:
                break
            await asyncio.sleep(0)

        assert owner.outputs == [text.OutputData(content="HELLO")]

    asyncio.run(run())


def test_text_generation_routes_wrong_output_type_to_failure() -> None:
    async def run() -> None:
        generation = RecordingTextGeneration(generator=WrongTextGenerator())
        ctx, owner = await start_recorded_generation(generation)

        await hsm.dispatch(
            ctx,
            generation,
            generation.input_event.with_data_and_id(
                text.InputData(messages=(text.TextMessage(role=text.TextRole.USER, content="hello"),)),
                uuid.uuid4().hex,
            ),
        )
        for _ in range(100):
            if owner.failures:
                break
            await asyncio.sleep(0)

        assert generation.state() == "/RecordingTextGenerationLifecycle/attached/behavior/idle"
        assert owner.outputs == []
        assert len(owner.failures) == 1
        assert "output schema" in typing.cast(abilities.FailureData, owner.failures[0]).message

    asyncio.run(run())


def test_ability_production_stop_then_attach_restarts() -> None:
    """``hsm.stop(ability)`` then attach must restart without RuntimeError (hsm 1.3.2)."""

    async def run() -> None:
        generation = RecordingTextGeneration(generator=EchoTextGenerator())
        ctx, owner = await start_recorded_generation(generation)
        assert bot.lifecycle.is_started(generation) is True
        await hsm.stop(generation)
        assert bot.lifecycle.is_started(generation) is False
        await generation.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        assert bot.lifecycle.is_started(generation) is True

    asyncio.run(run())


def test_ability_detach_when_stopped_emits_detach_failed() -> None:
    """Stopped ability detach with reply sink must surface DetachFailed, not silent drop."""

    async def run() -> list[hsm.Event[typing.Any]]:
        generation = RecordingTextGeneration(generator=EchoTextGenerator())
        ctx, owner = await start_recorded_generation(generation)
        await hsm.stop(generation)
        failures: list[hsm.Event[typing.Any]] = []

        def record_failure(_ctx: hsm.Context, _instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
            failures.append(event)

        reply_model = bot.define(
            "DetachReply",
            hsm.initial(hsm.target("s")),
            hsm.state(
                "s",
                hsm.transition(hsm.on(attachment.DetachFailedEvent), hsm.effect(record_failure)),
            ),
        )
        reply = hsm.Instance()
        await bot.started(ctx, reply, reply_model)
        await generation.detach(
            ctx,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner, reply_to=reply)),
        )
        for _ in range(20):
            if failures:
                break
            await asyncio.sleep(0)
        return failures

    failures = asyncio.run(run())
    assert len(failures) == 1
    assert failures[0].name == attachment.DetachFailedEvent.name
