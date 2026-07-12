from bot import abilities
from bot.abilities.language import text

import asyncio
import ast
import collections.abc
import importlib.util
import pathlib
import typing
import uuid
from typing import override

import hsm
import pytest
import bot.abilities as abilities_module
import bot.abilities.ability as ability_module

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
    model: typing.ClassVar[hsm.Model | None] = hsm.define(
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

    def __init__(self) -> None:
        super().__init__()
        self.outputs = []
        self.failures = []

    def record(self, event: hsm.Event[typing.Any]) -> None:
        if event.name.endswith(".output"):
            self.outputs.append(event.data)
        if event.name.endswith(".failed"):
            self.failures.append(event.data)

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

async def start_recorded_generation(
    generation: RecordingTextGeneration,
) -> tuple[hsm.Context, AbilityTerminalOwner]:
    ctx = hsm.Context()
    owner = AbilityTerminalOwner()
    _ = await hsm.started(ctx, owner, require_model(owner.model))
    _ = await generation.attach(owner=owner, ctx=ctx)
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
    assert input_schema["description"]
    assert input_schema["examples"] == ["Summarize this note."]
    assert "input" not in object_dict(input_schema.get("properties", {}))
    assert abilities.OutputEvent.name == "bot.ability.output"
    assert output_schema["description"]
    assert output_schema["examples"] == ["Summary text."]
    assert "output" not in object_dict(output_schema.get("properties", {}))

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
    async def run() -> tuple[bool, hsm.Instance | None]:
        ctx = hsm.Context()
        child = abilities.Ability[object, object]()
        owner = abilities.Ability[object, object]()

        _ = await child.attach(owner=owner, ctx=ctx)
        claimed_owner = abilities.Ability.current_owner(child)
        _ = await child.detach(ctx=ctx)

        return claimed_owner is owner, abilities.Ability.current_owner(child)

    claimed_expected_owner, detached_owner = asyncio.run(run())

    assert claimed_expected_owner
    assert detached_owner is None

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
        _ = await hsm.started(ctx, owner, require_model(owner.model))

        _ = await generation.attach(owner=owner, ctx=ctx)

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
