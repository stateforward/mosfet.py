"""Shared fixtures for memory tests after SQL-transaction hard cut."""

from bot import abilities
from bot.abilities import memory
from bot.protocols import attachment

import asyncio
import collections.abc
import typing

import hsm

from tests.hsm_instance_state import ability_terminal_owner, remember_ability_terminal_owner

_TResult = typing.TypeVar("_TResult")


class MemoryEncoder(abilities.Encoder[str, bytes]):
    @typing.override
    async def encode(self, input: str) -> bytes:
        return input.encode()


class MemoryDecoder(abilities.Decoder[bytes, str]):
    @typing.override
    async def decode(self, input: bytes) -> str:
        return input.decode()


class PreferenceMemoryGenerator(memory.MemoryGenerator):
    @typing.override
    async def generate(self, input: memory.SourceData) -> memory.GeneratedMemory:
        assert input.decoded == "Gabe prefers terse handoff notes."
        return memory.GeneratedMemory(
            content="The operator prefers terse handoff notes.",
            kind=memory.MemoryClassificationKind.PREFERENCE,
            subject_ref="operator",
        )


class GeneratedMemoryEncoder(abilities.Encoder[memory.GeneratedMemory, memory.EncodedMemory]):
    @typing.override
    async def encode(self, input: memory.GeneratedMemory) -> memory.EncodedMemory:
        return memory.EncodedMemory(format="text/plain", value=input.content)


class GeneratedMemoryDecoder(abilities.Decoder[memory.EncodedMemory, memory.GeneratedMemory]):
    @typing.override
    async def decode(self, input: memory.EncodedMemory) -> memory.GeneratedMemory:
        content = input.value if isinstance(input.value, str) else str(input.value)
        return memory.GeneratedMemory(content=content)


class WrongGeneratedMemoryEncoder(abilities.Encoder[memory.GeneratedMemory, memory.EncodedMemory]):
    @typing.override
    async def encode(self, input: memory.GeneratedMemory) -> memory.EncodedMemory:
        del input
        raise RuntimeError("encoder failed")


class RetainPreferenceClassifier(memory.MemoryClassifier):
    @typing.override
    async def classify(self, input: memory.classification.InputData) -> memory.classification.OutputData:
        del input
        return memory.classification.OutputData(
            retention=memory.MemoryClassificationRetention.RETAIN,
            kind=memory.MemoryClassificationKind.PREFERENCE,
            sensitivity=memory.MemoryClassificationSensitivity.STANDARD,
            confidence=0.93,
        )


class WrongMemoryClassifier(memory.MemoryClassifier):
    @typing.override
    async def classify(self, input: memory.classification.InputData) -> memory.classification.OutputData:
        del input
        raise RuntimeError("classifier failed")


class RecordingMemoryClassification(memory.MemoryClassification):
    outputs: list[memory.classification.OutputData]
    failures: list[object]

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.outputs = []
        self.failures = []


def classification_output(
    retention: memory.MemoryClassificationRetention = memory.MemoryClassificationRetention.RETAIN,
    *,
    kind: memory.MemoryClassificationKind = memory.MemoryClassificationKind.PREFERENCE,
    sensitivity: memory.MemoryClassificationSensitivity = memory.MemoryClassificationSensitivity.STANDARD,
    confidence: float = 0.93,
) -> memory.classification.OutputData:
    return memory.classification.OutputData(
        retention=retention,
        kind=kind,
        sensitivity=sensitivity,
        confidence=confidence,
    )


def preference_memory_generation() -> memory.MemoryGeneration:
    return memory.MemoryGeneration(
        generator=PreferenceMemoryGenerator(),
        encoder=GeneratedMemoryEncoder(),
    )


def generated_candidate() -> memory.CandidateData:
    return memory.CandidateData(
        memory=memory.GeneratedMemory(
            content="The operator prefers terse handoff notes.",
            kind=memory.MemoryClassificationKind.PREFERENCE,
            subject_ref="operator",
        ),
        encoded=memory.EncodedMemory(
            format="text/plain",
            value="The operator prefers terse handoff notes.",
        ),
    )


class RecordingShortTermMemory(memory.ShortTermMemory):
    inputs: list[memory.InputData]
    outputs: list[memory.OutputData]
    failures: list[object]

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.inputs = []
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.input_event.name and isinstance(event.data, memory.InputData):
            self.inputs.append(event.data)
        return super().dispatch(ctx, event)


class RecordingLongTermMemory(memory.LongTermMemory):
    inputs: list[memory.InputData]
    outputs: list[memory.OutputData]
    failures: list[object]

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.inputs = []
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.input_event.name and isinstance(event.data, memory.InputData):
            self.inputs.append(event.data)
        return super().dispatch(ctx, event)


class RecordingMemory(memory.Memory):
    inputs: list[memory.InputData]
    outputs: list[memory.OutputData]
    failures: list[object]

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.inputs = []
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.input_event.name and isinstance(event.data, memory.InputData):
            self.inputs.append(event.data)
        return super().dispatch(ctx, event)


def _record_memory_fixture_terminal_mirror_event(
    ctx: hsm.Context,
    instance: "MemoryFixtureAbilityTerminalMirror",
    event: hsm.Event[typing.Any],
) -> None:
    del ctx
    instance.record(event)


class MemoryFixtureAbilityTerminalMirror(hsm.Instance):
    model: typing.ClassVar[hsm.Model | None] = hsm.define(
        "MemoryFixtureAbilityTerminalMirror",
        hsm.initial(hsm.target("/MemoryFixtureAbilityTerminalMirror/recording")),
        hsm.state(
            "recording",
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.effect(_record_memory_fixture_terminal_mirror_event),
            ),
        ),
    )
    ability: abilities.Ability[typing.Any, typing.Any]
    results: dict[str, asyncio.Future[object]]

    def __init__(self, ability: abilities.Ability[typing.Any, typing.Any]) -> None:
        super().__init__()
        self.ability = ability
        self.results = {}

    def result_for(self, operation_id: str) -> asyncio.Future[object]:
        result = self.results.get(operation_id)
        if result is None:
            result = asyncio.get_running_loop().create_future()
            self.results[operation_id] = result
        return result

    def record(self, event: hsm.Event[typing.Any]) -> None:
        operation_id = event.id if event.id else None
        result = self.results.get(operation_id) if operation_id is not None else None
        if event.name == self.ability.output_event.name:
            values = getattr(self.ability, "outputs", None)
            if isinstance(values, list):
                values.append(event.data)
            if result is not None and not result.done():
                result.set_result(event.data)
        if event.name == self.ability.failed_event.name:
            values = getattr(self.ability, "failures", None)
            if isinstance(values, list):
                values.append(event.data)
            if result is not None and not result.done():
                failure = event.data
                message = failure.message if isinstance(failure, abilities.FailureData) else str(failure)
                result.set_exception(RuntimeError(message))


async def start_ability_tree(*abilities: abilities.Ability[typing.Any, typing.Any]) -> None:
    ctx = hsm.Context()
    for ability in abilities:
        owner = MemoryFixtureAbilityTerminalMirror(ability)
        assert owner.model is not None
        _ = await hsm.started(ctx, owner, owner.model)
        _ = await ability.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        remember_ability_terminal_owner(ability, owner)


async def stop_ability_tree(*abilities: abilities.Ability[typing.Any, typing.Any]) -> None:
    for ability in abilities:
        owner = ability_terminal_owner(ability)
        assert owner is not None
        _ = await ability.detach(
            ability.context(),
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met before timeout")


def require_model(ability: abilities.Ability[typing.Any, typing.Any] | hsm.Model | None) -> hsm.Model:
    if isinstance(ability, abilities.Ability):
        assert ability.model is not None
        return ability.model
    assert ability is not None
    return ability
