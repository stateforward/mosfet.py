from bot import abilities
import bot.lifecycle
import bot
from bot.abilities import cognition
from bot.abilities import encoding
from bot.abilities import listening
from bot.abilities import memory
from bot.abilities import processing
from bot.abilities import speaking
from bot.abilities.cognition import Cognition
from bot.abilities.hearing import sound as sound_hearing
from bot.abilities.hearing import speech
from bot.abilities.hearing import voice
from bot.devices import audio
from bot.devices import phone as phone_device

import asyncio
from pathlib import Path
import collections.abc
import dataclasses
import datetime
import inspect
import typing
import uuid

import hsm
import pydantic
import pytest

from tests.bot.abilities.cognition.metadata_contract import assert_metadata_key_prefix_is_absent

from bot.bot import Bot
import bot.bot as bot_module
from bot.device import Device
from bot import event_schema
from bot.event_schema import event_json_schema
from bot.protocols import attachment

from bot.environment import SoundData, SoundEvent, VisualData, VisualEvent, Environment, space
from tests.hsm_instance_state import (
    device_firmware,
    bot_has_focus,
    device_bots,
    phone_microphone,
    phone_speaker,
    start_ability_tree,
)
from tests.hsm_model import transition_map
from tests.type_helpers import object_dict


def no_output(reason: str = "") -> cognition.types.OutputData:
    del reason
    return ()


def focus_output(device: str, reason: str) -> cognition.types.OutputData:
    return (
        cognition.types.EventData(
            event=bot.FocusDeviceEvent.name,
            data=bot.FocusDeviceEventData(device=device).model_dump(),
            reason=reason,
        ),
    )


def clear_output(reason: str) -> cognition.types.OutputData:
    return (cognition.types.EventData(event=bot.ClearFocusEvent.name, reason=reason),)


def event_data_schema(event: hsm.Event[typing.Any]) -> dict[str, object]:
    return event_json_schema(event)


def input_priority(input: bot.BotInputData) -> int:
    if isinstance(input, bot.InputEventData):
        return input.priority
    return 0


def assert_heard_phone_ring(
    input: bot.BotInputData,
    *,
    phone: phone_device.Phone | None = None,
    caller: str | None = None,
) -> None:
    """Ringing is environment.sound from the phone; cognition sees that sound stimulus, not phone.ringing."""

    assert isinstance(input, hsm.Event)
    assert input.name == SoundEvent.name
    assert isinstance(input.data, SoundData)
    assert input.data.kind == "phone.ringing"
    if caller is not None:
        assert isinstance(input.data, phone_device.PhoneSoundData)
        assert input.data.caller == caller
    if phone is not None:
        assert input.source == hsm.id(phone)


class ProbeProcessor(processing.Processor):
    """Minimal ability used to exercise innate/acquired lifecycle on Bot."""

    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return processing.coerce_event_selections(no_output("probe")) or ()


class ProbeAbility(processing.Processing):
    def __init__(self) -> None:
        self._probe = ProbeProcessor()
        super().__init__(processor=self._probe)

    @property
    def calls(self) -> list[processing.InputData]:
        return self._probe.calls


class IgnoreProcessor(processing.Processor):
    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return processing.coerce_event_selections(no_output(f"priority:{input_priority(input.input)}")) or ()


class IgnoreAbility(processing.Processing):
    def __init__(self) -> None:
        self._ignore = IgnoreProcessor()
        super().__init__(processor=self._ignore)

    @property
    def calls(self) -> list[processing.InputData]:
        return self._ignore.calls


class MetadataRecordingProcessor(IgnoreProcessor):
    input_event_metadata: list[dict[str, object]]

    def __init__(self) -> None:
        super().__init__()
        self.input_event_metadata = []


class MetadataRecordingAbility(processing.Processing):
    def __init__(self) -> None:
        self._meta = MetadataRecordingProcessor()
        super().__init__(processor=self._meta)

    @property
    def calls(self) -> list[processing.InputData]:
        return self._meta.calls

    @property
    def input_event_metadata(self) -> list[dict[str, object]]:
        return self._meta.input_event_metadata

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.input_event.name:
            self._meta.input_event_metadata.append(dict(event.metadata))
        return super().dispatch(ctx, event)


def _as_output_data(
    value: cognition.types.OutputData | cognition.types.EventData,
) -> cognition.types.OutputData:
    if isinstance(value, cognition.types.EventData):
        return (value,)
    return value


class SequenceProcessor(processing.Processor):
    calls: list[processing.InputData]
    outputs: list[cognition.types.OutputData]

    def __init__(
        self,
        *outputs: cognition.types.OutputData | cognition.types.EventData,
    ) -> None:
        self.calls = []
        self.outputs = [_as_output_data(item) for item in outputs]

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return processing.coerce_event_selections(self.outputs.pop(0)) or ()


class SequenceAbility(processing.Processing):
    def __init__(
        self,
        *outputs: cognition.types.OutputData | cognition.types.EventData,
    ) -> None:
        self._seq = SequenceProcessor(*outputs)
        super().__init__(processor=self._seq)

    @property
    def calls(self) -> list[processing.InputData]:
        return self._seq.calls


class BlockingSequenceProcessor(processing.Processor):
    calls: list[processing.InputData]
    outputs: list[cognition.types.OutputData]
    release: asyncio.Event
    block_on_call: int

    def __init__(
        self, *, release: asyncio.Event, block_on_call: int, outputs: tuple[cognition.types.OutputData, ...]
    ) -> None:
        self.calls = []
        self.outputs = list(outputs)
        self.release = release
        self.block_on_call = block_on_call

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        if len(self.calls) == self.block_on_call:
            _ = await self.release.wait()
        return processing.coerce_event_selections(self.outputs.pop(0)) or ()


class BlockingSequenceAbility(processing.Processing):
    def __init__(
        self, *, release: asyncio.Event, block_on_call: int, outputs: tuple[cognition.types.OutputData, ...]
    ) -> None:
        self._seq = BlockingSequenceProcessor(release=release, block_on_call=block_on_call, outputs=outputs)
        super().__init__(processor=self._seq)

    @property
    def calls(self) -> list[processing.InputData]:
        return self._seq.calls


class BlockingProcessor(processing.Processor):
    calls: list[int]
    release: asyncio.Event

    def __init__(self, *, release: asyncio.Event) -> None:
        self.calls = []
        self.release = release

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        priority = input_priority(input.input)
        self.calls.append(priority)
        if len(self.calls) == 1:
            _ = await self.release.wait()
        return processing.coerce_event_selections(no_output(f"priority:{priority}")) or ()


class BlockingAbility(processing.Processing):
    def __init__(self, *, release: asyncio.Event) -> None:
        self._blocking = BlockingProcessor(release=release)
        super().__init__(processor=self._blocking)

    @property
    def calls(self) -> list[int]:
        return self._blocking.calls


class FailingProcessor(processing.Processor):
    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        raise RuntimeError("ability unavailable")


class FailingAbility(processing.Processing):
    def __init__(self) -> None:
        super().__init__(processor=FailingProcessor())


class NestedPrimaryProcessor(processing.Processor):
    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return processing.coerce_event_selections(no_output("nested affordance")) or ()


class NestedPrimaryAbility(processing.Processing):
    nested: ProbeAbility
    child_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...]

    def __init__(self) -> None:
        self.nested = ProbeAbility()
        self._nested_proc = NestedPrimaryProcessor()
        super().__init__(processor=self._nested_proc)
        self.child_abilities = (self.nested,)

    @property
    def calls(self) -> list[processing.InputData]:
        return self._nested_proc.calls


class HangingProcessor(processing.Processor):
    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        _ = await asyncio.Event().wait()
        raise AssertionError("unreachable")


class HangingAbility(processing.Processing):
    def __init__(self) -> None:
        super().__init__(processor=HangingProcessor())


class CancellableHangingProcessor(processing.Processor):
    calls: list[int]
    cancelled: bool

    def __init__(self) -> None:
        self.calls = []
        self.cancelled = False

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        priority = input_priority(input.input)
        self.calls.append(priority)
        if len(self.calls) > 1:
            return processing.coerce_event_selections(no_output(f"priority:{priority}")) or ()
        try:
            _ = await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


class CancellableHangingAbility(processing.Processing):
    def __init__(self) -> None:
        self._hang = CancellableHangingProcessor()
        super().__init__(processor=self._hang)

    @property
    def calls(self) -> list[int]:
        return self._hang.calls

    @property
    def cancelled(self) -> bool:
        return self._hang.cancelled


class CapturingCognition(cognition.Cognition):
    input_events: list[hsm.Event[typing.Any]]
    swallow_cancel: bool

    def __init__(self, processor: processing.Processor, *, swallow_cancel: bool = False) -> None:
        self.input_events = []
        self.swallow_cancel = swallow_cancel
        super().__init__(
            intuition=cognition.Intuition(processor=processor),
            reasoning=cognition.Reasoning(processor=_NoopReasoningProcessor()),
            reflection=_reflection_for_test(),
        )

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == cognition.InputEvent.name:
            self.input_events.append(event)
        if self.swallow_cancel and event.name == cognition.CancelEvent.name:
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return future
        return super().dispatch(ctx, event)


class _NoopReasoningProcessor(processing.Processor):
    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        return ()


def _reflection_for_test() -> cognition.Reflection:
    return cognition.Reflection(processor=_NoopReasoningProcessor(), memory=memory.Memory())


class InputRecordingCognition(Cognition):
    inputs: list[cognition.InputData]

    def __init__(self, processing_ability: processing.Processing) -> None:
        self.inputs = []
        super().__init__(
            intuition=cognition.Intuition(processor=processing_ability.processor),
            reasoning=cognition.Reasoning(processor=_NoopReasoningProcessor()),
            reflection=_reflection_for_test(),
        )

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == cognition.InputEvent.name and isinstance(event.data, cognition.InputData):
            self.inputs.append(event.data)
        return super().dispatch(ctx, event)


def as_cognition(
    value: abilities.Ability[typing.Any, typing.Any] | processing.Processing | processing.Processor,
) -> abilities.Ability[typing.Any, typing.Any]:
    """Bot cognition is Cognition; wrap a leaf Processor as intuition for tests."""

    if isinstance(value, cognition.Cognition):
        return value
    if isinstance(value, processing.Processor) and not isinstance(value, processing.Processing):
        processor: processing.Processor = value
    elif isinstance(value, processing.Processing):
        processor = value.processor
    else:
        raise TypeError("as_cognition expects Cognition, Processing, or Processor")
    return cognition.Cognition(
        intuition=cognition.Intuition(processor=processor),
        reasoning=cognition.Reasoning(processor=_NoopReasoningProcessor()),
        reflection=_reflection_for_test(),
    )


class _StubCancelData(pydantic.BaseModel):
    """Cancel payload for the contract stub cognition."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)
    token: str = pydantic.Field(min_length=1)


_StubCancelEvent = hsm.Event[_StubCancelData](name="test.stub.cancel", schema=_StubCancelData)


class _BaseStubCognition(abilities.Ability[cognition.InputData, typing.Any]):
    """Cognition stub that records cancel dispatches and never produces output."""

    submodel: typing.ClassVar[hsm.Model] = hsm.define(
        "StubCognition",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
    )
    received: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.received = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name in {_StubCancelEvent.name, processing.CancelEvent.name}:
            self.received.append(event)
        return super().dispatch(ctx, event)


class StubCognition(_BaseStubCognition):
    cancel_event: typing.ClassVar[hsm.Event[typing.Any] | None] = _StubCancelEvent


class DefaultCognition(_BaseStubCognition):
    """No cancel_event override: the bot must fall back to processing.CancelEvent."""


class TimeoutStubCognitionAgent(Bot):
    _processing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=1)

    def __init__(self, cognition: abilities.Ability[cognition.InputData, typing.Any]) -> None:
        super().__init__(devices=configured_devices("phone"), cognition=cognition)


class BasicAgent(Bot):
    actions: list[cognition.types.OutputData]
    failures: list[bot.ProcessingFailedEventData]

    def __init__(self, devices: collections.abc.Mapping[str, Device]) -> None:
        super().__init__(devices=devices, cognition=as_cognition(IgnoreAbility()))
        self.actions = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == bot.ProcessingCompletedEvent.name:
            completed = event.data
            assert isinstance(completed, bot.ProcessingCompletedEventData)
            self.actions.append(typing.cast(cognition.types.OutputData, completed.output))
        if event.name == bot.ProcessingFailedEvent.name:
            failure = event.data
            assert isinstance(failure, bot.ProcessingFailedEventData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


class FocusedAgent(Bot):
    """Bot subclass with an innate probe ability (not attention/focus semantics)."""

    _innate_abilities: typing.ClassVar[tuple[type[abilities.Ability[typing.Any, typing.Any]], ...]] = (ProbeAbility,)
    probe: ProbeAbility

    def __init__(
        self,
        devices: collections.abc.Mapping[str, Device],
        *,
        cognition: processing.Processing | None = None,
        acquired_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
    ) -> None:
        super().__init__(
            devices=devices,
            cognition=as_cognition(cognition or IgnoreAbility()),
            acquired_abilities=acquired_abilities,
        )
        probe = self._innate_ability_instances[0]
        assert isinstance(probe, ProbeAbility)
        self.probe = probe


class AbilityAgent(Bot):
    actions: list[cognition.types.OutputData]
    failures: list[bot.ProcessingFailedEventData]

    def __init__(
        self,
        devices: collections.abc.Mapping[str, Device],
        *,
        cognition: abilities.Ability[typing.Any, typing.Any] | processing.Processing,
        input: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        output: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        acquired_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
    ) -> None:
        super().__init__(
            devices=devices,
            cognition=as_cognition(cognition),
            input=input,
            output=output,
            acquired_abilities=acquired_abilities,
        )
        self.actions = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == bot.ProcessingCompletedEvent.name:
            completed = event.data
            assert isinstance(completed, bot.ProcessingCompletedEventData)
            self.actions.append(typing.cast(cognition.types.OutputData, completed.output))
        if event.name == bot.ProcessingFailedEvent.name:
            failure = event.data
            assert isinstance(failure, bot.ProcessingFailedEventData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


class IdleAgent(AbilityAgent):
    """Bot whose moments come often enough to observe inside a test.

    Only the interval is shortened. The occasion is otherwise the production one: no guard,
    no directive, nothing about this bot that makes it want anything.
    """

    _idle_interval: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=5)
    occasions: list[hsm.Event[typing.Any]]

    def __init__(
        self,
        devices: collections.abc.Mapping[str, Device],
        *,
        cognition: abilities.Ability[typing.Any, typing.Any] | processing.Processing,
    ) -> None:
        self.occasions = []
        super().__init__(devices=devices, cognition=cognition)

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == bot.IdleEvent.name:
            self.occasions.append(event)
        return super().dispatch(ctx, event)


def idle_occasion(active_bot: Bot) -> hsm.Event[bot.IdleEventData]:
    """The occasion exactly as the body mints it: empty payload, self-addressed envelope."""

    return dataclasses.replace(
        bot.IdleEvent.with_data(bot.IdleEventData()),
        id=uuid.uuid4().hex,
        source=hsm.id(active_bot),
        target=hsm.id(active_bot),
    )


class TimeoutAbilityAgent(AbilityAgent):
    _processing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=1)


class QuickTimeoutAgent(AbilityAgent):
    _processing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=20)


class FastDeactivationTimeoutAgent(BasicAgent):
    _deactivation_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=10)


class SlowInitializingDevice(Device):
    _firmware_initializing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(seconds=1)
    release: asyncio.Event

    def __init__(self, release: asyncio.Event) -> None:
        super().__init__()
        self.release = release

    @typing.override
    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event
        _ = await self.release.wait()


class FailingInitializingDevice(SlowInitializingDevice):
    @typing.override
    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        await super()._after_firmware_started(ctx, event)
        raise RuntimeError("firmware failed")


class ImmediatelyFailingInitializingDevice(Device):
    @typing.override
    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event
        raise RuntimeError("firmware failed")


class LifecycleDispatchFailingDevice(Device):
    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name in {attachment.AttachEvent.name, attachment.DetachEvent.name}:
            raise RuntimeError("device lifecycle dispatch override failed")
        return super().dispatch(ctx, event)


class SnapshotFailingDevice(Device):
    fail_snapshots: bool

    def __init__(self) -> None:
        super().__init__()
        self.fail_snapshots = False

    @typing.override
    def take_snapshot(self) -> hsm.Snapshot:
        if not self.fail_snapshots:
            return super().take_snapshot()
        raise RuntimeError("snapshot unavailable")


class SnapshotFailingPhone(phone_device.Phone):
    fail_snapshots: bool

    def __init__(self) -> None:
        super().__init__()
        self.fail_snapshots = False

    @typing.override
    def take_snapshot(self) -> hsm.Snapshot:
        if not self.fail_snapshots:
            return super().take_snapshot()
        raise RuntimeError("snapshot unavailable")


class ContextRecordingPhone(phone_device.Phone):
    event_metadata: list[dict[str, object]]

    def __init__(self) -> None:
        super().__init__()
        self.event_metadata = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == phone_device.AnswerCallEvent.name:
            self.event_metadata.append(dict(event.metadata))
        return super().dispatch(ctx, event)


def basic_agent(*, devices: collections.abc.Mapping[str, Device] | None = None) -> BasicAgent:
    return BasicAgent(devices=devices or {})


def configured_devices(*references: str) -> dict[str, Device]:
    return {reference: Device() for reference in references}


async def start_bot_with_devices(active_bot: Bot, *, placement: space.Placement | None = None) -> Environment:
    environment = Environment()
    _ = await active_bot.attach(environment, placement=placement)
    await wait_until(lambda: active_bot.state() != "/Bot/activating")
    return environment


def test_cognition_reboot_request_cycles_bot_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[object] = []
    started = bot_module._BotProcessingOperation.started

    @classmethod
    async def spy_started(
        cls,
        ctx: hsm.Context,
        operation: bot_module._BotProcessingOperation,
        **kwargs: typing.Any,
    ) -> bot_module._BotProcessingOperation:
        operations.append(operation)
        return await started(ctx, operation, **kwargs)

    monkeypatch.setattr(bot_module._BotProcessingOperation, "started", spy_started)

    async def run() -> tuple[str, str, str, bool, list[bot.ProcessingFailedEventData]]:
        release = asyncio.Event()
        ability = BlockingSequenceAbility(
            release=release, block_on_call=1, outputs=(no_output("turn"), no_output("turn"))
        )
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)
        environment = await start_bot_with_devices(active_bot)
        cognition_ability = active_bot._cognition
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/processing" and len(ability.calls) == 1)
        _ = await hsm.dispatch(
            environment,
            active_bot,
            dataclasses.replace(
                bot.RebootEvent.with_data(bot.RebootEventData(reason="cognition_child_teardown_failed")),
                id="reboot-turn",
                source=hsm.id(cognition_ability),
                target=hsm.id(active_bot),
            ),
        )
        reboot_state = active_bot.state()
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        timer_stopped = bool(operations) and all(operation.state() in {"", "/BotProcessingTimer"} for operation in operations)
        _ = release.set()
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=4)),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused")
        return (
            reboot_state,
            active_bot.state(),
            cognition_ability.state(),
            timer_stopped,
            active_bot.failures,
            active_bot.actions,
        )

    reboot_state, final_state, cognition_state, timer_stopped, failures, actions = asyncio.run(run())

    assert reboot_state in {"/Bot/reboot_deactivating", "/Bot/reboot_cleanup_preserve", "/Bot/reboot_cleanup_reset"}
    assert final_state == "/Bot/active/focused"
    assert cognition_state.endswith("/idle")
    # Exiting processing stops the interrupted turn's timer; it can never fire into a later turn.
    assert timer_stopped
    assert failures == []
    assert actions == [no_output("turn")]


def test_cognition_reboot_request_during_activation_forces_cleanup_then_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, int]:
        active_bot = basic_agent()
        cognition_ability = active_bot._cognition
        attachment_group = active_bot._attachments
        original_attach = attachment.Group.attach
        first_attach_started = asyncio.Event()
        blocked = asyncio.Event()
        attach_calls = 0

        async def block_first_bot_attach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            nonlocal attach_calls
            if isinstance(event.data, attachment.AttachData) and event.data.actor is active_bot:
                attach_calls += 1
                if attach_calls == 1:
                    assert group is attachment_group
                    first_attach_started.set()
                    await blocked.wait()
                    return
            await original_attach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "attach", block_first_bot_attach)
        environment = Environment()
        _ = await active_bot.attach(environment)
        await first_attach_started.wait()
        assert active_bot.state() == "/Bot/activating"
        _ = await hsm.dispatch(
            environment,
            active_bot,
            dataclasses.replace(
                bot.RebootEvent.with_data(bot.RebootEventData(reason="cognition_detach_rollback_failed")),
                id="activation-reboot",
                source=hsm.id(cognition_ability),
                target=hsm.id(active_bot),
            ),
        )
        await wait_until(lambda: active_bot.state().startswith("/Bot/active/"))
        return active_bot.state(), attach_calls

    state, attach_calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert attach_calls == 2


def test_cognition_reboot_detach_timeout_resets_stuck_group_before_restart() -> None:
    class FirstDetachHangsAbility(ProbeAbility):
        detach_calls: int

        def __init__(self) -> None:
            super().__init__()
            self.detach_calls = 0

        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
            if event.name == attachment.DetachEvent.name:
                self.detach_calls += 1
                if self.detach_calls == 1:

                    async def hang() -> None:
                        _ = await asyncio.Event().wait()

                    return hang()
            return super().dispatch(ctx, event)

    class FastRebootAbilityAgent(AbilityAgent):
        _deactivation_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=10)

    async def run() -> tuple[str, str, int]:
        stuck = FirstDetachHangsAbility()
        active_bot = FastRebootAbilityAgent(
            devices={},
            cognition=IgnoreAbility(),
            acquired_abilities=(stuck,),
        )
        environment = await start_bot_with_devices(active_bot)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        cognition_ability = active_bot._cognition
        _ = await hsm.dispatch(
            environment,
            active_bot,
            dataclasses.replace(
                bot.RebootEvent.with_data(bot.RebootEventData(reason="cognition_child_teardown_failed")),
                id="timeout-reboot",
                source=hsm.id(cognition_ability),
                target=hsm.id(active_bot),
            ),
        )
        for _ in range(100):
            if not active_bot.state().startswith("/Bot/active/"):
                break
            await asyncio.sleep(0.001)
        for _ in range(100):
            if active_bot.state() == "/Bot/active/unfocused":
                break
            await asyncio.sleep(0.005)
        return active_bot.state(), active_bot._attachments.state(), stuck.detach_calls

    state, group_state, detach_calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert group_state == "/AttachmentGroup/attached"
    assert detach_calls == 1


def test_bot_attach_rejects_started_agent_from_another_environment() -> None:
    async def run() -> None:
        active_bot = basic_agent(devices={})
        first_environment = Environment()
        second_environment = Environment()

        _ = await active_bot.attach(first_environment)

        with pytest.raises(RuntimeError, match="Bot is already started in another environment"):
            _ = await active_bot.attach(second_environment)

    asyncio.run(run())


def test_bot_activation_deduplicates_device_aliases() -> None:
    async def run() -> tuple[str, bool]:
        shared_device = Device()
        active_bot = basic_agent(devices={"one": shared_device, "two": shared_device})
        environment = Environment()

        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")

        return active_bot.state(), device_bots(shared_device) == (active_bot,)

    state, attached_to_agent = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert attached_to_agent


def test_bot_and_nested_cognition_use_private_attachment_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[int, int, bool, str]:
        active_bot = basic_agent(devices={"device": Device()})
        environment = Environment()
        attach_calls = 0
        detach_calls = 0
        group_is_private = False
        group_attach = attachment.Group.attach
        group_detach = attachment.Group.detach

        def attach_group(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> collections.abc.Awaitable[None]:
            nonlocal attach_calls, group_is_private
            attach_calls += 1
            group_is_private = group.context().value(hsm.Keys.Instances) is not environment.value(hsm.Keys.Instances)
            return group_attach(group, ctx, event)

        def detach_group(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> collections.abc.Awaitable[None]:
            nonlocal detach_calls
            detach_calls += 1
            return group_detach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "attach", attach_group)
        monkeypatch.setattr(attachment.Group, "detach", detach_group)

        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        return attach_calls, detach_calls, group_is_private, active_bot.state()

    attach_calls, detach_calls, group_is_private, state = asyncio.run(run())

    assert attach_calls == 4
    assert detach_calls == 4
    assert group_is_private
    assert state == "/Bot/inactive"


def test_bot_rejects_environment_started_lifecycle_ability_without_stopping_it() -> None:
    async def run() -> tuple[str, str, bool]:
        cognition_ability = as_cognition(IgnoreAbility())
        active_bot = AbilityAgent(devices={}, cognition=cognition_ability)
        environment = Environment()
        model = cognition_ability.model
        assert model is not None
        _ = await hsm.started(environment, cognition_ability, model)

        _ = await active_bot.attach(environment)
        await asyncio.sleep(0.05)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return (
            active_bot.state(),
            cognition_ability.state(),
            cognition_ability.context().value(hsm.Keys.Instances) is environment.value(hsm.Keys.Instances),
        )

    state, ability_state, ability_stayed_in_environment = asyncio.run(run())

    assert state == "/Bot/inactive"
    assert ability_state != "/IgnoreAbilityLifecycle"
    assert ability_stayed_in_environment


def test_bot_deduplicates_repeated_lifecycle_ability_instance() -> None:
    async def run() -> str:
        shared = ProbeAbility()
        active_bot = AbilityAgent(
            devices={},
            cognition=IgnoreAbility(),
            input=(shared,),
            output=(shared,),
        )
        environment = Environment()

        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        return active_bot.state()

    assert asyncio.run(run()) == "/Bot/inactive"


def test_bot_attachment_group_preserves_preexisting_device_attachment() -> None:
    async def run() -> tuple[str, bool]:
        first_device = Device()
        failing_device = ImmediatelyFailingInitializingDevice()
        active_bot = basic_agent(devices={"first": first_device, "failing": failing_device})
        environment = Environment()

        _ = await hsm.started(environment, active_bot, active_bot.model)
        _ = await hsm.started(environment, first_device, first_device.model)
        await first_device.attach(
            environment,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=active_bot)),
        )
        await wait_until(lambda: device_bots(first_device) == (active_bot,))
        assert device_bots(first_device) == (active_bot,)

        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return active_bot.state(), device_bots(first_device) == (active_bot,)

    state, still_attached = asyncio.run(run())

    assert state == "/Bot/inactive"
    assert still_attached


def test_bot_activation_waits_for_device_attach_events() -> None:
    async def run() -> tuple[str, str, tuple[hsm.Instance, ...]]:
        release = asyncio.Event()
        device = SlowInitializingDevice(release)
        active_bot = basic_agent(devices={"slow": device})
        environment = Environment()

        _ = await active_bot.attach(environment)
        await wait_until(lambda: device.state() == "/Device/initializing")
        await asyncio.sleep(0)

        activating_state = active_bot.state()
        release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")

        return activating_state, active_bot.state(), device_bots(device)

    activating_state, active_state, agents = asyncio.run(run())

    assert activating_state == "/Bot/activating"
    assert active_state == "/Bot/active/unfocused"
    assert len(agents) == 1


def test_bot_detach_during_activation_stops_owned_lifecycle_tree() -> None:
    async def run() -> tuple[str, str, str, tuple[str, ...]]:
        release = asyncio.Event()
        device = SlowInitializingDevice(release)
        cognition_ability = as_cognition(IgnoreAbility())
        input_ability = ProbeAbility()
        output_ability = ProbeAbility()
        active_bot = AbilityAgent(
            devices={"slow": device},
            cognition=cognition_ability,
            input=(input_ability,),
            output=(output_ability,),
        )
        environment = Environment()

        _ = await active_bot.attach(environment)
        await wait_until(lambda: device.state() == "/Device/initializing")
        group = typing.cast(attachment.Group, vars(active_bot)["_attachments"])
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() in {"/Bot/inactive", "/Bot/degraded"})

        return (
            active_bot.state(),
            group.state(),
            device.state(),
            tuple(ability.state() for ability in (cognition_ability, input_ability, output_ability)),
        )

    state, group_state, device_state, ability_states = asyncio.run(run())

    assert state == "/Bot/inactive"
    # hsm 1.3.2: stopped machines report empty state() (was model-root previously).
    assert group_state in {"", "/AttachmentGroup"}
    assert device_state in {"", "/Device"}
    assert ability_states == ("", "", "")


def test_bot_activation_rolls_back_when_device_firmware_initialization_fails() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...], bool, bool]:
        release = asyncio.Event()
        first_device = Device()
        failing_device = FailingInitializingDevice(release)
        active_bot = basic_agent(devices={"first": first_device, "failing": failing_device})
        environment = Environment()

        _ = await active_bot.attach(environment)
        await wait_until(lambda: failing_device.state() == "/Device/initializing")
        release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return (
            active_bot.state(),
            device_bots(first_device),
            first_device.state() in {"", "/Device"},
            failing_device.state() in {"", "/Device"},
        )

    state, first_bots, first_stopped, failing_stopped = asyncio.run(run())

    assert state == "/Bot/inactive"
    assert first_bots == ()
    assert first_stopped
    assert failing_stopped


def test_bot_activation_rolls_back_when_device_failed_before_attachment() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...], bool, bool]:
        first_device = Device()
        failing_device = ImmediatelyFailingInitializingDevice()
        active_bot = basic_agent(devices={"first": first_device, "failing": failing_device})
        environment = Environment()

        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return (
            active_bot.state(),
            device_bots(first_device),
            first_device.state() in {"", "/Device"},
            failing_device.state() in {"", "/Device"},
        )

    state, first_bots, first_stopped, failing_stopped = asyncio.run(run())

    assert state == "/Bot/inactive"
    assert first_bots == ()
    assert first_stopped
    assert failing_stopped


def test_bot_attachment_group_uses_modeled_device_events_not_dispatch_override() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...], bool]:
        first_device = LifecycleDispatchFailingDevice()
        failing_device = ImmediatelyFailingInitializingDevice()
        active_bot = basic_agent(devices={"first": first_device, "failing": failing_device})
        environment = Environment()

        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return active_bot.state(), device_bots(first_device), first_device.state() in {"", "/Device"}

    state, first_bots, first_stopped = asyncio.run(run())

    assert state == "/Bot/inactive"
    assert first_bots == ()
    assert first_stopped


def test_bot_activation_dispatch_failure_uses_modeled_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...], bool, bool]:
        first_device = Device()
        failing_device = Device()
        active_bot = basic_agent(devices={"first": first_device, "failing": failing_device})
        environment = Environment()
        original_dispatch = hsm.Instance.dispatch

        def dispatch(instance: hsm.Instance, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
            if instance is failing_device and event.name == attachment.AttachEvent.name:
                raise RuntimeError("device attach dispatch failed")
            return original_dispatch(instance, ctx, event)

        monkeypatch.setattr(hsm.Instance, "dispatch", dispatch)

        _ = await active_bot.attach(environment)
        # Activation cleanup now stops nested ability/group holds; give wall-clock time
        # beyond pure yield scheduling for the modeled rollback path.
        for _ in range(200):
            if active_bot.state() == "/Bot/inactive":
                break
            await asyncio.sleep(0.01)

        return (
            active_bot.state(),
            device_bots(first_device),
            first_device.state() in {"", "/Device"},
            failing_device.state() in {"", "/Device"},
        )

    state, first_bots, first_stopped, failing_stopped = asyncio.run(run())

    assert state == "/Bot/inactive"
    assert first_bots == ()
    assert first_stopped
    assert failing_stopped


def test_bot_attachment_group_handles_multiple_firmware_initialization_failures() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...], bool, bool, bool]:
        release = asyncio.Event()
        first_device = Device()
        first_failing_device = FailingInitializingDevice(release)
        second_failing_device = FailingInitializingDevice(release)
        active_bot = basic_agent(
            devices={
                "first": first_device,
                "first_failing": first_failing_device,
                "second_failing": second_failing_device,
            }
        )
        environment = Environment()

        _ = await active_bot.attach(environment)
        await wait_until(
            lambda: first_failing_device.state() == "/Device/initializing"
            and second_failing_device.state() == "/Device/initializing"
        )
        release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return (
            active_bot.state(),
            device_bots(first_device),
            first_device.state() in {"", "/Device"},
            first_failing_device.state() in {"", "/Device"},
            second_failing_device.state() in {"", "/Device"},
        )

    state, first_bots, first_stopped, first_failing_stopped, second_failing_stopped = asyncio.run(run())

    assert state == "/Bot/inactive"
    assert first_bots == ()
    assert first_stopped
    assert first_failing_stopped
    assert second_failing_stopped


def test_bot_deactivation_detaches_shared_device_alias() -> None:
    async def run() -> tuple[str, tuple[hsm.Instance, ...], bool]:
        shared_device = LifecycleDispatchFailingDevice()
        active_bot = basic_agent(devices={"one": shared_device, "two": shared_device})
        environment = await start_bot_with_devices(active_bot)

        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return active_bot.state(), device_bots(shared_device), shared_device.state() in {"", "/Device"}

    state, agents, shared_stopped = asyncio.run(run())

    assert state == "/Bot/inactive"
    assert agents == ()
    assert not shared_stopped


async def wait_until(condition: typing.Callable[[], bool], *, timeout: float = 2.0) -> None:
    """Wait until ``condition`` is true.

    Uses tight yields so short-lived HSM states remain observable, with a wall-clock
    deadline so async cleanup under hsm 1.3.2 cannot hang the suite forever.
    """

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0)
    raise TimeoutError("wait_until condition not met")


async def ring_phone(phone: phone_device.Phone, call_id: str = "call-123", caller: str | None = None) -> None:
    await emit_phone_service_event(
        phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id=call_id, caller=caller))
    )
    await wait_until(lambda: device_firmware(phone) is not None and device_firmware(phone).state() == "/Phone/ringing")


async def answer_phone(phone: phone_device.Phone, call_id: str = "call-123") -> None:
    await ring_phone(phone, call_id=call_id)
    await phone.dispatch(
        phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData())
    )
    await emit_phone_service_event(
        phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id=call_id))
    )
    await wait_until(
        lambda: device_firmware(phone) is not None
        and device_firmware(phone).state() == "/Phone/answered/media_connecting"
    )


async def emit_phone_service_event(phone: phone_device.Phone, event: hsm.Event[typing.Any]) -> None:
    firmware = device_firmware(phone)
    assert isinstance(firmware, phone_device.PhoneFirmware)
    recorder = firmware.event_recorder()
    assert isinstance(recorder, phone_device.PhoneEventRecorder)
    await recorder.receive(phone.context(), event)


async def answered_phone_in_environment() -> tuple[Environment, phone_device.Phone]:
    environment = Environment()
    phone = phone_device.Phone()
    _ = await hsm.started(environment, phone, phone.model)
    await answer_phone(phone)
    return environment, phone


def probe_input() -> processing.InputData:
    return processing.InputData(input=bot.InputEventData(target_device="phone", priority=0))


def test_bot_uses_hsm_instance_identity() -> None:
    bot_instance = basic_agent()

    assert isinstance(bot_instance, hsm.Instance)
    assert not hasattr(bot_instance, "name")


def test_bot_is_abstract_with_no_hard_coded_innate_abilities() -> None:
    assert inspect.isabstract(Bot)


def test_concrete_agent_can_declare_and_instantiate_innate_ability() -> None:
    bot_instance = FocusedAgent(devices={})

    assert isinstance(bot_instance.probe, ProbeAbility)


def test_bot_events_use_pydantic_schemas() -> None:
    activate_schema = object_dict(bot.ActivateEvent.schema)
    deactivate_schema = object_dict(bot.DeactivateEvent.schema)
    reboot_schema = object_dict(bot.RebootEvent.schema)
    input_schema = object_dict(bot.InputEvent.schema)
    idle_schema = object_dict(bot.IdleEvent.schema)
    completed_schema = object_dict(bot.ProcessingCompletedEvent.schema)
    failed_schema = object_dict(bot.ProcessingFailedEvent.schema)
    focus_device_schema = object_dict(bot.FocusDeviceEvent.schema)
    clear_focus_schema = object_dict(bot.ClearFocusEvent.schema)
    activating_done_schema = object_dict(bot.ActivatingDoneEvent.schema)
    activating_failed_schema = object_dict(bot.ActivatingFailedEvent.schema)
    deactivating_done_schema = object_dict(bot.DeactivatingDoneEvent.schema)

    assert bot.ActivateEvent.name == "bot.activate"
    assert activate_schema == bot.ActivateEventData.model_json_schema()
    assert activate_schema["description"]
    assert activate_schema["examples"] == [{}]
    assert bot.DeactivateEvent.name == "bot.deactivate"
    assert deactivate_schema == bot.DeactivateEventData.model_json_schema()
    assert deactivate_schema["description"]
    assert deactivate_schema["examples"] == [{}]
    assert bot.RebootEvent.name == "bot.reboot"
    assert reboot_schema == bot.RebootEventData.model_json_schema()
    assert reboot_schema["description"]
    assert reboot_schema["required"] == ["reason"]
    assert bot.InputEvent.name == "bot.input"
    assert input_schema == bot.InputEventData.model_json_schema()
    assert input_schema["required"] == ["target_device", "priority"]
    input_properties = typing.cast(collections.abc.Mapping[str, object], input_schema["properties"])
    assert "source_event" in input_properties
    assert "payload" in input_properties
    assert bot.IdleEvent.name == "bot.idle"
    assert idle_schema == bot.IdleEventData.model_json_schema()
    assert idle_schema["description"]
    assert idle_schema["examples"] == [{}]
    # Body ingress, never a model tool: plain event kind, not the tool-offerable kind.
    assert bot.IdleEvent.kind == hsm.EventKind
    assert bot.IdleEvent.kind != event_schema.EventKind
    assert bot.FocusDeviceEvent.name == "bot.focus_device"
    assert focus_device_schema == bot.FocusDeviceEventData.model_json_schema()
    assert bot.ClearFocusEvent.name == "bot.clear_focus"
    assert clear_focus_schema == bot.ClearFocusEventData.model_json_schema()
    assert bot.ProcessingCompletedEvent.name == "bot.processing.completed"
    assert completed_schema == bot.ProcessingCompletedEventData.model_json_schema()
    assert completed_schema["description"]
    assert completed_schema["required"] == ["output", "focus_candidates"]
    assert bot.ProcessingFailedEvent.name == "bot.processing.failed"
    assert failed_schema == bot.ProcessingFailedEventData.model_json_schema()
    assert failed_schema["description"]
    assert bot.ActivatingDoneEvent.name == "bot.activated"
    assert activating_done_schema == bot.ActivatingDoneEventData.model_json_schema()
    assert bot.ActivatingFailedEvent.name == "bot.activating.failed"
    assert activating_failed_schema == bot.ActivatingFailedEventData.model_json_schema()
    assert bot.DeactivatingDoneEvent.name == "bot.deactivated"
    assert deactivating_done_schema == bot.DeactivatingDoneEventData.model_json_schema()
    assert deactivating_done_schema["description"]
    assert deactivating_done_schema["examples"] == [{}]


def test_bot_input_priority_is_bounded() -> None:
    assert bot.InputEventData(target_device="phone", priority=0).priority == 0
    assert bot.InputEventData(target_device="phone", priority=10).priority == 10

    with pytest.raises(ValueError):
        _ = bot.InputEventData(target_device="phone", priority=-1)
    with pytest.raises(ValueError):
        _ = bot.InputEventData(target_device="phone", priority=11)


def test_bot_input_can_carry_modeled_source_event_payload() -> None:
    data = bot.InputEventData(
        target_device="phone",
        priority=0,
        source_event="phone.incoming_call",
        payload={"call_id": "call-123"},
    )

    assert data.source_event == "phone.incoming_call"
    assert data.payload == {"call_id": "call-123"}

    with pytest.raises(ValueError):
        _ = bot.InputEventData(target_device="phone", priority=0, source_event="")


def test_bot_model_tracks_activation_focus_and_processing_state() -> None:
    model = Bot.model
    transitions = transition_map(model)
    deferred_map = typing.cast(
        collections.abc.Mapping[str, collections.abc.Mapping[str, str]],
        getattr(model, "deferred_map"),
    )

    assert model.qualified_name == "/Bot"
    assert model.initial == "/Bot/.initial"
    assert "/Bot/inactive" in model.members
    assert "/Bot/activating" in model.members
    assert "/Bot/activation_cleanup" in model.members
    assert "/Bot/deactivating" in model.members
    assert "/Bot/deactivation_cleanup" in model.members
    assert "/Bot/reboot_deactivating" in model.members
    assert "/Bot/reboot_cleanup_preserve" in model.members
    assert "/Bot/reboot_cleanup_reset" in model.members
    assert "/Bot/active" in model.members
    assert "/Bot/active/unfocused" in model.members
    assert "/Bot/active/focused" in model.members
    assert "/Bot/active/processing" in model.members
    assert "/Bot/active/cancelling_processing" in model.members
    assert "bot.activate" in transitions["/Bot/inactive"]
    assert "bot.lifecycle.completed" in transitions["/Bot/activating"]
    assert "bot.lifecycle.failed" in transitions["/Bot/activating"]
    assert "attachment.attach.complete" not in transitions["/Bot/activating"]
    assert "bot.activating.failed" not in transitions["/Bot/activating"]
    assert "bot.lifecycle.cleanup.done" in transitions["/Bot/activation_cleanup"]
    assert "bot.lifecycle.completed" in transitions["/Bot/deactivating"]
    assert "bot.lifecycle.failed" in transitions["/Bot/deactivating"]
    assert "bot.lifecycle.cleanup.done" in transitions["/Bot/deactivation_cleanup"]
    assert "bot.lifecycle.completed" in transitions["/Bot/reboot_deactivating"]
    assert "bot.lifecycle.failed" in transitions["/Bot/reboot_deactivating"]
    assert "bot.lifecycle.cleanup.done" in transitions["/Bot/reboot_cleanup_preserve"]
    assert "bot.lifecycle.cleanup.done" in transitions["/Bot/reboot_cleanup_reset"]
    assert any("_bot_deactivation_timeout" in event for event in transitions["/Bot/deactivating"])
    assert any("_bot_deactivation_timeout" in event for event in transitions["/Bot/deactivation_cleanup"])
    assert "environment.sound" in transitions["/Bot/active"]
    assert "environment.visual" in transitions["/Bot/active"]
    assert "bot.deactivate" in transitions["/Bot/active"]
    assert "bot.reboot" in transitions["/Bot/active"]
    assert "bot.reboot" in transitions["/Bot/activating"]
    assert "bot.reboot" in transitions["/Bot/activation_cleanup"]
    assert "bot.reboot" in transitions["/Bot/deactivating"]
    assert "bot.reboot" in transitions["/Bot/deactivation_cleanup"]
    assert "bot.input" in transitions["/Bot/active/unfocused"]
    assert "bot.ability.cognition.input" in transitions["/Bot/active/unfocused"]
    assert "*" not in transitions["/Bot/active/unfocused"]
    assert "bot.input" in transitions["/Bot/active/focused"]
    assert "bot.ability.cognition.input" in transitions["/Bot/active/focused"]
    assert "*" not in transitions["/Bot/active/focused"]
    # Being awake is the occasion: a recurring moment on both idle attention states, and the
    # occasion event itself enters processing. Judgment decides what (if anything) it is for.
    for idle_state in ("/Bot/active/unfocused", "/Bot/active/focused"):
        assert any("_bot_idle_interval" in event for event in transitions[idle_state])
        assert "bot.idle" in transitions[idle_state]
        idle_transition = transitions[idle_state]["bot.idle"][0]
        assert idle_transition.target == "/Bot/active/processing"
        # An idle turn is not an attention change: no focus effect rides the occasion.
        assert not any("_focus_event_target" in effect for effect in idle_transition.effect)
    # A moment missed while busy is simply gone; stale occasions must not queue up.
    assert "bot.idle" not in deferred_map.get("/Bot/active/processing", {})
    assert "bot.idle" not in deferred_map.get("/Bot/active/cancelling_processing", {})
    assert "bot.ability.cognition.input" in deferred_map["/Bot/active/processing"]
    assert "bot.ability.cognition.output" in transitions["/Bot/active/processing"]
    assert "bot.ability.failed" in transitions["/Bot/active/processing"]
    # Focus is declared on active (transition_map may inherit it into child keys). Clear is
    # declared only on focused + processing so idle unfocused does not offer clear_focus.
    assert "bot.focus_device" in transitions["/Bot/active"]
    assert "bot.clear_focus" not in transitions.get("/Bot/active", {})
    assert "bot.clear_focus" not in transitions["/Bot/active/unfocused"]
    assert "bot.clear_focus" in transitions["/Bot/active/focused"]
    assert "bot.clear_focus" in transitions["/Bot/active/processing"]
    assert "bot.processing.completed" in transitions["/Bot/active/processing"]
    assert "bot.processing.failed" in transitions["/Bot/active/processing"]
    assert "bot.processing.timed_out" in transitions["/Bot/active/processing"]
    assert "bot.ability.cognition.cancelled" in transitions["/Bot/active/cancelling_processing"]
    assert "bot.ability.processing.cancelled" in transitions["/Bot/active/cancelling_processing"]
    assert "bot.processing.cancel.timed_out" in transitions["/Bot/active/cancelling_processing"]


def test_unfocused_agent_focuses_target_device_before_processing_input() -> None:
    async def run() -> tuple[str, str | None, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), ability.calls, active_bot.actions

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert len(calls) == 1
    assert calls[0].input == bot.InputEventData(target_device="phone", priority=3)
    assert actions == [no_output("priority:3")]


def test_unfocused_agent_rejects_input_from_unconfigured_target_device() -> None:
    async def run() -> tuple[str, str | None, list[processing.InputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="browser", priority=3)),
        )
        await asyncio.sleep(0)

        return active_bot.state(), bot_has_focus(active_bot), ability.calls

    state, focused_device, calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert not focused_device
    assert calls == []


def test_unfocused_agent_does_not_focus_anonymous_device_type_name() -> None:
    async def run() -> tuple[str, str | None, list[processing.InputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="Device", priority=3)),
        )
        await asyncio.sleep(0)

        return active_bot.state(), bot_has_focus(active_bot), ability.calls

    state, focused_device, calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert not focused_device
    assert calls == []


def test_bot_copies_configured_devices_at_construction() -> None:
    async def run() -> tuple[str, str | None, list[processing.InputData]]:
        ability = IgnoreAbility()
        devices = configured_devices("phone")
        active_bot = AbilityAgent(devices=devices, cognition=ability, input=(ring_hearing(),))
        devices["browser"] = Device()

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="browser", priority=3)),
        )
        await asyncio.sleep(0)

        return active_bot.state(), bot_has_focus(active_bot), ability.calls

    state, focused_device, calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert not focused_device
    assert calls == []


def test_bot_uses_configured_device_keys_for_processing_input() -> None:
    async def run() -> tuple[str, str | None, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="browser", priority=3)),
        )
        await asyncio.sleep(0)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=2)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), ability.calls, active_bot.actions

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert len(calls) == 1
    assert actions == [no_output("priority:2")]


def test_bot_processing_input_includes_event_derived_operations() -> None:
    async def run() -> list[processing.InputData]:
        ability = IgnoreAbility()
        phone = phone_device.Phone()
        active_bot = AbilityAgent(
            devices={"phone": phone},
            cognition=ability,
            input=(ring_hearing(),),
        )

        _ = await start_bot_with_devices(active_bot)
        await ring_phone(phone)
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")

        return ability.calls

    calls = asyncio.run(run())

    assert len(calls) == 1
    assert_heard_phone_ring(calls[0].input)
    operations = {operation.name: operation for operation in calls[0].schemas}
    # Body attention + device call tools from live snapshots; ignore from Cognition host topology.
    # First ring turn enters processing from unfocused: focus is offered, clear is not
    # (clear lives on focused + processing leaf, and entry snapshot is still unfocused).
    assert set(operations) == {
        bot.FocusDeviceEvent.name,
        phone_device.AnswerCallEvent.name,
        phone_device.DeclineCallEvent.name,
        cognition.types.IgnoreEvent.name,
    }
    assert event_data_schema(operations[bot.FocusDeviceEvent.name]) == bot.FocusDeviceEventData.model_json_schema()
    answer_operation = operations[phone_device.AnswerCallEvent.name]
    assert event_data_schema(answer_operation) == event_json_schema(phone_device.AnswerCallEvent)


def test_bot_snapshot_merges_focused_device_transitions() -> None:
    async def run() -> tuple[set[str], set[str]]:
        ability = IgnoreAbility()
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        before_focus = {
            event_name
            for transition in hsm.take_snapshot(None, active_bot).Transitions
            for event_name in transition.events
        }
        await ring_phone(phone)
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        after_focus = {
            event_name
            for transition in hsm.take_snapshot(None, active_bot).Transitions
            for event_name in transition.events
        }

        return before_focus, after_focus

    before_focus, after_focus = asyncio.run(run())

    assert phone_device.AnswerCallEvent.name not in before_focus
    assert phone_device.DeclineCallEvent.name not in before_focus
    # Focus is available while active; clear only when focused (or mid-processing).
    # Device call tools appear only with focus/device state.
    assert bot.FocusDeviceEvent.name in before_focus
    assert bot.ClearFocusEvent.name not in before_focus
    assert bot.FocusDeviceEvent.name in after_focus
    assert bot.ClearFocusEvent.name in after_focus
    assert phone_device.AnswerCallEvent.name in after_focus
    assert phone_device.DeclineCallEvent.name in after_focus


def test_bot_processing_operations_follow_focused_device_not_observed_device() -> None:
    async def run() -> list[processing.InputData]:
        ability = IgnoreAbility()
        phone = phone_device.Phone()
        browser_phone = phone_device.Phone()
        active_bot = AbilityAgent(
            devices={"phone": phone, "browser": browser_phone}, cognition=ability, input=(ring_hearing(),)
        )

        _ = await start_bot_with_devices(active_bot)
        await ring_phone(phone)
        await wait_until(lambda: len(ability.calls) == 1 and bot_has_focus(active_bot))
        await ring_phone(browser_phone, call_id="call-456", caller="Front desk")
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused")

        return ability.calls

    calls = asyncio.run(run())

    assert len(calls) == 2
    observed_browser_input = calls[1]
    assert isinstance(observed_browser_input.input, hsm.Event)
    assert observed_browser_input.input.name == SoundEvent.name
    assert isinstance(observed_browser_input.input.data, SoundData)
    # What a ringing phone carries out is who is calling, not which session is ringing.
    assert observed_browser_input.input.data.caller == "Front desk"
    offered = {event.name for event in observed_browser_input.schemas}
    assert bot.FocusDeviceEvent.name in offered
    # Second turn while already focused: clear is on focused snapshot during processing entry.
    assert bot.ClearFocusEvent.name in offered
    assert phone_device.AnswerCallEvent.name in offered
    assert phone_device.DeclineCallEvent.name in offered


def test_bot_processes_environment_broadcast_from_configured_device_event() -> None:
    async def run() -> tuple[str, str | None, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = IgnoreAbility()
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        await emit_phone_service_event(
            phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123"))
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), ability.calls, active_bot.actions

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert len(calls) == 1
    assert_heard_phone_ring(calls[0].input)
    assert actions == [no_output("priority:0")]


class FixedVoiceDetector(voice.detection.VoiceDetector):
    is_voice: bool
    confidence: float

    def __init__(self, *, is_voice: bool = True, confidence: float = 0.91) -> None:
        self.is_voice = is_voice
        self.confidence = confidence

    @typing.override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        return voice.detection.OutputData(is_voice=self.is_voice, confidence=self.confidence)


class RecordingSpeechDecoder(speech.SpeechDecoder):
    calls: list[bytes]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return b"decoded:" + input


class RecordingListening(listening.Listening):
    handoffs: list[cognition.InputData]
    failures: list[listening.FailedEventData]
    received: list[hsm.Event[typing.Any]]
    speech_decoder: RecordingSpeechDecoder | None

    def __init__(
        self,
        *,
        is_voice: bool = True,
        speech_decoder: RecordingSpeechDecoder | None | typing.Literal[False] = None,
        sound_classifier: sound_hearing.classification.SoundClassifier | None = None,
    ) -> None:
        decoder: RecordingSpeechDecoder | None
        if speech_decoder is False:
            decoder = None
        elif speech_decoder is None:
            decoder = RecordingSpeechDecoder()
        else:
            decoder = speech_decoder
        super().__init__(
            voice_detector=FixedVoiceDetector(is_voice=is_voice),
            sound_classifier=sound_classifier,
            speech_decoder=decoder,
        )
        self.handoffs = []
        self.failures = []
        self.received = []
        self.speech_decoder = decoder

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        self.received.append(event)
        if event.name == cognition.InputEvent.name:
            handoff = event.data
            assert isinstance(handoff, cognition.InputData)
            self.handoffs.append(handoff)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, listening.FailedEventData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


def ring_hearing(*, is_voice: bool = False) -> RecordingListening:
    """Sensory Listening for phone ring: no-voice sound labeled from SoundData.kind."""

    return RecordingListening(
        is_voice=is_voice,
        speech_decoder=False,
        sound_classifier=sound_hearing.classification.KindSoundClassifier(),
    )


def test_bot_fans_out_sound_event_to_input_listening() -> None:
    async def run() -> tuple[
        list[processing.InputData],
        list[hsm.Event[typing.Any]],
        list[bytes],
        str,
    ]:
        ability = IgnoreAbility()
        listening_ability = RecordingListening()
        active_bot = AbilityAgent(devices={}, cognition=ability, input=(listening_ability,))
        _ = await start_bot_with_devices(active_bot)
        sound = SoundEvent.with_data(
            SoundData(audio=b"heard-chunk", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)
        )
        await active_bot.dispatch(active_bot.context(), sound)
        await wait_until(lambda: len(ability.calls) == 1)
        assert listening_ability.speech_decoder is not None
        return (
            ability.calls,
            listening_ability.received,
            listening_ability.speech_decoder.calls,
            active_bot.state() or "",
        )

    calls, received, decoder_calls, state = asyncio.run(run())

    assert len(calls) == 1
    assert isinstance(calls[0].input, hsm.Event)
    assert calls[0].input.name == speech.SpeechDecoding.output_event.name
    assert calls[0].input.data == b"decoded:heard-chunk"
    assert decoder_calls == [b"heard-chunk"]
    assert (
        state.endswith("/active/focused") or state.endswith("/active/processing") or state.endswith("/active/unfocused")
    )


def test_bot_fans_out_visual_event_to_input_without_cognition() -> None:
    async def run() -> tuple[list[processing.InputData], str]:
        ability = IgnoreAbility()
        listening_ability = RecordingListening()
        active_bot = AbilityAgent(devices={}, cognition=ability, input=(listening_ability,))
        _ = await start_bot_with_devices(active_bot)
        visual = VisualEvent.with_data(VisualData(image=b"frame-bytes", media_type="image/png"))
        await active_bot.dispatch(active_bot.context(), visual)
        await asyncio.sleep(0.05)
        return ability.calls, active_bot.state() or ""

    calls, state = asyncio.run(run())

    # Listening has no visual transition; fan-out must not feed raw visual into cognition.
    assert calls == []
    assert state.endswith("/active/unfocused")


def test_bot_does_not_send_speaker_environment_sound_to_cognition() -> None:
    async def run() -> tuple[
        str,
        str | None,
        list[processing.InputData],
        list[cognition.types.OutputData],
    ]:
        ability = IgnoreAbility()
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))
        data = audio.AudioOutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)

        environment = await start_bot_with_devices(active_bot)
        await ring_phone(phone)
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")
        ability.calls.clear()
        active_bot.actions.clear()
        # Signal into the speaker; the speaker is the transducer that makes it environment sound.
        await phone_speaker(phone).dispatch(environment, audio.OutputEvent.with_data(data))
        await asyncio.sleep(0.05)

        return (
            active_bot.state(),
            bot_has_focus(active_bot),
            ability.calls,
            active_bot.actions,
        )

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    # Speaker elevates to environment.sound; Listening skips ordinary no-voice playback (not cognition).
    assert calls == []
    assert actions == []


def test_same_environment_sibling_bots_do_not_send_speaker_sound_to_cognition() -> None:
    async def run() -> tuple[
        list[processing.InputData],
        list[processing.InputData],
        str | None,
        str | None,
    ]:
        owner_ability = IgnoreAbility()
        sibling_ability = IgnoreAbility()
        phone = phone_device.Phone()
        owning_agent = AbilityAgent(devices={"phone": phone}, cognition=owner_ability, input=(ring_hearing(),))
        sibling_agent = AbilityAgent(devices={}, cognition=sibling_ability)
        data = audio.AudioOutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)

        environment = await start_bot_with_devices(owning_agent)
        _ = await sibling_agent.attach(environment)
        await wait_until(lambda: sibling_agent.state() == "/Bot/active/unfocused")
        await ring_phone(phone)
        await wait_until(lambda: len(owner_ability.calls) == 1 and owning_agent.state() == "/Bot/active/focused")
        owner_ability.calls.clear()
        owning_agent.actions.clear()

        # Signal into the speaker; the speaker is the transducer that makes it environment sound.
        await phone_speaker(phone).dispatch(environment, audio.OutputEvent.with_data(data))
        await asyncio.sleep(0.05)

        return (
            owner_ability.calls,
            sibling_ability.calls,
            bot_has_focus(owning_agent),
            bot_has_focus(sibling_agent),
        )

    owner_calls, sibling_calls, owner_focus, sibling_focus = asyncio.run(run())

    assert sibling_calls == []
    assert owner_calls == []
    assert owner_focus
    assert not sibling_focus


def test_same_environment_sibling_agent_without_phone_config_ignores_phone_broadcast() -> None:
    async def run() -> tuple[
        list[processing.InputData],
        list[processing.InputData],
        str | None,
        str | None,
    ]:
        owner_ability = IgnoreAbility()
        sibling_ability = IgnoreAbility()
        phone = phone_device.Phone()
        owning_agent = AbilityAgent(devices={"phone": phone}, cognition=owner_ability, input=(ring_hearing(),))
        sibling_agent = AbilityAgent(devices={}, cognition=sibling_ability)

        environment = await start_bot_with_devices(owning_agent)
        _ = await sibling_agent.attach(environment)
        await wait_until(lambda: sibling_agent.state() == "/Bot/active/unfocused")

        await emit_phone_service_event(
            phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123"))
        )
        await wait_until(lambda: owning_agent.state() == "/Bot/active/focused" and len(owner_ability.calls) == 1)
        await asyncio.sleep(0)

        return (
            owner_ability.calls,
            sibling_ability.calls,
            bot_has_focus(owning_agent),
            bot_has_focus(sibling_agent),
        )

    owner_calls, sibling_calls, owner_focus, sibling_focus = asyncio.run(run())

    assert len(owner_calls) == 1
    assert_heard_phone_ring(owner_calls[0].input)
    assert sibling_calls == []
    assert owner_focus
    assert not sibling_focus


def test_bot_processes_observed_device_event_with_trace_metadata() -> None:
    async def run() -> tuple[
        list[processing.InputData],
        list[dict[str, object]],
    ]:
        ability = MetadataRecordingAbility()
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}

        _ = await start_bot_with_devices(active_bot)
        await emit_phone_service_event(
            phone,
            dataclasses.replace(
                phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
                metadata=metadata,
            ),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return ability.calls, ability.input_event_metadata

    calls, input_event_metadata = asyncio.run(run())

    assert len(calls) == 1
    assert_heard_phone_ring(calls[0].input)
    # Intuition/Reasoning inject processing.Processor (not Processing), so model process()
    # sees the decision input only; HSM event metadata stays on the ability event chain.
    assert input_event_metadata == []


def test_bot_preserves_trace_metadata_when_dispatching_operation_to_device() -> None:
    async def run() -> list[dict[str, object]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                target="phone",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            )
        )
        phone = ContextRecordingPhone()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}

        _ = await start_bot_with_devices(active_bot)
        await emit_phone_service_event(
            phone,
            dataclasses.replace(
                phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
                metadata=metadata,
            ),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused" and bool(phone.event_metadata))

        return phone.event_metadata

    event_metadata = asyncio.run(run())

    assert len(event_metadata) == 1
    assert event_metadata[0]["traceparent"] == "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"


def test_bot_routes_operation_snapshot_failure_to_processing_failed_event() -> None:
    async def run() -> tuple[str, list[processing.InputData], list[bot.ProcessingFailedEventData]]:
        ability = IgnoreAbility()
        phone = SnapshotFailingPhone()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        phone.fail_snapshots = True
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=2)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), ability.calls, active_bot.failures

    state, calls, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert calls == []
    assert len(failures) == 1
    assert failures[0].message == "snapshot unavailable"


def test_bot_processing_input_includes_primary_ability_affordance() -> None:
    async def run() -> tuple[list[processing.InputData], list[cognition.types.OutputData]]:
        ability = NestedPrimaryAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=2)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return ability.calls, active_bot.actions

    calls, actions = asyncio.run(run())

    assert len(calls) == 1
    assert actions == [no_output("nested affordance")]


def test_focused_agent_processes_other_target_without_automatic_focus_change() -> None:
    async def run() -> tuple[str, str | None, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="browser", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), ability.calls, active_bot.actions

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert calls[1].input == bot.InputEventData(target_device="browser", priority=2)
    assert actions == [no_output("priority:3"), no_output("priority:2")]


def test_focused_agent_dispatches_operation_output_to_target_device() -> None:
    async def run() -> tuple[str, str, str | None, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                target="phone",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            )
        )
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        firmware = device_firmware(phone)
        assert firmware is not None
        await ring_phone(phone)
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused" and firmware.state() == "/Phone/answering")

        return (
            active_bot.state(),
            firmware.state(),
            bot_has_focus(active_bot),
            ability.calls,
            active_bot.actions,
        )

    state, phone_state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert phone_state == "/Phone/answering"
    assert focused_device
    assert len(calls) == 1
    assert actions == [
        (
            cognition.types.EventData(
                target="phone",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            ),
        )
    ]


def test_focused_agent_stale_device_selection_drops_at_device() -> None:
    """Stale device selections dispatch; the device drops them per its own topology.

    Delivery validation resolves against declared events (HSM-CONTEXT-001: no probed peer
    state), so a selection whose target moved on is no longer rejected pre-dispatch. The
    device ignores it as unmatched, the turn completes, and the hangup is observed as the
    next stimulus.
    """

    async def run() -> tuple[
        str, str, str | None, list[cognition.types.OutputData], list[bot.ProcessingFailedEventData]
    ]:
        release = asyncio.Event()
        answer_selection = cognition.types.EventData(
            target="phone",
            event=phone_device.AnswerCallEvent.name,
            data={"call_id": "call-123"},
            reason="answer incoming call",
        )
        ability = BlockingSequenceAbility(release=release, block_on_call=1, outputs=(answer_selection,))
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        firmware = device_firmware(phone)
        assert firmware is not None
        await ring_phone(phone)
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/processing")

        await emit_phone_service_event(
            phone, phone_device.RemoteHangUpEvent.with_data(phone_device.RemoteHangUpData(call_id="call-123"))
        )
        await wait_until(lambda: firmware.state() == "/Phone/hung_up")
        release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return (
            active_bot.state(),
            firmware.state(),
            bot_has_focus(active_bot),
            active_bot.actions,
            active_bot.failures,
        )

    state, phone_state, focused_device, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert phone_state == "/Phone/hung_up"
    assert focused_device
    assert actions == [
        (
            cognition.types.EventData(
                target="phone",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            ),
        )
    ]
    assert failures == []


def test_focused_agent_rejects_operation_output_that_does_not_match_event_schema() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData], list[bot.ProcessingFailedEventData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                event=phone_device.HangUpCallEvent.name,
                reason="hang up a call the phone is not on",
            )
        )
        active_bot = AbilityAgent(devices={"phone": phone_device.Phone()}, cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.actions, active_bot.failures

    state, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert actions == []
    assert len(failures) == 1
    assert failures[0].message == f"Processing selected unavailable event: {phone_device.HangUpCallEvent.name}."


def test_focused_agent_rejects_operation_event_not_offered_by_input() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData], list[bot.ProcessingFailedEventData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                target="phone",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            )
        )
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.actions, active_bot.failures

    state, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert actions == []
    assert len(failures) == 1
    assert (
        failures[0].message
        == f"Processing selected unavailable event for target phone: {phone_device.AnswerCallEvent.name}."
    )


def test_focused_agent_rejects_operation_target_not_offered_by_input() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData], list[bot.ProcessingFailedEventData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                target="browser",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            )
        )
        active_bot = AbilityAgent(devices={"phone": phone_device.Phone(), "browser": Device()}, cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.actions, active_bot.failures

    state, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert actions == []
    assert len(failures) == 1
    assert (
        failures[0].message
        == f"Processing selected unavailable event for target browser: {phone_device.AnswerCallEvent.name}."
    )


def test_focused_agent_rejects_agent_local_operation_with_target() -> None:
    async def run() -> tuple[str, str | None, list[cognition.types.OutputData], list[bot.ProcessingFailedEventData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                target="phone",
                event=bot.FocusDeviceEvent.name,
                data={"device": "browser"},
                reason="target mismatch",
            )
        )
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), active_bot.actions, active_bot.failures

    state, focused_device, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert actions == []
    assert len(failures) == 1
    assert failures[0].message == "Processing selected focus_device outside available device candidates."


def test_focused_agent_dispatches_operation_with_ref_backed_event_data_schema() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData], list[bot.ProcessingFailedEventData]]:
        data: dict[str, object] = {
            "call_id": "call-123",
            "transfer_id": "transfer-123",
            "target": {"kind": "address", "value": "operator@example.com"},
        }
        ability = SequenceAbility(
            cognition.types.EventData(
                target="phone",
                event=phone_device.TransferCallEvent.name,
                data=data,
                reason="transfer call",
            ),
        )
        environment, phone = await answered_phone_in_environment()
        await emit_phone_service_event(
            phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123"))
        )
        await wait_until(
            lambda: device_firmware(phone) is not None
            and device_firmware(phone).state() == "/Phone/answered/media_ready"
        )
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))

        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.actions) and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.actions, active_bot.failures

    state, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert failures == []
    assert actions == [
        (
            cognition.types.EventData(
                target="phone",
                event=phone_device.TransferCallEvent.name,
                data={
                    "call_id": "call-123",
                    "transfer_id": "transfer-123",
                    "target": {"kind": "address", "value": "operator@example.com"},
                },
                reason="transfer call",
            ),
        )
    ]


def test_focused_agent_rejects_operation_data_that_does_not_match_event_schema() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData], list[bot.ProcessingFailedEventData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                target="phone",
                event=phone_device.TransferCallEvent.name,
                data={
                    "call_id": "call-123",
                    "transfer_id": "transfer-123",
                    "target": {"kind": "queue", "value": "operator"},
                },
                reason="transfer call",
            )
        )
        environment, phone = await answered_phone_in_environment()
        await emit_phone_service_event(
            phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123"))
        )
        await wait_until(
            lambda: device_firmware(phone) is not None
            and device_firmware(phone).state() == "/Phone/answered/media_ready"
        )
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))

        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.actions, active_bot.failures

    state, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert actions == []
    assert len(failures) == 1
    assert failures[0].message == (
        f"Processing selected invalid event data for event: {phone_device.TransferCallEvent.name}."
    )


def test_focused_agent_dispatches_multi_event_focus_selection() -> None:
    async def run() -> tuple[str, str | None, list[cognition.types.OutputData]]:
        ability = SequenceAbility(focus_output("phone", "multi focus"))
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bot_has_focus(active_bot) and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), active_bot.actions

    state, focus, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focus
    assert actions == [focus_output("phone", "multi focus")]


def test_focused_agent_dispatches_multi_event_answer_then_focus() -> None:
    async def run() -> tuple[str, str, list[cognition.types.OutputData]]:
        phone = phone_device.Phone()
        ability = SequenceAbility(
            (
                cognition.types.EventData(
                    target="phone",
                    event=phone_device.AnswerCallEvent.name,
                    data={"call_id": "call-123"},
                    reason="answer incoming call",
                ),
                *focus_output("phone", "also keep focus"),
            )
        )
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        await ring_phone(phone)
        await wait_until(lambda: bot_has_focus(active_bot) and active_bot.state() == "/Bot/active/focused")
        assert device_firmware(phone) is not None

        return active_bot.state(), device_firmware(phone).state(), active_bot.actions

    state, phone_state, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert phone_state != ""
    assert actions == [
        (
            cognition.types.EventData(
                target="phone",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            ),
            *focus_output("phone", "also keep focus"),
        )
    ]


def test_focused_agent_ability_can_change_focus_device() -> None:
    async def run() -> tuple[
        str, list[cognition.InputData], list[processing.InputData], list[cognition.types.OutputData]
    ]:
        ability = SequenceAbility(
            no_output("stay on phone"),
            focus_output("browser", "change focus"),
            no_output("observe changed focus"),
        )
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=cognitive)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="browser", priority=1)),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 3 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), cognitive.inputs, ability.calls, active_bot.actions

    state, inputs, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert inputs[2].focus == "browser"
    assert len(calls) == 3
    assert calls[1].input == bot.InputEventData(target_device="browser", priority=1)
    assert actions == [
        no_output("stay on phone"),
        focus_output("browser", "change focus"),
        no_output("observe changed focus"),
    ]


def test_focused_agent_rejects_stale_completion_focus_outside_current_input() -> None:
    async def run() -> tuple[
        str, list[cognition.InputData], list[processing.InputData], list[cognition.types.OutputData]
    ]:
        release = asyncio.Event()
        ability = BlockingSequenceAbility(
            release=release,
            block_on_call=2,
            outputs=(
                no_output("initial focus"),
                no_output("real completion"),
                no_output("observe retained focus"),
            ),
        )
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser", "screen"), cognition=cognitive)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="phone", priority=3),
                "stale-focus-seed",
            ),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="browser", priority=1),
                "stale-focus-live",
            ),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/processing")

        # Correct cognition source/target, wrong turn id — must not move focus.
        stale_focus = dataclasses.replace(
            bot.FocusDeviceEvent.with_data(bot.FocusDeviceEventData(device="screen", reason="stale")),
            id="not-stale-focus-live",
            source=hsm.id(cognitive),
            target=hsm.id(active_bot),
        )
        await active_bot.dispatch(active_bot.context(), stale_focus)
        assert active_bot.state() == "/Bot/active/processing"

        _ = release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="browser", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 3 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), cognitive.inputs, ability.calls, active_bot.actions

    state, inputs, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert inputs[2].focus == "phone"
    assert len(calls) == 3
    assert actions == [
        no_output("initial focus"),
        no_output("real completion"),
        no_output("observe retained focus"),
    ]


def test_bot_rejects_forged_focus_for_unconfigured_device() -> None:
    async def run() -> str:
        active_bot = basic_agent(devices={})
        _ = await start_bot_with_devices(active_bot)
        forged = dataclasses.replace(
            bot.FocusDeviceEvent.with_data(bot.FocusDeviceEventData(device="ghost", reason="forged")),
            metadata={"bot.focus_candidates": ("ghost",)},
        )

        await active_bot.dispatch(active_bot.context(), forged)
        await asyncio.sleep(0)

        return active_bot.state()

    assert asyncio.run(run()) == "/Bot/active/unfocused"


def test_bot_rejects_forged_focus_action_source_during_current_processing() -> None:
    async def run() -> bool:
        release = asyncio.Event()
        processor = BlockingSequenceProcessor(
            release=release,
            block_on_call=2,
            outputs=(no_output("initial"), no_output("current")),
        )
        cognitive = CapturingCognition(processor)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognitive)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="phone", priority=1),
                "initial-focus",
            ),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="phone", priority=2),
                "current-focus-source",
            ),
        )
        await wait_until(lambda: len(cognitive.input_events) == 2)
        request = cognitive.input_events[1]
        forged = dataclasses.replace(
            bot.ClearFocusEvent.with_data(bot.ClearFocusEventData(reason="forged source")),
            id=f"{request.id}:intuition",
            source="forged-source",
            target=hsm.id(active_bot),
            metadata=dict(request.metadata),
        )
        await active_bot.dispatch(active_bot.context(), forged)
        await asyncio.sleep(0)
        release.set()
        await wait_until(lambda: active_bot.state() != "/Bot/active/processing")
        return bot_has_focus(active_bot)

    assert asyncio.run(run())


def test_focused_agent_rejects_focus_device_outside_processing_candidates() -> None:
    async def run() -> tuple[
        str,
        list[cognition.InputData],
        list[cognition.types.OutputData],
        list[bot.ProcessingFailedEventData],
    ]:
        ability = SequenceAbility(
            no_output("stay on phone"),
            focus_output("screen", "bad target"),
            no_output("observe retained focus"),
        )
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser", "screen"), cognition=cognitive)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="browser", priority=1)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="browser", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 3 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), cognitive.inputs, active_bot.actions, active_bot.failures

    state, inputs, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert inputs[2].focus == "phone"
    assert actions == [no_output("stay on phone"), no_output("observe retained focus")]
    assert len(failures) == 1
    assert failures[0].message == "Processing selected focus_device outside available device candidates."


def test_focused_agent_ability_can_clear_focus() -> None:
    async def run() -> tuple[str, str | None, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = SequenceAbility(
            no_output("stay on phone"),
            clear_output("done"),
        )
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=5)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")

        return active_bot.state(), bot_has_focus(active_bot), ability.calls, active_bot.actions

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert not focused_device
    assert len(calls) == 2
    assert calls[1].input == bot.InputEventData(target_device="phone", priority=5)
    assert actions == [no_output("stay on phone"), clear_output("done")]


def test_bot_processing_state_defers_repeated_input_until_processing_completes() -> None:
    async def run() -> tuple[str, list[int], list[cognition.types.OutputData]]:
        release = asyncio.Event()
        ability = BlockingAbility(release=release)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=1)),
        )
        await wait_until(lambda: ability.calls == [1])

        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=2)),
        )
        assert ability.calls == [1]
        _ = release.set()
        await wait_until(lambda: ability.calls == [1, 2] and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), ability.calls, active_bot.actions

    state, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert calls == [1, 2]
    assert actions == [no_output("priority:1"), no_output("priority:2")]


def test_bot_focus_change_during_processing_does_not_swallow_completion() -> None:
    async def run() -> tuple[str, str | None, list[cognition.types.OutputData]]:
        release = asyncio.Event()
        ability = BlockingAbility(release=release)
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="phone", priority=3),
                "processing-focus-change",
            ),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/processing")

        focus = dataclasses.replace(
            bot.FocusDeviceEvent.with_data(bot.FocusDeviceEventData(device="browser", reason="operator choice")),
            metadata={"bot.focus_candidates": ("phone", "browser")},
        )
        await active_bot.dispatch(active_bot.context(), focus)
        assert active_bot.state() == "/Bot/active/processing"

        release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        return active_bot.state() or "", bot_has_focus(active_bot), active_bot.actions

    state, focused_device, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert actions == [no_output("priority:3")]


def test_bot_processing_state_defers_observed_phone_event_until_processing_completes() -> None:
    async def run() -> tuple[str, list[processing.InputData], list[cognition.types.OutputData]]:
        release = asyncio.Event()
        ability = BlockingSequenceAbility(
            release=release,
            block_on_call=1,
            outputs=(
                no_output("initial"),
                no_output("observed"),
            ),
        )
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=1)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/processing")

        await emit_phone_service_event(
            phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123"))
        )
        assert len(ability.calls) == 1
        _ = release.set()
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), ability.calls, active_bot.actions

    state, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert calls[0].input == bot.InputEventData(target_device="phone", priority=1)
    assert_heard_phone_ring(calls[1].input)
    assert actions == [
        no_output("initial"),
        no_output("observed"),
    ]


def test_bot_processing_state_ignores_device_event_while_processing() -> None:
    async def run() -> tuple[str, list[processing.InputData], list[cognition.types.OutputData]]:
        release = asyncio.Event()
        ability = BlockingSequenceAbility(
            release=release,
            block_on_call=1,
            outputs=(no_output("initial"),),
        )
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=1)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/processing")

        await active_bot.dispatch(
            active_bot.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number="phone-bot-bob")),
        )
        assert len(ability.calls) == 1
        _ = release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), ability.calls, active_bot.actions

    state, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert len(calls) == 1
    assert calls[0].input == bot.InputEventData(target_device="phone", priority=1)
    assert actions == [no_output("initial")]


def test_deferred_input_replays_after_focus_device_completion() -> None:
    async def run() -> tuple[str, str | None, list[processing.InputData], list[cognition.types.OutputData]]:
        release = asyncio.Event()
        ability = BlockingSequenceAbility(
            release=release,
            block_on_call=2,
            outputs=(
                no_output("initial focus"),
                focus_output("browser", "change focus"),
                no_output("after focus change"),
            ),
        )
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=1)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")

        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="browser", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/processing")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        assert len(ability.calls) == 2
        _ = release.set()
        await wait_until(lambda: len(ability.calls) == 3 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), ability.calls, active_bot.actions

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert calls[2].input == bot.InputEventData(target_device="phone", priority=3)
    assert actions == [
        no_output("initial focus"),
        focus_output("browser", "change focus"),
        no_output("after focus change"),
    ]


def test_deferred_input_replays_after_clear_focus_completion() -> None:
    async def run() -> tuple[str, str | None, list[processing.InputData], list[cognition.types.OutputData]]:
        release = asyncio.Event()
        ability = BlockingSequenceAbility(
            release=release,
            block_on_call=2,
            outputs=(
                no_output("initial focus"),
                clear_output("clear before replay"),
                no_output("after clear"),
            ),
        )
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=1)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")

        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/processing")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="browser", priority=3)),
        )
        assert len(ability.calls) == 2
        _ = release.set()
        await wait_until(lambda: len(ability.calls) == 3 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), ability.calls, active_bot.actions

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert calls[2].input == bot.InputEventData(target_device="browser", priority=3)
    assert actions == [
        no_output("initial focus"),
        clear_output("clear before replay"),
        no_output("after clear"),
    ]


def test_bot_processing_state_rejects_malformed_input_event_data() -> None:
    async def run() -> tuple[str, list[processing.InputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices={}, cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        malformed = bot.InputEvent.with_data(typing.cast(bot.InputEventData, object()))
        await active_bot.dispatch(active_bot.context(), malformed)
        await asyncio.sleep(0)

        return active_bot.state(), ability.calls

    state, calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert calls == []


def test_bot_processing_state_routes_ability_failure_to_failed_event() -> None:
    async def run() -> tuple[str, list[bot.ProcessingFailedEventData], list[cognition.types.OutputData]]:
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=FailingAbility())

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.failures, active_bot.actions

    state, failures, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert len(failures) == 1
    assert failures[0].message == "ability unavailable"
    assert actions == []


@pytest.mark.parametrize("fails", [False, True])
def test_bot_processing_terminal_finishes_operation_timer_actor(fails: bool) -> None:
    async def run() -> tuple[str, dict[str, object]]:
        processor: processing.Processor = FailingProcessor() if fails else NestedPrimaryProcessor()
        cognition_ability = CapturingCognition(processor)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognition_ability)
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(bot.InputEventData(target_device="phone", priority=3), "timer-turn"),
        )
        await wait_until(lambda: bool(cognition_ability.input_events))
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        return active_bot.state(), cognition_ability.input_events[0].metadata

    state, metadata = asyncio.run(run())
    assert state == "/Bot/active/focused"
    assert "bot.processing.actor" not in metadata
    assert "bot.processing.operation" not in metadata


def test_bot_rejects_stale_processing_completion_for_blocked_turn() -> None:
    async def run() -> str:
        release = asyncio.Event()
        ability = BlockingAbility(release=release)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="phone", priority=3),
                "live-turn",
            ),
        )
        await wait_until(lambda: ability.calls == [3] and active_bot.state() == "/Bot/active/processing")

        # Correct endpoints, wrong turn id: topology alone must not complete the live turn.
        stale = dataclasses.replace(
            bot.ProcessingCompletedEvent.with_data(
                bot.ProcessingCompletedEventData(output=no_output("stale"), focus_candidates=("phone",))
            ),
            id="stale-operation",
            source=hsm.id(active_bot._cognition),
            target=hsm.id(active_bot),
        )
        await active_bot.dispatch(active_bot.context(), stale)
        state_after_stale = active_bot.state()
        release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return state_after_stale

    assert asyncio.run(run()) == "/Bot/active/processing"


def test_bot_rejects_stale_cognition_output_with_correct_endpoints() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData]]:
        release = asyncio.Event()
        processor = BlockingSequenceProcessor(
            release=release,
            block_on_call=1,
            outputs=(no_output("live"),),
        )
        cognitive = CapturingCognition(processor)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognitive)
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="phone", priority=3),
                "live-output-turn",
            ),
        )
        await wait_until(lambda: bool(cognitive.input_events) and active_bot.state() == "/Bot/active/processing")
        stale_output = dataclasses.replace(
            cognition.OutputEvent.with_data(no_output("stale-output")),
            id="not-the-live-turn",
            source=hsm.id(cognitive),
            target=hsm.id(active_bot),
        )
        await active_bot.dispatch(active_bot.context(), stale_output)
        await asyncio.sleep(0)
        assert active_bot.state() == "/Bot/active/processing"
        assert active_bot.actions == []
        release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        return active_bot.state(), active_bot.actions

    state, actions = asyncio.run(run())
    assert state == "/Bot/active/focused"
    assert actions == [no_output("live")]


def test_bot_rejects_focus_for_wrong_turn_id_during_processing() -> None:
    async def run() -> tuple[str, str | None]:
        release = asyncio.Event()
        processor = BlockingSequenceProcessor(
            release=release,
            block_on_call=2,
            outputs=(no_output("seed"), no_output("live"), no_output("observe")),
        )
        cognitive = InputRecordingCognition(
            processing.Processing(processor=processor),
        )
        active_bot = AbilityAgent(
            devices=configured_devices("phone", "browser"),
            cognition=cognitive,
        )
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="phone", priority=1),
                "focus-seed",
            ),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused" and len(cognitive.inputs) == 1)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="phone", priority=2),
                "focus-live-turn",
            ),
        )
        await wait_until(lambda: len(cognitive.inputs) == 2 and active_bot.state() == "/Bot/active/processing")
        forged = dataclasses.replace(
            bot.FocusDeviceEvent.with_data(bot.FocusDeviceEventData(device="browser", reason="wrong turn")),
            id="not-focus-live-turn",
            source=hsm.id(cognitive),
            target=hsm.id(active_bot),
        )
        await active_bot.dispatch(active_bot.context(), forged)
        await asyncio.sleep(0)
        release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(cognitive.inputs) == 3 and active_bot.state() == "/Bot/active/focused")
        return active_bot.state(), cognitive.inputs[2].focus

    state, focus = asyncio.run(run())
    assert state == "/Bot/active/focused"
    assert focus == "phone"


def test_bot_rejects_cancelled_with_wrong_token() -> None:
    async def run() -> str:
        processor = CancellableHangingProcessor()
        cognitive = CapturingCognition(processor)
        active_bot = TimeoutAbilityAgent(devices=configured_devices("phone"), cognition=cognitive)
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="phone", priority=3),
                "cancel-token-turn",
            ),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/cancelling_processing")
        forged = dataclasses.replace(
            cognition.CancelledEvent.with_data(
                cognition.CancelledData(operation_id="cancel-token-turn", token="not-the-timer-token")
            ),
            id="cancel-token-turn",
            source=hsm.id(cognitive),
            target=hsm.id(active_bot),
        )
        await active_bot.dispatch(active_bot.context(), forged)
        await asyncio.sleep(0)
        return active_bot.state()

    assert asyncio.run(run()) == "/Bot/active/cancelling_processing"


@pytest.mark.parametrize("wrong_endpoint", ["source", "target"])
def test_bot_rejects_current_processing_terminal_from_wrong_endpoint(wrong_endpoint: str) -> None:
    async def run() -> str:
        processor = CancellableHangingProcessor()
        cognition_ability = CapturingCognition(processor)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognition_ability)
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data_and_id(
                bot.InputEventData(target_device="phone", priority=3),
                "current-operation",
            ),
        )
        await wait_until(lambda: bool(cognition_ability.input_events))
        request = cognition_ability.input_events[0]
        forged = dataclasses.replace(
            bot.ProcessingCompletedEvent.with_data(
                bot.ProcessingCompletedEventData(output=no_output("forged"), focus_candidates=("phone",))
            ),
            id=request.id,
            source="forged-source" if wrong_endpoint == "source" else hsm.id(cognition_ability),
            target="forged-target" if wrong_endpoint == "target" else hsm.id(active_bot),
            metadata=dict(request.metadata),
        )
        await active_bot.dispatch(active_bot.context(), forged)
        await asyncio.sleep(0)
        return active_bot.state()

    assert asyncio.run(run()) == "/Bot/active/processing"


def test_bot_processing_state_times_out_hanging_ability() -> None:
    async def run() -> tuple[
        str, bool, str | None, list[int], list[bot.ProcessingFailedEventData], list[cognition.types.OutputData]
    ]:
        ability = CancellableHangingAbility()
        # First turn hangs forever (timeout cancels). Second turn returns immediately; use a
        # recovery-scale timeout so the full Cognition pipeline can finish without racing 1ms.
        active_bot = QuickTimeoutAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused" and ability.cancelled)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=4)),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused")

        return (
            active_bot.state(),
            ability.cancelled,
            ability.state(),
            ability.calls,
            active_bot.failures,
            active_bot.actions,
        )

    state, cancelled, ability_state, calls, failures, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert cancelled
    # Cognition injects the leaf Processor only; the Processing shell is not started.
    assert ability_state in (None, "")
    assert calls == [3, 4]
    assert len(failures) == 1
    assert "timed out" in failures[0].message
    assert actions == [no_output("priority:4")]


def test_bot_processing_timeout_recovery_finishes_cancellation() -> None:
    async def run() -> tuple[str, bool, list[bot.ProcessingFailedEventData]]:
        ability = CancellableHangingAbility()
        active_bot = TimeoutAbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await asyncio.sleep(0.05)

        return active_bot.state(), ability.cancelled, active_bot.failures

    state, cancelled, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert cancelled
    assert len(failures) == 1
    assert "timed out" in failures[0].message


def test_bot_processing_cancellation_timeout_fails_closed_and_rejects_next_turn() -> None:
    async def run() -> tuple[str, list[int], list[bot.ProcessingFailedEventData]]:
        processor = CancellableHangingProcessor()
        cognition_ability = CapturingCognition(processor, swallow_cancel=True)
        active_bot = TimeoutAbilityAgent(devices=configured_devices("phone"), cognition=cognition_ability)
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        for _ in range(100):
            if active_bot.state() == "/Bot/degraded":
                break
            await asyncio.sleep(0.002)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=4)),
        )
        await asyncio.sleep(0.01)
        return active_bot.state(), processor.calls, active_bot.failures

    state, calls, failures = asyncio.run(run())
    assert state == "/Bot/degraded"
    assert calls == [3]
    assert len(failures) == 1
    assert "timed out" in failures[0].message


def test_completed_turn_cancels_processing_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    operations: list[object] = []
    started = bot_module._BotProcessingOperation.started

    @classmethod
    async def spy_started(
        cls,
        ctx: hsm.Context,
        operation: bot_module._BotProcessingOperation,
        **kwargs: typing.Any,
    ) -> bot_module._BotProcessingOperation:
        operations.append(operation)
        return await started(ctx, operation, **kwargs)

    monkeypatch.setattr(bot_module._BotProcessingOperation, "started", spy_started)

    async def run() -> tuple[list[bot.ProcessingFailedEventData], list[cognition.types.OutputData]]:
        active_bot = QuickTimeoutAgent(devices=configured_devices("phone"), cognition=IgnoreAbility())
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused" and bool(active_bot.actions))
        await asyncio.sleep(0.05)  # well past the 20ms turn timeout
        return active_bot.failures, active_bot.actions

    failures, actions = asyncio.run(run())

    timer_stopped = bool(operations) and all(operation.state() in {"", "/BotProcessingTimer"} for operation in operations)

    # Completing the turn exits processing; the owning activity stops the timer explicitly.
    assert timer_stopped
    assert failures == []
    assert actions == [no_output("priority:3")]


def test_bot_timeout_cancel_uses_cognition_cancel_event_contract() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        cognition_ability = StubCognition()
        active_bot = TimeoutStubCognitionAgent(cognition_ability)
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(cognition_ability.received))
        return cognition_ability.received

    received = asyncio.run(run())

    assert [event.name for event in received] == [_StubCancelEvent.name]
    data = received[0].data
    assert isinstance(data, _StubCancelData)
    assert data.operation_id
    assert data.token


def test_bot_timeout_cancel_falls_back_to_processing_cancel_event() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        cognition_ability = DefaultCognition()
        active_bot = TimeoutStubCognitionAgent(cognition_ability)
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(cognition_ability.received))
        return cognition_ability.received

    received = asyncio.run(run())

    assert [event.name for event in received] == [processing.CancelEvent.name]
    data = received[0].data
    assert isinstance(data, processing.CancelData)
    assert data.operation_id
    assert data.token


def test_attach_is_idempotent_for_started_bot() -> None:
    async def run() -> str:
        active_bot = basic_agent()
        environment = await start_bot_with_devices(active_bot)
        await wait_until(lambda: active_bot.state().startswith("/Bot/active/"))
        _ = await active_bot.attach(environment)
        await asyncio.sleep(0)
        return active_bot.state()

    assert asyncio.run(run()).startswith("/Bot/active/")


def test_detach_is_idempotent_for_unstarted_bot() -> None:
    async def run() -> None:
        detached = basic_agent()
        _ = await detached.detach(Environment())

    asyncio.run(run())


def test_bot_activation_dispatches_completion_after_queueing_device_notifications() -> None:
    async def run() -> None:
        device = Device()
        active_bot = basic_agent(devices={"phone": device})

        environment = await start_bot_with_devices(active_bot)
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        assert active_bot.state() == "/Bot/inactive"
        assert device_bots(device) == ()

        _ = await active_bot.attach(environment)
        await wait_until(lambda: device_bots(device) == (active_bot,))
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")

        assert active_bot.state() == "/Bot/active/unfocused"
        assert device_bots(device) == (active_bot,)

    asyncio.run(run())


def test_bot_rejects_stale_attachment_terminal_from_previous_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, str]:
        active_bot = basic_agent(devices={"phone": Device()})
        environment = Environment()
        original_attach = attachment.Group.attach
        prior_request: hsm.Event[attachment.AttachData] | None = None
        bot_group: attachment.Group | None = None
        release = asyncio.Event()
        bot_attach_count = 0

        async def attach_group(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            nonlocal prior_request, bot_group, bot_attach_count
            if isinstance(event.data, attachment.AttachData) and event.data.actor is active_bot:
                bot_attach_count += 1
                bot_group = group
                if bot_attach_count == 1:
                    prior_request = event
                else:
                    await release.wait()
            await original_attach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "attach", attach_group)
        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/activating")

        assert prior_request is not None
        assert bot_group is not None
        stale = dataclasses.replace(
            attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=active_bot, created=True)),
            id=prior_request.id,
            source=hsm.id(bot_group),
            target=hsm.id(active_bot),
            metadata=dict(prior_request.metadata),
        )
        await active_bot.dispatch(active_bot.context(), stale)
        state_after_stale = active_bot.state()
        release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        return state_after_stale, active_bot.state()

    state_after_stale, final_state = asyncio.run(run())

    assert state_after_stale == "/Bot/activating"
    assert final_state == "/Bot/active/unfocused"


def test_bot_deactivation_clears_focus() -> None:
    async def run() -> tuple[str, str | None]:
        active_bot = basic_agent(devices=configured_devices("phone"))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bot_has_focus(active_bot))

        environment = Environment.from_context(active_bot.context())
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return active_bot.state(), bot_has_focus(active_bot)

    state, focused_device = asyncio.run(run())

    assert state == "/Bot/inactive"
    assert not focused_device


def test_bot_deactivation_stops_input_output_abilities() -> None:
    """Normal deactivate stops input/output abilities; cognition stays started."""

    async def run() -> tuple[bool, bool, bool, bool]:
        cognition_ability = as_cognition(IgnoreAbility())
        input_ability = ProbeAbility()
        output_ability = ProbeAbility()
        active_bot = AbilityAgent(
            devices={},
            cognition=cognition_ability,
            input=(input_ability,),
            output=(output_ability,),
        )
        environment = await start_bot_with_devices(active_bot)
        assert bot.lifecycle.is_started(input_ability) is True
        assert bot.lifecycle.is_started(output_ability) is True
        assert bot.lifecycle.is_started(cognition_ability) is True

        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return (
            bot.lifecycle.is_started(input_ability),
            bot.lifecycle.is_started(output_ability),
            bot.lifecycle.is_started(cognition_ability),
            bot.lifecycle.is_started(active_bot),
        )

    input_live, output_live, cognition_live, bot_live = asyncio.run(run())

    assert input_live is False
    assert output_live is False
    assert cognition_live is True
    assert bot_live is True


def test_bot_focus_state_has_no_public_accessor() -> None:
    active_bot = basic_agent()

    assert "focused_device" not in vars(type(active_bot))
    assert "focused_device" not in vars(active_bot)


def test_bot_core_does_not_import_device_audio_events() -> None:
    import bot

    assert bot.InputEvent.name == "bot.input"
    assert not hasattr(bot_module, "OutputEvent")
    assert not hasattr(bot_module, "_RAW_AUDIO_EVENT_NAMES")
    assert "bot.devices.audio" not in Path(bot_module.__file__).read_text()


def test_bot_ignores_unhandled_device_event() -> None:
    async def run() -> tuple[str, list[processing.InputData], str | None]:
        probe = hsm.Event[dict[str, object]](name="phone.probe.signal")
        ability = IgnoreAbility()
        phone = Device()
        active_bot = AbilityAgent(devices={"phone": phone}, cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        assert active_bot.state() == "/Bot/active/unfocused"
        await active_bot.dispatch(
            active_bot.context(),
            dataclasses.replace(probe.with_data({"kind": "ping"}), source=hsm.id(phone)),
        )
        await asyncio.sleep(0)

        return active_bot.state(), ability.calls, bot_has_focus(active_bot)

    state, calls, focused = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert calls == []
    assert not focused


def test_bot_cleanup_timeout_reports_degraded_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class HangOnDetachDevice(Device):
        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
            if event.name == attachment.DetachEvent.name:

                async def hang() -> None:
                    _ = await asyncio.Event().wait()

                return hang()
            return super().dispatch(ctx, event)

    async def run() -> str:
        active_bot = FastDeactivationTimeoutAgent(devices={"stuck": HangOnDetachDevice()})
        environment = await start_bot_with_devices(active_bot)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        original_stop = hsm.stop

        async def stop(instance: hsm.Instance, ctx: hsm.Context) -> None:
            if isinstance(instance, attachment.Group):
                _ = await asyncio.Event().wait()
            await original_stop(instance, ctx)

        monkeypatch.setattr(hsm, "stop", stop)
        _ = await active_bot.detach(environment)
        # FastDeactivationTimeoutAgent uses a 10ms HSM after; wait_until only yields, so give
        # the timer wall-clock time to fire deactivating -> cleanup -> degraded.
        for _ in range(50):
            if active_bot.state() == "/Bot/degraded":
                break
            await asyncio.sleep(0.005)
        return active_bot.state() or ""

    assert asyncio.run(run()) == "/Bot/degraded"


def test_bot_activation_attaches_started_and_unstarted_devices() -> None:
    async def run() -> None:
        started_device = Device()
        unstarted_device = Device()
        active_bot = basic_agent(devices={"started": started_device, "unstarted": unstarted_device})

        environment = Environment()
        _ = await hsm.started(environment, started_device, started_device.model)
        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        await wait_until(lambda: device_bots(started_device) == (active_bot,))
        await wait_until(lambda: device_bots(unstarted_device) == (active_bot,))

        assert active_bot.state() == "/Bot/active/unfocused"
        assert device_bots(started_device) == (active_bot,)
        assert device_bots(unstarted_device) == (active_bot,)

    asyncio.run(run())


def test_bot_activation_starts_phone_peripherals_in_agent_environment() -> None:
    async def run() -> tuple[str, str, str, str, bool, bool, bool, bool]:
        phone = phone_device.Phone()
        active_bot = basic_agent(devices={"phone": phone})

        environment = await start_bot_with_devices(active_bot)
        assert device_firmware(phone) is not None
        environment_scope = environment.value(hsm.Keys.Instances)

        microphone = phone_microphone(phone)
        speaker = phone_speaker(phone)
        return (
            phone.state(),
            device_firmware(phone).state(),
            microphone.state(),
            speaker.state(),
            active_bot.context().value(hsm.Keys.Instances) is environment_scope,
            phone.context().value(hsm.Keys.Instances) is environment_scope,
            microphone.context().value(hsm.Keys.Instances) is environment_scope,
            speaker.context().value(hsm.Keys.Instances) is environment_scope,
        )

    phone_state, firmware_state, microphone_state, speaker_state, agent_scope, phone_scope, mic_scope, speaker_scope = (
        asyncio.run(run())
    )

    assert phone_state == "/Device/attached"
    assert firmware_state == "/Phone/hung_up"
    # Wired, not idle: the phone powers its transducers and its firmware attaches to them during
    # bring-up, so they sit attached for the life of the phone. Call state gates the uplink in
    # firmware topology, not the wiring.
    assert microphone_state == "/Device/attached"
    assert speaker_state == "/Device/attached"
    assert agent_scope
    assert phone_scope
    assert mic_scope
    assert speaker_scope


def test_bot_lifecycle_starts_and_stops_innate_ability() -> None:
    async def run() -> tuple[list[processing.InputData], str]:
        bot_instance = FocusedAgent(devices={})
        probe = bot_instance.probe

        environment = await start_bot_with_devices(bot_instance)
        _ = await probe.apply(probe_input())
        await wait_until(lambda: bool(probe.calls))

        _ = await bot_instance.detach(environment)
        await wait_until(lambda: bot_instance.state() == "/Bot/inactive")

        return probe.calls, bot_instance.state()

    calls, state = asyncio.run(run())

    assert calls == [probe_input()]
    assert state == "/Bot/inactive"


def test_bot_lifecycle_starts_and_stops_acquired_abilities() -> None:
    async def run() -> tuple[list[processing.InputData], str]:
        acquired = ProbeAbility()
        bot_instance = AbilityAgent(
            devices={},
            cognition=IgnoreAbility(),
            acquired_abilities=(acquired,),
        )

        environment = await start_bot_with_devices(bot_instance)
        _ = await acquired.apply(probe_input())
        await wait_until(lambda: bool(acquired.calls))

        _ = await bot_instance.detach(environment)
        await wait_until(lambda: bot_instance.state() == "/Bot/inactive")

        return acquired.calls, acquired.state()

    calls, acquired_state = asyncio.run(run())

    assert calls == [probe_input()]
    assert acquired_state in {"", "/ProbeAbilityLifecycle"}  # hsm 1.3.2: stopped state is empty


def test_bot_processing_does_not_define_coordination_metadata_keys() -> None:
    assert_metadata_key_prefix_is_absent(bot_module, "_PROCESSING_")


def test_a_placed_bot_only_hears_what_is_loud_enough_where_it_stands() -> None:
    """A bot's placement is its ears: it hears the near source and not the far one.

    Pins that ``Bot.attach`` carries the placement through to presence. Without it the bot is
    unplaced, which means unselective, and both broadcasts arrive.
    """

    async def run() -> tuple[int, int]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices={}, cognition=ability, input=(ring_hearing(),))
        here = space.Position(x=0.0, y=0.0)

        environment = await start_bot_with_devices(
            active_bot, placement=space.Placement(position=here, threshold_db=20.0)
        )
        ability.calls.clear()

        def sound(audio_bytes: bytes) -> hsm.Event[typing.Any]:
            return SoundEvent.with_data(
                SoundData(
                    audio=audio_bytes,
                    media_type="audio/pcm",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="ambient",
                    amplitude_db=60.0,
                )
            )

        await environment.broadcast(sound(b"far-away"), origin=space.Position(x=500.0, y=0.0))
        await asyncio.sleep(0)
        after_far = len(ability.calls)

        await environment.broadcast(sound(b"right-here"), origin=here)
        await wait_until(lambda: len(ability.calls) > after_far)

        return after_far, len(ability.calls)

    after_far, after_near = asyncio.run(run())

    assert after_far == 0
    assert after_near == 1


class UtteranceEncoder(encoding.Encoder[bytes, bytes]):
    """Stands in for a vocal tract: text in, acoustic bytes out, on this machine.

    A real synthesizer (macOS ``say``, Moonshine) is the same contract and is what the phone-bot
    example uses. Here the audio only has to be traceable back to the words, so the pipeline can
    be checked without a model download.
    """

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        return b"spoken:" + input


async def somebody_speaks(
    environment: Environment,
    text: str,
    *,
    position: space.Position,
    amplitude_db: float = 60.0,
) -> audio.Speaker:
    """Put a mouth in the room at ``position`` and say ``text`` out loud through it.

    A mouth is a ``Speaker``: exactly the device a bot uses for its own voice, because a person's
    mouth and a robot's mouth are the same object acoustically. Everything downstream goes through
    ``Environment.broadcast`` — no dispatch at the body, no forged ``environment.sound``.
    """

    mouth = audio.Speaker(placement=space.Placement(position=position), amplitude_db=amplitude_db)
    _ = await hsm.started(environment, mouth, typing.cast(hsm.Model, mouth.model))
    voice = speaking.Speaking(
        encoder=UtteranceEncoder(),
        speaker=mouth,
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/wav",
    )
    await start_ability_tree(environment, voice)
    _ = await voice.apply(speaking.InputData(text=text), ctx=environment)
    return mouth


def test_a_bot_hears_words_somebody_speaks_in_its_environment() -> None:
    """Somebody stands a metre away and says something; the bot's turn arrives as words.

    The whole path is real: a mouth in the environment, ``Environment.broadcast`` with a level
    and an origin, the bot's own placement deciding it is loud enough where it stands, body
    fan-out to Listening, voice detection, speech decoding, and the decoded product handed to
    judgment. The decoded bytes carry the utterance forward through every stage, so this cannot
    pass on a pipeline that dropped the audio and substituted something else.

    What it refuses to assert is anything the bot does about it. No device is dialed, nothing is
    answered, and no output is expected — the turn is the whole claim. A bot that hears this
    sentence and does nothing for the rest of its life passes.
    """

    async def run() -> list[hsm.Event[typing.Any]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices={}, cognition=ability, input=(RecordingListening(),))
        environment = await start_bot_with_devices(
            active_bot,
            placement=space.Placement(position=space.Position(x=0.0, y=0.0), threshold_db=20.0),
        )
        ability.calls.clear()

        _ = await somebody_speaks(environment, "Call Bob at phone-bot-bob.", position=space.Position(x=0.0, y=1.0))

        await wait_until(lambda: bool(ability.calls), timeout=10.0)
        return [call.input for call in ability.calls if isinstance(call.input, hsm.Event)]

    stimuli = asyncio.run(run())

    assert [stimulus.name for stimulus in stimuli] == [speech.SpeechDecoding.output_event.name]
    assert stimuli[0].data == b"decoded:spoken:Call Bob at phone-bot-bob."


def test_words_spoken_from_across_the_room_never_reach_the_bot() -> None:
    """The same sentence, said 500 metres away, is a sentence the bot did not hear.

    This is what proves the utterance went *through* the environment rather than around it: the
    only difference between this and the test above is where the mouth is standing, and a
    dispatch aimed at the body would ignore that entirely.
    """

    async def run() -> tuple[int, int]:
        ability = IgnoreAbility()
        listening_ability = RecordingListening()
        active_bot = AbilityAgent(devices={}, cognition=ability, input=(listening_ability,))
        environment = await start_bot_with_devices(
            active_bot,
            placement=space.Placement(position=space.Position(x=0.0, y=0.0), threshold_db=20.0),
        )
        ability.calls.clear()

        _ = await somebody_speaks(environment, "Call Bob at phone-bot-bob.", position=space.Position(x=0.0, y=500.0))
        # Long enough for the near case to have finished decoding twice over.
        await asyncio.sleep(0.5)
        shouted_from_far = len(ability.calls)

        _ = await somebody_speaks(environment, "Call Bob at phone-bot-bob.", position=space.Position(x=0.0, y=1.0))
        await wait_until(lambda: bool(ability.calls), timeout=10.0)
        return shouted_from_far, len(ability.calls)

    from_far, from_near = asyncio.run(run())

    assert from_far == 0
    assert from_near == 1


def test_a_bot_that_was_told_nothing_still_gets_a_turn() -> None:
    """Topology grants the occasion; it never grants the action.

    Nothing is dispatched at this bot: no input, no sound, no device, no directive, and no
    memory to have recalled one from. Being awake is the whole condition, so the turn happens
    anyway — and judgment is the one that gets to decide the turn is a turn to do nothing.

    This is the test that would fail if the idle transition ever grew a guard that read a
    directive, a goal, or any other content.
    """

    async def run() -> tuple[list[processing.InputData], list[bot.ProcessingFailedEventData], str]:
        ability = IgnoreAbility()
        active_bot = IdleAgent(devices={}, cognition=ability)
        environment = await start_bot_with_devices(active_bot)

        await wait_until(lambda: bool(ability.calls))
        turns = list(ability.calls)
        failures = list(active_bot.failures)

        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        return turns, failures, active_bot.state()

    turns, failures, state = asyncio.run(run())

    stimulus = turns[0].input
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == bot.IdleEvent.name
    # The occasion is content-blind: an empty payload is the entire stimulus.
    assert stimulus.data == bot.IdleEventData()
    assert failures == []
    assert state == "/Bot/inactive"


def test_the_idle_occasion_enters_processing_carrying_its_own_id() -> None:
    """Time events arrive with no id and no data, so the body mints the occasion itself.

    The minted envelope is what processing correlates the turn against, and it is what makes
    ``bot.idle`` the recorded stimulus of the turn.
    """

    async def run() -> tuple[str, list[hsm.Event[typing.Any]], list[int]]:
        release = asyncio.Event()
        ability = BlockingAbility(release=release)
        active_bot = IdleAgent(devices={}, cognition=ability)
        environment = await start_bot_with_devices(active_bot)

        await wait_until(lambda: active_bot.state() == "/Bot/active/processing")
        processing_state = active_bot.state()
        occasions = list(active_bot.occasions)

        _ = release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        _ = await active_bot.detach(environment)
        return processing_state, occasions, list(ability.calls)

    processing_state, occasions, calls = asyncio.run(run())

    assert processing_state == "/Bot/active/processing"
    assert occasions
    assert all(occasion.id for occasion in occasions)
    assert len({occasion.id for occasion in occasions}) == len(occasions)
    assert calls


def test_a_moment_missed_while_busy_does_not_queue_up_for_later() -> None:
    """A person busy through a moment simply does not have that moment.

    ``bot.input`` is deferred through processing because an interrupt is still owed an answer.
    The occasion deliberately is not: queueing stale occasions builds a backlog of turns about
    a world that has already moved on.

    The bot here keeps the production interval, so its timer cannot fire inside this test —
    every occasion observed is one the test minted.
    """

    async def run() -> tuple[int, list[bot.ProcessingFailedEventData], str]:
        release = asyncio.Event()
        ability = BlockingAbility(release=release)
        active_bot = AbilityAgent(devices={}, cognition=ability)
        environment = await start_bot_with_devices(active_bot)

        await active_bot.dispatch(active_bot.context(), idle_occasion(active_bot))
        await wait_until(lambda: active_bot.state() == "/Bot/active/processing")

        # A second occasion arrives while the body is occupied by the first turn.
        await active_bot.dispatch(active_bot.context(), idle_occasion(active_bot))

        _ = release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        await asyncio.sleep(0)
        turns = len(ability.calls)
        failures = list(active_bot.failures)

        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        return turns, failures, active_bot.state()

    turns, failures, state = asyncio.run(run())

    assert turns == 1
    assert failures == []
    assert state == "/Bot/inactive"


def test_a_bot_that_is_not_awake_has_no_moments() -> None:
    """Occasions belong to being awake, so an unattached or stopped body has none."""

    async def run() -> tuple[int, int, int, int]:
        ability = IgnoreAbility()
        active_bot = IdleAgent(devices={}, cognition=ability)

        # Constructed but never attached: no environment, no lifetime, no moments.
        await asyncio.sleep(0.05)
        before_attach = len(active_bot.occasions)

        environment = await start_bot_with_devices(active_bot)
        await wait_until(lambda: bool(active_bot.occasions))
        while_awake = len(active_bot.occasions)

        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        at_rest = len(active_bot.occasions)
        await asyncio.sleep(0.05)
        return before_attach, while_awake, at_rest, len(active_bot.occasions)

    before_attach, while_awake, at_rest, after_rest = asyncio.run(run())

    assert before_attach == 0
    assert while_awake > 0
    assert after_rest == at_rest
