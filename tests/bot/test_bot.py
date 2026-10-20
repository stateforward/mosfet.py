from mosfet import abilities
import mosfet.lifecycle
import mosfet
from mosfet.abilities import cognition
from mosfet.abilities import listening
from mosfet.abilities import memory
from mosfet.abilities import processing
from mosfet.abilities import speaking
from mosfet.abilities import vocal
from mosfet.abilities.cognition import Cognition
from mosfet.abilities.hearing import sound as sound_hearing
from mosfet.abilities.hearing import speech
from mosfet.abilities.hearing import voice
from mosfet.devices import audio
from mosfet.devices import computer
from mosfet.devices import phone as phone_device
import asyncio
from pathlib import Path
import collections.abc
import dataclasses
import datetime
import io
import json
import inspect
import math
import os
import subprocess
import sys
import typing
import uuid
import wave

import hsm
import pydantic
import pytest

from tests.bot.abilities.cognition.metadata_contract import assert_metadata_key_prefix_is_absent
from tests.bot.behavior.support import ManualTimers

from mosfet.bot import body
from mosfet.bot.body import Bot
from mosfet.device import Device, DeviceId
from mosfet import event as event_contract
from mosfet import media
from mosfet.event import event_json_schema
from mosfet.protocols import attachment

from mosfet.abilities.speaking import EfferenceData, EfferenceEvent
from mosfet.environment import (
    SoundData,
    SoundEvent,
    VisualData,
    VisualEvent,
    Environment,
    space,
)
from tests.hsm_instance_state import (
    device_firmware,
    bot_has_focus,
    device_bots,
    phone_firmware,
    phone_microphone,
    phone_speaker,
    start_ability_tree,
)
from tests import metric_reader
from tests.devices.phone.handset import carrier_spans, published_names
from tests.bot.lifecycle_replies import LifecycleReplies, started_replies
from tests.hsm_model import choice_transitions, transition_map
from tests.type_helpers import object_dict


def no_output(reason: str = "") -> cognition.types.OutputData:
    del reason
    return ()


def focus_output(device: str, reason: str) -> cognition.types.OutputData:
    return (
        cognition.types.EventData(
            event=mosfet.FocusDeviceEvent.name,
            data=mosfet.FocusDeviceEventData(device=device).model_dump(),
            reason=reason,
        ),
    )


def event_data_schema(event: hsm.Event[typing.Any]) -> dict[str, object]:
    return event_json_schema(event)


def input_priority(input: object) -> int:
    if isinstance(input, mosfet.InputEventData):
        return input.priority
    return 0


def assert_heard_phone_ring(
    input: object,
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
        assert isinstance(input.data, phone_device.SoundData)
        assert input.data.caller == caller
    if phone is not None:
        assert input.source == hsm.id(phone)


class ProbeProcessor(processing.Processor):
    """Minimal ability used to exercise innate/acquired lifecycle on Bot."""

    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        return processing.coerce_event_selections(no_output("probe")) or ()


class ProbeAbility(processing.Processing):
    def __init__(self) -> None:
        self._probe = ProbeProcessor()
        super().__init__(processor=self._probe)

    @property
    def calls(self) -> list[processing.InputData]:
        return self._probe.calls


class IgnoreProcessor(processing.DecisionProcessor):
    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == self.input_event.name:
            self._meta.input_event_metadata.append(dict(event.metadata))
        return super().dispatch(ctx, event)


def _as_output_data(
    value: cognition.types.OutputData | cognition.types.EventData,
) -> cognition.types.OutputData:
    if isinstance(value, cognition.types.EventData):
        return (value,)
    return value


class SequenceProcessor(processing.DecisionProcessor):
    calls: list[processing.InputData]
    outputs: list[cognition.types.OutputData]

    def __init__(
        self,
        *outputs: cognition.types.OutputData | cognition.types.EventData,
    ) -> None:
        self.calls = []
        self.outputs = [_as_output_data(item) for item in outputs]

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        if not self.outputs:
            # Out of scripted answers is not an error; see BlockingSequenceProcessor.
            return ()
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


class CyclingSelectionProcessor(processing.DecisionProcessor):
    calls: list[processing.InputData]
    outputs: tuple[cognition.types.OutputData, ...]

    def __init__(self, *outputs: cognition.types.OutputData) -> None:
        self.calls = []
        self.outputs = outputs

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        output = self.outputs[(len(self.calls) - 1) % len(self.outputs)]
        return processing.coerce_event_selections(output) or ()


class BlockingSequenceProcessor(processing.DecisionProcessor):
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
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        if len(self.calls) == self.block_on_call:
            _ = await self.release.wait()
        if not self.outputs:
            # Out of scripted answers is not an error. These tests pin what the bot is given
            # and what it can do, never how many turns it gets — a turn it has nothing to say
            # about is a turn it says nothing about.
            return ()
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


class BlockingProcessor(processing.DecisionProcessor):
    calls: list[int]
    release: asyncio.Event

    def __init__(self, *, release: asyncio.Event) -> None:
        self.calls = []
        self.release = release

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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


class FailingProcessor(processing.DecisionProcessor):
    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input
        raise RuntimeError("ability unavailable")


class FailingAbility(processing.Processing):
    def __init__(self) -> None:
        super().__init__(processor=FailingProcessor())


class NestedPrimaryProcessor(processing.DecisionProcessor):
    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input
        _ = await asyncio.Event().wait()
        raise AssertionError("unreachable")


class HangingAbility(processing.Processing):
    def __init__(self) -> None:
        super().__init__(processor=HangingProcessor())


class CancellableHangingProcessor(processing.DecisionProcessor):
    calls: list[int]
    cancelled: bool

    def __init__(self) -> None:
        self.calls = []
        self.cancelled = False

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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

    def __init__(self, processor: processing.DecisionProcessor, *, swallow_cancel: bool = False) -> None:
        self.input_events = []
        self.swallow_cancel = swallow_cancel
        store = memory.Memory()
        super().__init__(
            intuition=cognition.Intuition(processor=processor),
            reasoning=cognition.Reasoning(processor=_NoopReasoningProcessor()),
            reflection=_reflection_for_test(store),
            # Cognition attaches the bot's one memory; Reflection is its peer.
            memory=store,
        )

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == cognition.InputEvent.name:
            self.input_events.append(event)
        if self.swallow_cancel and event.name == cognition.CancelEvent.name:
            # Swallowed: the cancel is never delivered.
            future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            future.set_result(False)
            return future
        return super().dispatch(ctx, event)


class _NoopReasoningProcessor(processing.Processor):
    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input
        return ()


class _PassingIntuitionProcessor(processing.DecisionProcessor):
    """Selects nothing, which intuition leaves unhandled, so every turn goes on to reasoning."""

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input
        return ()


def _reflection_for_test(store: memory.Memory) -> cognition.Reflection:
    return cognition.Reflection(processor=_NoopReasoningProcessor(), memory=store)


class InputRecordingCognition(Cognition):
    inputs: list[cognition.InputData]

    def __init__(self, processing_ability: processing.Processing) -> None:
        processor = processing_ability.processor
        if not isinstance(processor, processing.DecisionProcessor):
            raise TypeError("InputRecordingCognition expects Processing on a processing.DecisionProcessor")
        self.inputs = []
        store = memory.Memory()
        super().__init__(
            intuition=cognition.Intuition(processor=processor),
            reasoning=cognition.Reasoning(processor=_NoopReasoningProcessor()),
            reflection=_reflection_for_test(store),
            # Cognition attaches the bot's one memory; Reflection is its peer.
            memory=store,
        )

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == cognition.InputEvent.name and isinstance(event.data, cognition.InputData):
            self.inputs.append(event.data)
        return super().dispatch(ctx, event)


def _reasoning_on(processor: processing.Processor) -> cognition.Cognition:
    """A cognition whose intuition passes every turn, so ``processor`` decides each one as reasoning."""

    store = memory.Memory()
    return cognition.Cognition(
        intuition=cognition.Intuition(processor=_PassingIntuitionProcessor()),
        reasoning=cognition.Reasoning(processor=processor),
        reflection=_reflection_for_test(store),
        memory=store,
    )


def as_cognition(
    value: abilities.Ability[typing.Any, typing.Any] | processing.Processing | processing.DecisionProcessor,
) -> abilities.Ability[typing.Any, typing.Any]:
    """Bot cognition is Cognition; wrap a leaf DecisionProcessor as intuition for tests."""

    if isinstance(value, cognition.Cognition):
        return value
    if isinstance(value, processing.DecisionProcessor):
        processor: processing.DecisionProcessor = value
    elif isinstance(value, processing.Processing) and isinstance(value.processor, processing.DecisionProcessor):
        processor = value.processor
    else:
        raise TypeError("as_cognition expects Cognition, or Processing on a DecisionProcessor, or a DecisionProcessor")
    store = memory.Memory()
    return cognition.Cognition(
        intuition=cognition.Intuition(processor=processor),
        reasoning=cognition.Reasoning(processor=_NoopReasoningProcessor()),
        reflection=_reflection_for_test(store),
        memory=store,
    )


class _StubCancelData(pydantic.BaseModel):
    """Cancel payload for the contract stub cognition."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)
    token: str = pydantic.Field(min_length=1)


_StubCancelEvent = hsm.Event[_StubCancelData](name="test.stub.cancel", schema=_StubCancelData)


class _BaseStubCognition(abilities.Ability[cognition.InputData, typing.Any]):
    """Cognition stub that records cancel dispatches and never produces output."""

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "StubCognition",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
    )
    received: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.received = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
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
    failures: list[mosfet.ProcessingFailedEventData]

    def __init__(self, devices: collections.abc.Sequence[Device]) -> None:
        super().__init__(devices=devices, cognition=as_cognition(IgnoreAbility()))
        self.actions = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == mosfet.ProcessingCompletedEvent.name:
            completed = event.data
            assert isinstance(completed, mosfet.ProcessingCompletedEventData)
            self.actions.append(typing.cast(cognition.types.OutputData, completed.output))
        if event.name == mosfet.ProcessingFailedEvent.name:
            failure = event.data
            assert isinstance(failure, mosfet.ProcessingFailedEventData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


class LifecycleRecordingAgent(BasicAgent):
    lifecycle_events: list[hsm.Event[typing.Any]]

    def __init__(self, devices: collections.abc.Sequence[Device]) -> None:
        super().__init__(devices)
        self.lifecycle_events = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if isinstance(
            event.data,
            (
                mosfet.ActivatingDoneEventData,
                mosfet.ActivatingFailedEventData,
                mosfet.DeactivatingDoneEventData,
            ),
        ):
            self.lifecycle_events.append(event)
        return super().dispatch(ctx, event)


class FocusedAgent(Bot):
    """Bot subclass with an innate probe ability (not attention/focus semantics)."""

    _innate_abilities: typing.ClassVar[tuple[type[abilities.Ability[typing.Any, typing.Any]], ...]] = (ProbeAbility,)
    probe: ProbeAbility

    def __init__(
        self,
        devices: collections.abc.Sequence[Device],
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
    failures: list[mosfet.ProcessingFailedEventData]

    def __init__(
        self,
        devices: collections.abc.Sequence[Device],
        *,
        cognition: abilities.Ability[typing.Any, typing.Any] | processing.Processing,
        input: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        output: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        acquired_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        stm_memory: memory.StmMemory | None = None,
    ) -> None:
        super().__init__(
            devices=devices,
            cognition=as_cognition(cognition),
            input=input,
            output=output,
            acquired_abilities=acquired_abilities,
            stm_memory=stm_memory,
        )
        self.actions = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == mosfet.ProcessingCompletedEvent.name:
            completed = event.data
            assert isinstance(completed, mosfet.ProcessingCompletedEventData)
            self.actions.append(typing.cast(cognition.types.OutputData, completed.output))
        if event.name == mosfet.ProcessingFailedEvent.name:
            failure = event.data
            assert isinstance(failure, mosfet.ProcessingFailedEventData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


class HappeningData(pydantic.BaseModel):
    """Something that happened to a device, stated plainly and asking for nothing.

    Deliberately has no hint, suggestion, priority override, or requested action: a device
    says what it became, and what that is worth is the bot's to decide.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    situation: str = pydantic.Field(min_length=1)


HappeningEvent = hsm.Event[HappeningData](
    name="test.device.happening",
    schema=HappeningData,
)


def _dispatch_to_holders(device: Device, event: hsm.Event[typing.Any], *, priority: int = 5) -> None:
    """What a device does when something happens to it: dispatch bot.input carrying the event to its holders."""
    for owner in tuple(device._attachments):  # pyright: ignore[reportPrivateUsage]
        _ = hsm.dispatch(
            device.context(),
            owner,
            dataclasses.replace(
                mosfet.InputEvent.with_data(
                    mosfet.InputEventData(priority=priority, observation=mosfet.StimulusData.from_event(event))
                ),
                id=event.id or uuid.uuid4().hex,
                source=hsm.id(device),
                target=hsm.id(owner),
            ),
        )


class ReportingDevice(Device):
    """A device a test can make something happen to.

    Stands in for any device at all — the mechanism under test is on ``Device``, so nothing
    here is phone-shaped and no phone is needed to exercise it.
    """

    def happen(self, situation: str = "changed") -> None:
        _dispatch_to_holders(self, HappeningEvent.with_data(HappeningData(situation=situation)))


class OccasionAgent(AbilityAgent):
    """Bot that records every occasion it is handed, so a test can count its turns."""

    occasions: list[hsm.Event[typing.Any]]

    def __init__(
        self,
        devices: collections.abc.Sequence[Device],
        *,
        cognition: abilities.Ability[typing.Any, typing.Any] | processing.Processing,
    ) -> None:
        self.occasions = []
        super().__init__(devices=devices, cognition=cognition)

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == mosfet.InputEvent.name:
            self.occasions.append(event)
        return super().dispatch(ctx, event)


class TimeoutAbilityAgent(AbilityAgent):
    _processing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=1)


QUICK_PROCESSING_TIMEOUT = datetime.timedelta(milliseconds=20)


class QuickTimeoutAgent(AbilityAgent):
    _processing_timeout: typing.ClassVar[datetime.timedelta] = QUICK_PROCESSING_TIMEOUT


class FastDeactivationTimeoutAgent(BasicAgent):
    _deactivation_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=10)


class SlowInitializingDevice(Device):
    _firmware_initializing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(seconds=1)
    release: asyncio.Event

    def __init__(self, release: asyncio.Event, *, id: DeviceId | None = None) -> None:
        super().__init__(id=id)
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
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
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
        super().__init__(id="phone")
        self.fail_snapshots = False

    @typing.override
    def take_snapshot(self) -> hsm.Snapshot:
        if not self.fail_snapshots:
            return super().take_snapshot()
        raise RuntimeError("snapshot unavailable")


class ContextRecordingPhone(phone_device.Phone):
    event_metadata: list[dict[str, object]]

    def __init__(self) -> None:
        super().__init__(id="phone")
        self.event_metadata = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == phone_device.AnswerCallEvent.name:
            self.event_metadata.append(dict(event.metadata))
        return super().dispatch(ctx, event)


def basic_agent(*, devices: collections.abc.Sequence[Device] = ()) -> BasicAgent:
    return BasicAgent(devices=devices)


def configured_devices(*ids: str) -> tuple[Device, ...]:
    return tuple(Device(id=identity) for identity in ids)


async def start_bot_with_devices(
    active_bot: Bot, *, placement: space.Placement | None = None, clock: hsm.Clock | None = None
) -> Environment:
    environment = Environment(clock=clock)
    _ = await active_bot.attach(environment, placement=placement)
    await wait_until(lambda: active_bot.state() != "/Bot/activating")
    return environment


async def _processing_wait(timers: ManualTimers) -> "asyncio.Future[datetime.datetime]":
    """The newest unfired wait of a quick turn's processing timer, once its turn starts one."""

    deadline = asyncio.get_running_loop().time() + 30.0
    while asyncio.get_running_loop().time() < deadline:
        for duration, waiter in reversed(timers.pending):
            if duration == QUICK_PROCESSING_TIMEOUT and not waiter.done():
                return waiter
        await asyncio.sleep(0)
    raise TimeoutError("no processing timer wait became pending")


def _processing_wait_outcomes(timers: ManualTimers) -> list[str]:
    """How each quick processing-timer wait ended, oldest first: fired, cancelled or pending."""

    return [
        "pending" if not waiter.done() else "cancelled" if waiter.cancelled() else "fired"
        for duration, waiter in timers.pending
        if duration == QUICK_PROCESSING_TIMEOUT
    ]


def test_cognition_reboot_request_cycles_bot_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[hsm.Instance] = []
    hsm_started = hsm.started

    async def spy_started(
        ctx: hsm.Context | None,
        instance: hsm.Instance,
        model: hsm.Model,
        config: hsm.Config | None = None,
    ) -> hsm.Instance:
        if model.qualified_name == "/BotProcessingTimer":
            operations.append(instance)
        return await hsm_started(ctx, instance, model, config)

    monkeypatch.setattr(hsm, "started", spy_started)

    async def run() -> tuple[
        str, str, str, bool, list[mosfet.ProcessingFailedEventData], list[cognition.types.OutputData]
    ]:
        release = asyncio.Event()
        ability = BlockingSequenceAbility(
            release=release, block_on_call=1, outputs=(no_output("turn"), no_output("turn"))
        )
        cognition_ability = as_cognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognition_ability)
        environment = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/processing" and len(ability.calls) == 1)
        _ = await hsm.dispatch(
            environment,
            active_bot,
            dataclasses.replace(
                mosfet.RebootEvent.with_data(mosfet.RebootEventData(reason="cognition_child_teardown_failed")),
                id="reboot-turn",
                source=hsm.id(cognition_ability),
                target=hsm.id(active_bot),
            ),
        )
        reboot_state = active_bot.state()
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        timer_stopped = bool(operations) and all(
            operation.state() in {"", "/BotProcessingTimer"} for operation in operations
        )
        _ = release.set()
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=4)),
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
        cognition_ability = as_cognition(IgnoreAbility())
        active_bot = AbilityAgent(devices=(), cognition=cognition_ability)
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
                mosfet.RebootEvent.with_data(mosfet.RebootEventData(reason="cognition_detach_rollback_failed")),
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


def test_cognition_reboot_detach_timeout_resets_stuck_group_before_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    class FirstDetachHangsAbility(ProbeAbility):
        detach_calls: int

        def __init__(self) -> None:
            super().__init__()
            self.detach_calls = 0

        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
            if event.name == attachment.DetachEvent.name:
                self.detach_calls += 1
                if self.detach_calls == 1:

                    async def hang() -> bool:
                        return await asyncio.Event().wait()

                    return hang()
            return super().dispatch(ctx, event)

    class FastRebootAbilityAgent(AbilityAgent):
        _deactivation_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=10)

    async def run() -> tuple[str, str, int]:
        stuck = FirstDetachHangsAbility()
        cognition_ability = as_cognition(IgnoreAbility())
        active_bot = FastRebootAbilityAgent(
            devices=(),
            cognition=cognition_ability,
            acquired_abilities=(stuck,),
        )
        original_attach = attachment.Group.attach
        bot_attach_groups: list[attachment.Group] = []

        async def capture_bot_attach_group(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            if isinstance(event.data, attachment.AttachData) and event.data.actor is active_bot:
                bot_attach_groups.append(group)
            await original_attach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "attach", capture_bot_attach_group)
        environment = await start_bot_with_devices(active_bot)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        _ = await hsm.dispatch(
            environment,
            active_bot,
            dataclasses.replace(
                mosfet.RebootEvent.with_data(mosfet.RebootEventData(reason="cognition_child_teardown_failed")),
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
        return active_bot.state(), bot_attach_groups[-1].state(), stuck.detach_calls

    state, group_state, detach_calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert group_state == "/AttachmentGroup/attached"
    assert detach_calls == 1


def test_bot_attach_rejects_started_agent_from_another_environment() -> None:
    async def run() -> None:
        active_bot = basic_agent(devices=())
        first_environment = Environment()
        second_environment = Environment()

        _ = await active_bot.attach(first_environment)

        with pytest.raises(RuntimeError, match="Bot is already started in another environment"):
            _ = await active_bot.attach(second_environment)

    asyncio.run(run())


def test_bot_attach_with_otlp_endpoint_opens_no_event_injection_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The studio observes only: no OTLP-configured bot accepts injected events from it."""

    import grpc
    import importlib.util

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    opened: list[str] = []
    real_insecure_channel = grpc.insecure_channel

    def recording_insecure_channel(target: str, *args: object, **kwargs: object) -> grpc.Channel:
        del args, kwargs
        opened.append(target)
        return real_insecure_channel(target)

    monkeypatch.setattr(grpc, "insecure_channel", recording_insecure_channel)

    async def run() -> None:
        active_bot = basic_agent(devices=())
        environment = Environment()
        _ = await active_bot.attach(environment)
        await asyncio.sleep(0.2)
        _ = await active_bot.detach(environment)

    asyncio.run(run())

    assert opened == []
    assert importlib.util.find_spec("mosfet.telemetry.control") is None


def test_bot_and_nested_cognition_use_private_attachment_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[int, int, bool, str]:
        active_bot = basic_agent(devices=(Device(id="device"),))
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
        ) -> collections.abc.Awaitable[bool]:
            nonlocal attach_calls, group_is_private
            attach_calls += 1
            group_is_private = not environment.contains(group)
            return group_attach(group, ctx, event)

        def detach_group(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> collections.abc.Awaitable[bool]:
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

    # Memory is a composite ability too: its own (embedding) group is one of these.
    assert attach_calls == 5
    assert detach_calls == 5
    assert group_is_private
    assert state == "/Bot/inactive"


def test_bot_rejects_environment_started_lifecycle_ability_without_stopping_it() -> None:
    async def run() -> tuple[str, str, bool]:
        cognition_ability = as_cognition(IgnoreAbility())
        active_bot = AbilityAgent(devices=(), cognition=cognition_ability)
        environment = Environment()
        model = cognition_ability.model
        assert model is not None
        _ = await mosfet.started(environment, cognition_ability, model)

        _ = await active_bot.attach(environment)
        await asyncio.sleep(0.05)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return (
            active_bot.state(),
            cognition_ability.state(),
            environment.contains(cognition_ability),
        )

    state, ability_state, ability_stayed_in_environment = asyncio.run(run())

    assert state == "/Bot/inactive"
    assert ability_state != "/IgnoreAbilityLifecycle"
    assert ability_stayed_in_environment


def test_bot_deduplicates_repeated_lifecycle_ability_instance() -> None:
    async def run() -> str:
        shared = ProbeAbility()
        active_bot = AbilityAgent(
            devices=(),
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
        first_device = Device(id="first")
        failing_device = ImmediatelyFailingInitializingDevice(id="failing")
        active_bot = basic_agent(devices=(first_device, failing_device))
        environment = Environment()

        _ = await mosfet.started(environment, active_bot, active_bot.model)
        first_device_model = first_device.model
        assert first_device_model is not None
        _ = await mosfet.started(environment, first_device, first_device_model)
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
        active_bot = basic_agent(devices=(device,))
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
            devices=(device,),
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
        first_device = Device(id="first")
        failing_device = FailingInitializingDevice(release, id="failing")
        active_bot = basic_agent(devices=(first_device, failing_device))
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
        first_device = Device(id="first")
        failing_device = ImmediatelyFailingInitializingDevice(id="failing")
        active_bot = basic_agent(devices=(first_device, failing_device))
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
        first_device = LifecycleDispatchFailingDevice(id="first")
        failing_device = ImmediatelyFailingInitializingDevice(id="failing")
        active_bot = basic_agent(devices=(first_device, failing_device))
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
        first_device = Device(id="first")
        failing_device = Device(id="failing")
        active_bot = basic_agent(devices=(first_device, failing_device))
        environment = Environment()
        original_dispatch = hsm.Instance.dispatch

        def dispatch(instance: hsm.Instance, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
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
        first_device = Device(id="first")
        first_failing_device = FailingInitializingDevice(release, id="first_failing")
        second_failing_device = FailingInitializingDevice(release, id="second_failing")
        active_bot = basic_agent(devices=(first_device, first_failing_device, second_failing_device))
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
    await wait_until(lambda: (firmware := device_firmware(phone)) is not None and firmware.state() == "/Phone/ringing")


async def answer_phone(phone: phone_device.Phone, call_id: str = "call-123") -> None:
    await ring_phone(phone, call_id=call_id)
    await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
    await emit_phone_service_event(
        phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id=call_id))
    )
    await wait_until(
        lambda: (firmware := device_firmware(phone)) is not None
        and firmware.state() == "/Phone/answered/media_connecting"
    )


async def emit_phone_service_event(phone: phone_device.Phone, event: hsm.Event[typing.Any]) -> None:
    """What a carrier puts on the line: a typed service event, dispatched straight to firmware."""

    firmware = device_firmware(phone)
    assert isinstance(firmware, phone_device.Firmware)
    await hsm.dispatch(phone.context(), firmware, event)


async def answered_phone_in_environment() -> tuple[Environment, phone_device.Phone]:
    environment = Environment()
    phone = phone_device.Phone()
    phone_model = phone.model
    assert phone_model is not None
    _ = await mosfet.started(environment, phone, phone_model)
    await answer_phone(phone)
    return environment, phone


def probe_input() -> processing.InputData:
    return processing.InputData(input=mosfet.InputEventData(target_device="phone", priority=0))


def test_bot_uses_hsm_instance_identity() -> None:
    bot_instance = basic_agent()

    assert isinstance(bot_instance, hsm.Instance)
    assert not hasattr(bot_instance, "name")


def test_bot_snapshot_declares_owned_devices_by_their_ids() -> None:
    """DSL-30: the body keys each device by the id it was built with, bound to its live runtime id."""

    async def run() -> None:
        desk = Device(id="desk")
        radio = Device()
        active_bot = basic_agent(devices=(desk, radio))
        _ = await start_bot_with_devices(active_bot)
        snapshot = active_bot.take_snapshot()
        assert snapshot.Attributes is not None
        assert snapshot.Attributes["/Bot/owned_devices"] == {"desk": hsm.id(desk), "device": hsm.id(radio)}

    asyncio.run(run())


def test_a_named_bot_publishes_its_own_and_its_devices_topology_names() -> None:
    """DSL-25: a body given a name runs as ``/<name>`` and starts each device as ``/<name><Device>``."""

    async def run() -> tuple[str, str, list[str]]:
        desk = Device(id="desk")
        named = Bot((desk,), cognition=as_cognition(IgnoreAbility()), name="greeter")
        environment = Environment()
        _ = await named.attach(environment)
        await wait_until(lambda: named.state().startswith("/greeter/active"))
        try:
            return named.state(), desk.state(), list(named.take_snapshot().Attributes or {})
        finally:
            _ = await named.detach(environment)
            await wait_until(lambda: named.state() == "/greeter/inactive")
            await hsm.stop(named)

    state, device_state, attributes = asyncio.run(run())

    assert state.startswith("/greeter/active")
    assert device_state.startswith("/greeterDevice/")
    assert "/greeter/owned_devices" in attributes
    assert Bot.model.qualified_name == "/Bot"


def test_an_unnamed_bot_runs_the_body_model_itself() -> None:
    """DSL-25: without a name the body and its devices keep their own models' names."""

    async def run() -> tuple[str, str]:
        desk = Device(id="desk")
        active_bot = basic_agent(devices=(desk,))
        _ = await start_bot_with_devices(active_bot)
        return active_bot.state(), desk.state()

    state, device_state = asyncio.run(run())

    assert state.startswith("/Bot/active")
    assert device_state.startswith("/Device/")


def test_a_bots_observers_watch_its_own_model() -> None:
    """DSL-26: a body given observers runs them on its own model, and ``Bot.model`` itself keeps none."""

    seen: list[str] = []

    def watch(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        seen.append(typing.cast(dict[str, hsm.Event[typing.Any]], event.data)["event"].name)

    async def run() -> str:
        watched = Bot(
            (Device(id="desk"),),
            cognition=as_cognition(IgnoreAbility()),
            name="greeter",
            observe=(hsm.observe(mosfet.ActivateEvent, mosfet.DeactivateEvent, watch),),
        )
        environment = Environment()
        _ = await watched.attach(environment)
        await wait_until(lambda: watched.state().startswith("/greeter/active"))
        state = watched.state()
        _ = await watched.detach(environment)
        await wait_until(lambda: watched.state() == "/greeter/inactive")
        await hsm.stop(watched)
        return state

    state = asyncio.run(run())

    assert state.startswith("/greeter/active")
    assert seen == ["bot.activate", "bot.deactivate"]
    assert not [member for member in Bot.model.members.values() if member.qualified_name.endswith("/watch")]


def test_an_unnamed_bot_with_observers_keeps_the_body_models_name() -> None:
    """DSL-26: observers alone do not rename the body."""

    seen: list[str] = []

    def watch(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        seen.append("seen")

    async def run() -> str:
        watched = Bot((), cognition=as_cognition(IgnoreAbility()), observe=(hsm.observe(mosfet.ActivateEvent, watch),))
        environment = Environment()
        _ = await watched.attach(environment)
        await wait_until(lambda: watched.state().startswith("/Bot/active"))
        try:
            return watched.state()
        finally:
            await hsm.stop(watched)

    assert asyncio.run(run()).startswith("/Bot/active")
    assert seen == ["seen"]


def test_a_bot_observes_with_observers_only() -> None:
    """DSL-26: anything but an hsm observer would change what the body does, so it is refused."""

    with pytest.raises(TypeError, match="observe takes hsm observers"):
        _ = Bot(
            (),
            cognition=as_cognition(IgnoreAbility()),
            observe=typing.cast(collections.abc.Sequence[hsm.ObservationElement[typing.Any]], (hsm.state("extra"),)),
        )


@pytest.mark.parametrize("twice", ["one device twice", "two devices with one id"])
def test_bot_refuses_two_devices_with_one_id(twice: str) -> None:
    """DSL-30: one id names one device; the body never resolves a duplicate by order."""

    desk = Device(id="desk")
    other = desk if twice == "one device twice" else Device(id="desk")

    with pytest.raises(ValueError, match="'desk'"):
        _ = basic_agent(devices=(desk, other))


def test_bot_is_concrete_with_no_hard_coded_innate_abilities() -> None:
    """DSL-1: the body constructs as it is; a bot is defined by what it is given, not by a subclass."""

    instance = Bot((), cognition=as_cognition(IgnoreAbility()))

    assert not inspect.isabstract(Bot)
    assert type(instance) is Bot


def test_concrete_agent_can_declare_and_instantiate_innate_ability() -> None:
    bot_instance = FocusedAgent(devices=())

    assert isinstance(bot_instance.probe, ProbeAbility)


def test_bot_events_use_pydantic_schemas() -> None:
    activate_schema = object_dict(mosfet.ActivateEvent.schema)
    deactivate_schema = object_dict(mosfet.DeactivateEvent.schema)
    reboot_schema = object_dict(mosfet.RebootEvent.schema)
    input_schema = object_dict(mosfet.InputEvent.schema)
    completed_schema = object_dict(mosfet.ProcessingCompletedEvent.schema)
    failed_schema = object_dict(mosfet.ProcessingFailedEvent.schema)
    focus_device_schema = object_dict(mosfet.FocusDeviceEvent.schema)
    activating_done_schema = object_dict(mosfet.ActivatingDoneEvent.schema)
    activating_failed_schema = object_dict(mosfet.ActivatingFailedEvent.schema)
    deactivating_done_schema = object_dict(mosfet.DeactivatingDoneEvent.schema)

    assert mosfet.ActivateEvent.name == "bot.activate"
    assert activate_schema == mosfet.ActivateEventData.model_json_schema()
    assert activate_schema["description"]
    assert activate_schema["examples"] == [{}]
    assert mosfet.DeactivateEvent.name == "bot.deactivate"
    assert deactivate_schema == mosfet.DeactivateEventData.model_json_schema()
    assert deactivate_schema["description"]
    assert deactivate_schema["examples"] == [{}]
    assert mosfet.RebootEvent.name == "bot.reboot"
    assert reboot_schema == mosfet.RebootEventData.model_json_schema()
    assert reboot_schema["description"]
    assert reboot_schema["required"] == ["reason"]
    assert mosfet.InputEvent.name == "bot.input"
    assert input_schema == mosfet.InputEventData.model_json_schema()
    # A device does not know its bot-local name and need not rank itself, so an occasion can be
    # reported with neither; the body resolves the device from the envelope source.
    assert "required" not in input_schema
    input_properties = typing.cast(collections.abc.Mapping[str, object], input_schema["properties"])
    assert "observation" in input_properties
    # An occasion is ingress, never a model tool: a bot cannot select having a moment.
    assert mosfet.InputEvent.kind == hsm.EventKind
    assert mosfet.InputEvent.kind != event_contract.EventKind
    assert mosfet.FocusDeviceEvent.name == "bot.focus_device"
    assert focus_device_schema == mosfet.FocusDeviceEventData.model_json_schema()
    assert mosfet.ProcessingCompletedEvent.name == "bot.processing.completed"
    assert completed_schema == mosfet.ProcessingCompletedEventData.model_json_schema()
    assert completed_schema["description"]
    assert completed_schema["required"] == ["output", "focus_candidates"]
    assert mosfet.ProcessingFailedEvent.name == "bot.processing.failed"
    assert failed_schema == mosfet.ProcessingFailedEventData.model_json_schema()
    assert failed_schema["description"]
    assert mosfet.ActivatingDoneEvent.name == "bot.activated"
    assert activating_done_schema == mosfet.ActivatingDoneEventData.model_json_schema()
    assert mosfet.ActivatingFailedEvent.name == "bot.activating.failed"
    assert activating_failed_schema == mosfet.ActivatingFailedEventData.model_json_schema()
    assert mosfet.DeactivatingDoneEvent.name == "bot.deactivated"
    assert deactivating_done_schema == mosfet.DeactivatingDoneEventData.model_json_schema()
    assert deactivating_done_schema["description"]
    assert deactivating_done_schema["examples"] == [{}]
    degraded_schema = object_dict(mosfet.DegradedEvent.schema)
    assert mosfet.DegradedEvent.name == "bot.degraded"
    assert degraded_schema == mosfet.DegradedEventData.model_json_schema()
    assert degraded_schema["description"]
    assert degraded_schema["required"] == ["stage"]
    # Lifecycle requests carry their reply target at runtime only; it is never schema or wire data.
    assert "properties" not in activate_schema or "reply_to" not in object_dict(activate_schema["properties"])
    assert "properties" not in deactivate_schema or "reply_to" not in object_dict(deactivate_schema["properties"])


def test_bot_input_priority_is_bounded() -> None:
    assert mosfet.InputEventData(target_device="phone", priority=0).priority == 0
    assert mosfet.InputEventData(target_device="phone", priority=10).priority == 10

    with pytest.raises(ValueError):
        _ = mosfet.InputEventData(target_device="phone", priority=-1)
    with pytest.raises(ValueError):
        _ = mosfet.InputEventData(target_device="phone", priority=11)


def test_bot_input_can_carry_typed_observation() -> None:
    data = mosfet.InputEventData(
        target_device="phone",
        priority=0,
        observation=mosfet.StimulusData(event="phone.incoming_call", data=HappeningData(situation="call-123")),
    )

    assert data.observation == mosfet.StimulusData(
        event="phone.incoming_call", data=HappeningData(situation="call-123")
    )


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
    assert "/Bot/activation_cancelling" in model.members
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
    assert "attachment.attach.complete" in transitions["/Bot/activating"]
    assert "attachment.attach.failed" in transitions["/Bot/activating"]
    assert "bot.lifecycle.completed" not in transitions["/Bot/activating"]
    assert "bot.lifecycle.failed" not in transitions["/Bot/activating"]
    assert "bot.activating.failed" not in transitions["/Bot/activating"]
    assert "bot.lifecycle.cleanup.done" in transitions["/Bot/activation_cleanup"]
    assert "attachment.detached" in transitions["/Bot/deactivating"]
    assert "attachment.detach.failed" in transitions["/Bot/deactivating"]
    assert "bot.lifecycle.cleanup.done" in transitions["/Bot/deactivation_cleanup"]
    assert "attachment.detached" in transitions["/Bot/reboot_deactivating"]
    assert "attachment.detach.failed" in transitions["/Bot/reboot_deactivating"]
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
    # A stop the bot accepted is never turned back into an activation by a cognition reboot.
    assert "bot.reboot" not in transitions["/Bot/activation_cancelling"]
    assert "bot.reboot" not in transitions["/Bot/deactivating"]
    assert "bot.reboot" not in transitions["/Bot/deactivation_cleanup"]
    assert "bot.lifecycle.cleanup.done" in transitions["/Bot/activation_cancelling"]
    # Every lifecycle state answers, supersedes, or defers a deactivation request.
    for answering in ("/Bot/inactive", "/Bot/activating", "/Bot/degraded", "/Bot/active"):
        assert "bot.deactivate" in transitions[answering]
    for superseding in ("/Bot/activation_cancelling", "/Bot/deactivating", "/Bot/deactivation_cleanup"):
        assert "bot.deactivate" in transitions[superseding]
    for deferring in (
        "/Bot/activation_cleanup",
        "/Bot/reboot_deactivating",
        "/Bot/reboot_cleanup_preserve",
        "/Bot/reboot_cleanup_reset",
    ):
        assert "bot.deactivate" in deferred_map[deferring]
    # Each transition into degraded announces its own constant stage from its effect.
    for state, effect in (
        ("/Bot/activation_cleanup", "_dispatch_degraded_activation_cleanup"),
        ("/Bot/activation_cancelling", "_dispatch_degraded_activation_cancel"),
        ("/Bot/deactivation_cleanup", "_dispatch_degraded_deactivation_cleanup"),
        ("/Bot/reboot_cleanup_preserve", "_dispatch_degraded_reboot_cleanup"),
        ("/Bot/reboot_cleanup_reset", "_dispatch_degraded_reboot_cleanup"),
    ):
        timers = [
            transition
            for event_name, candidates in transitions[state].items()
            if "_bot_deactivation_timeout" in event_name
            for transition in candidates
        ]
        assert timers and all(any(effect in name for name in timer.effect) for timer in timers)
    cancel_timeout = transitions["/Bot/active/cancelling_processing"]["bot.processing.cancel.timed_out"]
    assert any("_dispatch_degraded_processing_cancel" in name for name in cancel_timeout[0].effect)
    assert "bot.input" in transitions["/Bot/active/unfocused"]
    assert "bot.ability.cognition.input" in transitions["/Bot/active/unfocused"]
    assert "*" not in transitions["/Bot/active/unfocused"]
    assert "bot.input" in transitions["/Bot/active/focused"]
    assert "bot.ability.cognition.input" in transitions["/Bot/active/focused"]
    assert "*" not in transitions["/Bot/active/focused"]
    # Something happening to the bot is the occasion, and it enters processing from either
    # attention state. Cognition decides what (if anything) it is for. Nothing wakes the body
    # on a clock: no timer transition survives on either attention state.
    for attention_state in ("/Bot/active/unfocused", "/Bot/active/focused"):
        occasion = transitions[attention_state]["bot.input"][0]
        assert occasion.target == "/Bot/active/processing"
        assert not any("interval" in event for event in transitions[attention_state])
    # An unoccupied body turns to look; a committed one does not get yanked. Both still get
    # the turn — an interrupt requests attention and only the body grants it.
    assert any(
        "_focus_event_target" in effect for effect in transitions["/Bot/active/unfocused"]["bot.input"][0].effect
    )
    assert not any(
        "_focus_event_target" in effect for effect in transitions["/Bot/active/focused"]["bot.input"][0].effect
    )

    # A happening survives a busy turn. Unlike a tick, which only ever asserted "it is now
    # later", a fact about the world is still true when the body finishes thinking, and
    # nothing re-announces a call that connected while the bot was blocked. The body holds it
    # itself, never in the runtime's defer list, and the occasion that wakes an idle body is
    # held the same way, so every turn takes what it perceives from one place.
    def behaviors(names: collections.abc.Iterable[str]) -> list[str]:
        # The boundary observer adds its own effect to every transition; only the body's count here.
        return [name.rpartition("/")[2] for name in names if "/observer/" not in name]

    for busy_state in ("/Bot/active/processing", "/Bot/active/cancelling_processing"):
        for name in ("bot.input", "bot.ability.cognition.input"):
            assert name not in deferred_map.get(busy_state, {})
            (hold,) = transitions[busy_state][name]
            assert not hold.target
            assert behaviors(hold.effect) == ["_hold_percept"]
    for attention_state in ("/Bot/active/unfocused", "/Bot/active/focused"):
        for name in ("bot.input", "bot.ability.cognition.input"):
            occasion = transitions[attention_state][name][0]
            assert behaviors(occasion.effect)[0] == "_hold_percept"
            assert occasion.target == "/Bot/active/processing"
    # Every turn terminal ends at one choice: something held starts the next turn at once (an
    # unoccupied body first turns to what arrived first), otherwise the body idles where its
    # attention is. The fallback is last and unguarded.
    for busy_state, name in (
        ("/Bot/active/processing", "bot.processing.completed"),
        ("/Bot/active/processing", "bot.processing.failed"),
        ("/Bot/active/cancelling_processing", "bot.ability.cognition.cancelled"),
        ("/Bot/active/cancelling_processing", "bot.ability.processing.cancelled"),
    ):
        assert [terminal.target for terminal in transitions[busy_state][name]] == ["/Bot/active/turn_ended"]
    branches = choice_transitions(model, "/Bot/active/turn_ended")
    assert [
        (
            (branch.guard or "").rpartition("/")[2],
            behaviors(branch.effect),
            branch.target,
        )
        for branch in branches
    ] == [
        ("_holds_percepts_with_focus", [], "/Bot/active/processing"),
        ("_holds_percepts", ["_focus_first_held"], "/Bot/active/processing"),
        ("_has_focused_device", [], "/Bot/active/focused"),
        ("", [], "/Bot/active/unfocused"),
    ]
    # Whatever is still held when the body leaves active is released and counted, not replayed.
    active_state = typing.cast(hsm.StateElement, model.members["/Bot/active"])
    assert behaviors(active_state.exit) == ["_release_held"]
    assert "bot.ability.cognition.output" in transitions["/Bot/active/processing"]
    assert "bot.ability.failed" in transitions["/Bot/active/processing"]
    # Focus is declared on active (transition_map may inherit it into child keys). Nothing clears
    # it: focus stays on a device until the bot focuses another (R-9).
    assert "bot.focus_device" in transitions["/Bot/active"]
    assert all("bot.clear_focus" not in state_transitions for state_transitions in transitions.values())
    assert "bot.processing.completed" in transitions["/Bot/active/processing"]
    assert "bot.processing.failed" in transitions["/Bot/active/processing"]
    assert "bot.processing.timed_out" in transitions["/Bot/active/processing"]
    assert "bot.ability.cognition.cancelled" in transitions["/Bot/active/cancelling_processing"]
    assert "bot.ability.processing.cancelled" in transitions["/Bot/active/cancelling_processing"]
    assert "bot.processing.cancel.timed_out" in transitions["/Bot/active/cancelling_processing"]


def test_unfocused_agent_focuses_target_device_before_processing_input() -> None:
    async def run() -> tuple[str, bool, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), ability.calls, active_bot.actions

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert len(calls) == 1
    assert calls[0].input == mosfet.InputEventData(target_device="phone", priority=3)
    assert actions == [no_output("priority:3")]


def test_unfocused_agent_rejects_input_from_unconfigured_target_device() -> None:
    async def run() -> tuple[str, bool, list[processing.InputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=3)),
        )
        await asyncio.sleep(0)

        return active_bot.state(), bot_has_focus(active_bot), ability.calls

    state, focused_device, calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert not focused_device
    assert calls == []


def test_unfocused_agent_does_not_focus_anonymous_device_type_name() -> None:
    async def run() -> tuple[str, bool, list[processing.InputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="Device", priority=3)),
        )
        await asyncio.sleep(0)

        return active_bot.state(), bot_has_focus(active_bot), ability.calls

    state, focused_device, calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert not focused_device
    assert calls == []


def test_bot_copies_configured_devices_at_construction() -> None:
    async def run() -> tuple[str, bool, list[processing.InputData]]:
        ability = IgnoreAbility()
        devices = [Device(id="phone")]
        active_bot = AbilityAgent(devices=devices, cognition=ability, input=(ring_hearing(),))
        devices.append(Device(id="browser"))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=3)),
        )
        await asyncio.sleep(0)

        return active_bot.state(), bot_has_focus(active_bot), ability.calls

    state, focused_device, calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert not focused_device
    assert calls == []


def test_bot_uses_configured_device_keys_for_processing_input() -> None:
    async def run() -> tuple[str, bool, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=3)),
        )
        await asyncio.sleep(0)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=2)),
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
            devices=(phone,),
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
        mosfet.FocusDeviceEvent.name,
        phone_device.AnswerCallEvent.name,
        phone_device.DeclineCallEvent.name,
        # A basic phone makes calls only: no texting or lock screen is on offer.
        cognition.types.IgnoreEvent.name,
    }
    assert (
        event_data_schema(operations[mosfet.FocusDeviceEvent.name]) == mosfet.FocusDeviceEventData.model_json_schema()
    )
    answer_operation = operations[phone_device.AnswerCallEvent.name]
    assert event_data_schema(answer_operation) == event_json_schema(phone_device.AnswerCallEvent)


def test_bot_snapshot_merges_focused_device_transitions() -> None:
    async def run() -> tuple[set[str], set[str]]:
        ability = IgnoreAbility()
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))

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
    # Focus is available while active and nothing clears it (R-9).
    # Device call tools appear only with focus/device state.
    assert mosfet.FocusDeviceEvent.name in before_focus
    assert "bot.clear_focus" not in before_focus
    assert mosfet.FocusDeviceEvent.name in after_focus
    assert "bot.clear_focus" not in after_focus
    assert phone_device.AnswerCallEvent.name in after_focus
    assert phone_device.DeclineCallEvent.name in after_focus


def test_bot_processing_operations_follow_focused_device_not_observed_device() -> None:
    async def run() -> list[processing.InputData]:
        ability = IgnoreAbility()
        phone = phone_device.Phone()
        browser_phone = phone_device.Phone(id="browser")
        active_bot = AbilityAgent(devices=(phone, browser_phone), cognition=ability, input=(ring_hearing(),))

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
    observed_sound = observed_browser_input.input.data
    # What a ringing phone carries out is who is calling, not which session is ringing.
    assert isinstance(observed_sound, phone_device.SoundData)
    assert observed_sound.caller == "Front desk"
    offered = {event.name for event in observed_browser_input.schemas}
    assert mosfet.FocusDeviceEvent.name in offered
    # Second turn while already focused: focus is offered, and nothing to clear it (R-9).
    assert "bot.clear_focus" not in offered
    assert phone_device.AnswerCallEvent.name in offered
    assert phone_device.DeclineCallEvent.name in offered


def test_bot_processes_environment_broadcast_from_configured_device_event() -> None:
    async def run() -> tuple[str, bool, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = IgnoreAbility()
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))

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


_VOICE_APPLY = voice.detection.ApplyData(
    segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.0, confidence=0.91),)
)
_SILENCE_APPLY = voice.detection.ApplyData(segments=())


class FixedVoiceActivityClassifier(voice.detection.VoiceActivityClassifier):
    """Always returns the same ApplyData (ring/no-voice fixtures)."""

    output: voice.detection.ApplyData

    def __init__(
        self,
        *,
        segments: tuple[voice.detection.VoiceDetectionSegment, ...] | None = None,
        is_voice: bool | None = None,
        confidence: float = 0.91,
        output: voice.detection.ApplyData | None = None,
    ) -> None:
        if output is not None:
            self.output = output
        elif segments is not None:
            self.output = voice.detection.ApplyData(segments=segments)
        elif is_voice is False:
            self.output = _SILENCE_APPLY
        else:
            self.output = voice.detection.ApplyData(
                segments=(
                    voice.detection.VoiceDetectionSegment(
                        start_seconds=0.0,
                        end_seconds=1.0,
                        confidence=confidence,
                    ),
                )
            )

    @typing.override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        return self.output


class ContentVoiceActivityClassifier(voice.detection.VoiceActivityClassifier):
    """Voice when audio is non-zero; silence when it is near-empty (closes HearingSpeech).

    Streaming VAD needs a silence frame after speech. Tests inject zero PCM via
    :func:`close_hearing_speech`; real utterance bytes stay voice so interrupts and
    residuals are not swallowed as End frames.
    """

    @typing.override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        if not input or not any(input):
            return _SILENCE_APPLY
        return _VOICE_APPLY


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
        # Voice path: content-based VAD so zero-PCM silence frames end HearingSpeech.
        voice_activity_classifier: voice.detection.VoiceActivityClassifier
        if is_voice:
            voice_activity_classifier = ContentVoiceActivityClassifier()
        else:
            voice_activity_classifier = FixedVoiceActivityClassifier(is_voice=False)
        super().__init__(
            voice_activity_classifier=voice_activity_classifier,
            sound_classifier=sound_classifier,
            speech_decoder=decoder,
        )
        self.handoffs = []
        self.failures = []
        self.received = []
        self.speech_decoder = decoder

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
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


def silence_sound(
    *,
    sample_rate_hz: int = 16_000,
    channels: int = 1,
    amplitude_db: float | None = None,
    received_level_db: float | None = None,
) -> SoundData:
    """A short no-voice frame used to close HearingSpeech after a voice frame."""

    return SoundData(
        audio=b"\x00" * 320,
        media_type="audio/pcm",
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        amplitude_db=amplitude_db,
        received_level_db=received_level_db,
    )


def wav_frame(audio_bytes: bytes, *, sample_rate_hz: int = 16_000, channels: int = 1) -> bytes:
    """Build the canonical PCM-in-WAV decoder input produced by Listening."""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate_hz)
        stream.writeframes(audio_bytes)
    return buffer.getvalue()


async def close_hearing_speech(
    bot: Bot,
    environment: Environment,
    *,
    sample_rate_hz: int = 16_000,
    channels: int = 1,
    amplitude_db: float | None = None,
    received_level_db: float | None = 0.0,
) -> None:
    """Dispatch a silence frame so streaming VAD can End an open HearingSpeech window.

    Default ``received_level_db=0.0`` keeps the silence frame below the product threshold so
    closing VAD does not invent a deliberative turn of its own.
    """

    await bot.dispatch(
        environment,
        SoundEvent.with_data(
            silence_sound(
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                amplitude_db=amplitude_db,
                received_level_db=received_level_db,
            )
        ),
    )


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
        active_bot = AbilityAgent(devices=(), cognition=ability, input=(listening_ability,))
        environment = await start_bot_with_devices(active_bot)
        sound = SoundEvent.with_data(
            SoundData(audio=b"heard-chunk", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)
        )
        await active_bot.dispatch(active_bot.context(), sound)
        # Streaming VAD: voice frame opens HearingSpeech; silence frame closes it for STT.
        await close_hearing_speech(active_bot, environment, sample_rate_hz=48_000)
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
    assert calls[0].input.name == listening.SpeechEvent.name
    assert isinstance(calls[0].input.data, listening.SpeechData)
    assert calls[0].input.data.content_type == "text/plain"
    assert calls[0].input.data.content == (b"decoded:heard-chunk" + silence_sound(sample_rate_hz=48_000).audio).decode(
        "utf-8", errors="replace"
    )
    assert decoder_calls == [b"heard-chunk" + silence_sound(sample_rate_hz=48_000).audio]
    assert (
        state.endswith("/active/focused") or state.endswith("/active/processing") or state.endswith("/active/unfocused")
    )


def test_bot_fans_out_visual_event_to_input_without_cognition() -> None:
    async def run() -> tuple[list[processing.InputData], str]:
        ability = IgnoreAbility()
        listening_ability = RecordingListening()
        active_bot = AbilityAgent(devices=(), cognition=ability, input=(listening_ability,))
        _ = await start_bot_with_devices(active_bot)
        frame = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"frame-bytes", media_type="image/png")
        visual = VisualEvent.with_data(VisualData(image=frame, kind="camera"))
        await active_bot.dispatch(active_bot.context(), visual)
        await asyncio.sleep(0.05)
        return ability.calls, active_bot.state() or ""

    calls, state = asyncio.run(run())

    # Listening has no visual transition; fan-out must not feed raw visual into cognition.
    assert calls == []
    assert state.endswith("/active/unfocused")


class _GlanceData(pydantic.BaseModel):
    """Glance around now; what you see arrives as your next observation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")


class _GestureData(pydantic.BaseModel):
    """Make a gesture."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")


_GlanceEvent = hsm.Event[_GlanceData](name="tests.bot.glance", kind=event_contract.EventKind, schema=_GlanceData)
_GestureEvent = hsm.Event[_GestureData](name="tests.bot.gesture", kind=event_contract.EventKind, schema=_GestureData)


def _accept_offered(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
    del ctx, instance, event


class Glancing(abilities.Ability[typing.Any, typing.Any]):
    """A test sense organ whose live topology enables one model-callable event."""

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Glancing",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle", hsm.transition(hsm.on(_GlanceEvent), hsm.effect(_accept_offered))),
    )


class Gesturing(abilities.Ability[typing.Any, typing.Any]):
    """A test output port whose live topology enables one model-callable event."""

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Gesturing",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle", hsm.transition(hsm.on(_GestureEvent), hsm.effect(_accept_offered))),
    )


def test_bot_offers_input_abilities_to_cognition_as_sense_organs() -> None:
    """An input ability is cognition-addressable; an output port never is."""

    async def run() -> tuple[list[processing.InputData], Glancing, Gesturing]:
        ability = IgnoreAbility()
        eye = Glancing()
        hand = Gesturing()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability, input=(eye,), output=(hand,))
        _ = await start_bot_with_devices(active_bot)
        _ = await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(ability.calls) == 1)
        return ability.calls, eye, hand

    calls, eye, hand = asyncio.run(run())

    offered = {event.name for event in calls[0].schemas}
    assert calls[0].actors["glancing"] is eye
    assert _GlanceEvent.name in offered
    assert calls[0].actor_events[_GlanceEvent.name] == ("glancing",)
    assert all(actor is not hand for actor in calls[0].actors.values())
    assert _GestureEvent.name not in offered


def test_bot_maps_an_ability_composed_as_acquired_and_input_once() -> None:
    """One actor has one key, so its events are offered from one target."""

    async def run() -> list[processing.InputData]:
        ability = IgnoreAbility()
        eye = Glancing()
        active_bot = AbilityAgent(
            devices=configured_devices("phone"), cognition=ability, input=(eye,), acquired_abilities=(eye,)
        )
        _ = await start_bot_with_devices(active_bot)
        _ = await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(ability.calls) == 1)
        return ability.calls

    calls = asyncio.run(run())

    assert [key for key, actor in calls[0].actors.items() if isinstance(actor, Glancing)] == ["glancing"]
    assert calls[0].actor_events[_GlanceEvent.name] == ("glancing",)


def test_bot_keys_abilities_by_their_ids() -> None:
    """DSL-32: an ability's actor key is the id it was built with; two of one kind are named apart."""

    async def run() -> list[processing.InputData]:
        ability = IgnoreAbility()
        left = Glancing(id="left_eye")
        right = Glancing(id="right_eye")
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability, input=(left, right))
        _ = await start_bot_with_devices(active_bot)
        _ = await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(ability.calls) == 1)
        return ability.calls

    calls = asyncio.run(run())

    assert {key for key, actor in calls[0].actors.items() if isinstance(actor, Glancing)} == {"left_eye", "right_eye"}
    assert set(calls[0].actor_events[_GlanceEvent.name]) == {"left_eye", "right_eye"}


@pytest.mark.parametrize("clash", ["two abilities of one kind", "an ability named like a device", "the body's key"])
def test_bot_refuses_two_actors_with_one_key(clash: str) -> None:
    """DSL-32: every actor cognition may address has its own key; the body never suffixes one."""

    devices: tuple[Device, ...] = ()
    if clash == "two abilities of one kind":
        given: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (Glancing(), Glancing())
    elif clash == "an ability named like a device":
        devices, given = (Device(id="glancing"),), (Glancing(),)
    else:
        given = (Glancing(id="bot"),)

    with pytest.raises(ValueError, match="'(glancing|bot)'"):
        _ = AbilityAgent(devices=devices, cognition=IgnoreAbility(), input=given)


class RestRecordingConsolidation(memory.Consolidation):
    """Consolidation that records the rest and wake notices Cognition sends it (test-only observation)."""

    notices: list[str]

    def __init__(self, *, memory: memory.Memory) -> None:
        super().__init__(memory=memory, processor=_NoopReasoningProcessor())
        self.notices = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name in (memory.RestEvent.name, memory.WakeEvent.name):
            self.notices.append(event.name.rsplit(".", 1)[-1])
        return super().dispatch(ctx, event)


def test_bot_stimuli_without_a_product_neither_wake_nor_hold_cognition_awake() -> None:
    """Only perception products wake cognition: a stimulus no input ability turns into one is not a turn."""

    async def run() -> tuple[list[str], int, list[processing.InputData]]:
        store = memory.Memory()
        consolidation = RestRecordingConsolidation(memory=store)
        ignore = IgnoreProcessor()
        ability = cognition.Cognition(
            intuition=cognition.Intuition(processor=ignore),
            reasoning=cognition.Reasoning(processor=_NoopReasoningProcessor()),
            reflection=_reflection_for_test(store),
            memory=store,
            consolidation=consolidation,
            rest_after=datetime.timedelta(milliseconds=50),
        )
        active_bot = AbilityAgent(devices=(), cognition=ability, input=())
        _ = await start_bot_with_devices(active_bot)
        frame = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"frame-bytes", media_type="image/png")
        sent = 0
        # A frame every 10 ms for up to a second: far more often than rest_after.
        while sent < 100 and "rest" not in consolidation.notices:
            await active_bot.dispatch(
                active_bot.context(), VisualEvent.with_data(VisualData(image=frame, kind="camera"))
            )
            sent += 1
            await asyncio.sleep(0.01)
        return list(consolidation.notices), sent, ignore.calls

    notices, sent, calls = asyncio.run(run())

    # It rested while the stimuli kept arriving, and none of them woke it.
    assert notices == ["rest"]
    assert sent < 100
    # The stimulus itself never reaches cognition.
    assert calls == []


def test_bot_records_environment_stimuli_in_stm_register() -> None:
    """Admitted environment stimuli land in the short-term register, grounded as observed.

    The body owns the register's lifecycle and records by message: a peer reading it back
    through ``RecentEvent`` finds the stimulus, so nothing called into the register directly.
    """

    async def run() -> memory.ObservedEvent | None:
        register = memory.StmMemory()
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=(), cognition=ability, input=(), stm_memory=register)
        _ = await start_bot_with_devices(active_bot)
        sound = SoundEvent.with_data(
            SoundData(audio=b"heard-chunk", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)
        )
        await active_bot.dispatch(active_bot.context(), sound)
        terminal = await abilities.run_terminal_operation(
            active_bot.context(),
            child=register,
            request=memory.stm.RecentEvent.with_data_and_id(
                memory.stm.RecentData(stimulus_name="environment.sound", limit=10), "read-register"
            ),
            terminals=(memory.stm.RecentOutputEvent, register.failed_event),
            timeout=datetime.timedelta(seconds=5),
        )
        assert isinstance(terminal.data, memory.stm.RecentOutputData), terminal.data
        return terminal.data.entries[0] if terminal.data.entries else None

    observed = asyncio.run(run())
    assert observed is not None
    assert observed.stimulus_name == "environment.sound"
    # Identity-like scalars are kept; media bytes are dropped by the projection.
    assert observed.payload["sample_rate_hz"] == 48_000
    assert observed.payload["channels"] == 1
    assert observed.payload["media_type"] == "audio/pcm"
    assert "audio" not in observed.payload


def test_bot_without_register_admits_stimuli_normally() -> None:
    """The register is optional: compositions without one behave exactly as before."""

    async def run() -> str:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=(), cognition=ability, input=())
        _ = await start_bot_with_devices(active_bot)
        sound = SoundEvent.with_data(
            SoundData(audio=b"heard-chunk", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)
        )
        await active_bot.dispatch(active_bot.context(), sound)
        return active_bot.state() or ""

    state = asyncio.run(run())
    # The stimulus was admitted through the fan-out without the register attached.
    assert state.startswith("/Bot/active"), state


def test_bot_does_not_send_speaker_environment_sound_to_cognition() -> None:
    async def run() -> tuple[
        str,
        bool,
        list[processing.InputData],
        list[cognition.types.OutputData],
    ]:
        ability = IgnoreAbility()
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))
        data = audio.OutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)

        environment = await start_bot_with_devices(active_bot)
        await ring_phone(phone)
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")
        ability.calls.clear()
        active_bot.actions.clear()
        # Signal into the speaker from its controller (the phone's firmware); the speaker is the
        # transducer that makes it environment sound.
        await phone_speaker(phone).dispatch(
            environment,
            dataclasses.replace(audio.OutputEvent.with_data(data), source=hsm.id(phone_firmware(phone))),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")

        return (
            active_bot.state(),
            bot_has_focus(active_bot),
            ability.calls,
            active_bot.actions,
        )

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    # Unlabeled no-voice without STT is a silence observation (sticky turn close), not deliberative speech.
    # ring_hearing has no speech_decoder, so playback still hands off SpeechData with empty VAD segments.
    assert len(calls) == 1
    stimulus = calls[0].input
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == listening.SpeechEvent.name
    assert isinstance(stimulus.data, listening.SpeechData)
    assert stimulus.data.content == b"playback-audio"
    assert stimulus.data.voice_detection.segments == ()
    # Silence observation is still a turn (sticky-turn close); IgnoreAbility yields empty output.
    assert actions == [()]


def test_same_environment_sibling_bots_do_not_send_speaker_sound_to_cognition() -> None:
    async def run() -> tuple[
        list[processing.InputData],
        list[processing.InputData],
        bool,
        bool,
    ]:
        owner_ability = IgnoreAbility()
        sibling_ability = IgnoreAbility()
        phone = phone_device.Phone()
        owning_agent = AbilityAgent(devices=(phone,), cognition=owner_ability, input=(ring_hearing(),))
        sibling_agent = AbilityAgent(devices=(), cognition=sibling_ability)
        data = audio.OutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)

        environment = await start_bot_with_devices(owning_agent)
        _ = await sibling_agent.attach(environment)
        await wait_until(lambda: sibling_agent.state() == "/Bot/active/unfocused")
        await ring_phone(phone)
        await wait_until(lambda: len(owner_ability.calls) == 1 and owning_agent.state() == "/Bot/active/focused")
        owner_ability.calls.clear()
        owning_agent.actions.clear()

        # Signal into the speaker from its controller (the phone's firmware); the speaker is the
        # transducer that makes it environment sound.
        await phone_speaker(phone).dispatch(
            environment,
            dataclasses.replace(audio.OutputEvent.with_data(data), source=hsm.id(phone_firmware(phone))),
        )
        await wait_until(lambda: len(owner_ability.calls) == 1 and owning_agent.state() == "/Bot/active/focused")

        return (
            owner_ability.calls,
            sibling_ability.calls,
            bot_has_focus(owning_agent),
            bot_has_focus(sibling_agent),
        )

    owner_calls, sibling_calls, owner_focus, sibling_focus = asyncio.run(run())

    # Sibling has no input Listening, so speaker sound cannot become cognition there.
    assert sibling_calls == []
    # Owner receives a silence observation (no STT on ring_hearing); not a deliberative speech product.
    assert len(owner_calls) == 1
    assert isinstance(owner_calls[0].input, hsm.Event)
    assert owner_calls[0].input.name == listening.SpeechEvent.name
    assert owner_focus
    assert not sibling_focus


def test_same_environment_sibling_agent_without_phone_config_ignores_phone_broadcast() -> None:
    async def run() -> tuple[
        list[processing.InputData],
        list[processing.InputData],
        bool,
        bool,
    ]:
        owner_ability = IgnoreAbility()
        sibling_ability = IgnoreAbility()
        phone = phone_device.Phone()
        owning_agent = AbilityAgent(devices=(phone,), cognition=owner_ability, input=(ring_hearing(),))
        sibling_agent = AbilityAgent(devices=(), cognition=sibling_ability)

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
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))
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
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))
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
    async def run() -> tuple[str, list[processing.InputData], list[mosfet.ProcessingFailedEventData]]:
        ability = IgnoreAbility()
        phone = SnapshotFailingPhone()
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        phone.fail_snapshots = True
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=2)),
        )
        # The bot perceives the failed step as one more turn (D-R-13); that turn fails the same way and is
        # not handed back again.
        await wait_until(lambda: len(active_bot.failures) == 2 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), ability.calls, active_bot.failures

    state, calls, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert calls == []
    assert [failure.message for failure in failures] == ["snapshot unavailable", "snapshot unavailable"]


def test_bot_processing_input_includes_primary_ability_affordance() -> None:
    async def run() -> tuple[list[processing.InputData], list[cognition.types.OutputData]]:
        ability = NestedPrimaryAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=2)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return ability.calls, active_bot.actions

    calls, actions = asyncio.run(run())

    assert len(calls) == 1
    assert actions == [no_output("nested affordance")]


def test_focused_agent_processes_other_target_without_automatic_focus_change() -> None:
    async def run() -> tuple[str, bool, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), ability.calls, active_bot.actions

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert calls[1].input == mosfet.InputEventData(target_device="browser", priority=2)
    assert actions == [no_output("priority:3"), no_output("priority:2")]


def test_focused_agent_dispatches_operation_output_to_target_device() -> None:
    async def run() -> tuple[str, str, bool, list[processing.InputData], list[cognition.types.OutputData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                target="phone",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            )
        )
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))

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


def test_focused_agent_stale_device_selection_drops_at_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale device selections dispatch; the device drops them per its own topology.

    Delivery validation resolves against declared events (HSM-CONTEXT-001: no probed peer
    state), so a selection whose target moved on is no longer rejected pre-dispatch. A phone
    answers only while it rings, so the stale answer lands on an idle handset that has no answer
    transition: the device drops it as unmatched and reports nothing. The turn completes, and
    the remote hang-up is observed as the next stimulus — the only one.
    """

    spans = carrier_spans(monkeypatch)

    async def run() -> tuple[
        str,
        str,
        bool,
        list[cognition.types.OutputData],
        list[mosfet.ProcessingFailedEventData],
        list[processing.InputData],
        list[str],
    ]:
        release = asyncio.Event()
        answer_selection = cognition.types.EventData(
            target="phone",
            event=phone_device.AnswerCallEvent.name,
            data={"call_id": "call-123"},
            reason="answer incoming call",
        )
        ability = BlockingSequenceAbility(release=release, block_on_call=1, outputs=((answer_selection,),))
        phone = phone_device.Phone()
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))

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
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused")
        await wait_until(lambda: bool(active_bot.actions))
        # Let the stale answer reach the handset and settle; nothing further may come of it.
        for _ in range(20):
            await asyncio.sleep(0)
        assert isinstance(firmware, phone_device.Firmware)

        return (
            active_bot.state(),
            firmware.state(),
            bot_has_focus(active_bot),
            active_bot.actions,
            active_bot.failures,
            list(ability.calls),
            published_names(spans),
        )

    state, phone_state, focused_device, actions, failures, calls, published = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert phone_state == "/Phone/hung_up"
    assert focused_device
    # The one thing it chose to do, dispatched even though the call had already gone.
    assert actions[0] == (
        cognition.types.EventData(
            target="phone",
            event=phone_device.AnswerCallEvent.name,
            data={"call_id": "call-123"},
            reason="answer incoming call",
        ),
    )
    # Then it finds out the far end hung up, which is how it learns the call it answered is gone.
    # Its own answer landed on an idle handset with nothing ringing, which has no answer to give:
    # the device dropped it and reported nothing, so no further turn follows.
    assert len(calls) == 2
    assert [
        turn.input.observation.event
        for turn in calls[1:]
        if isinstance(turn.input, mosfet.InputEventData) and turn.input.observation
    ] == [phone_device.HungUpEvent.name]
    assert phone_device.NoCallEvent.name not in published
    assert published[-1] == phone_device.HungUpEvent.name
    assert failures == []


def test_focused_agent_rejects_operation_output_that_does_not_match_event_schema() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData], list[mosfet.ProcessingFailedEventData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                event=phone_device.HangUpCallEvent.name,
                reason="hang up a call the phone is not on",
            ),
            cognition.types.EventData(
                event=phone_device.HangUpCallEvent.name,
                reason="hang up a call the phone is not on",
            ),
        )
        active_bot = AbilityAgent(devices=(phone_device.Phone(id="phone"),), cognition=_reasoning_on(ability.processor))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")
        # The bot perceives its failed step as one more turn (D-R-13), which selects nothing here.
        await wait_until(lambda: len(active_bot.actions) == 1 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.actions, active_bot.failures

    state, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert actions == [()]
    assert len(failures) == 1
    assert (
        failures[0].message
        == f"Selection rejected: Processing selected unavailable event: {phone_device.HangUpCallEvent.name}."
    )


def test_focused_agent_rejects_operation_event_not_offered_by_input() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData], list[mosfet.ProcessingFailedEventData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                target="phone",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            ),
            cognition.types.EventData(
                target="phone",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            ),
        )
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=_reasoning_on(ability.processor))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")
        # The bot perceives its failed step as one more turn (D-R-13), which selects nothing here.
        await wait_until(lambda: len(active_bot.actions) == 1 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.actions, active_bot.failures

    state, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert actions == [()]
    assert len(failures) == 1
    assert (
        failures[0].message
        == f"Selection rejected: Processing selected unavailable event for target phone: {phone_device.AnswerCallEvent.name}."
    )


def test_focused_agent_rejects_operation_target_not_offered_by_input() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData], list[mosfet.ProcessingFailedEventData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                target="browser",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            ),
            cognition.types.EventData(
                target="browser",
                event=phone_device.AnswerCallEvent.name,
                data={"call_id": "call-123"},
                reason="answer incoming call",
            ),
        )
        active_bot = AbilityAgent(
            devices=(phone_device.Phone(id="phone"), Device(id="browser")), cognition=_reasoning_on(ability.processor)
        )

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")
        # The bot perceives its failed step as one more turn (D-R-13), which selects nothing here.
        await wait_until(lambda: len(active_bot.actions) == 1 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.actions, active_bot.failures

    state, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert actions == [()]
    assert len(failures) == 1
    assert (
        failures[0].message
        == f"Selection rejected: Processing selected unavailable event for target browser: {phone_device.AnswerCallEvent.name}."
    )


def test_focused_agent_rejects_agent_local_operation_with_target() -> None:
    async def run() -> tuple[str, bool, list[cognition.types.OutputData], list[mosfet.ProcessingFailedEventData]]:
        ability = SequenceAbility(
            cognition.types.EventData(
                target="phone",
                event=mosfet.FocusDeviceEvent.name,
                data={"device": "browser"},
                reason="target mismatch",
            ),
            cognition.types.EventData(
                target="phone",
                event=mosfet.FocusDeviceEvent.name,
                data={"device": "browser"},
                reason="target mismatch",
            ),
        )
        active_bot = AbilityAgent(
            devices=configured_devices("phone", "browser"), cognition=_reasoning_on(ability.processor)
        )

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")
        # The bot perceives its failed step as one more turn (D-R-13), which selects nothing here.
        await wait_until(lambda: len(active_bot.actions) == 1 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), active_bot.actions, active_bot.failures

    state, focused_device, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert actions == [()]
    assert len(failures) == 1
    assert (
        failures[0].message
        == "Selection rejected: Processing selected focus_device outside available device candidates."
    )


def test_focused_agent_dispatches_operation_with_ref_backed_event_data_schema() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData], list[mosfet.ProcessingFailedEventData]]:
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
            lambda: (firmware := device_firmware(phone)) is not None
            and firmware.state() == "/Phone/answered/media_ready"
        )
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))

        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.actions) and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.actions, active_bot.failures

    state, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert failures == []
    # The selection it made. Later turns are the phone telling it how the transfer went, which
    # it is free to say nothing about — so this pins the action, never the number of turns.
    assert actions[0] == (
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


def test_focused_agent_rejects_operation_data_that_does_not_match_event_schema() -> None:
    async def run() -> tuple[str, list[cognition.types.OutputData], list[mosfet.ProcessingFailedEventData]]:
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
            ),
            cognition.types.EventData(
                target="phone",
                event=phone_device.TransferCallEvent.name,
                data={
                    "call_id": "call-123",
                    "transfer_id": "transfer-123",
                    "target": {"kind": "queue", "value": "operator"},
                },
                reason="transfer call",
            ),
        )
        environment, phone = await answered_phone_in_environment()
        await emit_phone_service_event(
            phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123"))
        )
        await wait_until(
            lambda: (firmware := device_firmware(phone)) is not None
            and firmware.state() == "/Phone/answered/media_ready"
        )
        active_bot = AbilityAgent(devices=(phone,), cognition=_reasoning_on(ability.processor), input=(ring_hearing(),))

        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: bool(active_bot.failures) and active_bot.state() == "/Bot/active/focused")
        # The bot perceives its failed step as one more turn (D-R-13), which selects nothing here.
        await wait_until(lambda: len(active_bot.actions) == 1 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.actions, active_bot.failures

    state, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert actions == [()]
    assert len(failures) == 1
    assert failures[0].message.startswith(
        "Selection rejected: Processing selected invalid event data for event: phone.transfer_call: phone.transfer_call|"
    )
    assert "input_value" not in failures[0].message


def test_bot_receives_one_redacted_terminal_for_bounded_alternating_selection_rejections() -> None:
    async def run() -> tuple[int, list[mosfet.ProcessingFailedEventData], list[cognition.types.OutputData]]:
        processor = CyclingSelectionProcessor(
            (cognition.types.EventData(event=mosfet.FocusDeviceEvent.name, data={"device": 0}),),
            (cognition.types.EventData(event=mosfet.FocusDeviceEvent.name, data={}),),
            (cognition.types.EventData(event=mosfet.FocusDeviceEvent.name, data={"device": 1}),),
        )
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=_reasoning_on(processor))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        # The bot perceives the failed turn as one more turn (D-R-13), whose selections are rejected the same
        # way; that failure is not handed back again.
        await wait_until(lambda: len(active_bot.failures) == 2 and active_bot.state() == "/Bot/active/focused")

        return len(processor.calls), active_bot.failures, active_bot.actions

    calls, failures, actions = asyncio.run(run())

    assert calls == 6
    assert len(failures) == 2
    assert all("input_value" not in failure.message for failure in failures)
    assert actions == []


def test_bot_receives_one_redacted_terminal_for_same_selection_rejection() -> None:
    async def run() -> tuple[int, list[mosfet.ProcessingFailedEventData]]:
        processor = CyclingSelectionProcessor(
            (cognition.types.EventData(event=mosfet.FocusDeviceEvent.name, data={"device": 0}),),
            (cognition.types.EventData(event=mosfet.FocusDeviceEvent.name, data={"device": 1}),),
        )
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=_reasoning_on(processor))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        # The bot perceives the failed turn as one more turn (D-R-13), whose selections are rejected the same
        # way; that failure is not handed back again.
        await wait_until(lambda: len(active_bot.failures) == 2 and active_bot.state() == "/Bot/active/focused")

        return len(processor.calls), active_bot.failures

    calls, failures = asyncio.run(run())

    assert calls == 4
    assert len(failures) == 2
    assert all("input_value" not in failure.message for failure in failures)


def test_focused_agent_dispatches_multi_event_focus_selection() -> None:
    async def run() -> tuple[str, bool, list[cognition.types.OutputData]]:
        ability = SequenceAbility(focus_output("phone", "multi focus"))
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
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
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        await ring_phone(phone)
        await wait_until(lambda: bot_has_focus(active_bot) and active_bot.state() == "/Bot/active/focused")
        firmware = device_firmware(phone)
        assert firmware is not None

        return active_bot.state(), firmware.state(), active_bot.actions

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
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=1)),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 3 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), cognitive.inputs, ability.calls, active_bot.actions

    state, inputs, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert inputs[2].focus == "browser"
    assert len(calls) == 3
    assert calls[1].input == mosfet.InputEventData(target_device="browser", priority=1)
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
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="phone", priority=3),
                "stale-focus-seed",
            ),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="browser", priority=1),
                "stale-focus-live",
            ),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/processing")

        # Correct cognition source/target, wrong turn id — must not move focus.
        stale_focus = dataclasses.replace(
            mosfet.FocusDeviceEvent.with_data(mosfet.FocusDeviceEventData(device="screen")),
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
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=2)),
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
        active_bot = basic_agent(devices=())
        _ = await start_bot_with_devices(active_bot)
        forged = dataclasses.replace(
            mosfet.FocusDeviceEvent.with_data(mosfet.FocusDeviceEventData(device="ghost")),
            metadata={"bot.focus_candidates": ("ghost",)},
        )

        await active_bot.dispatch(active_bot.context(), forged)
        await asyncio.sleep(0)

        return active_bot.state()

    assert asyncio.run(run()) == "/Bot/active/unfocused"


def test_bot_rejects_forged_focus_action_source_during_current_processing() -> None:
    async def run() -> str | None:
        release = asyncio.Event()
        processor = BlockingSequenceProcessor(
            release=release,
            block_on_call=2,
            outputs=(no_output("initial"), no_output("current"), no_output("observe retained focus")),
        )
        cognitive = CapturingCognition(processor)
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=cognitive)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="phone", priority=1),
                "initial-focus",
            ),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="browser", priority=2),
                "current-focus-source",
            ),
        )
        await wait_until(lambda: len(cognitive.input_events) == 2)
        request = cognitive.input_events[1]
        # The browser is a candidate of this turn, so only the forged source refuses the focus.
        forged = dataclasses.replace(
            mosfet.FocusDeviceEvent.with_data(mosfet.FocusDeviceEventData(device="browser")),
            id=f"{request.id}:intuition",
            source="forged-source",
            target=hsm.id(active_bot),
            metadata=dict(request.metadata),
        )
        await active_bot.dispatch(active_bot.context(), forged)
        await asyncio.sleep(0)
        release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(cognitive.input_events) == 3)
        observed = cognitive.input_events[2].data
        assert isinstance(observed, cognition.InputData)
        return observed.focus

    assert asyncio.run(run()) == "phone"


def test_focused_agent_rejects_focus_device_outside_processing_candidates() -> None:
    async def run() -> tuple[
        str,
        list[cognition.InputData],
        list[cognition.types.OutputData],
        list[mosfet.ProcessingFailedEventData],
    ]:
        ability = SequenceAbility(
            no_output("stay on phone"),
            focus_output("screen", "bad target"),
            focus_output("screen", "bad target"),
            no_output("observe retained focus"),
        )
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser", "screen"), cognition=cognitive)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=1)),
        )
        # Intuition's rejected selections fail it, so reasoning takes the same turn (D-R-36) and selects nothing.
        await wait_until(lambda: len(active_bot.actions) == 2 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 4 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), cognitive.inputs, active_bot.actions, active_bot.failures

    state, inputs, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    # The rejected focus never landed: the next turn still has the phone in focus.
    assert len(inputs) == 3
    assert inputs[2].focus == "phone"
    assert [tuple(action) for action in actions] == [
        no_output("stay on phone"),
        (),
        no_output("observe retained focus"),
    ]
    assert failures == []


def test_focus_stays_on_a_device_until_the_bot_focuses_another() -> None:
    async def run() -> tuple[list[str], list[str | None], list[processing.InputData]]:
        ability = SequenceAbility(
            no_output("look at the phone"),
            focus_output("browser", "turn to the browser"),
            no_output("still on the browser"),
            no_output("still on the browser later"),
        )
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=cognitive)

        _ = await start_bot_with_devices(active_bot)
        states: list[str] = []
        # The browser speaks up once, while the phone holds focus; then only the phone does.
        for turn, device in enumerate(("phone", "browser", "phone", "phone"), start=1):
            await active_bot.dispatch(
                active_bot.context(),
                mosfet.InputEvent.with_data(mosfet.InputEventData(target_device=device, priority=turn)),
            )
            await wait_until(
                lambda turn=turn: len(ability.calls) == turn and active_bot.state() == "/Bot/active/focused"
            )
            states.append(active_bot.state())

        return states, [turn.focus for turn in cognitive.inputs], ability.calls

    states, focuses, calls = asyncio.run(run())

    # Only the start is unfocused: the first turn focuses the phone, the bot's own choice moves
    # focus to the browser, and it stays there turn after turn; nothing offers to clear it (R-9).
    assert states == ["/Bot/active/focused"] * 4
    assert focuses == ["phone", "phone", "browser", "browser"]
    assert all("bot.clear_focus" not in {event.name for event in call.schemas} for call in calls)
    assert all(mosfet.FocusDeviceEvent.name in {event.name for event in call.schemas} for call in calls)


def test_mosfet_exports_no_clear_focus() -> None:
    assert not hasattr(mosfet, "ClearFocusEvent")
    assert not hasattr(mosfet, "ClearFocusEventData")
    assert "ClearFocusEvent" not in mosfet.__all__
    assert "ClearFocusEventData" not in mosfet.__all__


def test_bot_processing_state_holds_repeated_input_until_processing_completes() -> None:
    async def run() -> tuple[str, list[int], list[cognition.types.OutputData]]:
        release = asyncio.Event()
        ability = BlockingAbility(release=release)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=1)),
        )
        await wait_until(lambda: ability.calls == [1])

        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=2)),
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
    async def run() -> tuple[str, bool, list[cognition.types.OutputData]]:
        release = asyncio.Event()
        ability = BlockingAbility(release=release)
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="phone", priority=3),
                "processing-focus-change",
            ),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/processing")

        focus = dataclasses.replace(
            mosfet.FocusDeviceEvent.with_data(mosfet.FocusDeviceEventData(device="browser")),
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


def test_bot_processing_state_holds_observed_phone_event_until_processing_completes() -> None:
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
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=1)),
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
    assert calls[0].input == mosfet.InputEventData(target_device="phone", priority=1)
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
        active_bot = AbilityAgent(devices=(phone,), cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=1)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/processing")

        await active_bot.dispatch(
            active_bot.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number="5550142")),
        )
        assert len(ability.calls) == 1
        _ = release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), ability.calls, active_bot.actions

    state, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert len(calls) == 1
    assert calls[0].input == mosfet.InputEventData(target_device="phone", priority=1)
    assert actions == [no_output("initial")]


def test_held_input_is_perceived_after_focus_device_completion() -> None:
    async def run() -> tuple[str, bool, list[processing.InputData], list[cognition.types.OutputData]]:
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
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=1)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")

        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/processing")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        assert len(ability.calls) == 2
        _ = release.set()
        await wait_until(lambda: len(ability.calls) == 3 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), bot_has_focus(active_bot), ability.calls, active_bot.actions

    state, focused_device, calls, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert focused_device
    assert calls[2].input == mosfet.InputEventData(target_device="phone", priority=3)
    assert actions == [
        no_output("initial focus"),
        focus_output("browser", "change focus"),
        no_output("after focus change"),
    ]


def test_a_line_typed_while_the_bot_is_busy_is_perceived_on_the_next_turn() -> None:
    """A line typed during a turn waits one turn, not one turn per stimulus queued ahead of it.

    Three handed-back step failures queue behind the turn in progress, then the person types a
    line. When the turn ends, the next turn perceives all four together in arrival order, as
    one ``bot.perceived``, and no further turn is taken for any of them.
    """

    async def run() -> tuple[str, list[processing.InputData]]:
        release = asyncio.Event()
        ability = BlockingSequenceAbility(release=release, block_on_call=1, outputs=())
        active_bot = AbilityAgent(devices=configured_devices("desk"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data_and_id(mosfet.InputEventData(target_device="desk"), "turn-1"),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/processing")

        for index in range(3):
            failure = cognition.StepFailedData(
                message="Cognition's reasoning step failed, so the turn failed: provider unavailable",
                stage="reasoning",
                turn_id=f"failed-turn-{index}",
                stimulus=mosfet.InputEvent.name,
            )
            await active_bot.dispatch(
                active_bot.context(),
                dataclasses.replace(
                    cognition.InputEvent.with_data_and_id(
                        cognition.InputData(
                            stimulus=dataclasses.replace(
                                cognition.StepFailedEvent.with_data(failure), source="cognition"
                            )
                        ),
                        f"handed-back-{index}",
                    ),
                    source="cognition",
                ),
            )
        person = mosfet.Participant(ref="person-42", role=mosfet.ParticipantRole.SOURCE)
        line = computer.TextEnteredData(
            device_id="desk", person_id="person-42", participants=(person,), text="Are you there?"
        )
        await active_bot.dispatch(
            active_bot.context(),
            dataclasses.replace(
                mosfet.InputEvent.with_data_and_id(
                    mosfet.InputEventData(
                        target_device="desk",
                        observation=mosfet.StimulusData.from_event(computer.TextEnteredEvent.with_data(line)),
                        participants=(person,),
                    ),
                    "typed-line",
                ),
                source="desk",
            ),
        )
        assert len(ability.calls) == 1
        _ = release.set()
        await wait_until(lambda: len(ability.calls) >= 2 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state() or "", ability.calls

    state, calls = asyncio.run(run())

    assert state == "/Bot/active/focused"
    perceived = calls[1].input
    assert isinstance(perceived, hsm.Event)
    assert perceived.name == "bot.perceived"
    percepts = object_dict(object_dict(event_contract.event_json_value(perceived))["data"])["percepts"]
    assert isinstance(percepts, list)
    stimuli = [object_dict(object_dict(percept)["stimulus"]) for percept in typing.cast(list[object], percepts)]
    assert [stimulus.get("name") for stimulus in stimuli[:3]] == [cognition.StepFailedEvent.name] * 3
    assert [object_dict(stimulus["data"])["turn_id"] for stimulus in stimuli[:3]] == [
        "failed-turn-0",
        "failed-turn-1",
        "failed-turn-2",
    ]
    typed = object_dict(stimuli[3]["observation"])
    assert typed["event"] == computer.TextEnteredEvent.name
    assert object_dict(typed["data"])["text"] == "Are you there?"
    assert object_dict(typing.cast(list[object], percepts)[3])["participants"] == [
        {"ref": "person-42", "role": "source", "own": False}
    ]
    assert len(calls) == 2


# A busy body holds what reaches it (wave 12): the next turn takes the longest arrival-order prefix of
# what is held within 8 percepts, 64 KiB and 32 participants, and always at least one.


class GatedProcessor(processing.DecisionProcessor):
    """Records what each call perceived; call ``n`` waits at ``gates[n]`` when it has one. Chooses nothing."""

    calls: list[processing.InputData]
    gates: dict[int, asyncio.Event]

    def __init__(self, gates: dict[int, asyncio.Event] | None = None) -> None:
        self.calls = []
        self.gates = dict(gates or {})

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        gate = self.gates.get(len(self.calls))
        if gate is not None:
            _ = await gate.wait()
        return ()


class CancelGatedCognition(CapturingCognition):
    """Cognition whose cancel request waits at a gate, so the body stays cancelling while things arrive."""

    gate: asyncio.Event
    pending: set[asyncio.Future[bool]]

    def __init__(self, processor: processing.DecisionProcessor, *, gate: asyncio.Event) -> None:
        super().__init__(processor)
        self.gate = gate
        self.pending = set()

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name != cognition.CancelEvent.name:
            return super().dispatch(ctx, event)
        deliver = super().dispatch

        async def later() -> bool:
            _ = await self.gate.wait()
            return await deliver(ctx, event)

        task = asyncio.ensure_future(later())
        self.pending.add(task)
        return task


class HoldingTimeoutAgent(AbilityAgent):
    # Long enough for a recovered turn to finish; the cancellation bound is then 100 ms too.
    _processing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=100)


def happening_input(
    device: str,
    situation: str,
    *,
    participants: tuple[mosfet.Participant, ...] = (),
) -> mosfet.InputEventData:
    return mosfet.InputEventData(
        target_device=device,
        observation=mosfet.StimulusData.from_event(HappeningEvent.with_data(HappeningData(situation=situation))),
        participants=participants,
    )


async def arrive(active_bot: Bot, data: mosfet.InputEventData, arrival_id: str) -> None:
    _ = await active_bot.dispatch(
        active_bot.context(),
        dataclasses.replace(mosfet.InputEvent.with_data_and_id(data, arrival_id), source=data.target_device or ""),
    )


def held_percepts(active_bot: Bot) -> object:
    """The body's declared held count, as its snapshot shows it."""

    attributes = typing.cast(collections.abc.Mapping[str, object], active_bot.take_snapshot().Attributes or {})
    (count,) = [value for key, value in attributes.items() if key.endswith("/held_percepts")]
    return count


def perceived_stimuli(given: object) -> list[mosfet.InputData]:
    assert isinstance(given, hsm.Event)
    batch = typing.cast(hsm.Event[typing.Any], given).data
    assert isinstance(batch, mosfet.PerceivedData)
    return [percept.stimulus for percept in batch.percepts]


def test_a_single_held_stimulus_is_handed_to_cognition_exactly_as_it_arrived() -> None:
    """One held item is its own perception: the turn id is its id and cognition gets it unchanged."""

    ada = mosfet.Participant(ref="ada", role=mosfet.ParticipantRole.SOURCE)
    queued = happening_input("phone", "rang", participants=(ada,))

    async def run() -> tuple[list[hsm.Event[typing.Any]], object, str]:
        gate = asyncio.Event()
        cognitive = CapturingCognition(GatedProcessor({1: gate}))
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognitive)
        _ = await start_bot_with_devices(active_bot)
        await arrive(active_bot, happening_input("phone", "first"), "turn-1")
        await wait_until(lambda: len(cognitive.input_events) == 1)
        await arrive(active_bot, queued, "queued-1")
        _ = gate.set()
        await wait_until(lambda: len(cognitive.input_events) == 2 and active_bot.state() == "/Bot/active/focused")
        return cognitive.input_events, held_percepts(active_bot), active_bot.state() or ""

    handed, held, state = asyncio.run(run())

    assert handed[1].id == "queued-1"
    data = handed[1].data
    assert isinstance(data, cognition.InputData)
    assert data.stimulus == queued
    assert data.participants == (ada,)
    assert held == 0
    assert state == "/Bot/active/focused"


def test_a_backlog_longer_than_a_turn_takes_splits_into_consecutive_turns_in_order() -> None:
    queued = [happening_input("phone", f"happening-{index}") for index in range(10)]

    async def run() -> tuple[list[processing.InputData], object, object]:
        gate = asyncio.Event()
        processor = GatedProcessor({1: gate})
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=as_cognition(processor))
        _ = await start_bot_with_devices(active_bot)
        await arrive(active_bot, happening_input("phone", "busy"), "turn-1")
        await wait_until(lambda: len(processor.calls) == 1)
        for index, data in enumerate(queued):
            await arrive(active_bot, data, f"queued-{index}")
        held_while_busy = held_percepts(active_bot)
        _ = gate.set()
        await wait_until(lambda: len(processor.calls) == 3 and active_bot.state() == "/Bot/active/focused")
        await settle()
        return processor.calls, held_while_busy, held_percepts(active_bot)

    calls, held_while_busy, held_after = asyncio.run(run())

    assert held_while_busy == 10
    assert perceived_stimuli(calls[1].input) == queued[:8]
    assert perceived_stimuli(calls[2].input) == queued[8:]
    assert len(calls) == 3
    assert held_after == 0


def test_a_turn_takes_held_stimuli_up_to_its_byte_bound_and_a_lone_oversized_one_alone() -> None:
    small = happening_input("phone", "small")
    over_the_rest = happening_input("phone", "x" * (70 * 1024))
    oversized = happening_input("phone", "y" * (300 * 1024))
    after = happening_input("phone", "after")

    async def run() -> list[processing.InputData]:
        gate = asyncio.Event()
        processor = GatedProcessor({1: gate})
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=as_cognition(processor))
        _ = await start_bot_with_devices(active_bot)
        await arrive(active_bot, happening_input("phone", "busy"), "turn-1")
        await wait_until(lambda: len(processor.calls) == 1)
        for index, data in enumerate((small, over_the_rest, oversized, after)):
            await arrive(active_bot, data, f"queued-{index}")
        _ = gate.set()
        await wait_until(lambda: len(processor.calls) == 5 and active_bot.state() == "/Bot/active/focused")
        await settle()
        return processor.calls

    calls = asyncio.run(run())

    assert [call.input for call in calls[1:]] == [small, over_the_rest, oversized, after]


def test_a_turn_takes_held_stimuli_while_their_participants_fit_and_the_next_one_waits() -> None:
    first = tuple(mosfet.Participant(ref=f"person-{index}", role=mosfet.ParticipantRole.SOURCE) for index in range(20))
    others = tuple(
        mosfet.Participant(ref=f"person-{index}", role=mosfet.ParticipantRole.SOURCE) for index in range(20, 40)
    )
    a = happening_input("phone", "a", participants=first)
    again = happening_input("phone", "again", participants=first)
    b = happening_input("phone", "b", participants=others)

    async def run() -> list[cognition.InputData]:
        gate = asyncio.Event()
        ability = processing.Processing(processor=GatedProcessor({1: gate}))
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognitive)
        _ = await start_bot_with_devices(active_bot)
        await arrive(active_bot, happening_input("phone", "busy"), "turn-1")
        await wait_until(lambda: len(cognitive.inputs) == 1)
        for index, data in enumerate((a, again, b)):
            await arrive(active_bot, data, f"queued-{index}")
        _ = gate.set()
        await wait_until(lambda: len(cognitive.inputs) == 3 and active_bot.state() == "/Bot/active/focused")
        await settle()
        return cognitive.inputs

    inputs = asyncio.run(run())

    assert perceived_stimuli(inputs[1].stimulus) == [a, again]
    assert inputs[1].participants == first
    assert inputs[2].stimulus == b
    assert inputs[2].participants == others
    assert len(inputs) == 3


def test_what_arrives_during_a_turn_is_perceived_by_the_next_with_everything_else_held() -> None:
    a, b, c, d = (happening_input("phone", situation) for situation in "abcd")

    async def run() -> list[processing.InputData]:
        first_turn, second_turn = asyncio.Event(), asyncio.Event()
        processor = GatedProcessor({1: first_turn, 2: second_turn})
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=as_cognition(processor))
        _ = await start_bot_with_devices(active_bot)
        await arrive(active_bot, happening_input("phone", "busy"), "turn-1")
        await wait_until(lambda: len(processor.calls) == 1)
        await arrive(active_bot, a, "a")
        await arrive(active_bot, b, "b")
        _ = first_turn.set()
        await wait_until(lambda: len(processor.calls) == 2)
        await arrive(active_bot, c, "c")
        await arrive(active_bot, d, "d")
        _ = second_turn.set()
        await wait_until(lambda: len(processor.calls) == 3 and active_bot.state() == "/Bot/active/focused")
        await settle()
        return processor.calls

    calls = asyncio.run(run())

    assert perceived_stimuli(calls[1].input) == [a, b]
    assert perceived_stimuli(calls[2].input) == [c, d]
    assert len(calls) == 3


def test_a_cancelling_body_holds_what_arrives_and_the_next_turn_perceives_it() -> None:
    a, b = happening_input("phone", "a"), happening_input("phone", "b")

    async def run() -> tuple[list[processing.InputData], list[mosfet.ProcessingFailedEventData], str]:
        cancel_gate = asyncio.Event()
        processor = GatedProcessor({1: asyncio.Event()})
        active_bot = HoldingTimeoutAgent(
            devices=configured_devices("phone"), cognition=CancelGatedCognition(processor, gate=cancel_gate)
        )
        _ = await start_bot_with_devices(active_bot)
        await arrive(active_bot, happening_input("phone", "hangs"), "turn-1")
        await wait_until(lambda: active_bot.state() == "/Bot/active/cancelling_processing")
        await arrive(active_bot, a, "a")
        await arrive(active_bot, b, "b")
        _ = cancel_gate.set()
        await wait_until(lambda: len(processor.calls) == 2 and active_bot.state() == "/Bot/active/focused")
        return processor.calls, active_bot.failures, active_bot.state() or ""

    calls, failures, state = asyncio.run(run())

    assert perceived_stimuli(calls[1].input) == [a, b]
    assert [failure.message for failure in failures] == ["Bot processing timed out after 0.1 seconds."]
    assert state == "/Bot/active/focused"


def _released_count(metrics: metric_reader.Reader, reason: str) -> float:
    return sum(
        point.value
        for point in metrics.points("bot.turn.percept.released.count")
        if point.attributes == {"reason": reason}
    )


def test_a_body_that_degrades_while_holding_counts_what_it_released(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def run() -> tuple[str, object, int]:
        processor = GatedProcessor({1: asyncio.Event()})
        active_bot = HoldingTimeoutAgent(
            devices=configured_devices("phone"), cognition=CapturingCognition(processor, swallow_cancel=True)
        )
        _ = await start_bot_with_devices(active_bot)
        await arrive(active_bot, happening_input("phone", "hangs"), "turn-1")
        await wait_until(lambda: len(processor.calls) == 1)
        await arrive(active_bot, happening_input("phone", "a"), "a")
        await arrive(active_bot, happening_input("phone", "b"), "b")
        await wait_until(lambda: active_bot.state() == "/Bot/degraded")
        await settle()
        return active_bot.state() or "", held_percepts(active_bot), len(processor.calls)

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _released_count(metrics, "degraded")
        state, held, calls = asyncio.run(run())
        after = _released_count(metrics, "degraded")

    assert state == "/Bot/degraded"
    assert after - before == 2
    assert held == 0
    assert calls == 1


def test_a_body_deactivated_or_rebooted_while_holding_counts_what_it_released_and_never_replays_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def deactivate() -> tuple[str, object]:
        processor = GatedProcessor({1: asyncio.Event()})
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=as_cognition(processor))
        environment = await start_bot_with_devices(active_bot)
        await arrive(active_bot, happening_input("phone", "busy"), "turn-1")
        await wait_until(lambda: len(processor.calls) == 1)
        await arrive(active_bot, happening_input("phone", "a"), "a")
        await arrive(active_bot, happening_input("phone", "b"), "b")
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        return active_bot.state() or "", held_percepts(active_bot)

    async def reboot() -> tuple[str, object, int]:
        processor = GatedProcessor({1: asyncio.Event()})
        cognition_ability = as_cognition(processor)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognition_ability)
        environment = await start_bot_with_devices(active_bot)
        await arrive(active_bot, happening_input("phone", "busy"), "turn-1")
        await wait_until(lambda: len(processor.calls) == 1)
        for index in range(3):
            await arrive(active_bot, happening_input("phone", f"held-{index}"), f"held-{index}")
        _ = await hsm.dispatch(
            environment,
            active_bot,
            dataclasses.replace(
                mosfet.RebootEvent.with_data(mosfet.RebootEventData(reason="cognition_child_teardown_failed")),
                id="reboot-turn",
                source=hsm.id(cognition_ability),
                target=hsm.id(active_bot),
            ),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        await settle()
        return active_bot.state() or "", held_percepts(active_bot), len(processor.calls)

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        deactivated_before = _released_count(metrics, "deactivated")
        rebooted_before = _released_count(metrics, "rebooted")
        deactivated_state, deactivated_held = asyncio.run(deactivate())
        rebooted_state, rebooted_held, rebooted_calls = asyncio.run(reboot())
        deactivated_after = _released_count(metrics, "deactivated")
        rebooted_after = _released_count(metrics, "rebooted")

    assert deactivated_state == "/Bot/inactive"
    assert deactivated_after - deactivated_before == 2
    assert deactivated_held == 0
    # Nothing held survives into the next activation: the rebooted body idles, unprompted.
    assert rebooted_state == "/Bot/active/unfocused"
    assert rebooted_after - rebooted_before == 3
    assert rebooted_held == 0
    assert rebooted_calls == 1


def test_an_unoccupied_body_turns_to_what_arrived_first_and_an_occupied_one_keeps_its_focus() -> None:
    """Reflex floor at a turn's end, by arrival order: never by rank."""

    browser, phone = happening_input("browser", "opened"), happening_input("phone", "rang")

    async def run(*, occupied: bool) -> cognition.InputData:
        release = asyncio.Event()
        ability = BlockingSequenceAbility(
            release=release, block_on_call=2 if occupied else 1, outputs=(no_output("look"), no_output("stay"))
        )
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=cognitive)
        _ = await start_bot_with_devices(active_bot)
        if occupied:
            await arrive(active_bot, happening_input("phone", "first"), "turn-1")
            await wait_until(lambda: len(cognitive.inputs) == 1 and active_bot.state() == "/Bot/active/focused")
            await arrive(active_bot, happening_input("phone", "busy"), "turn-2")
        else:
            # Only the start is unfocused (R-9): a first turn that arrived on no device keeps it so.
            failure = cognition.StepFailedData(
                message="Cognition's reasoning step failed, so the turn failed: provider unavailable",
                stage="reasoning",
                turn_id="failed-turn",
                stimulus=mosfet.InputEvent.name,
            )
            await active_bot.dispatch(
                active_bot.context(),
                dataclasses.replace(
                    cognition.InputEvent.with_data_and_id(
                        cognition.InputData(
                            stimulus=dataclasses.replace(
                                cognition.StepFailedEvent.with_data(failure), source="cognition"
                            )
                        ),
                        "handed-back",
                    ),
                    source="cognition",
                ),
            )
        turns = 3 if occupied else 2
        await wait_until(lambda: len(cognitive.inputs) == turns - 1)
        await arrive(active_bot, browser, "browser-first")
        await arrive(active_bot, phone, "phone-second")
        _ = release.set()
        await wait_until(lambda: len(cognitive.inputs) == turns and active_bot.state() == "/Bot/active/focused")
        return cognitive.inputs[-1]

    unoccupied = asyncio.run(run(occupied=False))
    occupied = asyncio.run(run(occupied=True))

    assert perceived_stimuli(unoccupied.stimulus) == [browser, phone]
    assert unoccupied.focus == "browser"
    assert unoccupied.focus_candidates == ("browser", "phone")
    assert perceived_stimuli(occupied.stimulus) == [browser, phone]
    assert occupied.focus == "phone"
    assert occupied.focus_candidates == ("phone", "browser")


def test_a_turn_records_how_many_percepts_it_took_and_how_long_each_waited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def run() -> None:
        gate = asyncio.Event()
        processor = GatedProcessor({1: gate})
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=as_cognition(processor))
        _ = await start_bot_with_devices(active_bot)
        await arrive(active_bot, happening_input("phone", "busy"), "turn-1")
        await wait_until(lambda: len(processor.calls) == 1)
        await arrive(active_bot, happening_input("phone", "a"), "a")
        product = cognition.InputData(stimulus=HappeningEvent.with_data(HappeningData(situation="heard")))
        _ = await active_bot.dispatch(
            active_bot.context(),
            dataclasses.replace(cognition.InputEvent.with_data_and_id(product, "product"), source="ears"),
        )
        _ = gate.set()
        await wait_until(lambda: len(processor.calls) == 2 and active_bot.state() == "/Bot/active/focused")

    def totals(points: tuple[metric_reader.Point, ...]) -> dict[str, tuple[int, float]]:
        return {json.dumps(point.attributes, sort_keys=True): (point.count, point.sum) for point in points}

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        count_before = totals(metrics.points("bot.turn.percept.count"))
        wait_before = totals(metrics.points("bot.turn.percept.wait"))
        asyncio.run(run())
        count_after = totals(metrics.points("bot.turn.percept.count"))
        wait_after = totals(metrics.points("bot.turn.percept.wait"))

    # Closed attribute sets: none on the count, only the handoff kind on the wait.
    assert set(count_after) == {"{}"}
    assert set(wait_after) <= {
        '{"bot.handoff.kind": "ability_product"}',
        '{"bot.handoff.kind": "device_stimulus"}',
    }
    turns, percepts = count_after["{}"]
    prior_turns, prior_percepts = count_before.get("{}", (0, 0.0))
    assert (turns - prior_turns, percepts - prior_percepts) == (2, 3)
    waited = {key: count - wait_before.get(key, (0, 0.0))[0] for key, (count, _total) in wait_after.items()}
    assert waited == {
        '{"bot.handoff.kind": "ability_product"}': 1,
        '{"bot.handoff.kind": "device_stimulus"}': 2,
    }


def test_bot_processing_state_rejects_malformed_input_event_data() -> None:
    async def run() -> tuple[str, list[processing.InputData]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=(), cognition=ability, input=(ring_hearing(),))

        _ = await start_bot_with_devices(active_bot)
        malformed = mosfet.InputEvent.with_data(typing.cast(mosfet.InputEventData, object()))
        await active_bot.dispatch(active_bot.context(), malformed)
        await asyncio.sleep(0)

        return active_bot.state(), ability.calls

    state, calls = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert calls == []


def test_bot_processing_state_routes_ability_failure_to_failed_event() -> None:
    async def run() -> tuple[str, list[mosfet.ProcessingFailedEventData], list[cognition.types.OutputData]]:
        store = memory.Memory()
        failing = cognition.Cognition(
            intuition=cognition.Intuition(processor=FailingProcessor()),
            reasoning=cognition.Reasoning(processor=FailingProcessor()),
            reflection=_reflection_for_test(store),
            memory=store,
        )
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=failing)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        # Intuition's failure hands the turn to reasoning (D-R-36), which fails too. The bot perceives that
        # failed step as one more turn (D-R-13); that turn fails the same way and is not handed back again.
        await wait_until(lambda: len(active_bot.failures) == 2 and active_bot.state() == "/Bot/active/focused")

        return active_bot.state(), active_bot.failures, active_bot.actions

    state, failures, actions = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert [failure.message for failure in failures] == ["ability unavailable", "ability unavailable"]
    assert actions == []


@pytest.mark.parametrize("fails", [False, True])
def test_bot_processing_terminal_finishes_operation_timer_actor(fails: bool) -> None:
    async def run() -> tuple[str, dict[str, object]]:
        processor: processing.DecisionProcessor = FailingProcessor() if fails else NestedPrimaryProcessor()
        cognition_ability = CapturingCognition(processor)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognition_ability)
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data_and_id(mosfet.InputEventData(target_device="phone", priority=3), "timer-turn"),
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
        cognition_ability = as_cognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognition_ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="phone", priority=3),
                "live-turn",
            ),
        )
        await wait_until(lambda: ability.calls == [3] and active_bot.state() == "/Bot/active/processing")

        # Correct endpoints, wrong turn id: topology alone must not complete the live turn.
        stale = dataclasses.replace(
            mosfet.ProcessingCompletedEvent.with_data(
                mosfet.ProcessingCompletedEventData(output=no_output("stale"), focus_candidates=("phone",))
            ),
            id="stale-operation",
            source=hsm.id(cognition_ability),
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
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="phone", priority=3),
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
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="phone", priority=1),
                "focus-seed",
            ),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused" and len(cognitive.inputs) == 1)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="phone", priority=2),
                "focus-live-turn",
            ),
        )
        await wait_until(lambda: len(cognitive.inputs) == 2 and active_bot.state() == "/Bot/active/processing")
        forged = dataclasses.replace(
            mosfet.FocusDeviceEvent.with_data(mosfet.FocusDeviceEventData(device="browser")),
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
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
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
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="phone", priority=3),
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
            mosfet.InputEvent.with_data_and_id(
                mosfet.InputEventData(target_device="phone", priority=3),
                "current-operation",
            ),
        )
        await wait_until(lambda: bool(cognition_ability.input_events))
        request = cognition_ability.input_events[0]
        forged = dataclasses.replace(
            mosfet.ProcessingCompletedEvent.with_data(
                mosfet.ProcessingCompletedEventData(output=no_output("forged"), focus_candidates=("phone",))
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
        str,
        bool,
        str | None,
        list[int],
        list[mosfet.ProcessingFailedEventData],
        list[cognition.types.OutputData],
        list[str],
    ]:
        ability = CancellableHangingAbility()
        timers = ManualTimers()
        active_bot = QuickTimeoutAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot, clock=timers)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        # The first turn hangs until its processing timer fires; nothing else ends it.
        first = await _processing_wait(timers)
        first.set_result(datetime.datetime.now(datetime.UTC))
        await wait_until(lambda: active_bot.state() == "/Bot/active/focused" and ability.cancelled, timeout=30.0)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=4)),
        )
        # The second turn completes with its timer never fired, however long the pipeline takes.
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused", timeout=30.0)

        return (
            active_bot.state(),
            ability.cancelled,
            ability.state(),
            ability.calls,
            active_bot.failures,
            active_bot.actions,
            _processing_wait_outcomes(timers),
        )

    state, cancelled, ability_state, calls, failures, actions, outcomes = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert cancelled
    # Cognition injects the leaf Processor only; the Processing shell is not started.
    assert ability_state in (None, "")
    assert calls == [3, 4]
    assert len(failures) == 1
    assert "timed out" in failures[0].message
    assert actions == [no_output("priority:4")]
    assert outcomes == ["fired", "cancelled"]


def test_bot_processing_timeout_recovery_finishes_cancellation() -> None:
    async def run() -> tuple[str, bool, list[mosfet.ProcessingFailedEventData]]:
        ability = CancellableHangingAbility()
        active_bot = TimeoutAbilityAgent(devices=configured_devices("phone"), cognition=ability)

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await asyncio.sleep(0.05)

        return active_bot.state(), ability.cancelled, active_bot.failures

    state, cancelled, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert cancelled
    assert len(failures) == 1
    assert "timed out" in failures[0].message


def test_bot_processing_cancellation_timeout_fails_closed_and_rejects_next_turn() -> None:
    async def run() -> tuple[str, list[int], list[mosfet.ProcessingFailedEventData]]:
        processor = CancellableHangingProcessor()
        cognition_ability = CapturingCognition(processor, swallow_cancel=True)
        active_bot = TimeoutAbilityAgent(devices=configured_devices("phone"), cognition=cognition_ability)
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        for _ in range(100):
            if active_bot.state() == "/Bot/degraded":
                break
            await asyncio.sleep(0.002)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=4)),
        )
        await asyncio.sleep(0.01)
        return active_bot.state(), processor.calls, active_bot.failures

    state, calls, failures = asyncio.run(run())
    assert state == "/Bot/degraded"
    assert calls == [3]
    assert len(failures) == 1
    assert "timed out" in failures[0].message


def test_completed_turn_cancels_processing_timer() -> None:
    async def run() -> tuple[list[mosfet.ProcessingFailedEventData], list[cognition.types.OutputData], list[str]]:
        timers = ManualTimers()
        active_bot = QuickTimeoutAgent(devices=configured_devices("phone"), cognition=IgnoreAbility())
        _ = await start_bot_with_devices(active_bot, clock=timers)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        _ = await _processing_wait(timers)
        await wait_until(
            lambda: active_bot.state() == "/Bot/active/focused"
            and bool(active_bot.actions)
            and _processing_wait_outcomes(timers) != ["pending"],
            timeout=30.0,
        )
        return active_bot.failures, active_bot.actions, _processing_wait_outcomes(timers)

    failures, actions, outcomes = asyncio.run(run())

    # Completing the turn exits processing; the owning activity stops the timer, so its wait is
    # cancelled rather than ever fired.
    assert outcomes == ["cancelled"]
    assert failures == []
    assert actions == [no_output("priority:3")]


def test_bot_timeout_cancel_uses_cognition_cancel_event_contract() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        cognition_ability = StubCognition()
        active_bot = TimeoutStubCognitionAgent(cognition_ability)
        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
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
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
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


def test_stopping_the_body_stops_every_part_it_started_and_what_each_part_started() -> None:
    """D-W7-9: the body stops its devices and its abilities, the register and cognition among them, and each
    part stops what it started: a device its peripherals, cognition its faculties and its memory."""

    async def run() -> dict[str, bool]:
        speaker = audio.Speaker()
        handset = phone_device.Phone(id="desk", speaker=speaker)
        register = memory.StmMemory()
        store = memory.Memory()
        intuition = cognition.Intuition(processor=IgnoreProcessor())
        mind = cognition.Cognition(
            intuition=intuition,
            reasoning=cognition.Reasoning(processor=_NoopReasoningProcessor()),
            reflection=_reflection_for_test(store),
            memory=store,
        )
        active_bot = Bot(devices=(handset,), cognition=mind, stm_memory=register)
        environment = await start_bot_with_devices(active_bot)
        assert active_bot.state().startswith("/Bot/active")
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        await hsm.stop(active_bot, environment)
        parts = {
            "body": active_bot,
            "phone": handset,
            "speaker": speaker,
            "register": register,
            "cognition": mind,
            "intuition": intuition,
            "memory": store,
        }
        return {name: mosfet.lifecycle.is_started(part) for name, part in parts.items()}

    assert asyncio.run(run()) == dict.fromkeys(
        ("body", "phone", "speaker", "register", "cognition", "intuition", "memory"), False
    )


def test_bot_activation_dispatches_completion_after_queueing_device_notifications() -> None:
    async def run() -> None:
        device = Device()
        active_bot = basic_agent(devices=(device,))

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


def test_bot_activation_emits_correlated_public_completion_after_attachment() -> None:
    async def run() -> hsm.Event[typing.Any]:
        active_bot = LifecycleRecordingAgent(devices=(Device(id="phone"),))
        environment = Environment()
        _ = await mosfet.started(environment, active_bot, active_bot.model)
        environment.join(active_bot)
        request = dataclasses.replace(
            mosfet.ActivateEvent.with_data(mosfet.ActivateEventData()),
            id="activation-request",
            source="activation-owner",
            target=hsm.id(active_bot),
            metadata={"traceparent": "00-activation"},
        )
        await active_bot.dispatch(environment, request)
        await wait_until(lambda: len(active_bot.lifecycle_events) == 1)
        terminal = active_bot.lifecycle_events[0]
        await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        return terminal

    terminal = asyncio.run(run())

    assert isinstance(terminal.data, mosfet.ActivatingDoneEventData)
    assert terminal.id == "activation-request"
    assert terminal.source == terminal.target
    assert terminal.source
    assert terminal.metadata == {"traceparent": "00-activation"}


def test_bot_activation_emits_public_failure_without_public_completion() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], ...]:
        active_bot = LifecycleRecordingAgent(devices=(ImmediatelyFailingInitializingDevice(id="failing"),))
        environment = Environment()
        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(active_bot.lifecycle_events) == 1)
        terminals = tuple(active_bot.lifecycle_events)
        await active_bot.detach(environment)
        return terminals

    terminals = asyncio.run(run())

    assert len(terminals) == 1
    assert isinstance(terminals[0].data, mosfet.ActivatingFailedEventData)


def test_bot_deactivation_emits_correlated_public_completion_after_cleanup() -> None:
    async def run() -> hsm.Event[typing.Any]:
        active_bot = LifecycleRecordingAgent(devices=(Device(id="phone"),))
        environment = await start_bot_with_devices(active_bot)
        active_bot.lifecycle_events.clear()
        request = dataclasses.replace(
            mosfet.DeactivateEvent.with_data(mosfet.DeactivateEventData()),
            id="deactivation-request",
            source="deactivation-owner",
            target=hsm.id(active_bot),
            metadata={"traceparent": "00-deactivation"},
        )

        await active_bot.dispatch(environment, request)
        await wait_until(lambda: len(active_bot.lifecycle_events) == 1)
        terminal = active_bot.lifecycle_events[0]
        environment.leave(active_bot)
        return terminal

    terminal = asyncio.run(run())

    assert isinstance(terminal.data, mosfet.DeactivatingDoneEventData)
    assert terminal.id == "deactivation-request"
    assert terminal.source == terminal.target
    assert terminal.source
    assert terminal.metadata == {"traceparent": "00-deactivation"}


def test_bot_rejects_stale_attachment_terminal_from_previous_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, str]:
        active_bot = basic_agent(devices=(Device(id="phone"),))
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
    async def run() -> tuple[str, bool]:
        active_bot = basic_agent(devices=configured_devices("phone"))

        _ = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
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
            devices=(),
            cognition=cognition_ability,
            input=(input_ability,),
            output=(output_ability,),
        )
        environment = await start_bot_with_devices(active_bot)
        assert mosfet.lifecycle.is_started(input_ability) is True
        assert mosfet.lifecycle.is_started(output_ability) is True
        assert mosfet.lifecycle.is_started(cognition_ability) is True

        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")

        return (
            mosfet.lifecycle.is_started(input_ability),
            mosfet.lifecycle.is_started(output_ability),
            mosfet.lifecycle.is_started(cognition_ability),
            mosfet.lifecycle.is_started(active_bot),
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
    import mosfet

    assert mosfet.InputEvent.name == "bot.input"
    assert not hasattr(body, "OutputEvent")
    assert not hasattr(body, "_RAW_AUDIO_EVENT_NAMES")
    assert "bot.devices.audio" not in Path(body.__file__).read_text()


def test_bot_ignores_unhandled_device_event() -> None:
    async def run() -> tuple[str, list[processing.InputData], bool]:
        probe = hsm.Event[dict[str, object]](name="phone.probe.signal")
        ability = IgnoreAbility()
        phone = Device()
        active_bot = AbilityAgent(devices=(phone,), cognition=ability)

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
        def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
            if event.name == attachment.DetachEvent.name:

                async def hang() -> bool:
                    return await asyncio.Event().wait()

                return hang()
            return super().dispatch(ctx, event)

    async def run() -> str:
        active_bot = FastDeactivationTimeoutAgent(devices=(HangOnDetachDevice(id="stuck"),))
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
        started_device = Device(id="started")
        unstarted_device = Device(id="unstarted")
        active_bot = basic_agent(devices=(started_device, unstarted_device))

        environment = Environment()
        started_device_model = started_device.model
        assert started_device_model is not None
        _ = await mosfet.started(environment, started_device, started_device_model)
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
        active_bot = basic_agent(devices=(phone,))

        environment = await start_bot_with_devices(active_bot)
        firmware = device_firmware(phone)
        assert firmware is not None

        microphone = phone_microphone(phone)
        speaker = phone_speaker(phone)
        return (
            phone.state(),
            firmware.state(),
            microphone.state(),
            speaker.state(),
            environment.contains(active_bot),
            environment.contains(phone),
            environment.contains(microphone),
            environment.contains(speaker),
        )

    phone_state, firmware_state, microphone_state, speaker_state, agent_scope, phone_scope, mic_scope, speaker_scope = (
        asyncio.run(run())
    )

    assert phone_state == "/Device/attached"
    assert firmware_state == "/Phone/hung_up"
    # Wired, not idle: the phone powers its transducers and its firmware attaches to them during
    # bring-up, so they sit attached for the life of the phone. Call state gates the uplink in
    # firmware topology, not the wiring.
    assert microphone_state == "/DeviceMicrophone/attached"
    assert speaker_state == "/DeviceSpeaker/attached/silent"
    assert agent_scope
    assert phone_scope
    assert mic_scope
    assert speaker_scope


def test_bot_lifecycle_starts_and_stops_innate_ability() -> None:
    async def run() -> tuple[list[processing.InputData], str]:
        bot_instance = FocusedAgent(devices=())
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
            devices=(),
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
    assert_metadata_key_prefix_is_absent(body, "_PROCESSING_")


def test_a_placed_bot_only_hears_what_is_loud_enough_where_it_stands() -> None:
    """A bot's placement is its ears: it hears the near source and not the far one.

    Pins that ``Bot.attach`` carries the placement through to presence. Without it the bot is
    unplaced, which means unselective, and both broadcasts arrive.
    """

    async def run() -> tuple[int, int]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=(), cognition=ability, input=(ring_hearing(),))
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


def spoken(text: str) -> bytes:
    """The audio a vocal tract below makes for ``text``: its words, marked and whole-sample.

    A real encoder (macOS ``say``, Moonshine) is the same contract and is what the phone-bot example
    uses. Here the audio only has to be traceable back to the words, so the pipeline can be checked
    without a model download.
    """

    audio = b"spoken:" + text.encode()
    return audio + b" " if len(audio) % 2 else audio


class UtteranceSpeechEncoder(vocal.speech.SpeechEncoder):
    """Stands in for a vocal tract: the whole text in, its marked words out as 16 kHz PCM, on this machine."""

    def __init__(self) -> None:
        self.format = vocal.speech.Format(sample_rate_hz=16_000, channels=1)

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        yield vocal.speech.Audio(pcm=spoken(said), text=said)


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
    _ = await mosfet.started(environment, mouth, typing.cast(hsm.Model, mouth.model))
    voice = speaking.Speaking(encoder=UtteranceSpeechEncoder(), speaker=mouth)
    await start_ability_tree(environment, voice)
    _ = await voice.apply(speaking.InputData(parts=(vocal.speech.Part(words=text),)), ctx=environment)
    return mouth


def test_a_bot_hears_words_somebody_speaks_in_its_environment() -> None:
    """Somebody stands a metre away and says something; the bot's turn arrives as words.

    The whole path is real: a mouth in the environment, ``Environment.broadcast`` with a level
    and an origin, the bot's own placement deciding it is loud enough where it stands, body
    fan-out to Listening, voice detection, speech decoding, and the decoded product handed to
    cognition. The decoded bytes carry the utterance forward through every stage, so this cannot
    pass on a pipeline that dropped the audio and substituted something else.

    What it refuses to assert is anything the bot does about it. No device is dialed, nothing is
    answered, and no output is expected — the turn is the whole claim. A bot that hears this
    sentence and does nothing for the rest of its life passes.
    """

    async def run() -> list[hsm.Event[typing.Any]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=(), cognition=ability, input=(RecordingListening(),))
        environment = await start_bot_with_devices(
            active_bot,
            placement=space.Placement(position=space.Position(x=0.0, y=0.0), threshold_db=20.0),
        )
        ability.calls.clear()

        _ = await somebody_speaks(environment, "Call Bob at 555-0142.", position=space.Position(x=0.0, y=1.0))
        await close_hearing_speech(active_bot, environment)

        await wait_until(lambda: bool(ability.calls), timeout=10.0)
        return [call.input for call in ability.calls if isinstance(call.input, hsm.Event)]

    stimuli = asyncio.run(run())

    assert [stimulus.name for stimulus in stimuli] == [listening.SpeechEvent.name]
    assert isinstance(stimuli[0].data, listening.SpeechData)
    assert stimuli[0].data.content_type == "text/plain"
    assert stimuli[0].data.content == (b"decoded:" + spoken("Call Bob at 555-0142.") + silence_sound().audio).decode(
        "utf-8", errors="replace"
    )


def test_words_spoken_from_across_the_room_never_reach_the_bot() -> None:
    """The same sentence, said 500 metres away, is a sentence the bot did not hear.

    This is what proves the utterance went *through* the environment rather than around it: the
    only difference between this and the test above is where the mouth is standing, and a
    dispatch aimed at the body would ignore that entirely.
    """

    async def run() -> tuple[int, int]:
        ability = IgnoreAbility()
        listening_ability = RecordingListening()
        active_bot = AbilityAgent(devices=(), cognition=ability, input=(listening_ability,))
        environment = await start_bot_with_devices(
            active_bot,
            placement=space.Placement(position=space.Position(x=0.0, y=0.0), threshold_db=20.0),
        )
        ability.calls.clear()

        _ = await somebody_speaks(environment, "Call Bob at 555-0142.", position=space.Position(x=0.0, y=500.0))
        await close_hearing_speech(active_bot, environment)
        # Long enough for the near case to have finished decoding twice over.
        await asyncio.sleep(0.5)
        shouted_from_far = len(ability.calls)

        _ = await somebody_speaks(environment, "Call Bob at 555-0142.", position=space.Position(x=0.0, y=1.0))
        await close_hearing_speech(active_bot, environment)
        await wait_until(lambda: bool(ability.calls), timeout=10.0)
        return shouted_from_far, len(ability.calls)

    from_far, from_near = asyncio.run(run())

    assert from_far == 0
    assert from_near == 1


MOUTH_OFFSET_M = 0.15
"""How far a robot's mouth is from its own ears. Close enough that it is the loudest thing there."""


class TimedUtteranceSpeechEncoder(vocal.speech.SpeechEncoder):
    """A vocal tract that takes a realistic amount of time to say something.

    Byte count is what a linear-PCM duration is measured from, so an utterance that is only its
    own text would be a couple of milliseconds long — shorter than anything can be observed
    happening during. The marker stays at the front so the audio is still traceable to the words.
    """

    def __init__(self, *, seconds: float, sample_rate_hz: int) -> None:
        self.format = vocal.speech.Format(sample_rate_hz=sample_rate_hz, channels=1)
        self._bytes = int(sample_rate_hz * seconds) * 2

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        yield vocal.speech.Audio(pcm=spoken(said).ljust(self._bytes, b"\x00"), text=said)


class SlowSpeechDecoder(RecordingSpeechDecoder):
    """A speech decoder that takes as long over a sentence as an off-device one does.

    Interpretation is the slow part of hearing, and it is slower than the utterances it runs on.
    A decoder that returns instantly hides every question about what perception was doing when a
    sound arrived, because there is never a moment when it was doing anything.
    """

    _seconds: float

    def __init__(self, *, seconds: float) -> None:
        super().__init__()
        self._seconds = seconds

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        self.calls.append(input)
        await asyncio.sleep(self._seconds)
        return b"decoded:" + input


async def a_bot_with_a_voice(
    *,
    utterance_seconds: float,
    listening_ability: RecordingListening | None = None,
) -> tuple[AbilityAgent, IgnoreAbility, RecordingListening, speaking.Speaking, Environment, audio.Speaker]:
    """A robot standing at the origin with its own mouth 15 cm away, able to hear the room.

    The whole loop is real: the mouth is a ``Speaker`` in the same environment the bot is placed
    in, so the bot's own voice reaches its ears through ``Environment.broadcast`` exactly as
    anybody else's would — which is the defect this wiring exists to expose.
    """

    sample_rate_hz = 16_000
    mouth = audio.Speaker(
        placement=space.Placement(position=space.Position(x=MOUTH_OFFSET_M, y=0.0)),
        amplitude_db=60.0,
    )
    ability = IgnoreAbility()
    listening_ability = listening_ability if listening_ability is not None else RecordingListening()
    voice = speaking.Speaking(
        encoder=TimedUtteranceSpeechEncoder(seconds=utterance_seconds, sample_rate_hz=sample_rate_hz),
        speaker=mouth,
        listening=(listening_ability,),
    )
    active_bot = AbilityAgent(devices=(), cognition=ability, input=(listening_ability,), output=(voice,))
    # The mouth is powered by the Speaking ability it was injected into — part of the bot, not a
    # device the body starts — so bringing the bot up is what brings the mouth up.
    environment = await start_bot_with_devices(
        active_bot,
        placement=space.Placement(position=space.Position(x=0.0, y=0.0), threshold_db=20.0),
    )
    ability.calls.clear()
    return active_bot, ability, listening_ability, voice, environment, mouth


async def own_voice_from_the_mouth(environment: Environment, mouth: audio.Speaker, audio_bytes: bytes) -> None:
    """The bot's own voice reaching its ears again, later — an echo, a line, feedback.

    Emitted from the mouth's position with the mouth's level, so the environment computes the
    same arrival level it computed the first time. Only the timing is different.
    """

    await environment.broadcast(
        dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=audio_bytes,
                    media_type="audio/pcm",
                    sample_rate_hz=16_000,
                    channels=1,
                    amplitude_db=60.0,
                )
            ),
            source=hsm.id(mouth),
        ),
        origin=space.Position(x=MOUTH_OFFSET_M, y=0.0),
    )


def test_perception_sees_the_copy_of_a_command_before_the_sound_it_predicts() -> None:
    """The ordering the whole mechanism rests on, over the real chain rather than a stub.

    The copy leaves ``Speaking``'s entry into playout and goes Speaking → linked Listening; the
    sound leaves the playout activity and travels mouth → environment → body → input abilities.
    The copy is dispatched strictly earlier in program order and the environment adds a hop, but
    that is an argument about queue semantics, not a proof, and if the copy ever lost the race
    the mechanism would fail silently for that utterance.

    Two claims, because either alone is weak. Arrival order at perception is the invariant
    itself. The attenuated outcome is its consequence, and it is the part that cannot be faked:
    a sound scored before its copy arrived would carry its full 76.5 dB and become a turn.
    Repeated, because a race that holds once proves less than a race that holds every time.
    """

    utterances = 20

    async def run() -> tuple[list[list[str]], list[processing.InputData]]:
        _, ability, listening_ability, voice, environment, mouth = await a_bot_with_a_voice(utterance_seconds=0.05)
        mouth_id = hsm.id(mouth)
        orders: list[list[str]] = []

        for index in range(utterances):
            listening_ability.received.clear()

            def arrivals() -> list[str]:
                seen: list[str] = []
                for event in list(listening_ability.received):
                    if isinstance(event.data, EfferenceData):
                        seen.append("copy")
                    elif isinstance(event.data, SoundData) and event.source == mouth_id:
                        seen.append("sound")
                return seen

            _ = await voice.apply(
                speaking.InputData(parts=(vocal.speech.Part(words=f"Utterance {index}."),)), ctx=environment
            )
            await wait_until(lambda: {"copy", "sound"} <= set(arrivals()), timeout=5.0)
            orders.append(arrivals())
            # Let the window close so each utterance is scored against its own copy.
            await asyncio.sleep(0.15)

        return orders, list(ability.calls)

    orders, turns = asyncio.run(run())

    assert len(orders) == utterances
    assert all(order[:2] == ["copy", "sound"] for order in orders), orders
    assert turns == []


def test_a_bot_does_not_deliberate_its_own_utterance() -> None:
    """The robot says something and its own voice comes straight back at 76.5 dB. No turn.

    Nothing is filtered on the way in. The sound is broadcast, the bot is placed close enough to
    hear it, it clears the 20 dB floor by a mile, and the whole pipeline runs on it — voice
    detection and speech decoding both execute. What stops it becoming something the bot has to
    answer is that the body told its own senses it was producing, and the arrival matched.
    """

    async def run() -> tuple[list[processing.InputData], list[bytes]]:
        active_bot, ability, listening_ability, voice, environment, _ = await a_bot_with_a_voice(utterance_seconds=0.4)

        _ = await voice.apply(
            speaking.InputData(parts=(vocal.speech.Part(words="Hello, this is Alice."),)), ctx=environment
        )
        # Close streaming VAD so speech decoding can finish on the attenuated own-voice frame.
        await close_hearing_speech(active_bot, environment)
        # Well past playout, and past anything the pipeline could still be chewing on.
        await asyncio.sleep(1.0)
        assert listening_ability.speech_decoder is not None
        return list(ability.calls), list(listening_ability.speech_decoder.calls)

    turns, decoded = asyncio.run(run())

    assert turns == []
    # The bot did hear itself, all the way through speech decoding. It simply had nothing to
    # deliberate about it, which is a very different thing from having been deafened.
    assert len(decoded) == 1
    assert decoded[0].startswith(b"spoken:Hello, this is Alice.")


def test_a_bot_hears_its_own_voice_coming_back_late() -> None:
    """The same audio, arriving after the utterance is over, is a turn.

    This is the residual and it is the whole reason the window is bounded rather than the audio
    being dropped. Own voice that took longer to come back than the command lasted is a delayed
    line, an echo, or feedback — real information about the call, and a design that suppressed
    own audio outright would throw it away.
    """

    async def run() -> tuple[list[processing.InputData], list[processing.InputData]]:
        active_bot, ability, listening_ability, voice, environment, mouth = await a_bot_with_a_voice(
            utterance_seconds=0.2
        )

        _ = await voice.apply(
            speaking.InputData(parts=(vocal.speech.Part(words="Hello, this is Alice."),)), ctx=environment
        )
        await close_hearing_speech(active_bot, environment)
        await asyncio.sleep(1.0)
        during = list(ability.calls)

        await own_voice_from_the_mouth(environment, mouth, b"spoken:Hello, this is Alice.")
        await close_hearing_speech(active_bot, environment)
        await wait_until(lambda: bool(ability.calls), timeout=5.0)
        return during, list(ability.calls)

    during, after = asyncio.run(run())

    assert during == []
    assert len(after) == 1
    stimulus = after[0].input
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == listening.SpeechEvent.name
    assert isinstance(stimulus.data, listening.SpeechData)
    assert stimulus.data.content_type == "text/plain"
    assert stimulus.data.content == (b"decoded:spoken:Hello, this is Alice." + silence_sound().audio).decode(
        "utf-8", errors="replace"
    )


def test_somebody_cutting_in_while_the_bot_talks_is_still_heard() -> None:
    """The window is not a mute button, and a caller who interrupts still gets a turn.

    Their voice arrives from a different mouth, at a level nothing about the bot's own
    production predicts. Suppressing everything for the duration of an utterance would be a
    scheduled deafness, which is exactly what this is not.
    """

    async def run() -> list[processing.InputData]:
        active_bot, ability, listening_ability, voice, environment, mouth = await a_bot_with_a_voice(
            utterance_seconds=0.6
        )
        mouth_id = hsm.id(mouth)

        _ = await voice.apply(
            speaking.InputData(parts=(vocal.speech.Part(words="Hello, this is Alice."),)), ctx=environment
        )
        # Mid-utterance: the mouth is still committed and the sensitivity window is still open.
        await asyncio.sleep(0.15)
        # End own-voice HearingSpeech (attenuated, no turn) before the interrupt is heard.
        await close_hearing_speech(active_bot, environment)
        caller = await somebody_speaks(environment, "Actually, wait.", position=space.Position(x=0.0, y=1.0))
        caller_id = hsm.id(caller)

        def the_bot_kept_talking_over_the_caller() -> bool:
            sources = [event.source for event in list(listening_ability.received) if isinstance(event.data, SoundData)]
            return caller_id in sources and mouth_id in sources[sources.index(caller_id) + 1 :]

        # The mouth plays in real time, so the caller's words and the bot's next frame share the air:
        # wait for that overlap instead of racing it, so the turn holds the same sound every run.
        await wait_until(the_bot_kept_talking_over_the_caller, timeout=5.0)
        await close_hearing_speech(active_bot, environment)

        await wait_until(lambda: bool(ability.calls), timeout=5.0)
        await asyncio.sleep(0.6)
        return list(ability.calls)

    turns = asyncio.run(run())

    stimuli = [turn.input for turn in turns if isinstance(turn.input, hsm.Event)]
    heard = [getattr(stimulus.data, "content", None) for stimulus in stimuli]
    words = (b"decoded:" + spoken("Actually, wait.")).decode("utf-8", errors="replace")
    assert len(heard) == 1, heard
    assert isinstance(heard[0], str) and heard[0].startswith(words), heard
    # What follows the caller's words is silence only: the closing frame and the bot's own
    # playout past its marker, which shared the air with the caller and carries no words.
    assert heard[0].removeprefix(words).strip("\x00") == "", heard


def test_a_bot_does_not_answer_itself_because_it_spoke_while_perception_was_busy() -> None:
    """The bot answers somebody, and its own answer comes back while the ear is still working.

    This is the ordinary case, not an edge one. Interpreting a sentence takes longer than saying
    one, so a bot that replies is replying while its ears are still busy with what it is
    replying to — and its own voice arrives into a pipeline that is occupied and queues it.

    What the copy commits to is a duration of *sound*, and the sound arrived on time. Whether
    perception had got round to it is a fact about perception's backlog, not about the mouth. A
    body that scores an arrival against whatever it happens to be doing when it reaches the front
    of the queue is not correlating the arrival with the command at all: it is correlating the
    command with the clock.
    """

    async def run() -> list[processing.InputData]:
        decoder = SlowSpeechDecoder(seconds=0.5)
        active_bot, ability, _, voice, environment, _ = await a_bot_with_a_voice(
            utterance_seconds=0.2,
            listening_ability=RecordingListening(speech_decoder=decoder),
        )

        _ = await somebody_speaks(environment, "Are you there?", position=space.Position(x=0.0, y=1.0))
        # Close VAD so decoding starts; SlowSpeechDecoder holds the ear for the reply.
        await close_hearing_speech(active_bot, environment)
        # Perception is now committed to a sentence for longer than the reply will last.
        await wait_until(lambda: bool(decoder.calls), timeout=5.0)

        _ = await voice.apply(speaking.InputData(parts=(vocal.speech.Part(words="Yes, I am here."),)), ctx=environment)
        await close_hearing_speech(active_bot, environment)

        # Long enough for the backlog to drain and for anything it produced to become a turn.
        await asyncio.sleep(2.0)
        return list(ability.calls)

    turns = asyncio.run(run())

    stimuli = [turn.input for turn in turns if isinstance(turn.input, hsm.Event)]
    assert [getattr(stimulus.data, "content", None) for stimulus in stimuli] == [
        (b"decoded:" + spoken("Are you there?") + silence_sound().audio).decode("utf-8", errors="replace")
    ]


def test_speaking_delivers_the_copy_of_its_own_command_to_linked_listening() -> None:
    """A nerve from Speaking to Listening peers — not body fan-out, not environment broadcast.

    Pins the route rather than the consequence: every copy reaches the Listening ability it was wired to,
    with speaking as source and listening as target, and the last one says the mouth has nothing left.
    """

    async def run() -> tuple[list[hsm.Event[typing.Any]], str, str]:
        _, _, listening_ability, voice, environment, _ = await a_bot_with_a_voice(utterance_seconds=0.2)

        def copies() -> list[hsm.Event[typing.Any]]:
            return [event for event in listening_ability.received if isinstance(event.data, EfferenceData)]

        _ = await voice.apply(speaking.InputData(parts=(vocal.speech.Part(words="Hello."),)), ctx=environment)
        await wait_until(lambda: any(duration == 0.0 for duration in durations(copies())), timeout=5.0)
        return copies(), hsm.id(voice), hsm.id(listening_ability)

    def durations(copies: list[hsm.Event[typing.Any]]) -> list[float]:
        return [copy.data.duration for copy in copies if isinstance(copy.data, EfferenceData)]

    copies, voice_id, listening_id = asyncio.run(run())

    assert {copy.name for copy in copies} == {EfferenceEvent.name, "bot.ability.speaking.efference"}
    assert {(copy.source, copy.target) for copy in copies} == {(voice_id, listening_id)}
    assert math.isclose(durations(copies)[0], 0.2, abs_tol=1e-9)
    assert durations(copies)[-1] == 0.0


def test_a_bot_gets_a_turn_because_something_happened_and_never_because_of_what() -> None:
    """Topology grants the occasion; it never grants the action.

    Nothing is dispatched at this bot: no input is minted by the test, no sound, no directive,
    and no memory to have recalled one from. A device it owns simply becomes something, and
    that is the whole condition for the turn — the bot is the one that gets to decide the
    turn is a turn to do nothing.

    The guard reads which device the interrupt arrived from and nothing else. This is the test
    that would fail if it ever grew a condition on what happened: the two happenings here carry
    different content and both are worth exactly one turn.
    """

    async def run() -> tuple[list[processing.InputData], list[mosfet.ProcessingFailedEventData], str]:
        ability = IgnoreAbility()
        device = ReportingDevice()
        active_bot = OccasionAgent(devices=(device,), cognition=ability)
        environment = await start_bot_with_devices(active_bot)

        device.happen("became one thing")
        await wait_until(lambda: len(ability.calls) == 1)
        device.happen("became a different thing")
        await wait_until(lambda: len(ability.calls) == 2)

        turns = list(ability.calls)
        failures = list(active_bot.failures)
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        return turns, failures, active_bot.state()

    turns, failures, state = asyncio.run(run())

    stimulus = turns[0].input
    assert isinstance(stimulus, mosfet.InputEventData)
    # What happened rode along; what to do about it did not.
    assert stimulus.observation == mosfet.StimulusData(
        event=HappeningEvent.name,
        data=HappeningData(situation="became one thing"),
        id="",
        source="",
        target="",
    )
    assert set(mosfet.InputEventData.model_fields) == {"target_device", "priority", "observation", "participants"}
    # Doing nothing twice is not a failure, and the body is left exactly where it started.
    assert failures == []
    assert state == "/Bot/inactive"


def test_an_occasion_carries_the_envelope_of_the_happening_that_caused_it() -> None:
    """Each occasion is correlatable, and correlates to the thing that actually happened.

    The old clock-driven occasion had to mint an id out of nothing, because a time event
    arrives with neither id nor data. A happening already has an envelope worth keeping.
    """

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[processing.InputData]]:
        ability = IgnoreAbility()
        device = ReportingDevice()
        active_bot = OccasionAgent(devices=(device,), cognition=ability)
        environment = await start_bot_with_devices(active_bot)

        device.happen("first")
        device.happen("second")
        await wait_until(lambda: len(ability.calls) == 2)

        occasions = list(active_bot.occasions)
        turns = list(ability.calls)
        _ = await active_bot.detach(environment)
        return occasions, turns

    occasions, turns = asyncio.run(run())

    assert len(occasions) == 2
    assert all(occasion.id for occasion in occasions)
    assert len({occasion.id for occasion in occasions}) == 2
    # Sourced from the device that changed, addressed to the bot holding it — not broadcast.
    assert all(occasion.source and occasion.target for occasion in occasions)
    assert len(turns) == 2


def test_a_happening_during_a_busy_turn_is_still_waiting_afterwards() -> None:
    """A fact about the world outlives the turn the bot was busy with.

    This is where a happening and a clock tick part company. A tick only ever asserted "it is
    now later", which is stale the instant it is queued, so dropping it lost nothing. A call
    that connects while the bot is mid-thought is still connected when the thought ends, and
    nothing will announce it a second time — so the body owes it a turn, late rather than
    never.
    """

    async def run() -> tuple[int, list[mosfet.ProcessingFailedEventData], str]:
        release = asyncio.Event()
        ability = BlockingAbility(release=release)
        device = ReportingDevice()
        active_bot = OccasionAgent(devices=(device,), cognition=ability)
        environment = await start_bot_with_devices(active_bot)

        device.happen("first")
        await wait_until(lambda: active_bot.state() == "/Bot/active/processing")

        # A second thing happens while the body is occupied by the first turn.
        device.happen("second")

        _ = release.set()
        await wait_until(lambda: len(ability.calls) == 2)
        turns = len(ability.calls)
        failures = list(active_bot.failures)
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        return turns, failures, active_bot.state()

    turns, failures, state = asyncio.run(run())

    assert turns == 2
    assert failures == []
    assert state == "/Bot/inactive"


def test_a_bot_that_is_not_awake_has_no_moments() -> None:
    """Occasions belong to being awake, so an unattached or stopped body has none."""

    async def run() -> tuple[int, int, int, int]:
        ability = IgnoreAbility()
        device = ReportingDevice()
        active_bot = OccasionAgent(devices=(device,), cognition=ability)

        # Constructed but never attached: no environment, no lifetime, no moments. The device
        # is not attached to anything either, so there is nothing for it to report to.
        device.happen("before")
        await asyncio.sleep(0.05)
        before_attach = len(active_bot.occasions)

        environment = await start_bot_with_devices(active_bot)
        device.happen("while awake")
        await wait_until(lambda: bool(active_bot.occasions))
        while_awake = len(active_bot.occasions)

        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        at_rest = len(active_bot.occasions)
        device.happen("after")
        await asyncio.sleep(0.05)
        return before_attach, while_awake, at_rest, len(active_bot.occasions)

    before_attach, while_awake, at_rest, after_rest = asyncio.run(run())

    assert before_attach == 0
    assert while_awake > 0
    assert after_rest == at_rest


class _PriorityReportingDevice(ReportingDevice):
    """ReportingDevice that lets a test choose the report priority."""

    def happen_with_priority(self, situation: str, *, priority: int) -> None:
        _dispatch_to_holders(self, HappeningEvent.with_data(HappeningData(situation=situation)), priority=priority)


def test_body_handles_device_reports_in_arrival_order_regardless_of_priority() -> None:
    """The body never ranks stimuli: a background report first is handled first.

    Architectural negative: a priority table ("a ring outranks a thought") would be
    writing the bot's behavior for it. Urgency rides along on the handoff but must
    never reorder it; cognition decides what anything is worth.
    """

    async def run() -> list[processing.InputData]:
        ability = IgnoreAbility()
        device = _PriorityReportingDevice()
        active_bot = OccasionAgent(devices=(device,), cognition=ability)
        environment = await start_bot_with_devices(active_bot)

        device.happen_with_priority("background thought", priority=10)
        device.happen_with_priority("urgent ring", priority=0)
        await wait_until(lambda: len(ability.calls) == 2)

        turns = list(ability.calls)
        _ = await active_bot.detach(environment)
        return turns

    turns = asyncio.run(run())

    assert len(turns) == 2
    assert isinstance(turns[0].input, mosfet.InputEventData)
    assert isinstance(turns[1].input, mosfet.InputEventData)
    assert turns[0].input.observation == mosfet.StimulusData(
        event=HappeningEvent.name,
        data=HappeningData(situation="background thought"),
        id="",
        source="",
        target="",
    )
    assert turns[1].input.observation == mosfet.StimulusData(
        event=HappeningEvent.name,
        data=HappeningData(situation="urgent ring"),
        id="",
        source="",
        target="",
    )


class _OutputTap(ProbeAbility):
    """Output-slot ability that records every event name dispatched to it."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        self.seen.append(event.name)
        return super().dispatch(ctx, event)


def test_body_never_fans_environment_stimuli_out_to_output_abilities() -> None:
    """Output abilities are effectors, not stimulus recipients.

    The body fans environment stimuli to input abilities in parallel; output
    abilities join no fan-out path, so a heard sound never reaches them.
    """

    async def run() -> tuple[list[str], list[processing.InputData]]:
        ability = IgnoreAbility()
        listening_ability = RecordingListening()
        output_tap = _OutputTap()
        active_bot = AbilityAgent(devices=(), cognition=ability, input=(listening_ability,), output=(output_tap,))
        environment = await start_bot_with_devices(active_bot)
        sound = SoundEvent.with_data(
            SoundData(audio=b"heard-chunk", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)
        )
        await active_bot.dispatch(active_bot.context(), sound)
        await close_hearing_speech(active_bot, environment, sample_rate_hz=48_000)
        await wait_until(lambda: len(ability.calls) == 1)
        seen = list(output_tap.seen)
        _ = await active_bot.detach(environment)
        return seen, list(ability.calls)

    seen, calls = asyncio.run(run())

    assert len(calls) == 1
    assert SoundEvent.name not in seen
    assert listening.SpeechEvent.name not in seen


def test_body_turn_focus_candidates_do_not_carry_into_the_next_turn() -> None:
    """Focus legality is recomputed every turn, never inherited.

    Turn 1 focuses phone; turn 2 hears browser and turns to it; turn 3 hears browser. A
    turn-3 focus selection for phone must be rejected and retried even though phone was a
    legal candidate on turns 1 and 2 — the turn slot is set when the turn starts and retired
    with it, so a stale allowlist never gates a later turn and never becomes an action.
    """

    async def run() -> tuple[
        str,
        list[cognition.InputData],
        list[processing.InputData],
        list[cognition.types.OutputData],
        list[mosfet.ProcessingFailedEventData],
    ]:
        ability = SequenceAbility(
            no_output("stay on phone"),
            focus_output("browser", "look at the browser"),
            focus_output("phone", "stale grab"),
            no_output("backs off"),
        )
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone", "browser"), cognition=cognitive)
        environment = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(ability.calls) == 1 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=2)),
        )
        await wait_until(lambda: len(ability.calls) == 2 and active_bot.state() == "/Bot/active/focused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="browser", priority=1)),
        )
        await wait_until(lambda: len(ability.calls) == 4 and active_bot.state() == "/Bot/active/focused")
        result = (
            active_bot.state(),
            list(cognitive.inputs),
            list(ability.calls),
            list(active_bot.actions),
            list(active_bot.failures),
        )
        _ = await active_bot.detach(environment)
        return result

    state, inputs, calls, actions, failures = asyncio.run(run())

    assert state == "/Bot/active/focused"
    assert inputs[0].focus_candidates == ("phone",)
    assert inputs[1].focus_candidates == ("phone", "browser")
    assert inputs[2].focus == "browser"
    assert inputs[2].focus_candidates == ("browser",)
    # The stale grab was rejected and retried (4 processor calls for 3 turns) and never
    # became an action: only the back-off completed turn 3, with no terminal failure.
    assert len(calls) == 4
    assert actions == [
        no_output("stay on phone"),
        focus_output("browser", "look at the browser"),
        no_output("backs off"),
    ]
    assert failures == []


def _socket_guarded_import_probe() -> str:
    """Child preamble: record every connection attempt, block it, then import bot."""

    return (
        "import http.client\n"
        "import os\n"
        "import socket\n"
        "record_path = os.environ['BOT_SOCKET_RECORD_PATH']\n"
        "def _note(kind):\n"
        "    with open(record_path, 'a') as handle:\n"
        "        handle.write(kind + '\\n')\n"
        "def _blocked_create_connection(*args, **kwargs):\n"
        "    _note('create_connection')\n"
        "    raise RuntimeError('network blocked during import')\n"
        "def _blocked_http_connect(self):\n"
        "    _note('http_connect')\n"
        "    raise RuntimeError('network blocked during import')\n"
        "socket.create_connection = _blocked_create_connection\n"
        "http.client.HTTPConnection.connect = _blocked_http_connect\n"
        "import mosfet\n"
    )


def _run_import_probe(*, publish_opt_in: bool, record_path: Path) -> "subprocess.CompletedProcess[str]":
    probe_environment = dict(os.environ)
    probe_environment["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:9"
    if publish_opt_in:
        probe_environment["BOT_MODEL_PUBLISH"] = "1"
    else:
        probe_environment.pop("BOT_MODEL_PUBLISH", None)
    probe_environment["BOT_SOCKET_RECORD_PATH"] = str(record_path)
    return subprocess.run(
        [sys.executable, "-c", _socket_guarded_import_probe()],
        capture_output=True,
        text=True,
        env=probe_environment,
        timeout=180,
    )


def test_importing_bot_opens_zero_connections_without_publish_opt_in(tmp_path: Path) -> None:
    """Importing the package never attempts network I/O unless publishing is opted in.

    Model definition runs at import time, so ``define`` must not publish implicitly:
    with an OTLP endpoint configured (publish path armed) and connections blocked,
    ``import mosfet`` succeeds without a single connection attempt. Setting
    ``BOT_MODEL_PUBLISH=1`` re-arms the explicit path (attempt recorded, still
    non-fatal to the import).
    """

    record_path = tmp_path / "connections.log"
    completed = _run_import_probe(publish_opt_in=False, record_path=record_path)
    assert completed.returncode == 0, completed.stderr
    assert not record_path.exists()

    opted_in = _run_import_probe(publish_opt_in=True, record_path=record_path)
    assert opted_in.returncode == 0, opted_in.stderr
    assert record_path.exists()
    assert record_path.read_text() != ""


def test_model_publish_opt_in_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model publishing is explicit: only ``BOT_MODEL_PUBLISH=1`` enables it."""

    from mosfet.define import model_publish_enabled

    monkeypatch.delenv("BOT_MODEL_PUBLISH", raising=False)
    assert model_publish_enabled() is False
    monkeypatch.setenv("BOT_MODEL_PUBLISH", "1")
    assert model_publish_enabled() is True
    monkeypatch.setenv("BOT_MODEL_PUBLISH", "0")
    assert model_publish_enabled() is False


# --- Lifecycle requests reach their activator ---------------------------------------------------
#
# Every accepted bot.activate / bot.deactivate is answered with exactly one public terminal that
# carries the request's operation id and is delivered to the bot and its reply target. Each test
# asserts the complete reply ledger, so an accepted request answered twice or not at all fails.

_FAST_TEARDOWN = datetime.timedelta(milliseconds=20)


class LifecycleAgent(Bot):
    """Bot double with injectable devices, abilities and lifecycle bound."""

    def __init__(
        self,
        devices: collections.abc.Sequence[Device] = (),
        *,
        cognition: abilities.Ability[typing.Any, typing.Any] | None = None,
        deactivation_timeout: datetime.timedelta | None = None,
    ) -> None:
        super().__init__(
            devices=devices,
            cognition=cognition or as_cognition(IgnoreAbility()),
            deactivation_timeout=deactivation_timeout,
        )


class LifecycleGates:
    """Holds or fails one bot's own attachment-group operations so a lifecycle state stays observable."""

    groups: list[attachment.Group]
    held: list[str]
    hold_attach: bool
    fail_attach: bool
    hold_detach: bool
    fail_detach: bool
    hold_stop: bool
    release: asyncio.Event

    def __init__(self, monkeypatch: pytest.MonkeyPatch, active_bot: Bot) -> None:
        self.groups = []
        self.held = []
        self.hold_attach = False
        self.fail_attach = False
        self.hold_detach = False
        self.fail_detach = False
        self.hold_stop = False
        self.release = asyncio.Event()
        original_attach = attachment.Group.attach
        original_detach = attachment.Group.detach
        original_stop = hsm.stop

        async def attach(group: attachment.Group, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> None:
            if isinstance(event.data, attachment.AttachData) and event.data.actor is active_bot:
                if not any(known is group for known in self.groups):
                    self.groups.append(group)
                if self.fail_attach:
                    raise RuntimeError("bot attach dispatch failed")
                if self.hold_attach:
                    self.held.append("attach")
                    _ = await self.release.wait()
            await original_attach(group, ctx, event)

        async def detach(group: attachment.Group, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> None:
            if isinstance(event.data, attachment.DetachData) and event.data.actor is active_bot:
                if self.fail_detach:
                    raise RuntimeError("bot detach dispatch failed")
                if self.hold_detach:
                    self.held.append("detach")
                    _ = await self.release.wait()
            await original_detach(group, ctx, event)

        async def stop(instance: hsm.Instance, ctx: hsm.Context | None = None) -> None:
            if self.hold_stop and any(known is instance for known in self.groups):
                self.held.append("stop")
                _ = await self.release.wait()
            await original_stop(instance, ctx)

        monkeypatch.setattr(attachment.Group, "attach", attach)
        monkeypatch.setattr(attachment.Group, "detach", detach)
        monkeypatch.setattr(hsm, "stop", stop)


async def request_cognition_reboot(
    environment: Environment,
    active_bot: Bot,
    cognition_ability: abilities.Ability[typing.Any, typing.Any],
) -> None:
    _ = await hsm.dispatch(
        environment,
        active_bot,
        dataclasses.replace(
            mosfet.RebootEvent.with_data(mosfet.RebootEventData(reason="cognition_child_teardown_failed")),
            id=f"reboot-{uuid.uuid4().hex}",
            source=hsm.id(cognition_ability),
            target=hsm.id(active_bot),
        ),
    )


async def settle() -> None:
    """Give a duplicate or late terminal the chance to arrive before the ledger is read."""

    await asyncio.sleep(0.02)


async def wait_for_state(active_bot: Bot, state: str) -> None:
    await wait_until(lambda: active_bot.state() == state)


async def hold_stop_in(
    state: str,
    gates: LifecycleGates,
    active_bot: Bot,
    environment: Environment,
    replies: LifecycleReplies,
    operation_id: str,
) -> list[tuple[str, str]]:
    """Accept stop ``operation_id`` and hold the bot in ``state``; return the ledger before the stop."""

    if state == "activation_cancelling":
        gates.hold_attach = True
        gates.hold_stop = True
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_until(lambda: gates.held == ["attach"])
        before: list[tuple[str, str]] = []
        expected_hold = "stop"
    else:
        gates.hold_detach = state == "deactivating"
        gates.hold_stop = state == "deactivation_cleanup"
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        before = [("bot.activated", "op-a")]
        expected_hold = "detach" if state == "deactivating" else "stop"
    _ = await active_bot.detach(environment, reply_to=replies, operation_id=operation_id)
    await wait_until(lambda: active_bot.state() == f"/Bot/{state}" and gates.held[-1:] == [expected_hold])
    return before


def test_bot_answers_activation_and_deactivation_to_the_reply_target_under_the_operation_id() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, str, str]:
        replies = await started_replies()
        active_bot = LifecycleAgent(configured_devices("phone"))
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-1")
        await wait_until(lambda: len(replies.terminals) == 1)
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-2")
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == 2)
        await settle()
        return replies.terminals, hsm.id(active_bot), hsm.id(replies), active_bot.state()

    terminals, bot_id, replies_id, state = asyncio.run(run())

    assert [(terminal.name, terminal.id) for terminal in terminals] == [
        ("bot.activated", "op-1"),
        ("bot.deactivated", "op-2"),
    ]
    assert all(terminal.source == bot_id and terminal.target == replies_id for terminal in terminals)
    assert state == "/Bot/inactive"


def test_bot_answers_failed_activation_to_the_reply_target_under_the_operation_id() -> None:
    async def run() -> tuple[list[tuple[str, str]], str]:
        replies = await started_replies()
        active_bot = LifecycleAgent((ImmediatelyFailingInitializingDevice(id="failing"),))
        _ = await active_bot.attach(Environment(), reply_to=replies, operation_id="op-1")
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == 1)
        await settle()
        return replies.ledger(), active_bot.state()

    ledger, state = asyncio.run(run())

    assert ledger == [("bot.activating.failed", "op-1")]
    assert state == "/Bot/inactive"


def test_bot_mints_an_operation_id_when_the_activator_names_none() -> None:
    async def run() -> list[tuple[str, str]]:
        replies = await started_replies()
        active_bot = LifecycleAgent()
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies)
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        _ = await active_bot.detach(environment, reply_to=replies)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == 2)
        return replies.ledger()

    ledger = asyncio.run(run())

    assert [name for name, _ in ledger] == ["bot.activated", "bot.deactivated"]
    assert all(operation_id for _, operation_id in ledger)
    assert ledger[0][1] != ledger[1][1]


def test_bot_answers_a_deactivation_requested_while_activating(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[tuple[str, str]], str]:
        replies = await started_replies()
        active_bot = LifecycleAgent(configured_devices("phone"))
        gates = LifecycleGates(monkeypatch, active_bot)
        gates.hold_attach = True
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_until(lambda: gates.held == ["attach"])
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-stop")
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == 1)
        await settle()
        return replies.ledger(), active_bot.state()

    ledger, state = asyncio.run(run())

    assert ledger == [("bot.deactivated", "op-stop")]
    assert state == "/Bot/inactive"


def test_bot_answers_a_deactivation_deferred_through_reboot_deactivation_and_preserve_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, list[tuple[str, str]], str]:
        replies = await started_replies()
        cognition_ability = as_cognition(IgnoreAbility())
        active_bot = LifecycleAgent(cognition=cognition_ability)
        gates = LifecycleGates(monkeypatch, active_bot)
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        gates.hold_detach = True
        await request_cognition_reboot(environment, active_bot, cognition_ability)
        await wait_until(lambda: gates.held == ["detach"])
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-stop")
        state_while_rebooting = active_bot.state()
        _ = gates.release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == 2)
        await settle()
        return state_while_rebooting, replies.ledger(), active_bot.state()

    state_while_rebooting, ledger, state = asyncio.run(run())

    assert state_while_rebooting == "/Bot/reboot_deactivating"
    # Re-offered in reboot_cleanup_preserve, then accepted on the reboot's activating entry.
    assert ledger == [("bot.activated", "op-a"), ("bot.deactivated", "op-stop")]
    assert state == "/Bot/inactive"


def test_bot_answers_a_deactivation_deferred_through_reboot_reset_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[str, list[tuple[str, str]], str]:
        replies = await started_replies()
        cognition_ability = as_cognition(IgnoreAbility())
        active_bot = LifecycleAgent(cognition=cognition_ability)
        gates = LifecycleGates(monkeypatch, active_bot)
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        gates.fail_detach = True
        gates.hold_stop = True
        await request_cognition_reboot(environment, active_bot, cognition_ability)
        await wait_until(lambda: gates.held == ["stop"])
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-stop")
        state_while_resetting = active_bot.state()
        gates.fail_detach = False
        _ = gates.release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == 2)
        await settle()
        return state_while_resetting, replies.ledger(), active_bot.state()

    state_while_resetting, ledger, state = asyncio.run(run())

    assert state_while_resetting == "/Bot/reboot_cleanup_reset"
    assert ledger == [("bot.activated", "op-a"), ("bot.deactivated", "op-stop")]
    assert state == "/Bot/inactive"


def test_bot_answers_a_deactivation_deferred_through_activation_cleanup_and_a_reboot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, list[tuple[str, str]], list[str], str]:
        replies = await started_replies()
        cognition_ability = as_cognition(IgnoreAbility())
        active_bot = LifecycleAgent(cognition=cognition_ability)
        gates = LifecycleGates(monkeypatch, active_bot)
        gates.fail_attach = True
        gates.hold_stop = True
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_until(lambda: gates.held == ["stop"])
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-stop")
        state_while_cleaning = active_bot.state()
        # The reboot's activation now holds its attach, so only the deferred stop can end it.
        gates.fail_attach = False
        gates.hold_stop = False
        gates.hold_attach = True
        gates.release = asyncio.Event()
        await request_cognition_reboot(environment, active_bot, cognition_ability)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == 2)
        await settle()
        return state_while_cleaning, replies.ledger(), gates.held, active_bot.state()

    state_while_cleaning, ledger, held, state = asyncio.run(run())

    assert state_while_cleaning == "/Bot/activation_cleanup"
    # Deferred through the reboot states and accepted on the reboot's activating entry.
    assert held == ["stop", "attach"]
    assert ledger == [("bot.activating.failed", "op-a"), ("bot.deactivated", "op-stop")]
    assert state == "/Bot/inactive"


def test_bot_answers_a_deactivation_while_inactive() -> None:
    async def run() -> tuple[list[tuple[str, str]], str]:
        replies = await started_replies()
        active_bot = LifecycleAgent()
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-b")
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == 2)
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-c")
        await wait_until(lambda: len(replies.terminals) == 3)
        await settle()
        return replies.ledger(), active_bot.state()

    ledger, state = asyncio.run(run())

    assert ledger == [("bot.activated", "op-a"), ("bot.deactivated", "op-b"), ("bot.deactivated", "op-c")]
    assert state == "/Bot/inactive"


def test_bot_degrades_when_deactivation_cleanup_times_out_and_answers_later_stops_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[tuple[str, str]], list[str], str]:
        replies = await started_replies()
        active_bot = LifecycleAgent(deactivation_timeout=_FAST_TEARDOWN)
        gates = LifecycleGates(monkeypatch, active_bot)
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        gates.hold_stop = True
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-stop")
        await wait_until(lambda: active_bot.state() == "/Bot/degraded" and len(replies.terminals) == 2)
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-again")
        await wait_until(lambda: len(replies.terminals) == 3)
        await settle()
        return replies.ledger(), replies.stages(), active_bot.state()

    ledger, stages, state = asyncio.run(run())

    assert ledger == [("bot.activated", "op-a"), ("bot.degraded", "op-stop"), ("bot.degraded", "op-again")]
    assert stages == ["deactivation_cleanup", "degraded"]
    assert state == "/Bot/degraded"


def test_bot_degrades_when_activation_rollback_cleanup_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[tuple[str, str]], list[str], str]:
        replies = await started_replies()
        active_bot = LifecycleAgent(deactivation_timeout=_FAST_TEARDOWN)
        gates = LifecycleGates(monkeypatch, active_bot)
        gates.fail_attach = True
        gates.hold_stop = True
        _ = await active_bot.attach(Environment(), reply_to=replies, operation_id="op-a")
        await wait_until(lambda: active_bot.state() == "/Bot/degraded" and len(replies.terminals) == 2)
        await settle()
        return replies.ledger(), replies.stages(), active_bot.state()

    ledger, stages, state = asyncio.run(run())

    assert ledger == [("bot.activating.failed", "op-a"), ("bot.degraded", "op-a")]
    assert stages == ["activation_cleanup"]
    assert state == "/Bot/degraded"


def test_bot_degrades_when_cancelling_an_activation_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[tuple[str, str]], list[str], str]:
        replies = await started_replies()
        active_bot = LifecycleAgent(deactivation_timeout=_FAST_TEARDOWN)
        gates = LifecycleGates(monkeypatch, active_bot)
        gates.hold_attach = True
        gates.hold_stop = True
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_until(lambda: gates.held == ["attach"])
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-stop")
        await wait_until(lambda: active_bot.state() == "/Bot/degraded" and len(replies.terminals) == 1)
        await settle()
        return replies.ledger(), replies.stages(), active_bot.state()

    ledger, stages, state = asyncio.run(run())

    assert ledger == [("bot.degraded", "op-stop")]
    assert stages == ["activation_cancel"]
    assert state == "/Bot/degraded"


def test_bot_degrades_when_reboot_cleanup_times_out_under_the_last_activation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[tuple[str, str]], list[str], str]:
        replies = await started_replies()
        cognition_ability = as_cognition(IgnoreAbility())
        active_bot = LifecycleAgent(cognition=cognition_ability, deactivation_timeout=_FAST_TEARDOWN)
        gates = LifecycleGates(monkeypatch, active_bot)
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        gates.fail_detach = True
        gates.hold_stop = True
        await request_cognition_reboot(environment, active_bot, cognition_ability)
        await wait_until(lambda: active_bot.state() == "/Bot/degraded" and len(replies.terminals) == 2)
        await settle()
        return replies.ledger(), replies.stages(), active_bot.state()

    ledger, stages, state = asyncio.run(run())

    assert ledger == [("bot.activated", "op-a"), ("bot.degraded", "op-a")]
    assert stages == ["reboot_cleanup"]
    assert state == "/Bot/degraded"


def _undelivered_count(metrics: metric_reader.Reader, terminal: str) -> float:
    return sum(
        point.value
        for point in metrics.points("bot.lifecycle.answer.undelivered.count")
        if point.attributes == {"terminal": terminal}
    )


def test_a_reboot_after_its_activator_stopped_reactivates_and_counts_the_answer_nobody_heard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Willy live check (D2): ``bot.start`` waits for activation with a one-shot reply target that stops once
    answered. A cognition reboot later re-entered activation, and answering that stopped target raised, so the
    body never became active again. A reply target hears the answer only while it runs: the body is active
    again, still answers what it is asked next, and the answer nobody heard is counted."""

    async def run() -> tuple[str, list[tuple[str, str]], list[tuple[str, str]]]:
        replies = await started_replies()
        cognition_ability = as_cognition(IgnoreAbility())
        active_bot = LifecycleAgent(cognition=cognition_ability)
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        await hsm.stop(replies, hsm.Context())
        await request_cognition_reboot(environment, active_bot, cognition_ability)
        await wait_until(lambda: active_bot.state() != "/Bot/active/unfocused")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        later = await started_replies()
        _ = await active_bot.detach(environment, reply_to=later, operation_id="op-stop")
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(later.terminals) == 1)
        await settle()
        return active_bot.state(), replies.ledger(), later.ledger()

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _undelivered_count(metrics, mosfet.ActivatingDoneEvent.name)
        state, stopped, later = asyncio.run(run())
        after = _undelivered_count(metrics, mosfet.ActivatingDoneEvent.name)

    assert state == "/Bot/inactive"
    assert stopped == [("bot.activated", "op-a")]
    assert later == [("bot.deactivated", "op-stop")]
    assert after - before == 1


def test_bot_degrades_when_processing_cancellation_times_out_under_the_last_activation_id() -> None:
    async def run() -> tuple[list[tuple[str, str]], list[str], str]:
        replies = await started_replies()
        processor = CancellableHangingProcessor()
        active_bot = TimeoutAbilityAgent(
            devices=configured_devices("phone"),
            cognition=CapturingCognition(processor, swallow_cancel=True),
        )
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: active_bot.state() == "/Bot/degraded" and len(replies.terminals) == 2)
        await settle()
        return replies.ledger(), replies.stages(), active_bot.state()

    ledger, stages, state = asyncio.run(run())

    assert ledger == [("bot.activated", "op-a"), ("bot.degraded", "op-a")]
    assert stages == ["processing_cancel"]
    assert state == "/Bot/degraded"


def test_bot_keeps_the_operation_id_when_the_detach_answer_times_out() -> None:
    class NeverDetachingDevice(Device):
        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
            if event.name == attachment.DetachEvent.name:

                async def hang() -> bool:
                    return await asyncio.Event().wait()

                return hang()
            return super().dispatch(ctx, event)

    async def run() -> tuple[list[tuple[str, str]], str]:
        replies = await started_replies()
        # Long enough for the cleanup that follows the detach timer to finish inside its own bound.
        active_bot = LifecycleAgent(
            (NeverDetachingDevice(id="stuck"),),
            deactivation_timeout=datetime.timedelta(milliseconds=200),
        )
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-3")
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == 2)
        await settle()
        return replies.ledger(), active_bot.state()

    ledger, state = asyncio.run(run())

    # deactivating -after-> deactivation_cleanup: the timer event's own id never reaches the answer.
    assert ledger == [("bot.activated", "op-a"), ("bot.deactivated", "op-3")]
    assert state == "/Bot/inactive"


@pytest.mark.parametrize("stopping_state", ["activation_cancelling", "deactivating", "deactivation_cleanup"])
def test_cognition_reboot_never_turns_an_accepted_stop_back_into_an_activation(
    monkeypatch: pytest.MonkeyPatch,
    stopping_state: str,
) -> None:
    async def run() -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
        replies = await started_replies()
        cognition_ability = as_cognition(IgnoreAbility())
        active_bot = LifecycleAgent(configured_devices("phone"), cognition=cognition_ability)
        gates = LifecycleGates(monkeypatch, active_bot)
        environment = Environment()
        before = await hold_stop_in(stopping_state, gates, active_bot, environment, replies, "op-stop")
        await request_cognition_reboot(environment, active_bot, cognition_ability)
        _ = gates.release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == len(before) + 1)
        await settle()
        return before, replies.ledger(), active_bot.state()

    before, ledger, state = asyncio.run(run())

    assert ledger == [*before, ("bot.deactivated", "op-stop")]
    assert state == "/Bot/inactive"


@pytest.mark.parametrize("stopping_state", ["activation_cancelling", "deactivating", "deactivation_cleanup"])
def test_a_superseding_deactivation_is_answered_by_the_running_stop_without_a_timer(
    monkeypatch: pytest.MonkeyPatch,
    stopping_state: str,
) -> None:
    async def run() -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
        replies = await started_replies()
        # The default teardown bound is minutes: only the running step's own terminal can end
        # this stop inside wait_until's deadline.
        active_bot = LifecycleAgent(configured_devices("phone"))
        gates = LifecycleGates(monkeypatch, active_bot)
        environment = Environment()
        before = await hold_stop_in(stopping_state, gates, active_bot, environment, replies, "op-4")
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-5")
        _ = gates.release.set()
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == len(before) + 1)
        await settle()
        return before, replies.ledger(), active_bot.state()

    before, ledger, state = asyncio.run(run())

    assert ledger == [*before, ("bot.deactivated", "op-5")]
    assert state == "/Bot/inactive"


def _degraded_count(metrics: metric_reader.Reader, stage: str) -> float:
    return sum(
        point.value for point in metrics.points("bot.lifecycle.degraded.count") if point.attributes == {"stage": stage}
    )


def test_bot_counts_degraded_transitions_by_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def run() -> None:
        replies = await started_replies()
        active_bot = LifecycleAgent(deactivation_timeout=_FAST_TEARDOWN)
        gates = LifecycleGates(monkeypatch, active_bot)
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-a")
        await wait_for_state(active_bot, "/Bot/active/unfocused")
        gates.hold_stop = True
        _ = await active_bot.detach(environment, reply_to=replies, operation_id="op-stop")
        await wait_until(lambda: active_bot.state() == "/Bot/degraded" and len(replies.terminals) == 2)

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _degraded_count(metrics, "deactivation_cleanup")
        asyncio.run(run())
        after = _degraded_count(metrics, "deactivation_cleanup")

    assert after - before == 1


def test_body_rebuild_carries_a_products_stamped_participants_into_cognition() -> None:
    """The body copies who the producer stamped; it never works out who took part itself."""

    ada = mosfet.Participant(ref="ada", role=mosfet.ParticipantRole.SOURCE)

    async def run() -> list[cognition.InputData]:
        ability = IgnoreAbility()
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=(), cognition=cognitive, input=(RecordingListening(),))
        environment = await start_bot_with_devices(active_bot)
        heard = SoundData(
            audio=b"heard-chunk", media_type="audio/pcm", sample_rate_hz=48_000, channels=1, participants=(ada,)
        )
        await active_bot.dispatch(active_bot.context(), SoundEvent.with_data(heard))
        await close_hearing_speech(active_bot, environment, sample_rate_hz=48_000)
        await wait_until(lambda: len(ability.calls) == 1)
        inputs = list(cognitive.inputs)
        _ = await active_bot.detach(environment)
        return inputs

    (handed,) = asyncio.run(run())

    assert handed.participants == (ada,)


def test_body_rebuild_carries_a_device_reports_stamped_participants_into_cognition() -> None:
    """Live (builder check round 3, G1): a device report's stamped participants reach cognition unchanged."""

    ada = mosfet.Participant(ref="ada", role=mosfet.ParticipantRole.SOURCE)

    async def run() -> list[cognition.InputData]:
        ability = IgnoreAbility()
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognitive)
        environment = await start_bot_with_devices(active_bot)
        _ = await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", participants=(ada,))),
        )
        await wait_until(lambda: len(ability.calls) == 1)
        inputs = list(cognitive.inputs)
        _ = await active_bot.detach(environment)
        return inputs

    (handed,) = asyncio.run(run())

    assert handed.participants == (ada,)


def test_body_rebuild_of_a_device_report_names_no_participants() -> None:
    async def run() -> list[cognition.InputData]:
        ability = IgnoreAbility()
        cognitive = InputRecordingCognition(ability)
        active_bot = AbilityAgent(devices=configured_devices("phone"), cognition=cognitive)
        environment = await start_bot_with_devices(active_bot)
        await active_bot.dispatch(
            active_bot.context(),
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=3)),
        )
        await wait_until(lambda: len(ability.calls) == 1)
        inputs = list(cognitive.inputs)
        _ = await active_bot.detach(environment)
        return inputs

    (handed,) = asyncio.run(run())

    assert handed.participants == ()


# The body's one firmware (DSL-14, DSL-17, DSL-20): owned, started last, stopped first, shown and offered.


class _ChimeData(pydantic.BaseModel):
    """Chime once."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")


_ChimeEvent = hsm.Event[_ChimeData](name="tests.bot.firmware.chime", kind=event_contract.EventKind, schema=_ChimeData)


def _firmware_started(ctx: hsm.Context, instance: "RecordingFirmware", event: hsm.Event[typing.Any]) -> None:
    del ctx, event
    instance.started_with.append(tuple(bool(device_bots(device)) for device in instance.devices))


def _firmware_stopped(ctx: hsm.Context, instance: "RecordingFirmware", event: hsm.Event[typing.Any]) -> None:
    del ctx, event
    instance.stopped_with.append(tuple(bool(device_bots(device)) for device in instance.devices))


def _chime(ctx: hsm.Context, instance: "RecordingFirmware", event: hsm.Event[typing.Any]) -> None:
    del ctx
    instance.chimes.append(event)
    (count, _) = typing.cast(tuple[int, bool], instance.get("chimes"))
    _ = instance.set("chimes", count + 1)


_RECORDING_FIRMWARE_MODEL = mosfet.define(
    "TestFirmware",
    hsm.attribute("chimes", int, 0),
    hsm.initial(hsm.target("idle")),
    hsm.state(
        "idle",
        hsm.entry(_firmware_started),
        hsm.exit(_firmware_stopped),
        hsm.transition(hsm.on(_ChimeEvent), hsm.effect(_chime), hsm.target("../chimed")),
    ),
    hsm.state("chimed", hsm.exit(_firmware_stopped)),
)


class RecordingFirmware(hsm.Instance):
    """Machinery an owner wrote for a bot: records whether the bot's devices were held when it started and stopped."""

    model: hsm.Model
    devices: tuple[Device, ...]
    started_with: list[tuple[bool, ...]]
    stopped_with: list[tuple[bool, ...]]
    chimes: list[hsm.Event[typing.Any]]

    def __init__(self, *devices: Device, model: hsm.Model = _RECORDING_FIRMWARE_MODEL) -> None:
        super().__init__()
        self.model = model
        self.devices = devices
        self.started_with = []
        self.stopped_with = []
        self.chimes = []


class FailingFirmware(RecordingFirmware):
    """Firmware whose start raises."""

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        del ctx, data
        raise RuntimeError("firmware failed")


class HangingFirmware(RecordingFirmware):
    """Firmware whose start never answers."""

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        del ctx, data
        _ = await asyncio.Event().wait()
        return self


class FirmwareAgent(Bot):
    """A body with one device, one ignoring cognition and the firmware it is given."""

    def __init__(
        self,
        device: Device,
        firmware: hsm.Instance,
        *,
        cognition: abilities.Ability[typing.Any, typing.Any] | processing.Processing | None = None,
        attachment_timeout: datetime.timedelta | None = None,
    ) -> None:
        super().__init__(
            devices=(device,),
            cognition=as_cognition(cognition or IgnoreAbility()),
            attachment_timeout=attachment_timeout,
            firmware=firmware,
        )


async def start_firmware_bot(active_bot: Bot) -> Environment:
    """Start ``active_bot`` and wait past its firmware step, until it is active."""

    environment = await start_bot_with_devices(active_bot)
    await wait_until(lambda: active_bot.state().startswith("/Bot/active"))
    return environment


def test_bot_starts_its_firmware_once_its_parts_are_attached_and_declares_it_on_its_snapshot() -> None:
    """DSL-20: the firmware is the last activation step, run in the body's environment."""

    async def run() -> tuple[str, list[tuple[bool, ...]], object, str, bool, str | None]:
        desk = Device(id="desk")
        firmware = RecordingFirmware(desk)
        active_bot = FirmwareAgent(desk, firmware)
        environment = await start_firmware_bot(active_bot)
        try:
            attributes = active_bot.take_snapshot().Attributes or {}
            return (
                active_bot.state(),
                list(firmware.started_with),
                attributes.get("/Bot/firmware"),
                hsm.id(firmware),
                environment.contains(firmware),
                environment.model_snapshot(active_bot),
            )
        finally:
            _ = await active_bot.detach(environment)

    state, started_with, declared, runtime_id, in_environment, world = asyncio.run(run())

    assert state == "/Bot/active/unfocused"
    assert started_with == [(True,)]
    assert declared == runtime_id
    assert in_environment
    assert world is not None
    assert f'<firmware ref="firmware" id="{runtime_id}" state="/TestFirmware/idle">' in world
    assert "<chimes>0</chimes>" in world


def test_bot_offers_its_firmwares_commands_under_firmware_while_a_state_reacts_to_them() -> None:
    """DSL-17: cognition is offered the firmware's commands under the reserved key, and a selection reaches it."""

    async def run() -> tuple[list[processing.InputData], RecordingFirmware, list[hsm.Event[typing.Any]], str]:
        ability = IgnoreAbility()
        desk = Device(id="desk")
        firmware = RecordingFirmware(desk)
        active_bot = FirmwareAgent(desk, firmware, cognition=ability)
        environment = await start_firmware_bot(active_bot)
        try:
            _ = await active_bot.dispatch(
                active_bot.context(),
                mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="desk", priority=3)),
            )
            await wait_until(lambda: len(ability.calls) == 1)
            _ = await hsm.dispatch(environment, firmware, _ChimeEvent.with_data(_ChimeData()))
            await wait_until(lambda: firmware.state() == "/TestFirmware/chimed")
            _ = await active_bot.dispatch(
                active_bot.context(),
                mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="desk", priority=3)),
            )
            await wait_until(lambda: len(ability.calls) == 2)
            return ability.calls, firmware, list(firmware.chimes), firmware.state()
        finally:
            _ = await active_bot.detach(environment)

    calls, firmware, chimes, state = asyncio.run(run())

    assert calls[0].actors["firmware"] is firmware
    assert _ChimeEvent.name in {schema.name for schema in calls[0].schemas}
    assert calls[0].actor_events[_ChimeEvent.name] == ("firmware",)
    assert len(chimes) == 1
    assert state == "/TestFirmware/chimed"
    # Once no active state reacts to it, the command is no longer offered.
    assert _ChimeEvent.name not in {schema.name for schema in calls[1].schemas}


def test_bot_without_firmware_offers_and_declares_none() -> None:
    async def run() -> tuple[list[processing.InputData], dict[str, object]]:
        ability = IgnoreAbility()
        active_bot = AbilityAgent(devices=configured_devices("desk"), cognition=ability)
        environment = await start_firmware_bot(active_bot)
        try:
            _ = await active_bot.dispatch(
                active_bot.context(),
                mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="desk", priority=3)),
            )
            await wait_until(lambda: len(ability.calls) == 1)
            return ability.calls, dict(active_bot.take_snapshot().Attributes or {})
        finally:
            _ = await active_bot.detach(environment)

    calls, attributes = asyncio.run(run())

    assert "firmware" not in calls[0].actors
    assert attributes.get("/Bot/firmware") is None


@pytest.mark.parametrize(
    ("firmware_type", "outcome"),
    [(FailingFirmware, "failed"), (HangingFirmware, "timeout")],
    ids=["start raises", "start never answers"],
)
def test_bot_activation_fails_and_rolls_back_when_its_firmware_does_not_start(
    firmware_type: type[RecordingFirmware], outcome: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DSL-20: a firmware that fails or does not start in the attachment bound fails the activation."""

    async def run() -> tuple[list[tuple[str, str]], str, bool]:
        replies = await started_replies()
        desk = Device(id="desk")
        active_bot = FirmwareAgent(desk, firmware_type(desk), attachment_timeout=datetime.timedelta(milliseconds=500))
        environment = Environment()
        _ = await active_bot.attach(environment, reply_to=replies, operation_id="op-start")
        await wait_until(lambda: active_bot.state() == "/Bot/inactive" and len(replies.terminals) == 1)
        return replies.ledger(), active_bot.state(), mosfet.lifecycle.is_started(desk)

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = metrics.points("bot.firmware.start.count")
        ledger, state, desk_running = asyncio.run(run())
        after = metrics.points("bot.firmware.start.count")

    assert ledger == [("bot.activating.failed", "op-start")]
    assert state == "/Bot/inactive"
    assert not desk_running

    def total(points: tuple[metric_reader.Point, ...], label: str) -> float:
        return sum(point.value for point in points if point.attributes == {"bot.outcome": label})

    assert total(after, outcome) - total(before, outcome) == 1
    assert {key for point in after for key in point.attributes} == {"bot.outcome"}


def test_bot_counts_a_firmware_that_starts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def run() -> None:
        desk = Device(id="desk")
        active_bot = FirmwareAgent(desk, RecordingFirmware(desk))
        environment = await start_firmware_bot(active_bot)
        _ = await active_bot.detach(environment)

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = metrics.points("bot.firmware.start.count")
        asyncio.run(run())
        after = metrics.points("bot.firmware.start.count")

    def total(points: tuple[metric_reader.Point, ...]) -> float:
        return sum(point.value for point in points if point.attributes == {"bot.outcome": "ok"})

    assert total(after) - total(before) == 1


def test_bot_stops_its_firmware_first_and_starts_it_fresh_when_it_activates_again() -> None:
    """DSL-20: deactivation stops the firmware while its parts are still held; activation restarts it from fresh."""

    async def run() -> tuple[list[tuple[bool, ...]], list[tuple[bool, ...]], str, str, object]:
        desk = Device(id="desk")
        firmware = RecordingFirmware(desk)
        active_bot = FirmwareAgent(desk, firmware)
        environment = await start_firmware_bot(active_bot)
        first = hsm.id(firmware)
        _ = await hsm.dispatch(environment, firmware, _ChimeEvent.with_data(_ChimeData()))
        await wait_until(lambda: firmware.state() == "/TestFirmware/chimed")
        _ = await active_bot.detach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/inactive")
        _ = await active_bot.attach(environment)
        await wait_until(lambda: active_bot.state() == "/Bot/active/unfocused")
        try:
            (chimes, _) = typing.cast(tuple[int, bool], firmware.get("chimes"))
            return list(firmware.stopped_with), list(firmware.started_with), first, hsm.id(firmware), chimes
        finally:
            _ = await active_bot.detach(environment)

    stopped_with, started_with, first, second, chimes = asyncio.run(run())

    assert stopped_with[0] == (True,)
    assert started_with == [(True,), (True,)]
    assert first != second
    assert chimes == 0


def test_stopping_a_bot_stops_its_firmware_before_its_parts() -> None:
    async def run() -> tuple[list[tuple[bool, ...]], bool]:
        desk = Device(id="desk")
        firmware = RecordingFirmware(desk)
        active_bot = FirmwareAgent(desk, firmware)
        environment = await start_firmware_bot(active_bot)
        await active_bot.stop(environment)
        return list(firmware.stopped_with), mosfet.lifecycle.is_started(firmware)

    stopped_with, running = asyncio.run(run())

    assert stopped_with == [(True,)]
    assert not running


class _NoticedData(pydantic.BaseModel):
    """The firmware noticed something as it started."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


_NoticedEvent = hsm.Event[_NoticedData](name="tests.bot.firmware.noticed", schema=_NoticedData)


def _hand_over_on_start(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
    """What a firmware reporting as it starts does: hand the body that started it a product."""

    del event
    found = instance.context().value(hsm.Keys.Owner)
    assert isinstance(found, hsm.HSM)
    # The owner key holds the starting machine's runtime; which instance it runs is not typed.
    owner = typing.cast(hsm.HSM[typing.Any], found)
    stimulus = dataclasses.replace(
        _NoticedEvent.with_data(_NoticedData()), id=uuid.uuid4().hex, source=hsm.id(instance)
    )
    _ = hsm.dispatch(
        ctx,
        owner,
        dataclasses.replace(
            cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=stimulus), uuid.uuid4().hex),
            source=hsm.id(instance),
            target=owner.id,
        ),
    )


class ReportingOnStartFirmware(RecordingFirmware):
    """Firmware that hands the body a product while the body is still starting it."""

    def __init__(self, *devices: Device) -> None:
        super().__init__(
            *devices,
            model=mosfet.define(
                "ReportingFirmware",
                hsm.initial(hsm.target("on")),
                hsm.state("on", hsm.entry(_hand_over_on_start)),
            ),
        )


def test_a_product_handed_over_while_the_firmware_starts_waits_for_the_body_to_be_active() -> None:
    async def run() -> list[processing.InputData]:
        ability = IgnoreAbility()
        desk = Device(id="desk")
        active_bot = FirmwareAgent(desk, ReportingOnStartFirmware(desk), cognition=ability)
        environment = await start_firmware_bot(active_bot)
        try:
            await wait_until(lambda: len(ability.calls) == 1)
            return list(ability.calls)
        finally:
            _ = await active_bot.detach(environment)

    (call,) = asyncio.run(run())

    assert isinstance(call.input, hsm.Event)
    assert typing.cast(hsm.Event[typing.Any], call.input).name == _NoticedEvent.name
