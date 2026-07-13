from bot.abilities import cognition

import abc
import collections.abc
import dataclasses
import datetime
import typing
import weakref

import hsm
import pydantic

from bot import abilities
from bot.protocols import attachment
from . import events

from bot.device import Device
from bot.telemetry import observer
from bot.world import SoundEvent, VisualEvent, World, require_world_scope

_DEFAULT_BOT_PROCESSING_TIMEOUT = datetime.timedelta(minutes=5)
_DEFAULT_BOT_ACTIVATION_ROLLBACK_TIMEOUT = datetime.timedelta(minutes=5)
_DEFAULT_BOT_DEACTIVATION_TIMEOUT = datetime.timedelta(minutes=5)
_FOCUS_CANDIDATES_METADATA_KEY = "bot.focus_candidates"
_PROCESSING_OPERATION_METADATA_KEY = "bot.processing.operation"


_BotProcessingChildCancelledEvent = hsm.Event[object](
    name="bot.processing.child.cancelled",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)


class _BotActivationState(pydantic.BaseModel):
    """Private bot field: device join/rollback bookkeeping during activation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    pending_devices: tuple[str, ...]
    attached_devices: tuple[str, ...] = ()
    started_devices: tuple[Device, ...] = ()


_BotActivationRollbackReadyEvent = hsm.Event[object](
    name="bot.activation.rollback.ready",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)


class _BotActivationRollbackFailedEventData(pydantic.BaseModel):
    """Private signal that activation rollback cleanup could not complete."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    message: str


_BotActivationRollbackFailedEvent = hsm.Event[_BotActivationRollbackFailedEventData](
    name="bot.activation.rollback.failed",
    kind=hsm.ErrorEventKind,
    schema=_BotActivationRollbackFailedEventData,
)


class _BotDeactivatingFailedEventData(pydantic.BaseModel):
    """Private signal that bot deactivation cleanup failed or timed out."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    message: str


_BotDeactivatingFailedEvent = hsm.Event[_BotDeactivatingFailedEventData](
    name="bot.deactivating.failed",
    kind=hsm.ErrorEventKind,
    schema=_BotDeactivatingFailedEventData,
)


def _device_tree(*roots: Device) -> tuple[Device, ...]:
    return Device.device_tree(*roots)


def _instance_is_started(instance: hsm.Instance) -> bool:
    """Machine liveness via ``instance.state()`` (HSM-CONTEXT-001), not snapshots or ``is_done()``."""

    state = instance.state()
    if not state:
        return False
    model = getattr(instance, "model", None)
    root = getattr(model, "qualified_name", None)
    # Root-only qualified name means the machine is not in a region (unstarted/stopped).
    return not (isinstance(root, str) and state == root)


def _device_model_is_running(device: Device) -> bool:
    return _instance_is_started(device)


def _ability_attach_context(lifetime: hsm.Context) -> hsm.Context:
    """Parent abilities under bot lifetime without joining the world broadcast set.

    World ``dispatch_all`` delivers to every instance in the world's Instances map. Input
    abilities must only receive ``world.sound`` / ``world.visual`` via the bot's parallel
    fan-out, not a second direct world delivery.
    """

    return hsm.Context(parent=lifetime, values={hsm.Keys.Instances: weakref.WeakValueDictionary()})


def _instance_id(instance: hsm.Instance) -> str:
    """Stable HSM instance id; empty only when the machine is not started yet."""

    if not _instance_is_started(instance):
        return ""
    return hsm.id(instance)


def _event_map_for_model(model: object) -> dict[str, hsm.Event[typing.Any]]:
    events = getattr(model, "events", None)
    if not isinstance(events, collections.abc.Mapping):
        return {}
    event_map = typing.cast(collections.abc.Mapping[object, object], events)
    return {str(name): event for name, event in event_map.items() if isinstance(event, hsm.Event)}


def _device_event_map(device: Device) -> dict[str, hsm.Event[typing.Any]]:
    event_map = _event_map_for_model(getattr(device, "model", None))
    event_map.update(_event_map_for_model(getattr(device, "firmware_model", None)))
    return event_map


def _focus_candidates_for_event(event: hsm.Event[typing.Any]) -> tuple[str, ...]:
    value = event.metadata.get(_FOCUS_CANDIDATES_METADATA_KEY)
    if not isinstance(value, collections.abc.Sequence) or isinstance(value, str | bytes | bytearray):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


class Bot(hsm.Instance, abc.ABC):
    """Interrupt-driven bot that observes and processes events while active."""

    _innate_abilities: typing.ClassVar[tuple[type[abilities.Ability[typing.Any, typing.Any]], ...]] = ()
    _processing_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_PROCESSING_TIMEOUT
    _activation_rollback_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_ACTIVATION_ROLLBACK_TIMEOUT
    _deactivation_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_DEACTIVATION_TIMEOUT
    _devices: dict[str, Device]
    _cognition: abilities.Ability[cognition.InputData, typing.Any]
    _focused_device: str | None
    _innate_ability_instances: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _acquired_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _input: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _output: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    # Private activation join scratch. Processing correlation is event metadata only.
    _activation_join: _BotActivationState | None

    @abc.abstractmethod
    def __init__(
        self,
        devices: collections.abc.Mapping[str, Device],
        *,
        cognition: abilities.Ability[cognition.InputData, typing.Any],
        input: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        output: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        acquired_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
    ) -> None:
        super().__init__()
        if self._processing_timeout <= datetime.timedelta():
            raise ValueError("processing_timeout must be positive.")
        if self._activation_rollback_timeout <= datetime.timedelta():
            raise ValueError("activation_rollback_timeout must be positive.")
        if self._deactivation_timeout <= datetime.timedelta():
            raise ValueError("deactivation_timeout must be positive.")
        self._devices = dict(devices)
        self._cognition = cognition
        self._focused_device = None
        self._innate_ability_instances = tuple(ability_type() for ability_type in self._innate_abilities)
        self._input = tuple(input)
        self._output = tuple(output)
        self._acquired_abilities = tuple(acquired_abilities)
        self._activation_join = None

    async def attach(self, world: World) -> typing.Self:
        require_world_scope(world, self, participant="Bot")
        if not _instance_is_started(self):
            _ = await hsm.started(world.context, self, self.model)
        await self.dispatch(world.context, events.ActivateEvent.with_data(events.ActivateEventData()))
        return self

    async def detach(self, world: World) -> typing.Self:
        if not _instance_is_started(self):
            return self
        require_world_scope(world, self, participant="Bot")
        await self.dispatch(world.context, events.DeactivateEvent.with_data(events.DeactivateEventData()))
        return self

    @staticmethod
    def _lifecycle_abilities(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
        return (
            instance._cognition,
            *instance._input,
            *instance._output,
            *instance._innate_ability_instances,
            *instance._acquired_abilities,
        )

    @staticmethod
    async def _start_abilities(ctx: hsm.Context, instance: "Bot") -> None:
        del ctx
        ability_scope = _ability_attach_context(instance.context())
        lifecycle_abilities = Bot._lifecycle_abilities(instance)
        for ability in lifecycle_abilities:
            model = ability.model
            if model is None:
                continue
            _ = await hsm.started(ability_scope, ability, model)
        await hsm.Group(*lifecycle_abilities, ctx=ability_scope).dispatch(
            ability_scope,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=instance)),
        )

    @staticmethod
    async def _stop_abilities(ctx: hsm.Context, instance: "Bot") -> None:
        del ctx
        lifetime = instance.context()
        lifecycle_abilities = Bot._lifecycle_abilities(instance)
        group = hsm.Group(*lifecycle_abilities)
        await group.dispatch(
            lifetime,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=instance)),
        )
        await group.stop(lifetime)

    @staticmethod
    async def _deactivate_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        del event
        lifetime = instance.context()
        world = World.from_context(lifetime)
        try:
            detached_devices: set[int] = set()
            for device in instance._devices.values():
                identifier = id(device)
                if identifier in detached_devices:
                    continue
                detached_devices.add(identifier)
                if not _device_model_is_running(device):
                    continue
                await hsm.Instance.dispatch(
                    device,
                    world.context,
                    attachment.DetachEvent.with_data(attachment.DetachData(actor=instance)),
                )
            await Bot._stop_abilities(lifetime, instance)
        except Exception as error:
            _ = instance.dispatch(
                ctx,
                _BotDeactivatingFailedEvent.with_data(
                    _BotDeactivatingFailedEventData(message=f"Bot deactivation failed: {error}")
                ),
            )
            return
        _ = instance.dispatch(ctx, events.DeactivatingDoneEvent.with_data(events.DeactivatingDoneEventData()))

    @staticmethod
    def _device_reference_for_source(instance: "Bot", source: str) -> str | None:
        for reference, device in instance._devices.items():
            if any(_instance_id(candidate) == source for candidate in _device_tree(device)):
                return reference
        return None

    @staticmethod
    def _device_references_for_event(instance: "Bot", event: hsm.Event[typing.Any]) -> tuple[str, ...]:
        source_reference = Bot._device_reference_for_source(instance, event.source) if event.source else None
        if source_reference is not None:
            return (source_reference,)
        return tuple(
            reference for reference, device in instance._devices.items() if event.name in _device_event_map(device)
        )

    @staticmethod
    def _target_device_reference(instance: "Bot", input: events.BotInputData) -> str | None:
        if isinstance(input, events.InputEventData):
            return input.target_device
        references = Bot._device_references_for_event(instance, input)
        return references[0] if len(references) == 1 else None

    @staticmethod
    def _input_device_references(instance: "Bot", input: events.BotInputData) -> tuple[str, ...]:
        if isinstance(input, events.InputEventData):
            return (input.target_device,) if input.target_device in instance._devices else ()
        references = Bot._device_references_for_event(instance, input)
        if not input.source and instance._focused_device in references:
            assert instance._focused_device is not None
            return (instance._focused_device,)
        return references

    @staticmethod
    def _processing_device_references(instance: "Bot", input: events.BotInputData) -> tuple[str, ...]:
        references: list[str] = []
        if instance._focused_device in instance._devices:
            assert instance._focused_device is not None
            references.append(instance._focused_device)
        for target_device in Bot._input_device_references(instance, input):
            if target_device not in references:
                references.append(target_device)
        return tuple(references)

    @staticmethod
    def _input_targets_configured_device(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, events.InputEventData) and event.data.target_device in instance._devices

    @staticmethod
    def _fan_out_input(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        for ability in instance._input:
            _ = hsm.dispatch(
                ctx,
                ability,
                dataclasses.replace(event, target=hsm.id(ability), metadata=dict(event.metadata)),
            )

    @staticmethod
    def _has_focused_device(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._focused_device in instance._devices

    @staticmethod
    def _ability_selected_configured_focus_device(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        if not isinstance(event.data, events.FocusDeviceEventData):
            return False
        return event.data.device in _focus_candidates_for_event(event)

    @staticmethod
    def _ability_selected_clear_focus(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, events.ClearFocusEventData) and bool(_focus_candidates_for_event(event))

    @staticmethod
    def _processing_completed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, events.ProcessingCompletedEventData)

    @staticmethod
    def _processing_failed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, events.ProcessingFailedEventData)

    @staticmethod
    def _processing_completed_with_focus(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        return Bot._processing_completed(ctx, instance, event) and Bot._has_focused_device(ctx, instance, event)

    @staticmethod
    def _processing_failed_with_focus(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        return Bot._processing_failed(ctx, instance, event) and Bot._has_focused_device(ctx, instance, event)

    @staticmethod
    def _bot_processing_timeout(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._processing_timeout

    @staticmethod
    def _bot_activation_rollback_timeout(
        ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, event
        return instance._activation_rollback_timeout

    @staticmethod
    def _bot_deactivation_timeout(
        ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, event
        return instance._deactivation_timeout

    @staticmethod
    def _dispatch_actors(instance: "Bot") -> dict[str, hsm.Instance]:
        actors: dict[str, hsm.Instance] = {"bot": instance, **instance._devices}
        for ability in (
            *instance._input,
            *instance._output,
            *instance._innate_ability_instances,
            *instance._acquired_abilities,
        ):
            name = type(ability).__name__
            chars: list[str] = []
            for index, char in enumerate(name):
                if (
                    char.isupper()
                    and index > 0
                    and (name[index - 1].islower() or (index + 1 < len(name) and name[index + 1].islower()))
                ):
                    chars.append("_")
                chars.append(char.lower())
            key = "".join(chars) or "actor"
            if key in actors:
                suffix = 2
                while f"{key}_{suffix}" in actors:
                    suffix += 1
                key = f"{key}_{suffix}"
            actors[key] = ability
        return actors

    @staticmethod
    def _dispatch_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        if isinstance(event.data, events.InputEventData):
            stimulus = event.data
        elif isinstance(event.data, cognition.InputData):
            stimulus = event.data.stimulus
        else:
            raise AssertionError(f"unsupported body processing event data: {type(event.data)!r}")
        focus_candidates = Bot._processing_device_references(instance, stimulus)
        cognition_input = cognition.InputData(
            stimulus=stimulus,
            abilities=Bot._lifecycle_abilities(instance),
            actors=Bot._dispatch_actors(instance),
            focus=instance._focused_device if instance._focused_device in instance._devices else None,
            focus_candidates=focus_candidates,
        )
        input_event = dataclasses.replace(
            cognition.InputEvent.with_data(cognition_input),
            id=event.id,
            metadata={
                **event.metadata,
                _FOCUS_CANDIDATES_METADATA_KEY: focus_candidates,
                _PROCESSING_OPERATION_METADATA_KEY: event.id,
            },
        )
        _ = hsm.dispatch(ctx, instance._cognition, input_event)

    @staticmethod
    async def _cancel_bot_processing_child(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        del event
        ability = instance._cognition
        if _instance_is_started(ability):
            lifetime = instance.context()
            _ = await ability.detach(ctx=lifetime)
            await hsm.stop(ability, lifetime)
            _ = await ability.attach(owner=instance, ctx=lifetime)
        _ = hsm.dispatch(ctx, instance, _BotProcessingChildCancelledEvent.with_data(None))

    @staticmethod
    def _matches_bot_processing_output(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        ability = instance._cognition
        operation_id = event.metadata.get(_PROCESSING_OPERATION_METADATA_KEY)
        return (
            event.name == ability.output_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(ability)
            and isinstance(operation_id, str)
            and event.id == operation_id
        )

    @staticmethod
    def _matches_bot_processing_failure(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        ability = instance._cognition
        operation_id = event.metadata.get(_PROCESSING_OPERATION_METADATA_KEY)
        return (
            event.name == ability.failed_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(ability)
            and isinstance(operation_id, str)
            and event.id == operation_id
        )

    @staticmethod
    def _complete_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        focus_candidates = _focus_candidates_for_event(event)
        if not focus_candidates:
            if instance._focused_device is not None:
                focus_candidates = (instance._focused_device,)
            elif instance._devices:
                focus_candidates = (next(iter(instance._devices)),)
        completed = events.ProcessingCompletedEventData(output=event.data, focus_candidates=focus_candidates)
        _ = instance.dispatch(
            ctx,
            dataclasses.replace(
                events.ProcessingCompletedEvent.with_data(completed),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_processing_timeout_failure(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del event
        seconds = instance._processing_timeout.total_seconds()
        failure = events.ProcessingFailedEventData(message=f"Bot processing timed out after {seconds:g} seconds.")
        _ = instance.dispatch(ctx, events.ProcessingFailedEvent.with_data(failure))

    @staticmethod
    def _fail_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        failure_message = getattr(event.data, "message", "Bot processing ability failed.")
        failure = events.ProcessingFailedEventData(message=str(failure_message))
        _ = instance.dispatch(ctx, events.ProcessingFailedEvent.with_data(failure))

    @staticmethod
    def _clear_focus(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._focused_device = None

    @staticmethod
    def _focus_event_target(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx
        if isinstance(event.data, events.InputEventData):
            processing_data = event.data
        elif isinstance(event.data, cognition.InputData):
            processing_data = event.data.stimulus
        else:
            raise AssertionError(f"unsupported body processing event data: {type(event.data)!r}")
        instance._focused_device = Bot._target_device_reference(instance, processing_data)

    @staticmethod
    def _dispatch_focus_device_action(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, events.FocusDeviceEventData)
        instance._focused_device = data.device

    @staticmethod
    def _dispatch_clear_focus_action(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._focused_device = None

    def _focused_device_snapshot(self, *, strict: bool) -> hsm.Snapshot | None:
        focused_device = self._focused_device
        if focused_device not in self._devices:
            return None
        try:
            return self._devices[focused_device].take_snapshot()
        except Exception:
            if strict:
                raise
            return None

    @typing.override
    def take_snapshot(self) -> hsm.Snapshot:
        snapshot = super().take_snapshot()
        device_snapshot = self._focused_device_snapshot(strict=False)
        if device_snapshot is None:
            return snapshot
        return dataclasses.replace(
            snapshot,
            Transitions=(*snapshot.Transitions, *device_snapshot.Transitions),
        )

    @staticmethod
    def _activation_state(instance: "Bot") -> _BotActivationState | None:
        return instance._activation_join

    @staticmethod
    def _set_activation_state(instance: "Bot", state: _BotActivationState | None) -> None:
        instance._activation_join = state

    @staticmethod
    def _clear_activation_state(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Bot._set_activation_state(instance, None)

    @staticmethod
    async def _activate_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        del event
        # Devices and abilities outlive activate activity; parent under bot lifetime (HSM-CONTEXT-001).
        lifetime = instance.context()
        world = World.from_context(lifetime)
        devices = _device_tree(*instance._devices.values())
        device_references: list[tuple[str, Device]] = []
        started_devices: list[Device] = []
        seen_devices: set[int] = set()
        for reference, device in instance._devices.items():
            identifier = id(device)
            if identifier in seen_devices:
                continue
            seen_devices.add(identifier)
            device_references.append((reference, device))
        requested_devices: list[str] = []
        try:
            for device in devices:
                require_world_scope(world, device, participant="Device")
            for device in devices:
                if not _device_model_is_running(device):
                    _ = await hsm.started(world.context, device, device.model)
                    started_devices.append(device)
            await Bot._start_abilities(lifetime, instance)
            Bot._set_activation_state(
                instance,
                _BotActivationState(
                    pending_devices=tuple(reference for reference, _ in device_references),
                    started_devices=tuple(started_devices),
                ),
            )
            if not device_references:
                _ = instance.dispatch(ctx, events.ActivatingDoneEvent.with_data(events.ActivatingDoneEventData()))
                return
            for reference, device in device_references:
                await hsm.Instance.dispatch(
                    device,
                    world.context,
                    attachment.AttachEvent.with_data(attachment.AttachData(actor=instance)),
                )
                requested_devices.append(reference)
        except Exception:
            state = Bot._activation_state(instance)
            if state is not None:
                Bot._set_activation_state(
                    instance,
                    state.model_copy(
                        update={
                            "pending_devices": tuple(
                                reference for reference in requested_devices if reference in state.pending_devices
                            )
                        }
                    ),
                )
                _ = instance.dispatch(ctx, events.ActivatingFailedEvent.with_data(events.ActivatingFailedEventData()))
                return
            cleanup_error: Exception | None = None
            try:
                await Bot._stop_abilities(ctx, instance)
            except Exception as error:
                cleanup_error = error
            if state is None:
                for device in reversed(started_devices):
                    if _device_model_is_running(device):
                        try:
                            await hsm.stop(device, world.context)
                        except Exception as error:
                            if cleanup_error is None:
                                cleanup_error = error
            if cleanup_error is not None:
                _ = instance.dispatch(
                    ctx,
                    _BotActivationRollbackFailedEvent.with_data(
                        _BotActivationRollbackFailedEventData(
                            message=f"Bot activation rollback failed: {cleanup_error}"
                        )
                    ),
                )
                return
            _ = instance.dispatch(ctx, events.ActivatingFailedEvent.with_data(events.ActivatingFailedEventData()))

    @staticmethod
    def _activation_event_targets_agent(instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        bot_id = _instance_id(instance)
        return bool(bot_id) and event.target == bot_id

    @staticmethod
    def _matches_activation_device_attached(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        state = Bot._activation_state(instance)
        if state is None or event.name != attachment.AttachCompleteEvent.name:
            return False
        if not isinstance(event.data, attachment.AttachCompleteData):
            return False
        device_reference = Bot._device_reference_for_source(instance, event.source)
        return device_reference in state.pending_devices and Bot._activation_event_targets_agent(instance, event)

    @staticmethod
    def _matches_activation_device_failed(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        state = Bot._activation_state(instance)
        if state is None or event.name != attachment.AttachFailedEvent.name:
            return False
        if not isinstance(event.data, attachment.FailedData):
            return False
        device_reference = Bot._device_reference_for_source(instance, event.source)
        return device_reference in state.pending_devices and Bot._activation_event_targets_agent(instance, event)

    @staticmethod
    def _has_activation_state(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return Bot._activation_state(instance) is not None

    @staticmethod
    def _consume_pending_attach(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
        *,
        on_empty: collections.abc.Callable[[hsm.Context, "Bot"], None],
    ) -> None:
        """Shared attach bookkeeping for activation success and rollback paths."""

        state = Bot._activation_state(instance)
        if state is None:
            return
        device_reference = Bot._device_reference_for_source(instance, event.source)
        pending_devices = tuple(reference for reference in state.pending_devices if reference != device_reference)
        attached_devices = state.attached_devices
        if isinstance(event.data, attachment.AttachCompleteData) and event.data.created and device_reference is not None:
            attached_devices = (*state.attached_devices, device_reference)
        Bot._set_activation_state(
            instance,
            state.model_copy(update={"pending_devices": pending_devices, "attached_devices": attached_devices}),
        )
        if not pending_devices:
            on_empty(ctx, instance)

    @staticmethod
    def _mark_activation_device_attached(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> None:
        def _done(done_ctx: hsm.Context, done_instance: "Bot") -> None:
            _ = hsm.dispatch(
                done_ctx,
                done_instance,
                events.ActivatingDoneEvent.with_data(events.ActivatingDoneEventData()),
            )

        Bot._consume_pending_attach(ctx, instance, event, on_empty=_done)

    @staticmethod
    def _mark_activation_device_failed(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> None:
        state = Bot._activation_state(instance)
        if state is None:
            return
        device_reference = Bot._device_reference_for_source(instance, event.source)
        pending_devices = tuple(reference for reference in state.pending_devices if reference != device_reference)
        Bot._set_activation_state(instance, state.model_copy(update={"pending_devices": pending_devices}))
        if not pending_devices:
            _ = hsm.dispatch(ctx, instance, _BotActivationRollbackReadyEvent.with_data(None))

    @staticmethod
    def _mark_rollback_device_attached(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> None:
        def _ready(ready_ctx: hsm.Context, ready_instance: "Bot") -> None:
            _ = hsm.dispatch(ready_ctx, ready_instance, _BotActivationRollbackReadyEvent.with_data(None))

        Bot._consume_pending_attach(ctx, instance, event, on_empty=_ready)

    @staticmethod
    def _dispatch_activation_rollback_ready_if_no_pending(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        state = Bot._activation_state(instance)
        if state is None or not state.pending_devices:
            _ = hsm.dispatch(ctx, instance, _BotActivationRollbackReadyEvent.with_data(None))

    @staticmethod
    async def _rollback_activation_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        del event
        world = World.from_context(instance.context())
        state = Bot._activation_state(instance)
        pending_devices = () if state is None else state.pending_devices
        attached_devices = () if state is None else state.attached_devices
        started_devices = () if state is None else state.started_devices
        cleanup_error: Exception | None = None
        for reference in reversed(pending_devices):
            device = instance._devices.get(reference)
            if device is None:
                continue
            if not _device_model_is_running(device):
                continue
            try:
                await hsm.Instance.dispatch(
                    device,
                    world.context,
                    attachment.DetachEvent.with_data(attachment.DetachData(actor=instance)),
                )
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
        for reference in reversed(attached_devices):
            device = instance._devices.get(reference)
            if device is None:
                continue
            if not _device_model_is_running(device):
                continue
            try:
                await hsm.Instance.dispatch(
                    device,
                    world.context,
                    attachment.DetachEvent.with_data(attachment.DetachData(actor=instance)),
                )
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
        try:
            await Bot._stop_abilities(ctx, instance)
        except Exception as error:
            if cleanup_error is None:
                cleanup_error = error
        for device in reversed(started_devices):
            if _device_model_is_running(device):
                try:
                    await hsm.stop(device, world.context)
                except Exception as error:
                    if cleanup_error is None:
                        cleanup_error = error
        if cleanup_error is not None:
            _ = hsm.dispatch(
                ctx,
                instance,
                _BotActivationRollbackFailedEvent.with_data(
                    _BotActivationRollbackFailedEventData(message=f"Bot activation rollback failed: {cleanup_error}")
                ),
            )
            return
        _ = hsm.dispatch(ctx, instance, events.ActivatingFailedEvent.with_data(events.ActivatingFailedEventData()))

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "Bot",
        hsm.initial(hsm.target("inactive")),
        hsm.state(
            "inactive",
            hsm.transition(
                hsm.on(events.ActivateEvent),
                hsm.target("../activating"),
            ),
        ),
        hsm.state(
            "activating",
            hsm.activity(_activate_activity),
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.guard(_matches_activation_device_attached),
                hsm.effect(_mark_activation_device_attached),
            ),
            hsm.transition(
                hsm.on(attachment.AttachFailedEvent),
                hsm.guard(_matches_activation_device_failed),
                hsm.effect(_mark_activation_device_failed),
                hsm.target("../activation_rolling_back"),
            ),
            hsm.transition(
                hsm.on(events.ActivatingDoneEvent),
                hsm.effect(_clear_activation_state),
                hsm.target("../active"),
            ),
            hsm.transition(
                hsm.on(events.ActivatingFailedEvent),
                hsm.guard(_has_activation_state),
                hsm.target("../activation_rolling_back"),
            ),
            hsm.transition(
                hsm.on(events.ActivatingFailedEvent),
                hsm.effect(_clear_activation_state),
                hsm.target("../inactive"),
            ),
            hsm.transition(
                hsm.on(_BotActivationRollbackFailedEvent),
                hsm.effect(_clear_activation_state),
                hsm.target("../activation_failed"),
            ),
        ),
        hsm.state(
            "activation_rolling_back",
            hsm.entry(_dispatch_activation_rollback_ready_if_no_pending),
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.guard(_matches_activation_device_attached),
                hsm.effect(_mark_rollback_device_attached),
            ),
            hsm.transition(
                hsm.on(attachment.AttachFailedEvent),
                hsm.guard(_matches_activation_device_failed),
                hsm.effect(_mark_activation_device_failed),
            ),
            hsm.transition(
                hsm.on(_BotActivationRollbackReadyEvent),
                hsm.target("../activation_detaching"),
            ),
            hsm.transition(
                hsm.after(_bot_activation_rollback_timeout),
                hsm.target("../activation_detaching"),
            ),
        ),
        hsm.state(
            "activation_detaching",
            hsm.activity(_rollback_activation_activity),
            hsm.transition(
                hsm.on(events.ActivatingFailedEvent),
                hsm.effect(_clear_activation_state),
                hsm.target("../inactive"),
            ),
            hsm.transition(
                hsm.on(_BotActivationRollbackFailedEvent),
                hsm.effect(_clear_activation_state),
                hsm.target("../activation_failed"),
            ),
            hsm.transition(
                hsm.after(_bot_activation_rollback_timeout),
                hsm.effect(_clear_activation_state),
                hsm.target("../activation_failed"),
            ),
        ),
        hsm.state("activation_failed"),
        hsm.state(
            "deactivating",
            hsm.activity(_deactivate_activity),
            hsm.transition(
                hsm.on(events.DeactivatingDoneEvent),
                hsm.target("../inactive"),
            ),
            hsm.transition(
                hsm.on(_BotDeactivatingFailedEvent),
                hsm.target("../inactive"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.target("../inactive"),
            ),
        ),
        hsm.state(
            "active",
            hsm.initial(hsm.target("unfocused")),
            # Explicit world input events fan out in parallel to input abilities (never cognition).
            hsm.transition(
                hsm.on(SoundEvent, VisualEvent),
                hsm.effect(_fan_out_input),
            ),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_clear_focus),
                hsm.target("../deactivating"),
            ),
            hsm.state(
                "unfocused",
                hsm.entry(_clear_focus),
                hsm.transition(
                    hsm.on(events.InputEvent),
                    hsm.guard(_input_targets_configured_device),
                    hsm.effect(_focus_event_target, _dispatch_bot_processing),
                    hsm.target("../processing"),
                ),
                # Sensory products: explicit cognition.InputEvent handoff (never raw world media).
                hsm.transition(
                    hsm.on(cognition.InputEvent),
                    hsm.effect(_focus_event_target, _dispatch_bot_processing),
                    hsm.target("../processing"),
                ),
                hsm.transition(
                    hsm.on(events.FocusDeviceEvent),
                    hsm.guard(_ability_selected_configured_focus_device),
                    hsm.effect(_dispatch_focus_device_action),
                    hsm.target("../focused"),
                ),
            ),
            hsm.state(
                "focused",
                hsm.transition(
                    hsm.on(events.ClearFocusEvent),
                    hsm.guard(_ability_selected_clear_focus),
                    hsm.effect(_dispatch_clear_focus_action),
                    hsm.target("../unfocused"),
                ),
                hsm.transition(
                    hsm.on(events.FocusDeviceEvent),
                    hsm.guard(_ability_selected_configured_focus_device),
                    hsm.effect(_dispatch_focus_device_action),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(events.InputEvent),
                    hsm.guard(_input_targets_configured_device),
                    hsm.effect(_dispatch_bot_processing),
                    hsm.target("../processing"),
                ),
                hsm.transition(
                    hsm.on(cognition.InputEvent),
                    hsm.effect(_dispatch_bot_processing),
                    hsm.target("../processing"),
                ),
            ),
            hsm.state(
                "processing",
                hsm.defer(events.InputEvent),
                hsm.defer(cognition.InputEvent),
                hsm.transition(
                    hsm.on(cognition.OutputEvent),
                    hsm.guard(_matches_bot_processing_output),
                    hsm.effect(_complete_bot_processing),
                ),
                hsm.transition(
                    hsm.on(abilities.FailedEvent),
                    hsm.guard(_matches_bot_processing_failure),
                    hsm.effect(_fail_bot_processing),
                ),
                hsm.transition(
                    hsm.on(events.ClearFocusEvent),
                    hsm.guard(_ability_selected_clear_focus),
                    hsm.effect(_dispatch_clear_focus_action),
                ),
                hsm.transition(
                    hsm.on(events.FocusDeviceEvent),
                    hsm.guard(_ability_selected_configured_focus_device),
                    hsm.effect(_dispatch_focus_device_action),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingCompletedEvent),
                    hsm.guard(_processing_completed_with_focus),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingCompletedEvent),
                    hsm.guard(_processing_completed),
                    hsm.target("../unfocused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingFailedEvent),
                    hsm.guard(_processing_failed_with_focus),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingFailedEvent),
                    hsm.guard(_processing_failed),
                    hsm.target("../unfocused"),
                ),
                hsm.transition(
                    hsm.after(_bot_processing_timeout),
                    hsm.target("../cancelling_processing"),
                ),
            ),
            hsm.state(
                "cancelling_processing",
                hsm.defer(events.InputEvent),
                hsm.defer(cognition.InputEvent),
                hsm.activity(_cancel_bot_processing_child),
                hsm.transition(
                    hsm.on(_BotProcessingChildCancelledEvent),
                    hsm.guard(_has_focused_device),
                    hsm.effect(_dispatch_processing_timeout_failure),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(_BotProcessingChildCancelledEvent),
                    hsm.effect(_dispatch_processing_timeout_failure),
                    hsm.target("../unfocused"),
                ),
            ),
        ),
        hsm.observe(observer),
    )
