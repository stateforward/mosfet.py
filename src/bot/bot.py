from bot.abilities import cognition

import abc
import collections.abc
import dataclasses
import datetime
import typing
import weakref

import hsm
import pydantic

import bot.device
from bot import abilities
from . import events

from bot.device import Device
from bot.telemetry import observer
from bot.world import SoundEvent, VisualEvent, World, require_world_scope

_DEFAULT_BOT_PROCESSING_TIMEOUT = datetime.timedelta(minutes=5)
_DEFAULT_BOT_ACTIVATION_ROLLBACK_TIMEOUT = datetime.timedelta(minutes=5)
_FOCUS_CANDIDATES_METADATA_KEY = "bot.focus_candidates"
_BOT_ACTIVATION_STATE_ATTRIBUTE = "bot_activation_state"


_BotProcessingChildCancelledEvent = hsm.Event[object](
    name="bot.processing.child.cancelled",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)
_HsmObservationEvent = hsm.Event[dict[str, object]](
    name="hsm/observation",
    schema=pydantic.TypeAdapter(dict[str, object]),
)


class _BotActivationState(pydantic.BaseModel):
    """Private HSM attribute carrying the device attachments still needed for activation."""

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


def _device_tree(*roots: Device) -> tuple[Device, ...]:
    return Device.device_tree(*roots)


def _model_is_running(instance: hsm.Instance) -> bool:
    try:
        snapshot = hsm.take_snapshot(None, instance)
    except hsm.ErrorValidatingModel:
        return False
    model = getattr(instance, "model", None)
    root = getattr(model, "qualified_name", None)
    if not isinstance(root, str):
        return bool(snapshot.State)
    return bool(snapshot.State) and snapshot.State != root


def _device_model_is_running(device: Device) -> bool:
    return _model_is_running(device)


def _bot_cognition(instance: "Bot") -> abilities.Ability[cognition.InputData, typing.Any]:
    return Bot.cognition_for(instance)


def _bot_focused_device(instance: "Bot") -> str | None:
    return Bot.focused_device_for(instance)


def _bot_innate_ability_instances(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
    return Bot.innate_abilities_for(instance)


def _bot_acquired_abilities(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
    return Bot.acquired_abilities_for(instance)


def _bot_input(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
    return Bot.input_for(instance)


def _bot_output(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
    return Bot.output_for(instance)


def _lifecycle_ability_graph(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
    return (
        _bot_cognition(instance),
        *_bot_input(instance),
        *_bot_output(instance),
        *_bot_innate_ability_instances(instance),
        *_bot_acquired_abilities(instance),
    )


def _ability_attach_context(lifetime: hsm.Context) -> hsm.Context:
    """Parent abilities under bot lifetime without joining the world broadcast set.

    World ``dispatch_all`` delivers to every instance in the world's Instances map. Input
    abilities must only receive ``world.sound`` / ``world.visual`` via the bot's parallel
    fan-out, not a second direct world delivery.
    """

    return hsm.Context(parent=lifetime, values={hsm.Keys.Instances: weakref.WeakValueDictionary()})


async def _start_abilities(ctx: hsm.Context, instance: "Bot") -> None:
    # Abilities outlive activate activity; parent under bot lifetime context (HSM-CONTEXT-001).
    lifetime = instance.context()
    del ctx
    ability_scope = _ability_attach_context(lifetime)
    for ability in _lifecycle_ability_graph(instance):
        _ = await ability.attach(owner=instance, ctx=ability_scope)


async def _stop_abilities(ctx: hsm.Context, instance: "Bot") -> None:
    lifetime = instance.context()
    del ctx
    for ability in reversed(_lifecycle_ability_graph(instance)):
        try:
            _ = await ability.detach(ctx=lifetime)
        except RuntimeError as error:
            if "dispatch requires a started HSM" in str(error):
                continue
            raise
        await hsm.stop(ability, lifetime)


async def _deactivate_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
    del event
    # Devices/abilities were started under bot lifetime; coordinate on that scope (HSM-CONTEXT-001).
    lifetime = instance.context()
    world = World.from_context(lifetime)
    detached_devices: set[int] = set()
    for device in instance.devices.values():
        identifier = id(device)
        if identifier in detached_devices:
            continue
        detached_devices.add(identifier)
        if not _device_model_is_running(device):
            continue
        await hsm.Instance.dispatch(
            device,
            world.context,
            bot.device.DetachEvent.with_data(bot.device.DetachEventData(bot=instance)),
        )
    await _stop_abilities(lifetime, instance)
    _ = instance.dispatch(ctx, events.DeactivatingDoneEvent.with_data(events.DeactivatingDoneEventData()))


def _target_device_reference(instance: "Bot", input: events.BotInputData) -> str | None:
    if isinstance(input, events.InputEventData):
        return input.target_device
    references = _device_references_for_event(instance, input)
    if len(references) == 1:
        return references[0]
    return None


def _input_device_references(instance: "Bot", input: events.BotInputData) -> tuple[str, ...]:
    if isinstance(input, events.InputEventData):
        return (input.target_device,) if input.target_device in instance.devices else ()
    references = _device_references_for_event(instance, input)
    focused_device = _bot_focused_device(instance)
    if not input.source and focused_device in references:
        assert focused_device is not None
        return (focused_device,)
    return references


def _processing_device_references(instance: "Bot", input: events.BotInputData) -> tuple[str, ...]:
    references: list[str] = []
    focused_device = _bot_focused_device(instance)
    if focused_device in instance.devices:
        assert focused_device is not None
        references.append(focused_device)
    for target_device in _input_device_references(instance, input):
        if target_device not in references:
            references.append(target_device)
    return tuple(references)


def _instance_id(instance: hsm.Instance) -> str:
    try:
        return hsm.id(instance)
    except Exception:
        return ""


def _source_device_reference_for_event(instance: "Bot", event: hsm.Event[typing.Any]) -> str | None:
    """Map event.source to a configured device reference when present.

    Sensory handoffs (``cognition.InputEvent``) keep the acoustic origin id on the stimulus
    ``source`` so focus can follow the phone even though the outer event is cognitive.
    """

    if not event.source:
        return None
    return _device_reference_for_source(instance, event.source)


def _device_reference_for_source(instance: "Bot", source: str) -> str | None:
    for reference, device in instance.devices.items():
        for candidate in _device_tree(device):
            if _instance_id(candidate) == source:
                return reference
    return None


def _device_references_for_event(instance: "Bot", event: hsm.Event[typing.Any]) -> tuple[str, ...]:
    source_reference = _source_device_reference_for_event(instance, event)
    if source_reference is not None:
        return (source_reference,)
    return tuple(reference for reference, device in instance.devices.items() if event.name in _device_event_map(device))


def _input_targets_configured_device(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    return isinstance(event.data, events.InputEventData) and event.data.target_device in instance.devices


def _fan_out_input(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
    """Publish world stimuli (sound/visual) to every input ability in parallel (no ordering)."""

    for ability in _bot_input(instance):
        _ = hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                event,
                target=hsm.id(ability),
                metadata=dict(event.metadata),
            ),
        )


def _ability_selected_configured_focus_device(
    ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]
) -> bool:
    del ctx, instance
    if not isinstance(event.data, events.FocusDeviceEventData):
        return False
    return event.data.device in _focus_candidates_for_event(event)


def _ability_selected_clear_focus(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, events.ClearFocusEventData) and _has_focus_candidates(event)


def _processing_completed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, events.ProcessingCompletedEventData)


def _processing_failed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, events.ProcessingFailedEventData)


def _bot_processing_timeout(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> datetime.timedelta:
    del ctx, event
    return instance.processing_timeout


def _bot_activation_rollback_timeout(
    ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]
) -> datetime.timedelta:
    del ctx, event
    return instance.activation_rollback_timeout


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


def _has_focus_candidates(event: hsm.Event[typing.Any]) -> bool:
    return bool(_focus_candidates_for_event(event))


def _stimulus_from_body_event(event: hsm.Event[typing.Any]) -> events.BotInputData:
    data = event.data
    if isinstance(data, events.InputEventData):
        return data
    if isinstance(data, cognition.InputData):
        return data.stimulus
    raise AssertionError(f"unsupported body processing event data: {type(data)!r}")


def _actor_key(instance: hsm.Instance) -> str:
    """Stable snake_case actor name from the instance class (Speaking → speaking)."""

    name = type(instance).__name__
    chars: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0 and (name[index - 1].islower() or (index + 1 < len(name) and name[index + 1].islower())):
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars) or "actor"


def _dispatch_actors(instance: "Bot") -> dict[str, hsm.Instance]:
    """Named devices plus input/output/acquired abilities for cognition dispatch."""

    actors: dict[str, hsm.Instance] = dict(instance.devices)
    for ability in (
        *_bot_input(instance),
        *_bot_output(instance),
        *_bot_innate_ability_instances(instance),
        *_bot_acquired_abilities(instance),
    ):
        key = _actor_key(ability)
        if key in actors:
            # Avoid clobbering devices; suffix when needed.
            suffix = 2
            while f"{key}_{suffix}" in actors:
                suffix += 1
            key = f"{key}_{suffix}"
        actors[key] = ability
    return actors


def _dispatch_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
    """Enrich body context and dispatch ``cognition.InputEvent`` to the cognition ability."""

    stimulus = _stimulus_from_body_event(event)
    focused_reference = _bot_focused_device(instance)
    focus_candidates = _processing_device_references(instance, stimulus)
    cognition_input = cognition.InputData(
        stimulus=stimulus,
        abilities=_lifecycle_ability_graph(instance),
        actors=_dispatch_actors(instance),
        focus=focused_reference if focused_reference in instance.devices else None,
        focus_candidates=focus_candidates,
    )
    ability = _bot_cognition(instance)
    input_event = dataclasses.replace(
        cognition.InputEvent.with_data(cognition_input),
        id=event.id,
        metadata={
            **event.metadata,
            _FOCUS_CANDIDATES_METADATA_KEY: focus_candidates,
        },
    )
    _ = hsm.dispatch(ctx, ability, input_event)


async def _cancel_bot_processing_child(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
    del event
    ability = _bot_cognition(instance)
    owner = abilities.Ability.current_owner(ability)
    if owner is not None:
        # Re-attach under owner lifetime so cognition outlives this cancel activity (HSM-CONTEXT-001).
        lifetime = owner.context() if owner.state() else instance.context()
        try:
            _ = await ability.detach(ctx=lifetime)
        except RuntimeError as error:
            if "dispatch requires a started HSM" not in str(error):
                raise
        else:
            await hsm.stop(ability, lifetime)
            _ = await ability.attach(owner=owner, ctx=lifetime)
    _ = hsm.dispatch(ctx, instance, _BotProcessingChildCancelledEvent.with_data(None))


def _matches_bot_processing_output(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    ability = _bot_cognition(instance)
    return (
        event.name == ability.output_event.name and event.target == hsm.id(instance) and event.source == hsm.id(ability)
    )


def _matches_bot_processing_failure(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    ability = _bot_cognition(instance)
    return (
        event.name == ability.failed_event.name and event.target == hsm.id(instance) and event.source == hsm.id(ability)
    )


def _complete_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
    """Cognition finished; body only records completion (ops already dispatched by the cognition)."""

    focus_candidates = _focus_candidates_for_event(event)
    if not focus_candidates:
        focused = _bot_focused_device(instance)
        if focused is not None:
            focus_candidates = (focused,)
        elif instance.devices:
            focus_candidates = (next(iter(instance.devices)),)
        else:
            focus_candidates = ("_",)
    completed = events.ProcessingCompletedEventData(
        output=event.data,
        focus_candidates=focus_candidates,
    )
    _ = instance.dispatch(
        ctx,
        dataclasses.replace(
            events.ProcessingCompletedEvent.with_data(completed),
            metadata=dict(event.metadata),
        ),
    )


def _fail_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
    failure_message = getattr(event.data, "message", "Bot processing ability failed.")
    _ = instance.dispatch(
        ctx,
        events.ProcessingFailedEvent.with_data(events.ProcessingFailedEventData(message=str(failure_message))),
    )


def _dispatch_processing_timeout_failure(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
    del event
    seconds = instance.processing_timeout.total_seconds()
    failure = events.ProcessingFailedEventData(message=f"Bot processing timed out after {seconds:g} seconds.")
    _ = instance.dispatch(ctx, events.ProcessingFailedEvent.with_data(failure))


class Bot(hsm.Instance, abc.ABC):
    """Interrupt-driven bot that observes and processes events while active."""

    innate_abilities: typing.ClassVar[tuple[type[abilities.Ability[typing.Any, typing.Any]], ...]] = ()
    _processing_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_PROCESSING_TIMEOUT
    processing_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_PROCESSING_TIMEOUT
    _activation_rollback_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_ACTIVATION_ROLLBACK_TIMEOUT
    activation_rollback_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_ACTIVATION_ROLLBACK_TIMEOUT
    devices: dict[str, Device]
    _cognition: abilities.Ability[cognition.InputData, typing.Any]
    _focused_device: str | None
    _innate_ability_instances: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _acquired_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _input: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _output: tuple[abilities.Ability[typing.Any, typing.Any], ...]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "_processing_timeout" in cls.__dict__:
            cls.processing_timeout = cls._processing_timeout
        if "_activation_rollback_timeout" in cls.__dict__:
            cls.activation_rollback_timeout = cls._activation_rollback_timeout

    @staticmethod
    def cognition_for(instance: "Bot") -> abilities.Ability[cognition.InputData, typing.Any]:
        return instance._cognition

    @staticmethod
    def focused_device_for(instance: "Bot") -> str | None:
        return instance._focused_device

    @staticmethod
    def innate_abilities_for(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
        return instance._innate_ability_instances

    @staticmethod
    def acquired_abilities_for(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
        return instance._acquired_abilities

    @staticmethod
    def input_for(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
        return instance._input

    @staticmethod
    def output_for(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
        return instance._output

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
        if self.processing_timeout <= datetime.timedelta():
            raise ValueError("_processing_timeout must be positive.")
        if self.activation_rollback_timeout <= datetime.timedelta():
            raise ValueError("_activation_rollback_timeout must be positive.")
        self.devices = dict(devices)
        self._cognition = cognition
        self._focused_device = None
        self._innate_ability_instances = tuple(ability_type() for ability_type in self.innate_abilities)
        self._input = tuple(input)
        self._output = tuple(output)
        self._acquired_abilities = tuple(acquired_abilities)

    async def attach(self, world: World) -> typing.Self:
        require_world_scope(world, self, participant="Bot")
        if not _model_is_running(self):
            _ = await hsm.started(world.context, self, self.model)
        await self.dispatch(world.context, events.ActivateEvent.with_data(events.ActivateEventData()))
        return self

    async def detach(self, world: World) -> typing.Self:
        if not _model_is_running(self):
            return self
        require_world_scope(world, self, participant="Bot")
        await self.dispatch(world.context, events.DeactivateEvent.with_data(events.DeactivateEventData()))
        return self

    @staticmethod
    def _clear_focus(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._focused_device = None

    @staticmethod
    def _focus_event_target(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx
        processing_data = _stimulus_from_body_event(event)
        instance._focused_device = _target_device_reference(instance, processing_data)

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
        focused_device = _bot_focused_device(self)
        if focused_device not in self.devices:
            return None
        try:
            return self.devices[focused_device].take_snapshot()
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
        value, ok = typing.cast(tuple[object, bool], instance.get(_BOT_ACTIVATION_STATE_ATTRIBUTE))
        if ok and isinstance(value, _BotActivationState):
            return value
        return None

    @staticmethod
    def _set_activation_state(instance: "Bot", state: _BotActivationState | None) -> None:
        _ = instance.set(_BOT_ACTIVATION_STATE_ATTRIBUTE, state)

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
        devices = _device_tree(*instance.devices.values())
        device_references: list[tuple[str, Device]] = []
        started_devices: list[Device] = []
        seen_devices: set[int] = set()
        for reference, device in instance.devices.items():
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
            await _start_abilities(lifetime, instance)
            Bot._set_activation_state(
                instance,
                _BotActivationState(
                    pending_devices=tuple(reference for reference, _ in device_references),
                    started_devices=tuple(started_devices),
                ),
            )
            if not device_references:
                _ = instance.dispatch(
                    ctx, events.ActivatingDoneEvent.with_data(events.ActivatingDoneEventData())
                )
                return
            for reference, device in device_references:
                await hsm.Instance.dispatch(
                    device,
                    world.context,
                    bot.device.AttachEvent.with_data(bot.device.AttachEventData(bot=instance)),
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
                _ = instance.dispatch(
                    ctx, events.ActivatingFailedEvent.with_data(events.ActivatingFailedEventData())
                )
                return
            cleanup_error: Exception | None = None
            try:
                await _stop_abilities(ctx, instance)
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
            _ = instance.dispatch(
                ctx, events.ActivatingFailedEvent.with_data(events.ActivatingFailedEventData())
            )

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
        if state is None or event.name != bot.device.AttachEvent.name:
            return False
        if not isinstance(event.data, bot.device.AttachEventData):
            return False
        device_reference = _device_reference_for_source(instance, event.source)
        return device_reference in state.pending_devices and Bot._activation_event_targets_agent(instance, event)

    @staticmethod
    def _matches_activation_device_failed(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        state = Bot._activation_state(instance)
        if state is None or event.name != bot.device.FirmwareInitializingFailedEvent.name:
            return False
        device_reference = _device_reference_for_source(instance, event.source)
        return device_reference in state.pending_devices and Bot._activation_event_targets_agent(instance, event)

    @staticmethod
    def _has_activation_state(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return Bot._activation_state(instance) is not None

    @staticmethod
    def _mark_activation_device_attached(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> None:
        state = Bot._activation_state(instance)
        if state is None:
            return
        device_reference = _device_reference_for_source(instance, event.source)
        pending_devices = tuple(reference for reference in state.pending_devices if reference != device_reference)
        attached_devices = state.attached_devices
        if event.metadata.get(bot.device.ATTACH_CREATED_METADATA_KEY) is True and device_reference is not None:
            attached_devices = (*state.attached_devices, device_reference)
        Bot._set_activation_state(
            instance,
            state.model_copy(update={"pending_devices": pending_devices, "attached_devices": attached_devices}),
        )
        if not pending_devices:
            _ = hsm.dispatch(
                ctx, instance, events.ActivatingDoneEvent.with_data(events.ActivatingDoneEventData())
            )

    @staticmethod
    def _mark_activation_device_failed(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> None:
        state = Bot._activation_state(instance)
        if state is None:
            return
        device_reference = _device_reference_for_source(instance, event.source)
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
        state = Bot._activation_state(instance)
        if state is None:
            return
        device_reference = _device_reference_for_source(instance, event.source)
        pending_devices = tuple(reference for reference in state.pending_devices if reference != device_reference)
        attached_devices = state.attached_devices
        if event.metadata.get(bot.device.ATTACH_CREATED_METADATA_KEY) is True and device_reference is not None:
            attached_devices = (*state.attached_devices, device_reference)
        Bot._set_activation_state(
            instance,
            state.model_copy(update={"pending_devices": pending_devices, "attached_devices": attached_devices}),
        )
        if not pending_devices:
            _ = hsm.dispatch(ctx, instance, _BotActivationRollbackReadyEvent.with_data(None))

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
            device = instance.devices.get(reference)
            if device is None:
                continue
            if not _device_model_is_running(device):
                continue
            try:
                await hsm.Instance.dispatch(
                    device,
                    world.context,
                    bot.device.DetachEvent.with_data(bot.device.DetachEventData(bot=instance)),
                )
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
        for reference in reversed(attached_devices):
            device = instance.devices.get(reference)
            if device is None:
                continue
            if not _device_model_is_running(device):
                continue
            try:
                await hsm.Instance.dispatch(
                    device,
                    world.context,
                    bot.device.DetachEvent.with_data(bot.device.DetachEventData(bot=instance)),
                )
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
        try:
            await _stop_abilities(ctx, instance)
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
                    _BotActivationRollbackFailedEventData(
                        message=f"Bot activation rollback failed: {cleanup_error}"
                    )
                ),
            )
            return
        _ = hsm.dispatch(
            ctx, instance, events.ActivatingFailedEvent.with_data(events.ActivatingFailedEventData())
        )

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "Bot",
        hsm.attribute(_BOT_ACTIVATION_STATE_ATTRIBUTE),
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
                hsm.on(bot.device.AttachEvent),
                hsm.guard(_matches_activation_device_attached),
                hsm.effect(_mark_activation_device_attached),
            ),
            hsm.transition(
                hsm.on(bot.device.FirmwareInitializingFailedEvent),
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
                hsm.on(bot.device.AttachEvent),
                hsm.guard(_matches_activation_device_attached),
                hsm.effect(_mark_rollback_device_attached),
            ),
            hsm.transition(
                hsm.on(bot.device.FirmwareInitializingFailedEvent),
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
                # Sensory products: explicit cognition.InputEvent handoff (never raw world/device AnyEvent).
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
                hsm.transition(hsm.on(_HsmObservationEvent)),
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
                hsm.transition(hsm.on(_HsmObservationEvent)),
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
                    hsm.target("../unfocused"),
                ),
                hsm.transition(
                    hsm.on(events.FocusDeviceEvent),
                    hsm.guard(_ability_selected_configured_focus_device),
                    hsm.effect(_dispatch_focus_device_action),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingCompletedEvent),
                    hsm.guard(_processing_completed),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingFailedEvent),
                    hsm.guard(_processing_failed),
                    hsm.target("../focused"),
                ),
                hsm.transition(hsm.on(_HsmObservationEvent)),
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
                    hsm.effect(_dispatch_processing_timeout_failure),
                    hsm.target("../focused"),
                ),
            ),
        ),
        hsm.observe(observer),
    )
