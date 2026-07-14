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
_DEFAULT_BOT_DEACTIVATION_TIMEOUT = datetime.timedelta(minutes=5)
_FOCUS_CANDIDATES_METADATA_KEY = "bot.focus_candidates"
_PROCESSING_OPERATION_METADATA_KEY = "bot.processing.operation"
_STARTED_DEVICES_METADATA_KEY = "bot.activation.started_devices"
_STARTED_ABILITIES_METADATA_KEY = "bot.activation.started_abilities"
_STARTED_ATTACHMENT_GROUP_METADATA_KEY = "bot.activation.started_attachment_group"


_BotProcessingChildCancelledEvent = hsm.Event[object](
    name="bot.processing.child.cancelled",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)


def _device_tree(*roots: Device) -> tuple[Device, ...]:
    return Device.device_tree(*roots)


def _ability_attach_context(lifetime: hsm.Context) -> hsm.Context:
    """Parent abilities under bot lifetime without joining the world broadcast set.

    World ``dispatch_all`` delivers to every instance in the world's Instances map. Input
    abilities must only receive ``world.sound`` / ``world.visual`` via the bot's parallel
    fan-out, not a second direct world delivery.
    """

    values: dict[typing.Hashable, object] = {hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()}
    return hsm.Context(parent=lifetime, values=values)


def _instance_id(instance: hsm.Instance) -> str:
    """Return the stable id of an active configured actor."""

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
    _deactivation_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_DEACTIVATION_TIMEOUT
    _devices: dict[str, Device]
    _cognition: abilities.Ability[cognition.InputData, typing.Any]
    _focused_device: str | None
    _innate_ability_instances: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _acquired_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _input: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _output: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _attachments: attachment.Group

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
        if self._deactivation_timeout <= datetime.timedelta():
            raise ValueError("deactivation_timeout must be positive.")
        self._devices = dict(devices)
        self._cognition = cognition
        self._focused_device = None
        self._innate_ability_instances = tuple(ability_type() for ability_type in self._innate_abilities)
        self._input = tuple(input)
        self._output = tuple(output)
        self._acquired_abilities = tuple(acquired_abilities)
        members: list[hsm.Instance] = []
        seen_members: set[int] = set()
        for member in (*self._devices.values(), *Bot._lifecycle_abilities(self)):
            identifier = id(member)
            if identifier not in seen_members:
                seen_members.add(identifier)
                members.append(member)
        self._attachments = attachment.Group(*members)

    async def attach(self, world: World) -> typing.Self:
        require_world_scope(world, self, participant="Bot")
        if not self.state() or self.state() == self.model.qualified_name:
            _ = await hsm.started(world.context, self, self.model)
        await self.dispatch(world.context, events.ActivateEvent.with_data(events.ActivateEventData()))
        return self

    async def detach(self, world: World) -> typing.Self:
        if not self.state() or self.state() == self.model.qualified_name:
            return self
        require_world_scope(world, self, participant="Bot")
        await self.dispatch(world.context, events.DeactivateEvent.with_data(events.DeactivateEventData()))
        return self

    @staticmethod
    def _lifecycle_abilities(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
        configured = (
            instance._cognition,
            *instance._input,
            *instance._output,
            *instance._innate_ability_instances,
            *instance._acquired_abilities,
        )
        lifecycle: list[abilities.Ability[typing.Any, typing.Any]] = []
        seen: set[int] = set()
        for ability in configured:
            identifier = id(ability)
            if identifier not in seen:
                seen.add(identifier)
                lifecycle.append(ability)
        return tuple(lifecycle)

    @staticmethod
    async def _deactivate_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        del ctx
        world = World.from_context(instance.context())
        await instance._attachments.detach(
            world.context,
            dataclasses.replace(
                attachment.DetachEvent.with_data(attachment.DetachData(actor=instance)),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(instance._attachments),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _deactivation_cleanup_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        del event
        lifetime = instance.context()
        world = World.from_context(lifetime)
        await hsm.stop(instance._attachments, world.context)
        for ability in Bot._lifecycle_abilities(instance):
            await hsm.stop(ability, lifetime)
        _ = hsm.dispatch(ctx, instance, events.DeactivatingDoneEvent.with_data(events.DeactivatingDoneEventData()))

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
        lifetime = instance.context()
        _ = await ability.detach(
            lifetime,
            dataclasses.replace(
                attachment.DetachEvent.with_data(attachment.DetachData(actor=instance)),
                source=hsm.id(instance),
                target=hsm.id(ability),
            ),
        )
        await hsm.stop(ability, lifetime)
        _ = await ability.attach(
            lifetime,
            dataclasses.replace(
                attachment.AttachEvent.with_data(attachment.AttachData(actor=instance)),
                source=hsm.id(instance),
            ),
        )
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
    async def _activate_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        lifetime = instance.context()
        world = World.from_context(lifetime)
        started_devices: list[Device] = []
        started_abilities: list[abilities.Ability[typing.Any, typing.Any]] = []
        group_started = False
        try:
            group_scope = _ability_attach_context(lifetime)
            _ = await hsm.started(group_scope, instance._attachments, instance._attachments.model)
            group_started = True
            for device in _device_tree(*instance._devices.values()):
                require_world_scope(world, device, participant="Device")
                model = device.model
                if model is None:
                    raise RuntimeError(f"{type(device).__name__} has no lifecycle model.")
                try:
                    _ = await hsm.started(world.context, device, model)
                    started_devices.append(device)
                except hsm.ErrorValidatingModel as error:
                    if "already has a running HSM" not in str(error):
                        raise
            ability_scope = _ability_attach_context(lifetime)
            for ability in Bot._lifecycle_abilities(instance):
                model = ability.model
                if model is None:
                    raise RuntimeError(f"{type(ability).__name__} has no lifecycle model.")
                _ = await hsm.started(ability_scope, ability, model)
                started_abilities.append(ability)
            await instance._attachments.attach(
                world.context,
                dataclasses.replace(
                    attachment.AttachEvent.with_data(attachment.AttachData(actor=instance)),
                    id=event.id,
                    source=hsm.id(instance),
                    target=hsm.id(instance._attachments),
                    metadata={
                        **event.metadata,
                        _STARTED_DEVICES_METADATA_KEY: tuple(started_devices),
                        _STARTED_ABILITIES_METADATA_KEY: tuple(started_abilities),
                        _STARTED_ATTACHMENT_GROUP_METADATA_KEY: group_started,
                    },
                ),
            )
        except Exception:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    events.ActivatingFailedEvent.with_data(events.ActivatingFailedEventData()),
                    metadata={
                        **event.metadata,
                        _STARTED_DEVICES_METADATA_KEY: tuple(started_devices),
                        _STARTED_ABILITIES_METADATA_KEY: tuple(started_abilities),
                        _STARTED_ATTACHMENT_GROUP_METADATA_KEY: group_started,
                    },
                ),
            )

    @staticmethod
    def _matches_attachment_group(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return event.source == hsm.id(instance._attachments) and event.target == hsm.id(instance)

    @staticmethod
    async def _activation_cleanup_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        lifetime = instance.context()
        world = World.from_context(lifetime)
        if event.metadata.get(_STARTED_ATTACHMENT_GROUP_METADATA_KEY) is True:
            await hsm.stop(instance._attachments, world.context)
        started_abilities: object = event.metadata.get(_STARTED_ABILITIES_METADATA_KEY, ())
        if isinstance(started_abilities, tuple):
            values = typing.cast(tuple[object, ...], started_abilities)
            if all(isinstance(value, abilities.Ability) for value in values):
                owned = typing.cast(tuple[abilities.Ability[typing.Any, typing.Any], ...], values)
                for ability in reversed(owned):
                    await hsm.stop(ability, lifetime)
        started_devices: object = event.metadata.get(_STARTED_DEVICES_METADATA_KEY, ())
        if isinstance(started_devices, tuple):
            values = typing.cast(tuple[object, ...], started_devices)
            devices = tuple(value for value in values if isinstance(value, Device))
            for device in reversed(devices if len(devices) == len(values) else ()):
                await hsm.stop(device, world.context)
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
                hsm.guard(_matches_attachment_group),
                hsm.target("../active"),
            ),
            hsm.transition(
                hsm.on(attachment.AttachFailedEvent),
                hsm.guard(_matches_attachment_group),
                hsm.target("../activation_cleanup"),
            ),
            hsm.transition(
                hsm.on(events.ActivatingFailedEvent),
                hsm.target("../activation_cleanup"),
            ),
        ),
        hsm.state(
            "activation_cleanup",
            hsm.activity(_activation_cleanup_activity),
            hsm.transition(
                hsm.on(events.ActivatingFailedEvent),
                hsm.target("../inactive"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.target("../inactive"),
            ),
        ),
        hsm.state(
            "deactivating",
            hsm.activity(_deactivate_activity),
            hsm.transition(
                hsm.on(attachment.DetachedEvent, attachment.DetachFailedEvent),
                hsm.guard(_matches_attachment_group),
                hsm.target("../deactivation_cleanup"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.target("../deactivation_cleanup"),
            ),
        ),
        hsm.state(
            "deactivation_cleanup",
            hsm.activity(_deactivation_cleanup_activity),
            hsm.transition(
                hsm.on(events.DeactivatingDoneEvent),
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
