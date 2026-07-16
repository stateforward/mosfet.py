from bot.abilities import cognition

import abc
import asyncio
import collections.abc
import dataclasses
import datetime
import typing
import uuid
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
_PROCESSING_ACTOR_METADATA_KEY = "bot.processing.actor"
_ACTION_SOURCE_METADATA_KEY = "bot.cognition.action_source"
_STARTED_DEVICES_METADATA_KEY = "bot.activation.started_devices"
_STARTED_ABILITIES_METADATA_KEY = "bot.activation.started_abilities"
_STARTED_ATTACHMENT_GROUP_METADATA_KEY = "bot.activation.started_attachment_group"
_LIFECYCLE_OPERATION_METADATA_KEY = "bot.lifecycle.operation"
_REBOOT_CLEANUP_METADATA_KEY = "bot.lifecycle.reboot_cleanup"
_REBOOT_CLEANUP_PRESERVE = "preserve"
_REBOOT_CLEANUP_RESET = "reset"


class _BotLifecycleOperation(pydantic.BaseModel):
    """Immutable JSON-safe capability for one lifecycle operation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    token: str
    request_id: str
    kind: typing.Literal["attach", "detach"]
    _context: hsm.Context = pydantic.PrivateAttr()

    @classmethod
    def create(
        cls,
        ctx: hsm.Context,
        *,
        request_id: str,
        kind: typing.Literal["attach", "detach"],
    ) -> "_BotLifecycleOperation":
        operation = cls(token=uuid.uuid4().hex, request_id=request_id, kind=kind)
        operation._context = ctx
        return operation

    def is_active(self) -> bool:
        """Return whether the HSM activity scope that owns this capability is active."""

        return not self._context.is_done()


class _BotCleanupData(pydantic.BaseModel):
    """Private exact terminal for one lifecycle cleanup activity."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    token: str = pydantic.Field(min_length=1)
    request_id: str
    kind: typing.Literal["activation", "deactivation"]
    _context: hsm.Context | None = pydantic.PrivateAttr(default=None)

    @classmethod
    def create(
        cls,
        ctx: hsm.Context,
        *,
        request_id: str,
        kind: typing.Literal["activation", "deactivation"],
    ) -> "_BotCleanupData":
        operation = cls(token=uuid.uuid4().hex, request_id=request_id, kind=kind)
        operation._context = ctx
        return operation

    def is_active(self) -> bool:
        """Return whether the cleanup activity that owns this terminal is active."""

        return self._context is not None and not self._context.is_done()


_BotCleanupDoneEvent = hsm.Event[_BotCleanupData](
    name="bot.lifecycle.cleanup.done",
    kind=hsm.CompletionEventKind,
    schema=_BotCleanupData,
)


class _BotProcessingOperationData(pydantic.BaseModel):
    """Immutable JSON-safe capability for one Bot processing turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    token: str = pydantic.Field(min_length=1)
    request_id: str
    actor_id: str = pydantic.Field(min_length=1)


_BotProcessingTimedOutEvent = hsm.Event[_BotProcessingOperationData](
    name="bot.processing.timed_out",
    kind=hsm.ErrorEventKind,
    schema=_BotProcessingOperationData,
)
_BotProcessingCancelTimedOutEvent = hsm.Event[_BotProcessingOperationData](
    name="bot.processing.cancel.timed_out",
    kind=hsm.ErrorEventKind,
    schema=_BotProcessingOperationData,
)
_BotProcessingFinishedEvent = hsm.Event[_BotProcessingOperationData](
    name="bot.processing.finished",
    kind=hsm.CompletionEventKind,
    schema=_BotProcessingOperationData,
)


class _BotProcessingOperation(hsm.Instance):
    """Operation-scoped HSM actor that owns timeout progression."""

    _owner: "Bot"
    _data: _BotProcessingOperationData | None
    _metadata: dict[str, object]
    _timeout: datetime.timedelta
    _token: str
    _request_id: str

    def __init__(
        self,
        owner: "Bot",
        *,
        request_id: str,
        metadata: dict[str, object],
        timeout: datetime.timedelta,
    ) -> None:
        super().__init__()
        self._owner = owner
        self._metadata = metadata
        self._timeout = timeout
        self._data = None
        self._token = uuid.uuid4().hex
        self._request_id = request_id

    @classmethod
    def create(
        cls,
        owner: "Bot",
        *,
        request_id: str,
        metadata: dict[str, object],
        timeout: datetime.timedelta,
    ) -> "_BotProcessingOperation":
        operation = cls(
            owner,
            request_id=request_id,
            metadata=metadata,
            timeout=timeout,
        )
        return operation

    @classmethod
    def _model(cls) -> hsm.Model:
        def timeout_delay(
            ctx: hsm.Context,
            instance: _BotProcessingOperation,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, event
            return instance._timeout

        def dispatch_timeout(
            ctx: hsm.Context,
            instance: _BotProcessingOperation,
            event: hsm.Event[typing.Any],
        ) -> None:
            assert instance._data is not None
            _ = hsm.dispatch(
                ctx,
                instance._owner,
                dataclasses.replace(
                    _BotProcessingTimedOutEvent.with_data(instance._data),
                    id=instance._data.request_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance._owner),
                    metadata={
                        **instance._metadata,
                        _PROCESSING_OPERATION_METADATA_KEY: instance._data,
                        _PROCESSING_ACTOR_METADATA_KEY: instance,
                    },
                ),
            )

        def dispatch_cancel_timeout(
            ctx: hsm.Context,
            instance: _BotProcessingOperation,
            event: hsm.Event[typing.Any],
        ) -> None:
            assert instance._data is not None
            _ = hsm.dispatch(
                ctx,
                instance._owner,
                dataclasses.replace(
                    _BotProcessingCancelTimedOutEvent.with_data(instance._data),
                    id=instance._data.request_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance._owner),
                    metadata={
                        **instance._metadata,
                        _PROCESSING_OPERATION_METADATA_KEY: instance._data,
                        _PROCESSING_ACTOR_METADATA_KEY: instance,
                    },
                ),
            )

        def is_operation(
            ctx: hsm.Context,
            instance: _BotProcessingOperation,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            assert instance._data is not None
            return (
                event.data == instance._data
                and event.id == instance._data.request_id
                and event.source == hsm.id(instance._owner)
                and event.target == hsm.id(instance)
            )

        return hsm.define(
            "BotProcessingTimer",
            hsm.initial(hsm.target("waiting")),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(_BotProcessingFinishedEvent),
                    hsm.guard(is_operation),
                    hsm.target("/BotProcessingTimer/done"),
                ),
                hsm.transition(
                    hsm.after(timeout_delay),
                    hsm.effect(dispatch_timeout),
                    hsm.target("/BotProcessingTimer/cancelling"),
                ),
            ),
            hsm.state(
                "cancelling",
                hsm.transition(
                    hsm.on(_BotProcessingFinishedEvent),
                    hsm.guard(is_operation),
                    hsm.target("/BotProcessingTimer/done"),
                ),
                hsm.transition(
                    hsm.after(timeout_delay),
                    hsm.effect(dispatch_cancel_timeout),
                    hsm.target("/BotProcessingTimer/done"),
                ),
            ),
            hsm.final("done"),
        )

    @classmethod
    async def started(
        cls, ctx: hsm.Context, operation: "_BotProcessingOperation"
    ) -> tuple["_BotProcessingOperation", _BotProcessingOperationData]:
        private = hsm.Context(
            parent=ctx,
            values={hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()},
        )
        try:
            started = await hsm.started(private, operation, cls._model())
            operation._data = _BotProcessingOperationData(
                token=hsm.id(operation),
                request_id=operation._request_id,
                actor_id=hsm.id(operation),
            )
            return started, operation._data
        except asyncio.CancelledError:
            await hsm.stop(operation, hsm.Context())
            raise


def _processing_operation(event: hsm.Event[typing.Any]) -> _BotProcessingOperationData | None:
    operation = event.metadata.get(_PROCESSING_OPERATION_METADATA_KEY)
    return operation if isinstance(operation, _BotProcessingOperationData) else None


def _event_belongs_to_processing(event: hsm.Event[typing.Any], operation: _BotProcessingOperationData) -> bool:
    return event.id == operation.request_id or event.id.startswith(f"{operation.request_id}:")


def _active_processing_operation(
    event: hsm.Event[typing.Any],
    *,
    phases: tuple[str, ...] = ("/BotProcessingTimer/waiting", "/BotProcessingTimer/cancelling"),
) -> _BotProcessingOperationData | None:
    """Return the capability only while its exact operation actor owns an active phase."""

    operation = _processing_operation(event)
    actor = event.metadata.get(_PROCESSING_ACTOR_METADATA_KEY)
    if (
        operation is None
        or not isinstance(actor, _BotProcessingOperation)
        or hsm.id(actor) != operation.actor_id
        or operation.token != operation.actor_id
        or actor.state() not in phases
    ):
        return None
    return operation


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
        self._attachments = attachment.Group(*Bot._lifecycle_attachment_members(self))

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
    def _lifecycle_attachment_members(instance: "Bot") -> tuple[hsm.Instance, ...]:
        members: list[hsm.Instance] = []
        seen: set[int] = set()
        for member in (*instance._devices.values(), *Bot._lifecycle_abilities(instance)):
            identifier = id(member)
            if identifier not in seen:
                seen.add(identifier)
                members.append(member)
        return tuple(members)

    @staticmethod
    async def _deactivate_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        operation = _BotLifecycleOperation.create(
            ctx,
            request_id=event.id,
            kind="detach",
        )
        await instance._attachments.detach(
            ctx,
            dataclasses.replace(
                attachment.DetachEvent.with_data(attachment.DetachData(actor=instance)),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(instance._attachments),
                metadata={**event.metadata, _LIFECYCLE_OPERATION_METADATA_KEY: operation},
            ),
        )

    @staticmethod
    async def _deactivation_cleanup_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        lifetime = instance.context()
        cleanup_mode = event.metadata.get(_REBOOT_CLEANUP_METADATA_KEY)
        world = World.from_context(lifetime)
        if cleanup_mode == _REBOOT_CLEANUP_RESET:
            for member in Bot._lifecycle_attachment_members(instance):
                await hsm.Instance.dispatch(
                    member,
                    lifetime,
                    dataclasses.replace(
                        attachment.DetachEvent.with_data(attachment.DetachData(actor=instance)),
                        id=event.id,
                        source=hsm.id(instance),
                        target=hsm.id(member),
                        metadata=dict(event.metadata),
                    ),
                )
            await hsm.stop(instance._attachments, world.context)
            instance._attachments = attachment.Group(*Bot._lifecycle_attachment_members(instance))
        elif cleanup_mode != _REBOOT_CLEANUP_PRESERVE:
            await hsm.stop(instance._attachments, world.context)
            for ability in Bot._lifecycle_abilities(instance):
                # Cognition's required composite tree remains detached under Bot lifetime;
                # stopping it cancels the child contexts needed by a later activation.
                if ability is instance._cognition:
                    continue
                await hsm.stop(ability, lifetime)
        instances = lifetime.value(hsm.Keys.Instances)
        operation_actors = (
            tuple(actor for actor in instances.values() if isinstance(actor, _BotProcessingOperation))
            if isinstance(instances, collections.abc.Mapping)
            else ()
        )
        for actor in operation_actors:
            await actor.stop(lifetime)
            if isinstance(instances, collections.abc.MutableMapping):
                _ = instances.pop(hsm.id(actor), None)
        cleanup = _BotCleanupData.create(ctx, request_id=event.id, kind="deactivation")
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _BotCleanupDoneEvent.with_data(cleanup),
                id=cleanup.request_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
            ),
        )

    @staticmethod
    def _preserve_reboot_cleanup(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        if event.metadata.get(_REBOOT_CLEANUP_METADATA_KEY) != _REBOOT_CLEANUP_RESET:
            event.metadata[_REBOOT_CLEANUP_METADATA_KEY] = _REBOOT_CLEANUP_PRESERVE

    @staticmethod
    def _reset_reboot_cleanup(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        event.metadata[_REBOOT_CLEANUP_METADATA_KEY] = _REBOOT_CLEANUP_RESET

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
    def _reboot_requested_by_cognition(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, events.RebootEventData)
            and event.source == hsm.id(instance._cognition)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _has_current_ability_source(instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        """Require action provenance from the live cognition actor that selected it."""

        del instance
        source = event.metadata.get(_ACTION_SOURCE_METADATA_KEY)
        return isinstance(source, abilities.Ability) and event.source == hsm.id(source) and bool(source.state())

    @staticmethod
    def _ability_selected_configured_focus_device(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        if not isinstance(event.data, events.FocusDeviceEventData):
            return False
        operation = _active_processing_operation(event, phases=("/BotProcessingTimer/waiting",))
        return (
            operation is not None
            and _event_belongs_to_processing(event, operation)
            and event.target == hsm.id(instance)
            and Bot._has_current_ability_source(instance, event)
            and event.data.device in instance._devices
            and event.data.device in _focus_candidates_for_event(event)
        )

    @staticmethod
    def _ability_selected_clear_focus(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        operation = _active_processing_operation(event, phases=("/BotProcessingTimer/waiting",))
        return (
            isinstance(event.data, events.ClearFocusEventData)
            and operation is not None
            and _event_belongs_to_processing(event, operation)
            and event.target == hsm.id(instance)
            and Bot._has_current_ability_source(instance, event)
            and bool(_focus_candidates_for_event(event))
        )

    @staticmethod
    def _processing_completed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        operation = _active_processing_operation(event, phases=("/BotProcessingTimer/waiting",))
        return (
            isinstance(event.data, events.ProcessingCompletedEventData)
            and operation is not None
            and event.id == operation.request_id
            and event.source == hsm.id(instance._cognition)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _processing_failed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        operation = _active_processing_operation(event, phases=("/BotProcessingTimer/waiting",))
        return (
            isinstance(event.data, events.ProcessingFailedEventData)
            and operation is not None
            and event.id == operation.request_id
            and event.source == hsm.id(instance._cognition)
            and event.target == hsm.id(instance)
        )

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
    async def _dispatch_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
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
        operation_actor = _BotProcessingOperation.create(
            instance,
            request_id=event.id,
            metadata=dict(event.metadata),
            timeout=instance._processing_timeout,
        )
        _, operation = await _BotProcessingOperation.started(instance.context(), operation_actor)
        input_event = dataclasses.replace(
            cognition.InputEvent.with_data(cognition_input),
            id=event.id,
            metadata={
                **event.metadata,
                _FOCUS_CANDIDATES_METADATA_KEY: focus_candidates,
                _PROCESSING_OPERATION_METADATA_KEY: operation,
                _PROCESSING_ACTOR_METADATA_KEY: operation_actor,
            },
        )
        _ = hsm.dispatch(ctx, instance._cognition, input_event)

    @staticmethod
    def _cancel_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        operation = _active_processing_operation(event)
        assert operation is not None
        _ = hsm.dispatch(
            ctx,
            instance._cognition,
            dataclasses.replace(
                cognition.CancelEvent.with_data(
                    cognition.CancelData(operation_id=operation.request_id, token=operation.token)
                ),
                id=operation.request_id,
                source=hsm.id(instance),
                target=hsm.id(instance._cognition),
                metadata={**event.metadata, _PROCESSING_OPERATION_METADATA_KEY: operation},
            ),
        )

    @staticmethod
    def _matches_bot_processing_output(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        ability = instance._cognition
        operation = _active_processing_operation(event, phases=("/BotProcessingTimer/waiting",))
        return (
            event.name == ability.output_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(ability)
            and operation is not None
            and event.id == operation.request_id
        )

    @staticmethod
    def _matches_bot_processing_failure(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        ability = instance._cognition
        operation = _active_processing_operation(event, phases=("/BotProcessingTimer/waiting",))
        return (
            event.name == ability.failed_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(ability)
            and operation is not None
            and event.id == operation.request_id
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
                id=event.id,
                source=event.source,
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_processing_timeout_failure(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        operation = _active_processing_operation(event, phases=("/BotProcessingTimer/cancelling",))
        assert operation is not None
        seconds = instance._processing_timeout.total_seconds()
        failure = events.ProcessingFailedEventData(message=f"Bot processing timed out after {seconds:g} seconds.")
        _ = instance.dispatch(
            ctx,
            dataclasses.replace(
                events.ProcessingFailedEvent.with_data(failure),
                id=operation.request_id,
                source=hsm.id(instance._cognition),
                target=hsm.id(instance),
                metadata={**event.metadata, _PROCESSING_OPERATION_METADATA_KEY: operation},
            ),
        )

    @staticmethod
    def _fail_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        failure_message = getattr(event.data, "message", "Bot processing ability failed.")
        failure = events.ProcessingFailedEventData(message=str(failure_message))
        _ = instance.dispatch(
            ctx,
            dataclasses.replace(
                events.ProcessingFailedEvent.with_data(failure),
                id=event.id,
                source=event.source,
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _matches_processing_timeout(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        operation = _active_processing_operation(event)
        return (
            isinstance(event.data, _BotProcessingOperationData)
            and operation is not None
            and event.data == operation
            and event.id == operation.request_id
            and event.source == operation.actor_id
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _matches_cognition_cancelled(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        operation = _active_processing_operation(event, phases=("/BotProcessingTimer/cancelling",))
        data = event.data
        return (
            isinstance(data, cognition.CancelledData)
            and operation is not None
            and data.operation_id == operation.request_id
            and data.token == operation.token
            and event.id == operation.request_id
            and event.source == hsm.id(instance._cognition)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _consume_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        operation = _processing_operation(event)
        actor = event.metadata.get(_PROCESSING_ACTOR_METADATA_KEY)
        assert operation is not None
        assert isinstance(actor, _BotProcessingOperation)
        _ = hsm.dispatch(
            ctx,
            actor,
            dataclasses.replace(
                _BotProcessingFinishedEvent.with_data(operation),
                id=operation.request_id,
                source=hsm.id(instance),
                target=hsm.id(actor),
            ),
        )

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
        operation = _BotLifecycleOperation.create(
            ctx,
            request_id=event.id,
            kind="attach",
        )
        try:
            group_scope = _ability_attach_context(lifetime)
            try:
                _ = await hsm.started(group_scope, instance._attachments, instance._attachments.model)
                group_started = True
            except hsm.ErrorValidatingModel as error:
                if "already has a running HSM" not in str(error):
                    raise
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
                try:
                    _ = await hsm.started(ability_scope, ability, model)
                    started_abilities.append(ability)
                except hsm.ErrorValidatingModel as error:
                    shares_world_instances = ability.context().value(hsm.Keys.Instances) is world.context.value(
                        hsm.Keys.Instances
                    )
                    if "already has a running HSM" not in str(error) or shares_world_instances:
                        raise
            await instance._attachments.attach(
                ctx,
                dataclasses.replace(
                    attachment.AttachEvent.with_data(attachment.AttachData(actor=instance)),
                    id=event.id,
                    source=hsm.id(instance),
                    target=hsm.id(instance._attachments),
                    metadata={
                        **event.metadata,
                        _LIFECYCLE_OPERATION_METADATA_KEY: operation,
                        _STARTED_DEVICES_METADATA_KEY: tuple(started_devices),
                        _STARTED_ABILITIES_METADATA_KEY: tuple(started_abilities),
                        _STARTED_ATTACHMENT_GROUP_METADATA_KEY: group_started,
                    },
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    events.ActivatingFailedEvent.with_data(events.ActivatingFailedEventData()),
                    id=operation.request_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata={
                        **event.metadata,
                        _LIFECYCLE_OPERATION_METADATA_KEY: operation,
                        _STARTED_DEVICES_METADATA_KEY: tuple(started_devices),
                        _STARTED_ABILITIES_METADATA_KEY: tuple(started_abilities),
                        _STARTED_ATTACHMENT_GROUP_METADATA_KEY: group_started,
                    },
                ),
            )

    @staticmethod
    def _matches_attachment_group(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        operation = event.metadata.get(_LIFECYCLE_OPERATION_METADATA_KEY)
        if not isinstance(operation, _BotLifecycleOperation):
            return False
        expected_kind = (
            "attach"
            if event.name in {attachment.AttachCompleteEvent.name, attachment.AttachFailedEvent.name}
            else "detach"
        )
        return (
            operation.kind == expected_kind
            and operation.is_active()
            and event.id == operation.request_id
            and event.source == hsm.id(instance._attachments)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _matches_activation_failure(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        operation = event.metadata.get(_LIFECYCLE_OPERATION_METADATA_KEY)
        return (
            isinstance(operation, _BotLifecycleOperation)
            and operation.kind == "attach"
            and operation.is_active()
            and event.id == operation.request_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _matches_cleanup_done(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
        kind: typing.Literal["activation", "deactivation"],
    ) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _BotCleanupData)
            and data.kind == kind
            and data.is_active()
            and event.id == data.request_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _matches_activation_cleanup_done(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        return Bot._matches_cleanup_done(ctx, instance, event, "activation")

    @staticmethod
    def _matches_deactivation_cleanup_done(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        return Bot._matches_cleanup_done(ctx, instance, event, "deactivation")

    @staticmethod
    async def _activation_cleanup_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        lifetime = instance.context()
        world = World.from_context(lifetime)
        # Deactivation may cancel activation before the activity can publish its
        # incrementally-started ownership set. Cleanup therefore owns the complete
        # configured lifecycle surface; stopping an actor that never started is
        # intentionally idempotent at this boundary.
        await hsm.stop(instance._attachments, world.context)
        for ability in reversed(Bot._lifecycle_abilities(instance)):
            await hsm.stop(ability, lifetime)
        for device in reversed(_device_tree(*instance._devices.values())):
            await hsm.stop(device, world.context)
        cleanup = _BotCleanupData.create(ctx, request_id=event.id, kind="activation")
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _BotCleanupDoneEvent.with_data(cleanup),
                id=cleanup.request_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
            ),
        )

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
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus, _reset_reboot_cleanup),
                hsm.target("../reboot_deactivating"),
            ),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_clear_focus),
                hsm.target("../activation_cleanup"),
            ),
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
                hsm.guard(_matches_activation_failure),
                hsm.target("../activation_cleanup"),
            ),
        ),
        hsm.state(
            "activation_cleanup",
            hsm.activity(_activation_cleanup_activity),
            hsm.transition(
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus, _reset_reboot_cleanup),
                hsm.target("../reboot_deactivating"),
            ),
            hsm.transition(
                hsm.on(_BotCleanupDoneEvent),
                hsm.guard(_matches_activation_cleanup_done),
                hsm.target("../inactive"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.target("../degraded"),
            ),
        ),
        hsm.state(
            "deactivating",
            hsm.activity(_deactivate_activity),
            hsm.transition(
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus),
                hsm.target("../reboot_deactivating"),
            ),
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
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus, _reset_reboot_cleanup),
                hsm.target("../reboot_cleanup"),
            ),
            hsm.transition(
                hsm.on(_BotCleanupDoneEvent),
                hsm.guard(_matches_deactivation_cleanup_done),
                hsm.target("../inactive"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.target("../degraded"),
            ),
        ),
        hsm.state(
            "reboot_deactivating",
            hsm.activity(_deactivate_activity),
            hsm.transition(
                hsm.on(attachment.DetachedEvent),
                hsm.guard(_matches_attachment_group),
                hsm.effect(_preserve_reboot_cleanup),
                hsm.target("../reboot_cleanup"),
            ),
            hsm.transition(
                hsm.on(attachment.DetachFailedEvent),
                hsm.guard(_matches_attachment_group),
                hsm.effect(_reset_reboot_cleanup),
                hsm.target("../reboot_cleanup"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.effect(_reset_reboot_cleanup),
                hsm.target("../reboot_cleanup"),
            ),
        ),
        hsm.state(
            "reboot_cleanup",
            hsm.activity(_deactivation_cleanup_activity),
            hsm.transition(
                hsm.on(_BotCleanupDoneEvent),
                hsm.guard(_matches_deactivation_cleanup_done),
                hsm.target("../activating"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.target("../degraded"),
            ),
        ),
        hsm.state("degraded"),
        hsm.state(
            "active",
            hsm.initial(hsm.target("unfocused")),
            hsm.transition(
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus),
                hsm.target("../reboot_deactivating"),
            ),
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
                    hsm.effect(_focus_event_target),
                    hsm.target("../processing"),
                ),
                # Sensory products: explicit cognition.InputEvent handoff (never raw world media).
                hsm.transition(
                    hsm.on(cognition.InputEvent),
                    hsm.effect(_focus_event_target),
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
                    hsm.target("../processing"),
                ),
                hsm.transition(
                    hsm.on(cognition.InputEvent),
                    hsm.target("../processing"),
                ),
            ),
            hsm.state(
                "processing",
                hsm.activity(_dispatch_bot_processing),
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
                    hsm.effect(_consume_bot_processing),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingCompletedEvent),
                    hsm.guard(_processing_completed),
                    hsm.effect(_consume_bot_processing),
                    hsm.target("../unfocused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingFailedEvent),
                    hsm.guard(_processing_failed_with_focus),
                    hsm.effect(_consume_bot_processing),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingFailedEvent),
                    hsm.guard(_processing_failed),
                    hsm.effect(_consume_bot_processing),
                    hsm.target("../unfocused"),
                ),
                hsm.transition(
                    hsm.on(_BotProcessingTimedOutEvent),
                    hsm.guard(_matches_processing_timeout),
                    hsm.effect(_cancel_bot_processing),
                    hsm.target("../cancelling_processing"),
                ),
            ),
            hsm.state(
                "cancelling_processing",
                hsm.defer(events.InputEvent),
                hsm.defer(cognition.InputEvent),
                hsm.transition(
                    hsm.on(cognition.CancelledEvent),
                    hsm.guard(_matches_cognition_cancelled),
                    hsm.guard(_has_focused_device),
                    hsm.effect(_dispatch_processing_timeout_failure, _consume_bot_processing),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(cognition.CancelledEvent),
                    hsm.guard(_matches_cognition_cancelled),
                    hsm.effect(_dispatch_processing_timeout_failure, _consume_bot_processing),
                    hsm.target("../unfocused"),
                ),
                hsm.transition(
                    hsm.on(_BotProcessingCancelTimedOutEvent),
                    hsm.guard(_matches_processing_timeout),
                    hsm.effect(_dispatch_processing_timeout_failure, _consume_bot_processing),
                    hsm.target("/Bot/degraded"),
                ),
            ),
        ),
        hsm.observe(observer),
    )
