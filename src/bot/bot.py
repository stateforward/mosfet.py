from bot.abilities import cognition
from bot.abilities import processing

import abc
import asyncio
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
_MIN_BOT_CANCELLATION_TIMEOUT = datetime.timedelta(milliseconds=100)
_MAX_BOT_CANCELLATION_TIMEOUT = datetime.timedelta(seconds=5)

# hsm surfaces idempotency conditions only as exception messages ("already has a running HSM"
# on start, "dispatch requires a started HSM" on dispatch). The predicates below implement
# idempotent attach/detach and canonical-inventory reactivation; they are not peer-state
# gating (HSM-CONTEXT-001).
_HSM_ALREADY_RUNNING_MESSAGE = "already has a running HSM"
_HSM_NOT_STARTED_MESSAGE = "dispatch requires a started HSM"


def _is_already_running_error(error: Exception) -> bool:
    return isinstance(error, hsm.ErrorValidatingModel) and _HSM_ALREADY_RUNNING_MESSAGE in str(error)


def _is_not_started_error(error: Exception) -> bool:
    return isinstance(error, RuntimeError) and _HSM_NOT_STARTED_MESSAGE in str(error)


class _BotLifecycleTerminalData(pydantic.BaseModel):
    """Immutable typed terminal for one lifecycle run, correlated by request id."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    request_id: str = pydantic.Field(min_length=1)
    kind: typing.Literal["attach", "detach"]


_BotLifecycleCompletedEvent = hsm.Event[_BotLifecycleTerminalData](
    name="bot.lifecycle.completed",
    kind=hsm.CompletionEventKind,
    schema=_BotLifecycleTerminalData,
)
_BotLifecycleFailedEvent = hsm.Event[_BotLifecycleTerminalData](
    name="bot.lifecycle.failed",
    kind=hsm.ErrorEventKind,
    schema=_BotLifecycleTerminalData,
)


class _BotLifecycleReply(hsm.Instance):
    """One-shot reply inbox that hands the attachment group terminal to the owning activity."""


def _lifecycle_reply_model(request_id: str, terminal: asyncio.Future[hsm.Event[typing.Any]]) -> hsm.Model:
    def correlated(
        ctx: hsm.Context,
        instance: _BotLifecycleReply,
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return event.id == request_id and isinstance(
            event.data,
            (attachment.AttachCompleteData, attachment.DetachedData, attachment.FailedData),
        )

    def complete(
        ctx: hsm.Context,
        instance: _BotLifecycleReply,
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, instance
        if not terminal.done():
            terminal.set_result(event)

    return hsm.define(
        "BotLifecycleReply",
        hsm.initial(hsm.target("waiting")),
        hsm.state(
            "waiting",
            hsm.transition(
                hsm.on(
                    attachment.AttachCompleteEvent,
                    attachment.AttachFailedEvent,
                    attachment.DetachedEvent,
                    attachment.DetachFailedEvent,
                ),
                hsm.guard(correlated),
                hsm.effect(complete),
                hsm.target("/BotLifecycleReply/done"),
            ),
        ),
        hsm.final("done"),
    )


class _BotCleanupData(pydantic.BaseModel):
    """Private exact terminal for one lifecycle cleanup activity."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    request_id: str = pydantic.Field(min_length=1)
    kind: typing.Literal["activation", "deactivation"]


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


class _BotProcessingOperation(hsm.Instance):
    """Single-shot timer actor scoped to one Bot processing turn.

    Identity is closure-bound in the model, never recovered from shared registries. The
    owning state's activity starts the actor and stops it on state exit (hsm 1.1.4 does not
    stop machines on context cancel), and the actor withholds its timeout once the owning
    activity is canceled. A timeout event already dispatched before that stop lands is
    correlated only by its self-certifying payload; a later turn can still dequeue it first
    in a narrow RTC-ordering window. Closing that window would need per-turn correlation
    storage the contract forbids (HSM-COMPLETION-001), so it is an accepted residual.
    """

    @classmethod
    def _model(
        cls,
        *,
        owner: hsm.Instance,
        request_id: str,
        delay: datetime.timedelta,
        event_template: hsm.Event[_BotProcessingOperationData],
    ) -> hsm.Model:
        def timeout_delay(
            ctx: hsm.Context,
            instance: _BotProcessingOperation,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, instance, event
            return delay

        def dispatch_timeout(
            ctx: hsm.Context,
            instance: _BotProcessingOperation,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            if ctx.is_done():
                # The owning activity was canceled: this timer's turn is over, so the timeout
                # must not be dispatched (HSM-CONTEXT-001: is_done is a cancel signal for work
                # owned by the canceled activity, never a liveness gate).
                return
            capability = _BotProcessingOperationData(
                token=hsm.id(instance),
                request_id=request_id,
                actor_id=hsm.id(instance),
            )
            _ = hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    event_template.with_data(capability),
                    id=request_id,
                    source=hsm.id(instance),
                    target=hsm.id(owner),
                ),
            )

        return hsm.define(
            "BotProcessingTimer",
            hsm.initial(hsm.target("waiting")),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.after(timeout_delay),
                    hsm.effect(dispatch_timeout),
                    hsm.target("/BotProcessingTimer/done"),
                ),
            ),
            hsm.final("done"),
        )

    @classmethod
    async def started(
        cls,
        ctx: hsm.Context,
        operation: "_BotProcessingOperation",
        *,
        owner: "Bot",
        request_id: str,
        delay: datetime.timedelta,
        event_template: hsm.Event[_BotProcessingOperationData],
    ) -> "_BotProcessingOperation":
        try:
            return await hsm.started(
                _private_scope(ctx),
                operation,
                cls._model(
                    owner=owner,
                    request_id=request_id,
                    delay=delay,
                    event_template=event_template,
                ),
            )
        except asyncio.CancelledError:
            await hsm.stop(operation, hsm.Context())
            raise


def _device_tree(*roots: Device) -> tuple[Device, ...]:
    return Device.device_tree(*roots)


def _private_scope(parent: hsm.Context) -> hsm.Context:
    """Child context with a private Instances map, off the world broadcast set.

    World ``dispatch_all`` delivers to every instance in the world's Instances map. Actors
    started under this scope (bot-owned abilities, ephemeral reply and timer actors) receive
    events only through explicit dispatch, and cancel with ``parent``.
    """

    values: dict[typing.Hashable, object] = {hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()}
    return hsm.Context(parent=parent, values=values)


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
        try:
            _ = await hsm.started(world.context, self, self.model)
        except hsm.ErrorValidatingModel as error:
            if not _is_already_running_error(error):
                raise
        await self.dispatch(world.context, events.ActivateEvent.with_data(events.ActivateEventData()))
        return self

    async def detach(self, world: World) -> typing.Self:
        require_world_scope(world, self, participant="Bot")
        try:
            await self.dispatch(world.context, events.DeactivateEvent.with_data(events.DeactivateEventData()))
        except RuntimeError as error:
            # An unstarted or stopped bot is already detached; deactivation is idempotent.
            if not _is_not_started_error(error):
                raise
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
    async def _request_attachment_terminal(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event,
        *,
        kind: typing.Literal["attach", "detach"],
    ) -> None:
        """Run one attachment group operation and emit the bot-private lifecycle terminal.

        The group replies to a one-shot inbox addressed through ``reply_to``; the activity
        alone emits the terminal, so a canceled run can never deliver a stale completion.
        """

        terminal: asyncio.Future[hsm.Event[typing.Any]] = asyncio.get_running_loop().create_future()
        reply = _BotLifecycleReply()
        try:
            _ = await hsm.started(_private_scope(ctx), reply, _lifecycle_reply_model(event.id, terminal))
            request: hsm.Event[typing.Any]
            if kind == "attach":
                request = attachment.AttachEvent.with_data(attachment.AttachData(actor=instance, reply_to=reply))
            else:
                request = attachment.DetachEvent.with_data(attachment.DetachData(actor=instance, reply_to=reply))
            request = dataclasses.replace(
                request,
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(instance._attachments),
                metadata=dict(event.metadata),
            )
            if kind == "attach":
                await instance._attachments.attach(ctx, typing.cast(hsm.Event[attachment.AttachData], request))
            else:
                await instance._attachments.detach(ctx, typing.cast(hsm.Event[attachment.DetachData], request))
            outcome = await terminal
        finally:
            await hsm.stop(reply, hsm.Context())
        outcome_data = _BotLifecycleTerminalData(request_id=event.id, kind=kind)
        if isinstance(outcome.data, (attachment.AttachCompleteData, attachment.DetachedData)):
            terminal_event = _BotLifecycleCompletedEvent.with_data(outcome_data)
        else:
            terminal_event = _BotLifecycleFailedEvent.with_data(outcome_data)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                terminal_event,
                id=outcome_data.request_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _deactivate_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        await Bot._request_attachment_terminal(ctx, instance, event, kind="detach")

    @staticmethod
    async def _deactivation_cleanup(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event,
        *,
        cleanup: typing.Literal["preserve", "reset"] | None,
    ) -> None:
        lifetime = instance.context()
        world = World.from_context(lifetime)
        if cleanup == "reset":
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
        elif cleanup is None:
            await hsm.stop(instance._attachments, world.context)
            for ability in Bot._lifecycle_abilities(instance):
                # Cognition's required composite tree remains detached under Bot lifetime;
                # stopping it cancels the child contexts needed by a later activation.
                if ability is instance._cognition:
                    continue
                await hsm.stop(ability, lifetime)
        terminal = _BotCleanupData(request_id=event.id, kind="deactivation")
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _BotCleanupDoneEvent.with_data(terminal),
                id=terminal.request_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _deactivation_cleanup_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        await Bot._deactivation_cleanup(ctx, instance, event, cleanup=None)

    @staticmethod
    async def _reboot_cleanup_preserve_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        await Bot._deactivation_cleanup(ctx, instance, event, cleanup="preserve")

    @staticmethod
    async def _reboot_cleanup_reset_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        await Bot._deactivation_cleanup(ctx, instance, event, cleanup="reset")

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
        try:
            cognition_id = hsm.id(instance._cognition)
        except hsm.ErrorValidatingModel:
            # Cognition is not started yet (early activation); it cannot have requested reboot.
            return False
        return (
            isinstance(event.data, events.RebootEventData)
            and event.source == cognition_id
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _ability_selected_configured_focus_device(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, events.FocusDeviceEventData)
            and event.source == hsm.id(instance._cognition)
            and event.target == hsm.id(instance)
            and event.data.device in instance._devices
        )

    @staticmethod
    def _ability_selected_clear_focus(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, events.ClearFocusEventData)
            and event.source == hsm.id(instance._cognition)
            and event.target == hsm.id(instance)
            and instance._focused_device is not None
        )

    @staticmethod
    def _processing_completed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return (
            isinstance(event.data, events.ProcessingCompletedEventData)
            and event.source == hsm.id(instance._cognition)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _processing_failed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return (
            isinstance(event.data, events.ProcessingFailedEventData)
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
        # The turn timer rides this state's activity: the activity holds the state until HSM
        # exits it, then stops the timer (hsm 1.1.4 does not stop machines on context cancel;
        # the stop is explicit). The timer also withholds its event once this scope is done.
        operation = _BotProcessingOperation()
        await _BotProcessingOperation.started(
            ctx,
            operation,
            owner=instance,
            request_id=event.id,
            delay=instance._processing_timeout,
            event_template=_BotProcessingTimedOutEvent,
        )
        try:
            input_event = dataclasses.replace(
                cognition.InputEvent.with_data(cognition_input),
                id=event.id,
                metadata=dict(event.metadata),
            )
            _ = hsm.dispatch(ctx, instance._cognition, input_event)
            await asyncio.wrap_future(ctx.Done())
        finally:
            await hsm.stop(operation, hsm.Context())

    @staticmethod
    def _cancel_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _BotProcessingOperationData)
        ability = instance._cognition
        cancel_event = ability.cancel_event or processing.CancelEvent
        schema = cancel_event.schema
        assert isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)
        _ = hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                cancel_event.with_data(schema(operation_id=data.request_id, token=data.token)),
                id=data.request_id,
                source=hsm.id(instance),
                target=hsm.id(ability),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _matches_bot_processing_output(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        ability = instance._cognition
        return (
            event.name == ability.output_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(ability)
        )

    @staticmethod
    def _matches_bot_processing_failure(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        ability = instance._cognition
        return (
            event.name == ability.failed_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(ability)
        )

    @staticmethod
    def _complete_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        focus_candidates: tuple[str, ...] = ()
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
        data = event.data
        if isinstance(data, _BotProcessingOperationData):
            request_id = data.request_id
        elif isinstance(data, (cognition.CancelledData, processing.CancelledData)):
            request_id = data.operation_id
        else:
            raise AssertionError(f"unsupported processing timeout event data: {type(data)!r}")
        seconds = instance._processing_timeout.total_seconds()
        failure = events.ProcessingFailedEventData(message=f"Bot processing timed out after {seconds:g} seconds.")
        _ = instance.dispatch(
            ctx,
            dataclasses.replace(
                events.ProcessingFailedEvent.with_data(failure),
                id=request_id,
                source=hsm.id(instance._cognition),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
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
        data = event.data
        return (
            isinstance(data, _BotProcessingOperationData)
            and event.id == data.request_id
            and event.source == data.actor_id
            and data.token == data.actor_id
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _matches_cognition_cancelled(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        ability = instance._cognition
        cancelled_event = ability.cancelled_event or processing.CancelledEvent
        data = event.data
        return (
            event.name == cancelled_event.name
            and isinstance(data, (cognition.CancelledData, processing.CancelledData))
            and event.id == data.operation_id
            and event.source == hsm.id(ability)
            and event.target == hsm.id(instance)
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
        try:
            group_scope = _private_scope(lifetime)
            try:
                _ = await hsm.started(group_scope, instance._attachments, instance._attachments.model)
            except hsm.ErrorValidatingModel as error:
                if not _is_already_running_error(error):
                    raise
            for device in _device_tree(*instance._devices.values()):
                require_world_scope(world, device, participant="Device")
                model = device.model
                if model is None:
                    raise RuntimeError(f"{type(device).__name__} has no lifecycle model.")
                try:
                    _ = await hsm.started(world.context, device, model)
                except hsm.ErrorValidatingModel as error:
                    if not _is_already_running_error(error):
                        raise
            ability_scope = _private_scope(lifetime)
            for ability in Bot._lifecycle_abilities(instance):
                model = ability.model
                if model is None:
                    raise RuntimeError(f"{type(ability).__name__} has no lifecycle model.")
                try:
                    _ = await hsm.started(ability_scope, ability, model)
                except hsm.ErrorValidatingModel as error:
                    # Residual peer inspection: skip abilities already running under our
                    # private scope (idempotent reactivation), but raise when an injected
                    # ability runs under the world scope, where it would receive world
                    # broadcasts directly. Removing this needs typed errors from hsm.
                    shares_world_instances = ability.context().value(hsm.Keys.Instances) is world.context.value(
                        hsm.Keys.Instances
                    )
                    if not _is_already_running_error(error) or shares_world_instances:
                        raise
            await Bot._request_attachment_terminal(ctx, instance, event, kind="attach")
        except asyncio.CancelledError:
            raise
        except Exception:
            # Public notification for observers; the private terminal drives topology.
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    events.ActivatingFailedEvent.with_data(events.ActivatingFailedEventData()),
                    id=event.id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )
            failure = _BotLifecycleTerminalData(request_id=event.id, kind="attach")
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _BotLifecycleFailedEvent.with_data(failure),
                    id=failure.request_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _matches_lifecycle_terminal(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
        *,
        kind: typing.Literal["attach", "detach"],
    ) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _BotLifecycleTerminalData)
            and data.kind == kind
            and event.id == data.request_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _matches_attach_terminal(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        return Bot._matches_lifecycle_terminal(ctx, instance, event, kind="attach")

    @staticmethod
    def _matches_detach_terminal(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        return Bot._matches_lifecycle_terminal(ctx, instance, event, kind="detach")

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
        terminal = _BotCleanupData(request_id=event.id, kind="activation")
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _BotCleanupDoneEvent.with_data(terminal),
                id=terminal.request_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _cancelling_processing_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, _BotProcessingOperationData)
        operation = _BotProcessingOperation()
        await _BotProcessingOperation.started(
            ctx,
            operation,
            owner=instance,
            request_id=data.request_id,
            delay=min(
                max(instance._processing_timeout, _MIN_BOT_CANCELLATION_TIMEOUT),
                _MAX_BOT_CANCELLATION_TIMEOUT,
            ),
            event_template=_BotProcessingCancelTimedOutEvent,
        )
        try:
            await asyncio.wrap_future(ctx.Done())
        finally:
            await hsm.stop(operation, hsm.Context())

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
                hsm.effect(_clear_focus),
                hsm.target("../reboot_deactivating"),
            ),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_clear_focus),
                hsm.target("../activation_cleanup"),
            ),
            hsm.transition(
                hsm.on(_BotLifecycleCompletedEvent),
                hsm.guard(_matches_attach_terminal),
                hsm.target("../active"),
            ),
            hsm.transition(
                hsm.on(_BotLifecycleFailedEvent),
                hsm.guard(_matches_attach_terminal),
                hsm.target("../activation_cleanup"),
            ),
        ),
        hsm.state(
            "activation_cleanup",
            hsm.activity(_activation_cleanup_activity),
            hsm.transition(
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus),
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
                hsm.on(_BotLifecycleCompletedEvent, _BotLifecycleFailedEvent),
                hsm.guard(_matches_detach_terminal),
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
                hsm.effect(_clear_focus),
                hsm.target("../reboot_cleanup_reset"),
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
                hsm.on(_BotLifecycleCompletedEvent),
                hsm.guard(_matches_detach_terminal),
                hsm.target("../reboot_cleanup_preserve"),
            ),
            hsm.transition(
                hsm.on(_BotLifecycleFailedEvent),
                hsm.guard(_matches_detach_terminal),
                hsm.target("../reboot_cleanup_reset"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.target("../reboot_cleanup_reset"),
            ),
        ),
        hsm.state(
            "reboot_cleanup_preserve",
            hsm.activity(_reboot_cleanup_preserve_activity),
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
        hsm.state(
            "reboot_cleanup_reset",
            hsm.activity(_reboot_cleanup_reset_activity),
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
            ),
            hsm.state(
                "focused",
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
                    hsm.on(_BotProcessingTimedOutEvent),
                    hsm.guard(_matches_processing_timeout),
                    hsm.effect(_cancel_bot_processing),
                    hsm.target("../cancelling_processing"),
                ),
            ),
            hsm.state(
                "cancelling_processing",
                hsm.activity(_cancelling_processing_activity),
                hsm.defer(events.InputEvent),
                hsm.defer(cognition.InputEvent),
                hsm.transition(
                    hsm.on(cognition.CancelledEvent, processing.CancelledEvent),
                    hsm.guard(_matches_cognition_cancelled),
                    hsm.guard(_has_focused_device),
                    hsm.effect(_dispatch_processing_timeout_failure),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(cognition.CancelledEvent, processing.CancelledEvent),
                    hsm.guard(_matches_cognition_cancelled),
                    hsm.effect(_dispatch_processing_timeout_failure),
                    hsm.target("../unfocused"),
                ),
                hsm.transition(
                    hsm.on(_BotProcessingCancelTimedOutEvent),
                    hsm.guard(_matches_processing_timeout),
                    hsm.effect(_dispatch_processing_timeout_failure),
                    hsm.target("/Bot/degraded"),
                ),
            ),
        ),
        hsm.observe(observer),
    )
