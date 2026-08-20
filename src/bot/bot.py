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

import bot
from bot import abilities
from bot import lifecycle
from bot.protocols import attachment
from . import events

from bot.device import Device
from bot import telemetry
from bot.telemetry import observer
from bot.telemetry import span
from bot.environment import SoundEvent, VisualEvent, Environment, require_environment_scope, space

_DEFAULT_BOT_PROCESSING_TIMEOUT = datetime.timedelta(minutes=5)
_DEFAULT_BOT_DEACTIVATION_TIMEOUT = datetime.timedelta(minutes=5)
_MIN_BOT_CANCELLATION_TIMEOUT = datetime.timedelta(milliseconds=100)
_MAX_BOT_CANCELLATION_TIMEOUT = datetime.timedelta(seconds=5)
_OWNED_DEVICES_ATTRIBUTE = "owned_devices"

# Lifecycle idempotency for attach/detach/activate (not peer-state gating; HSM-CONTEXT-001).
# Prefer typed HSM errors when present. Stock stateforward-hsm still often surfaces these
# conditions as fixed exception prose (ErrorAlreadyStarted / ErrorMissingHSM are exported
# but not always raised). All residual message detection is confined to these two predicates
# — call sites must not open-code hsm exception text.


def _is_already_running_error(error: BaseException) -> bool:
    if isinstance(error, hsm.ErrorAlreadyStarted):
        return True
    if isinstance(error, hsm.ErrorValidatingModel | RuntimeError):
        message = str(error)
        return "already has a running HSM" in message or "already started HSM" in message
    return False


def _is_not_started_error(error: BaseException) -> bool:
    if isinstance(error, hsm.ErrorMissingHSM):
        return True
    if isinstance(error, RuntimeError):
        message = str(error)
        return (
            "dispatch requires a started HSM" in message
            or "take snapshot requires a started HSM" in message
            or "operation requires a started HSM" in message
            or "restart requires a started HSM" in message
            or "set requires a started HSM" in message
        )
    return False


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


def _bot_cancel_operation_id(request_id: str, token: str, owner: hsm.Instance) -> str:
    """Live cancel capability identity for one Bot-issued processing cancellation."""

    return processing.cancellation_operation_id(request_id, request_id, token, hsm.id(owner))


def _event_belongs_to_turn(event: hsm.Event[typing.Any], request_id: str) -> bool:
    """Return whether an event is the turn root or a child id under that turn."""

    return bool(event.id) and (event.id == request_id or event.id.startswith(f"{request_id}:"))


def _active_bot_turn_id(instance: "Bot", event: hsm.Event[typing.Any]) -> str | None:
    """Return the live Bot turn id this event belongs to, if any.

    Turn identity is a ``processing.Operation`` owned by Bot (same live-capability pattern as
    Cognition). Timers stay activity-scoped and are never consulted for correlation.
    """

    if not event.id:
        return None
    if processing.active_operation(instance, event.id) is not None:
        return event.id
    parent, separator, _suffix = event.id.partition(":")
    if separator and parent and processing.active_operation(instance, parent) is not None:
        return parent
    return None


class _BotProcessingOperation(hsm.Instance):
    """Single-shot timer actor scoped to one Bot processing turn.

    Identity is closure-bound in the model. The owning state's activity starts the actor and
    stops it on state exit (hsm 1.1.4 does not stop machines on context cancel), and the actor
    withholds its timeout once the owning activity is canceled. Turn correlation is owned by a
    separate ``processing.Operation`` on Bot, not by this timer or a shared Instances rewrite.
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


def _private_scope(parent: hsm.Context) -> hsm.Context:
    """Child context with a private Instances map, off the environment addressing map.

    ``hsm.dispatch_all`` delivers to every instance in the surrounding Instances map. Actors
    started under this scope (bot-owned abilities, ephemeral reply and timer actors) receive
    events only through explicit dispatch, and cancel with ``parent``. The environment presence set
    is deliberately *not* shadowed: an actor here can emit into the environment without ever being a
    broadcast recipient, which is presence's job to decide.
    """

    values: dict[typing.Hashable, object] = {hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()}
    return hsm.Context(parent=parent, values=values)


class Bot(hsm.Instance, abc.ABC):
    """Interrupt-driven bot that observes and processes events while active."""

    _innate_abilities: typing.ClassVar[tuple[type[abilities.Ability[typing.Any, typing.Any]], ...]] = ()
    _processing_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_PROCESSING_TIMEOUT
    _deactivation_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_DEACTIVATION_TIMEOUT
    _devices: dict[str, Device]
    # Live actor id → configured device reference. Built when owned devices (and the
    # peripherals they power) start; cleared when they stop. Hot-path ownership is map
    # lookup only — never a per-event DFS of the device tree.
    _device_source_refs: dict[str, str]
    _cognition: abilities.Ability[cognition.InputData, typing.Any]
    _focused_device: str | None
    # Turn-scoped attention policy for the active processing operation (body-owned).
    # Set when the processing activity starts; cleared when the turn retires.
    _processing_focus_candidates: tuple[str, ...]
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
        self._device_source_refs = {}
        self._cognition = cognition
        self._focused_device = None
        self._processing_focus_candidates = ()
        self._innate_ability_instances = tuple(ability_type() for ability_type in self._innate_abilities)
        self._input = tuple(input)
        self._output = tuple(output)
        self._acquired_abilities = tuple(acquired_abilities)
        self._attachments = attachment.Group(*Bot._lifecycle_attachment_members(self))

    async def attach(self, environment: Environment, *, placement: space.Placement | None = None) -> typing.Self:
        require_environment_scope(environment, self, participant="Bot")
        try:
            _ = await bot.started(environment, self, self._model_for_instance())
        except Exception as error:
            if not _is_already_running_error(error):
                raise
        # A Bot is not a Device: its presence is an attach-time decision, and detach ends it.
        # Unconditional: the already-running branch above swallows its error, and join is idempotent.
        environment.join(self, placement=placement)
        await self.dispatch(environment, events.ActivateEvent.with_data(events.ActivateEventData()))
        return self

    def _model_for_instance(self) -> hsm.Model:
        return self.model

    def _model_for_device(self, device: Device) -> hsm.Model:
        model = device.model
        if model is None:
            raise RuntimeError(f"{type(device).__name__} has no lifecycle model.")
        return model

    async def detach(self, environment: Environment) -> typing.Self:
        require_environment_scope(environment, self, participant="Bot")
        try:
            await self.dispatch(environment, events.DeactivateEvent.with_data(events.DeactivateEventData()))
        except Exception as error:
            # An unstarted or stopped bot is already detached; deactivation is idempotent.
            if not _is_not_started_error(error):
                raise
        # Presence is attachment-scoped, not activation-scoped: a deactivated but still attached
        # bot is legitimately in the environment, so only detach ends it — reboot deactivates without
        # detaching and keeps presence. Pairs with the join in attach; no test can observe the
        # removal (a stopped or inactive bot ignores broadcasts either way), so this is ownership
        # completeness and map hygiene. Do not delete it as untested.
        environment.leave(self)
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
        environment = Environment.from_context(lifetime)
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
            await hsm.stop(instance._attachments, environment)
            instance._attachments = attachment.Group(*Bot._lifecycle_attachment_members(instance))
        elif cleanup is None:
            await hsm.stop(instance._attachments, environment)
            for ability in Bot._lifecycle_abilities(instance):
                # Cognition's required composite tree remains detached under Bot lifetime;
                # stopping it cancels the child contexts needed by a later activation.
                if ability is instance._cognition:
                    continue
                await hsm.stop(ability, lifetime)
        processing.finish_operations(ctx, instance)
        # Ownership map is only valid while active; rebuild happens on next active entry.
        Bot._clear_owned_device_sources(ctx, instance, event)
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
        """Resolve a live actor id to a configured device ref via the start-time ownership map."""

        return instance._device_source_refs.get(source)

    @staticmethod
    def _device_references_for_event(instance: "Bot", event: hsm.Event[typing.Any]) -> tuple[str, ...]:
        """Map stimulus provenance to configured device refs via envelope source id only."""

        source_reference = Bot._device_reference_for_source(instance, event.source) if event.source else None
        if source_reference is not None:
            return (source_reference,)
        return ()

    @staticmethod
    def _interrupt_reference(instance: "Bot", input: events.InputEventData, source: str) -> str | None:
        """Which of this bot's devices an interrupt arrived from.

        Prefer a stamped ``target_device`` when the producer already carried ownership. Otherwise
        resolve the envelope source id through the body-owned flat map registered when devices
        (and known peripherals) started — never a per-event tree walk. Identity only: this never
        reads what happened to decide whose interrupt it is.
        """

        if input.target_device is not None:
            return input.target_device
        return Bot._device_reference_for_source(instance, source) if source else None

    @staticmethod
    def _target_device_reference(instance: "Bot", input: events.BotInputData, source: str = "") -> str | None:
        if isinstance(input, events.InputEventData):
            return Bot._interrupt_reference(instance, input, source)
        references = Bot._device_references_for_event(instance, input)
        return references[0] if len(references) == 1 else None

    @staticmethod
    def _input_device_references(instance: "Bot", input: events.BotInputData, source: str = "") -> tuple[str, ...]:
        if isinstance(input, events.InputEventData):
            reference = Bot._interrupt_reference(instance, input, source)
            return (reference,) if reference in instance._devices else ()
        references = Bot._device_references_for_event(instance, input)
        if not input.source and instance._focused_device in references:
            assert instance._focused_device is not None
            return (instance._focused_device,)
        return references

    @staticmethod
    def _processing_device_references(instance: "Bot", input: events.BotInputData, source: str = "") -> tuple[str, ...]:
        references: list[str] = []
        if instance._focused_device in instance._devices:
            assert instance._focused_device is not None
            references.append(instance._focused_device)
        for target_device in Bot._input_device_references(instance, input, source):
            if target_device not in references:
                references.append(target_device)
        return tuple(references)

    @staticmethod
    def _input_targets_configured_device(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        if not isinstance(event.data, events.InputEventData):
            return False
        return Bot._interrupt_reference(instance, event.data, event.source) in instance._devices

    @staticmethod
    def _fan_out_input(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        with span.operation(
            "bot.body.fan_out_stimulus",
            scope="bot.body",
            component="body",
            stage="stimulus_fan_out",
            context=telemetry.event_context(event),
        ) as active:
            # A body with no input abilities attached hears nothing, and looks exactly like a body
            # whose abilities dropped everything. The count is the difference.
            active.set_attribute("bot.stimulus.name", event.name)
            active.set_attribute("bot.ability.input.count", len(instance._input))
            for ability in instance._input:
                # Stamped on the way out: the ability handles this on its own task, where the
                # ambient context is bring-up's, not this stimulus's.
                _ = hsm.dispatch(
                    ctx,
                    ability,
                    telemetry.inject_context(
                        dataclasses.replace(event, target=hsm.id(ability), metadata=dict(event.metadata))
                    ),
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
        # Cognition not started yet (early activation) cannot have requested reboot.
        if not lifecycle.is_started(instance._cognition):
            return False
        cognition_id = hsm.id(instance._cognition)
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
        turn_id = _active_bot_turn_id(instance, event)
        candidates = instance._processing_focus_candidates
        return (
            isinstance(event.data, events.FocusDeviceEventData)
            and turn_id is not None
            and _event_belongs_to_turn(event, turn_id)
            and event.source == hsm.id(instance._cognition)
            and event.target == hsm.id(instance)
            and event.data.device in instance._devices
            # Fail-closed: empty turn candidates means no legal focus target (match
            # attention_selection_error). Never accept any configured device when candidates is ().
            and event.data.device in candidates
        )

    @staticmethod
    def _ability_selected_clear_focus(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        turn_id = _active_bot_turn_id(instance, event)
        return (
            isinstance(event.data, events.ClearFocusEventData)
            and turn_id is not None
            and _event_belongs_to_turn(event, turn_id)
            and event.source == hsm.id(instance._cognition)
            and event.target == hsm.id(instance)
            and instance._focused_device is not None
        )

    @staticmethod
    def attention_selection_error(
        selection: processing.SelectedEvent,
        *,
        focus_candidates: tuple[str, ...],
        configured_device_names: frozenset[str],
        enforce_candidates: bool,
        focused_device: str | None = None,
    ) -> str | None:
        """Return a fail-closed error for illegal body-attention selections, else None.

        Body owns attention policy. Cognition calls this before dispatch so illegal
        focus/clear selections fail the turn instead of silently dropping at a guard.
        """

        if selection.event == events.FocusDeviceEvent.name:
            if selection.target is not None and selection.target != "bot":
                return "Processing selected focus_device outside available device candidates."
            data = events.FocusDeviceEventData.model_validate(selection.data or {})
            if enforce_candidates and data.device not in focus_candidates:
                return "Processing selected focus_device outside available device candidates."
            if configured_device_names and data.device not in configured_device_names:
                return "Processing selected focus_device outside available device candidates."
            return None
        if selection.event == events.ClearFocusEvent.name:
            if selection.target not in (None, "bot"):
                return "Processing selected clear_focus for a non-bot target."
            if not focused_device:
                return "Processing selected clear_focus with no focused device."
            return None
        return None

    @staticmethod
    def _processing_completed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        turn_id = _active_bot_turn_id(instance, event)
        return (
            isinstance(event.data, events.ProcessingCompletedEventData)
            and turn_id is not None
            and event.id == turn_id
            and event.source == hsm.id(instance._cognition)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _processing_failed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        turn_id = _active_bot_turn_id(instance, event)
        return (
            isinstance(event.data, events.ProcessingFailedEventData)
            and turn_id is not None
            and event.id == turn_id
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
    def _ability_actor_key(ability: abilities.Ability[typing.Any, typing.Any], actors: dict[str, hsm.Instance]) -> str:
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
        return key

    @staticmethod
    def _dispatch_actors(instance: "Bot") -> dict[str, hsm.Instance]:
        """Map cognition-visible actors: devices, abilities, and nested ability actors.

        Abilities that expose ``nested_actors()`` (e.g. Communication → conversation) are
        flattened so tools resolve ``conversation.input`` without dual-acquiring Conversation.
        """

        actors: dict[str, hsm.Instance] = {"bot": instance, **instance._devices}
        for ability in (
            *instance._input,
            *instance._output,
            *instance._innate_ability_instances,
            *instance._acquired_abilities,
        ):
            key = Bot._ability_actor_key(ability, actors)
            actors[key] = ability
            nested_method = getattr(ability, "nested_actors", None)
            if not callable(nested_method):
                continue
            nested_map = nested_method()
            if not isinstance(nested_map, collections.abc.Mapping):
                continue
            typed_nested = typing.cast(collections.abc.Mapping[object, object], nested_map)
            for nested_key, nested_actor in typed_nested.items():
                if not isinstance(nested_key, str) or not nested_key:
                    continue
                if not isinstance(nested_actor, hsm.Instance):
                    continue
                resolved = nested_key
                if resolved in actors:
                    suffix = 2
                    while f"{resolved}_{suffix}" in actors:
                        suffix += 1
                    resolved = f"{resolved}_{suffix}"
                actors[resolved] = nested_actor
        return actors

    @staticmethod
    async def _dispatch_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        # The span covers admitting the stimulus and handing it to cognition, and stops there:
        # the wait below ends by cancellation on every ordinary turn, which is not a failure of
        # the handoff.
        with span.operation(
            "bot.body.cognition_handoff",
            scope="bot.body",
            component="body",
            stage="cognition_handoff",
            context=telemetry.event_context(event),
        ) as active:
            if isinstance(event.data, events.InputEventData):
                stimulus = event.data
                active.set_attribute("bot.handoff.kind", "device_stimulus")
            elif isinstance(event.data, cognition.InputData):
                # Sensory / contribution products hand cognition.InputEvent (Listening pattern).
                # Body admits the handoff; it does not inspect conversation or other product domains.
                stimulus = event.data.stimulus
                active.set_attribute("bot.handoff.kind", "ability_product")
            else:
                raise AssertionError(f"unsupported body processing event data: {type(event.data)!r}")
            focus_candidates = Bot._processing_device_references(instance, stimulus, event.source)
            instance._processing_focus_candidates = focus_candidates
            active.set_attribute("bot.device.focus_candidate.count", len(focus_candidates))
            active.set_attribute("bot.ability.count", len(Bot._lifecycle_abilities(instance)))
            cognition_input = cognition.InputData(
                stimulus=stimulus,
                abilities=Bot._lifecycle_abilities(instance),
                actors=Bot._dispatch_actors(instance),
                focus=instance._focused_device if instance._focused_device in instance._devices else None,
                focus_candidates=focus_candidates,
            )
        # Live turn capability + activity-owned timer. The cancel-token capability is minted with
        # the timer id so a cancelled confirmation can match immediately (before the cancelling
        # activity runs). Successful terminals retire both; timeout keeps only the cancel token.
        request_id = event.id
        assert request_id
        _ = await processing.start_operation(instance, request_id)
        timer = _BotProcessingOperation()
        _ = await _BotProcessingOperation.started(
            ctx,
            timer,
            owner=instance,
            request_id=request_id,
            delay=instance._processing_timeout,
            event_template=_BotProcessingTimedOutEvent,
        )
        cancel_id = _bot_cancel_operation_id(request_id, hsm.id(timer), instance)
        _ = await processing.start_operation(instance, cancel_id)
        try:
            with span.operation(
                "bot.body.cognition_dispatch",
                scope="bot.body",
                component="body",
                stage="cognition_dispatch",
                context=telemetry.event_context(event),
            ):
                input_event = telemetry.inject_context(
                    dataclasses.replace(
                        cognition.InputEvent.with_data_and_id(cognition_input, request_id),
                        source=hsm.id(instance),
                        target=hsm.id(instance._cognition),
                        metadata=dict(event.metadata),
                    )
                )
                _ = hsm.dispatch(ctx, instance._cognition, input_event)
            await asyncio.wrap_future(ctx.Done())
        finally:
            await hsm.stop(timer, hsm.Context())
            # Drop only the turn id if still live. Cancel-token ops are owned by the cancel path
            # (_retire_bot_turn on success, cancel confirmation/timeout, or deactivation cleanup).
            if processing.active_operation(instance, request_id) is not None:
                processing.finish_operation(ctx, instance, request_id)

    @staticmethod
    def _cancel_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _BotProcessingOperationData)
        ability = instance._cognition
        cancel_event = ability.cancel_event or processing.CancelEvent
        schema = cancel_event.schema
        assert isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)
        # Drop turn correlation; the pre-minted cancel-token capability remains for confirmation.
        if processing.active_operation(instance, data.request_id) is not None:
            processing.finish_operation(ctx, instance, data.request_id)
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
        turn_id = _active_bot_turn_id(instance, event)
        # Topology is hsm.on(cognition.OutputEvent); correlate by turn id + envelope identity.
        return (
            turn_id is not None
            and event.id == turn_id
            and event.target == hsm.id(instance)
            and event.source == hsm.id(ability)
        )

    @staticmethod
    def _matches_bot_processing_failure(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        ability = instance._cognition
        turn_id = _active_bot_turn_id(instance, event)
        # Topology is hsm.on(abilities.FailedEvent); correlate by turn id + envelope identity.
        return (
            turn_id is not None
            and event.id == turn_id
            and event.target == hsm.id(instance)
            and event.source == hsm.id(ability)
        )

    @staticmethod
    def _complete_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        # Keep the turn capability live until ProcessingCompleted is matched so that private
        # terminal still carries exact request-id correlation (same RTC as the output match).
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
    def _retire_bot_turn(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del event
        instance._processing_focus_candidates = ()
        # Successful terminals never consume the cancel token; retire every Bot-owned capability.
        processing.finish_operations(ctx, instance)

    @staticmethod
    def _dispatch_processing_timeout_failure(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        if isinstance(data, _BotProcessingOperationData):
            request_id = data.request_id
            token = data.token
        elif isinstance(data, (cognition.CancelledData, processing.CancelledData)):
            request_id = data.operation_id
            token = data.token
        else:
            raise AssertionError(f"unsupported processing timeout event data: {type(data)!r}")
        instance._processing_focus_candidates = ()
        cancel_id = _bot_cancel_operation_id(request_id, token, instance)
        if processing.active_operation(instance, cancel_id) is not None:
            processing.finish_operation(ctx, instance, cancel_id)
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
        # Keep the turn capability live until ProcessingFailed is matched (mirrors completion).
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
        data = event.data
        if not isinstance(data, (cognition.CancelledData, processing.CancelledData)):
            return False
        cancel_id = _bot_cancel_operation_id(data.operation_id, data.token, instance)
        # Topology is hsm.on(cognition.CancelledEvent, processing.CancelledEvent); payload + envelope.
        return (
            event.id == data.operation_id
            and processing.active_operation(instance, cancel_id) is not None
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
        instance._focused_device = Bot._target_device_reference(instance, processing_data, event.source)

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
        environment = Environment.from_context(lifetime)
        try:
            group_scope = _private_scope(lifetime)
            try:
                _ = await hsm.started(group_scope, instance._attachments, instance._attachments.model)
            except Exception as error:
                if not _is_already_running_error(error):
                    raise

            # Configured devices only: a device powers its own peripherals (Device.start), so
            # walking the tree here would start them a second time.
            for device in instance._devices.values():
                require_environment_scope(environment, device, participant="Device")
                model = instance._model_for_device(device)
                try:
                    _ = await bot.started(
                        environment,
                        device,
                        model,
                        hsm.Config(data=model),
                        owner=instance,
                    )
                except Exception as error:
                    if not _is_already_running_error(error):
                        raise
            ability_scope = _private_scope(lifetime)
            for ability in Bot._lifecycle_abilities(instance):
                model = ability.model
                if model is None:
                    raise RuntimeError(f"{type(ability).__name__} has no lifecycle model.")
                try:
                    _ = await hsm.started(ability_scope, ability, model)
                except Exception as error:
                    # Idempotent when already running under private scope; raise when the
                    # ability is already running under the environment instance map (would receive
                    # environment broadcasts directly).
                    shares_environment_instances = ability.context().value(hsm.Keys.Instances) is environment.value(
                        hsm.Keys.Instances
                    )
                    if not _is_already_running_error(error) or shares_environment_instances:
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
        environment = Environment.from_context(lifetime)
        # Deactivation may cancel activation before the activity can publish its
        # incrementally-started ownership set. Cleanup therefore owns the complete
        # configured lifecycle surface; stopping an actor that never started is
        # intentionally idempotent at this boundary.
        await hsm.stop(instance._attachments, environment)
        for ability in reversed(Bot._lifecycle_abilities(instance)):
            await hsm.stop(ability, lifetime)
        # Device.stop stops the peripherals that device powers, so this owns configured devices only.
        for device in reversed(list(instance._devices.values())):
            await device.stop(environment)
        Bot._clear_owned_device_sources(ctx, instance, event)
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
        cancel_id = _bot_cancel_operation_id(data.request_id, data.token, instance)
        # Cancel-token capability was minted with the turn timer; only the cancel-timeout timer
        # is owned here.
        timer = _BotProcessingOperation()
        _ = await _BotProcessingOperation.started(
            ctx,
            timer,
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
            await hsm.stop(timer, hsm.Context())
            if processing.active_operation(instance, cancel_id) is not None:
                processing.finish_operation(ctx, instance, cancel_id)

    @staticmethod
    def _describe_owned_devices(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        """Declare owned devices on the body snapshot and rebuild the flat ownership map.

        The devices map is fixed at construction, so the references are one unchanging truth;
        the runtime ids bind only once the devices are actually up, which is why this restates
        on every activation (a restarted device is a new actor with a new id) and never
        earlier — an id that does not exist yet is not stamped.

        Snapshot ``owned_devices`` is configured reference → shell device id (environment
        presence). ``_device_source_refs`` is the reverse hot-path map: every started actor id
        in each owned shell's powered tree (shell + peripherals) → configured reference, so
        interrupt and stimulus sources resolve by lookup without walking the tree per event.
        """

        del ctx, event
        owned: dict[str, str] = {}
        source_refs: dict[str, str] = {}
        for reference, device in instance._devices.items():
            if not lifecycle.is_started(device):
                continue
            owned[reference] = hsm.id(device)
            for candidate in Device.device_tree(device):
                if lifecycle.is_started(candidate):
                    source_refs[hsm.id(candidate)] = reference
        instance._device_source_refs = source_refs
        _ = instance.set(_OWNED_DEVICES_ATTRIBUTE, owned)

    @staticmethod
    def _clear_owned_device_sources(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        """Drop start-time ownership ids when devices are no longer live under this body."""

        del ctx, event
        instance._device_source_refs = {}

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "Bot",
        hsm.attribute(_OWNED_DEVICES_ATTRIBUTE),
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
            # Devices are up by now (started during activation), so their runtime ids are real.
            hsm.entry(_describe_owned_devices),
            hsm.transition(
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus),
                hsm.target("../reboot_deactivating"),
            ),
            # Explicit environment input events fan out in parallel to input abilities (never cognition).
            # Efference is Speaking→Listening only (speaking-owned product), not body fan-out.
            hsm.transition(
                hsm.on(SoundEvent, VisualEvent),
                hsm.effect(_fan_out_input),
            ),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_clear_focus),
                hsm.target("../deactivating"),
            ),
            # Focus lives on active so menus built during processing entry still see it:
            # the processing activity nested-dispatches cognition before enter finishes, so a
            # processing-only snapshot still reads as unfocused/focused. Clear lives only on
            # focused + processing so idle unfocused snapshots do not offer clear_focus.
            # Acceptance stays turn-gated by the guards (active bot turn + cognition source).
            hsm.transition(
                hsm.on(events.FocusDeviceEvent),
                hsm.guard(_ability_selected_configured_focus_device),
                hsm.effect(_dispatch_focus_device_action),
            ),
            hsm.state(
                "unfocused",
                hsm.entry(_clear_focus),
                # Something happening to this bot is the occasion. The guard reads device
                # identity and nothing else — never what happened — so the turn is granted
                # blind to content. What (if anything) to do with it is cognition's, and
                # ignoring is a real answer. An unoccupied body also turns to look: an
                # interrupt requests attention and the body is what grants it.
                hsm.transition(
                    hsm.on(events.InputEvent),
                    hsm.guard(_input_targets_configured_device),
                    hsm.effect(_focus_event_target),
                    hsm.target("../processing"),
                ),
                # Sensory / contribution products: explicit cognition.InputEvent handoff
                # (never raw environment media; never conversation.OutputEvent special-case).
                # Bot grants Cognition the turn; it does not inspect or route the stimulus.
                hsm.transition(
                    hsm.on(cognition.InputEvent),
                    hsm.effect(_focus_event_target),
                    hsm.target("../processing"),
                ),
            ),
            hsm.state(
                "focused",
                # Same blind occasion as unfocused. Attention is already committed here, so an
                # interrupt gets its turn without moving focus: a phone buzzing while you are
                # idle draws your eye, the same buzz mid-conversation does not.
                hsm.transition(
                    hsm.on(events.ClearFocusEvent),
                    hsm.guard(_ability_selected_clear_focus),
                    hsm.effect(_dispatch_clear_focus_action),
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
                    hsm.on(events.ClearFocusEvent),
                    hsm.guard(_ability_selected_clear_focus),
                    hsm.effect(_dispatch_clear_focus_action),
                ),
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
                    hsm.on(events.ProcessingCompletedEvent),
                    hsm.guard(_processing_completed_with_focus),
                    hsm.effect(_retire_bot_turn),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingCompletedEvent),
                    hsm.guard(_processing_completed),
                    hsm.effect(_retire_bot_turn),
                    hsm.target("../unfocused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingFailedEvent),
                    hsm.guard(_processing_failed_with_focus),
                    hsm.effect(_retire_bot_turn),
                    hsm.target("../focused"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingFailedEvent),
                    hsm.guard(_processing_failed),
                    hsm.effect(_retire_bot_turn),
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
                    # Relative so redefines (e.g. PhoneBot(Bot.model, ...)) keep a valid target.
                    hsm.target("../../degraded"),
                ),
            ),
        ),
        hsm.observe(observer),
    )
