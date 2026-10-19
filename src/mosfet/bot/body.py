from mosfet.abilities import cognition
from mosfet.abilities import memory
from mosfet.abilities import processing

import asyncio
import collections.abc
import dataclasses
import datetime
import json
import logging
import time
import typing
import uuid
import weakref

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet import abilities
from mosfet import lifecycle
from mosfet import scope
from mosfet.protocols import attachment
from mosfet import events

from mosfet.device import Device
from mosfet.start import bound_model
from mosfet import telemetry
from mosfet.telemetry import observer
from mosfet.telemetry import span
from mosfet.environment import SoundEvent, VisualEvent, Environment, require_environment_scope, space
from mosfet.event import event_json_value
from mosfet.telemetry.hsm import Traced, deliver

_LOG = logging.getLogger(__name__)

_DEFAULT_BOT_PROCESSING_TIMEOUT = datetime.timedelta(minutes=5)
_DEFAULT_BOT_DEACTIVATION_TIMEOUT = datetime.timedelta(minutes=5)
_DEFAULT_BOT_ATTACHMENT_TIMEOUT = datetime.timedelta(seconds=30)
_MIN_BOT_CANCELLATION_TIMEOUT = datetime.timedelta(milliseconds=100)
_MAX_BOT_CANCELLATION_TIMEOUT = datetime.timedelta(seconds=5)
_OWNED_DEVICES_ATTRIBUTE = "owned_devices"
# The body's one firmware, by its runtime id: rendered to cognition like a device, under ref "firmware".
_FIRMWARE_ATTRIBUTE = "firmware"
_LIFECYCLE_SCOPE = "bot.lifecycle"
_FIRMWARE_SCOPE = "bot.firmware"
_TURN_SCOPE = "bot.turn"
# How many arrivals the body holds for the next turn: a count only, visible in the snapshot.
_HELD_PERCEPTS_ATTRIBUTE = "held_percepts"
# A turn takes the longest arrival-order prefix of what is held within these bounds, and always at
# least one item: the bounds split a backlog into consecutive turns and never leave anything out.
# Bytes are the summed canonical JSON of the taken stimuli (a quarter of the prompt and episode
# budget); participants are their order-preserving union, under the payloads' own bound of 32.
_MAX_PERCEIVED_BYTES = 65_536
_MAX_PERCEIVED_PARTICIPANTS = 32
# Attributes are a closed set from code: whether the firmware started (ok), failed or timed out.
_FIRMWARE_STARTS = metrics.get_meter(_FIRMWARE_SCOPE).create_counter(
    "bot.firmware.start.count",
    unit="{start}",
    description="Starts of the bot's firmware during activation, by outcome (ok, failed, timeout).",
)
_DEGRADED = metrics.get_meter(_LIFECYCLE_SCOPE).create_counter(
    "bot.lifecycle.degraded.count",
    unit="{transition}",
    description="Count of Bot transitions into the degraded lifecycle state, by the stage whose bound expired.",
)
# No attributes: how many percepts each turn took together.
_PERCEPT_COUNT = metrics.get_meter(_TURN_SCOPE).create_histogram(
    "bot.turn.percept.count",
    unit="{stimulus}",
    description="Percepts each turn took together: one stimulus, or a bot.perceived batch of up to eight.",
)
# ``bot.handoff.kind`` is a closed set of two from code: device_stimulus, ability_product.
_PERCEPT_WAIT = metrics.get_meter(_TURN_SCOPE).create_histogram(
    "bot.turn.percept.wait",
    unit="s",
    description="Seconds each percept was held while the bot was busy before a turn took it, by handoff kind.",
)
# ``reason`` is a closed set of four from code: deactivated, rebooted, degraded, stopped.
_PERCEPTS_RELEASED = metrics.get_meter(_TURN_SCOPE).create_counter(
    "bot.turn.percept.released.count",
    unit="{stimulus}",
    description="Held percepts released unperceived because the bot left active, by why it left.",
)
# ``terminal`` is the lifecycle terminal's event name: a closed set of four from code.
_UNDELIVERED = metrics.get_meter(_LIFECYCLE_SCOPE).create_counter(
    "bot.lifecycle.answer.undelivered.count",
    unit="{answer}",
    description="Lifecycle answers a reply target did not take (it had stopped), by terminal event name.",
)

# Lifecycle idempotency predicates live in mosfet.lifecycle (single prose allowlist that may
# only shrink); call sites below use lifecycle.is_already_running_error /
# lifecycle.is_not_started_error and must not open-code hsm exception text.


class _CleanupData(pydantic.BaseModel):
    """Private exact terminal for one lifecycle cleanup activity."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    request_id: str = pydantic.Field(min_length=1)
    kind: typing.Literal["activation", "deactivation"]


_CleanupDoneEvent = hsm.Event[_CleanupData](
    name="bot.lifecycle.cleanup.done",
    kind=hsm.CompletionEventKind,
    schema=_CleanupData,
)


class _FirmwareStepData(pydantic.BaseModel):
    """Private exact terminal for the activation step that starts the body's firmware."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    request_id: str = pydantic.Field(min_length=1)


_FirmwareStartedEvent = hsm.Event[_FirmwareStepData](
    name="bot.lifecycle.firmware.started",
    kind=hsm.CompletionEventKind,
    schema=_FirmwareStepData,
)
_FirmwareFailedEvent = hsm.Event[_FirmwareStepData](
    name="bot.lifecycle.firmware.failed",
    kind=hsm.ErrorEventKind,
    schema=_FirmwareStepData,
)


@dataclasses.dataclass(frozen=True)
class _Held:
    """One arrival a busy body holds for its next turn: the event exactly as it arrived, and when.

    ``held_at`` is a monotonic reading for telemetry only (``bot.turn.percept.wait``); no behavior
    reads it.
    """

    event: hsm.Event[typing.Any]
    held_at: float


@dataclasses.dataclass(frozen=True)
class _LifecycleRequest:
    """The accepted activate/deactivate request the Bot's lifecycle is currently answering.

    Durable lifecycle ownership (same shape as ``Attachment._attachments``), not step scratch:
    written only when ``bot.activate`` / ``bot.deactivate`` is accepted, and read by every
    public lifecycle terminal, so timer paths answer under the request's id too.
    """

    operation_id: str
    reply_to: hsm.Instance | None


class _AttachmentOperation(Traced):
    """One-shot graph-visible correlation boundary for a Bot attachment request."""

    _owner: hsm.Instance
    _request_id: str

    def __init__(self, owner: hsm.Instance, request_id: str) -> None:
        super().__init__()
        self._owner = owner
        self._request_id = request_id

    @staticmethod
    def _matches(
        ctx: hsm.Context,
        instance: "_AttachmentOperation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        return (
            event.id == instance._request_id
            and event.target == hsm.id(instance)
            and isinstance(data, (attachment.AttachCompleteData, attachment.DetachedData, attachment.FailedData))
            and data.actor is instance._owner
        )

    @staticmethod
    def _forward(
        ctx: hsm.Context,
        instance: "_AttachmentOperation",
        event: hsm.Event[typing.Any],
    ) -> None:
        _ = hsm.dispatch(
            ctx,
            instance._owner,
            dataclasses.replace(
                event,
                source=instance._request_id,
                target=hsm.id(instance._owner),
                metadata=dict(event.metadata),
            ),
        )

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "BotAttachmentOperation",
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
                hsm.guard(_matches),
                hsm.effect(_forward),
                hsm.target("/BotAttachmentOperation/done"),
            ),
        ),
        hsm.final("done"),
    )


class _ProcessingOperationData(pydantic.BaseModel):
    """Immutable JSON-safe capability for one Bot processing turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    token: str = pydantic.Field(min_length=1)
    request_id: str
    actor_id: str = pydantic.Field(min_length=1)


_ProcessingTimedOutEvent = hsm.Event[_ProcessingOperationData](
    name="bot.processing.timed_out",
    kind=hsm.ErrorEventKind,
    schema=_ProcessingOperationData,
)
_ProcessingCancelTimedOutEvent = hsm.Event[_ProcessingOperationData](
    name="bot.processing.cancel.timed_out",
    kind=hsm.ErrorEventKind,
    schema=_ProcessingOperationData,
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


class _ProcessingOperation(Traced):
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
        event_template: hsm.Event[_ProcessingOperationData],
    ) -> hsm.Model:
        def timeout_delay(
            ctx: hsm.Context,
            instance: _ProcessingOperation,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, instance, event
            return delay

        def dispatch_timeout(
            ctx: hsm.Context,
            instance: _ProcessingOperation,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            if ctx.is_done():
                # The owning activity was canceled: this timer's turn is over, so the timeout
                # must not be dispatched (HSM-CONTEXT-001: is_done is a cancel signal for work
                # owned by the canceled activity, never a liveness gate).
                return
            capability = _ProcessingOperationData(
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

        return mosfet.define(
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
        operation: "_ProcessingOperation",
        *,
        owner: "Bot",
        request_id: str,
        delay: datetime.timedelta,
        event_template: hsm.Event[_ProcessingOperationData],
    ) -> "_ProcessingOperation":
        try:
            return await mosfet.started(
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

    The registry is weak (compare ability._private_instance_scope's strong registry): ephemeral
    actors here are referenced by their owning activity, so the map must not pin them alive.
    """

    values: dict[typing.Hashable, object] = {
        hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance](),
    }
    return hsm.Context(parent=parent, values=scope.mark_private(values))


class Bot(Traced):
    """Interrupt-driven bot that observes and processes events while active."""

    _innate_abilities: typing.ClassVar[tuple[type[abilities.Ability[typing.Any, typing.Any]], ...]] = ()
    _processing_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_PROCESSING_TIMEOUT
    _deactivation_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_BOT_DEACTIVATION_TIMEOUT
    # Constructor-injected operation bounds (None = class default, so subclass ClassVar
    # overrides keep working). Resolved through _resolved_*_timeout; never read the
    # ClassVar when an override is present.
    _processing_timeout_override: datetime.timedelta | None
    _deactivation_timeout_override: datetime.timedelta | None
    # Bot-owned attachment budget stamped on Bot-issued attach/detach requests
    # (ability.py is not owned by this change; the timeout travels in typed event
    # data, which the recipient adopts as its own attaching/detaching deadline).
    _attachment_timeout: datetime.timedelta
    # Device id (Device.identity(), fixed at construction) → device. The id is the body's
    # reference: cognition sees it on the tool menu, as the focus target, and in owned_devices.
    _devices: dict[str, Device]
    # Live actor id → configured device reference. Built when owned devices (and the
    # peripherals they power) start; cleared when they stop. Hot-path ownership is map
    # lookup only — never a per-event DFS of the device tree.
    _device_source_refs: dict[str, str]
    _cognition: abilities.Ability[cognition.InputData, typing.Any]
    # Address of the cognition this body attached with, recorded while it is live during
    # activation. Reboot correlation reads this own-state copy — never hsm.id() on the
    # peer inside a guard, which raises once the peer is stopped (HSM-CONTEXT-001).
    _cognition_id: str | None
    _focused_device: str | None
    # Turn-scoped attention allowlist for the active processing operation.
    # Set when the processing activity starts; cleared when the turn retires
    # (success, timeout failure) and when deactivation abandons the turn lifetime.
    # Guards read it only alongside a live turn operation — never alone.
    _processing_focus_candidates: tuple[str, ...]
    # What reached the body for a turn, oldest first: the defer queue made explicit, durable
    # across any number of turns (not step scratch). Only _hold_percept appends to it; only a
    # turn's take and the release on leaving active remove from it, so it is empty whenever the
    # body is idle. Its count is the declared held_percepts attribute.
    _held: tuple[_Held, ...]
    _innate_ability_instances: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    # The machinery the bot's owner wrote for it: started after every part is attached, stopped first.
    _firmware: hsm.Instance | None
    # The bot's name, fixed at construction: the body runs as /<name> and each device as /<name><Device>.
    _name: str | None
    # Boundary observers the bot's runner gave it: they watch its model and never act on it.
    _observers: tuple[hsm.ObservationElement[typing.Any], ...]
    _acquired_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _input: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _output: tuple[abilities.Ability[typing.Any, typing.Any], ...]
    _attachments: attachment.Group
    # The request every public lifecycle terminal answers; None until the first is accepted.
    _lifecycle: _LifecycleRequest | None
    # Monotonic count of lifecycle step entries and the private correlation id frozen on the
    # latest entry. Step activities stamp it on their attachment requests and cleanup terminal,
    # and the step guards accept only it, so a stale terminal from an earlier step never
    # completes a later one and superseding the request never re-keys the running step.
    _lifecycle_step: int
    _lifecycle_step_id: str
    # Answers still on their way to a reply target, held until delivery settles so none is lost
    # before its outcome is known.
    _answering: set[asyncio.Future[bool]]

    def __init__(
        self,
        devices: collections.abc.Sequence[Device],
        *,
        cognition: abilities.Ability[cognition.InputData, typing.Any],
        input: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        output: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        acquired_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        processing_timeout: datetime.timedelta | None = None,
        deactivation_timeout: datetime.timedelta | None = None,
        attachment_timeout: datetime.timedelta | None = None,
        stm_memory: memory.StmMemory | None = None,
        firmware: hsm.Instance | None = None,
        name: str | None = None,
        observe: collections.abc.Sequence[hsm.ObservationElement[typing.Any]] = (),
    ) -> None:
        """Build an unstarted body holding ``devices``, each known by its ``Device.identity()``.

        ``name``, when given, is the bot's own (``bot.define``'s, lower snake case): the body runs on
        ``Bot.model`` redefined as ``/<name>`` and starts each device it holds on that device's model
        redefined as ``/<name><DeviceClass>``, so two bots in one process publish distinct topology
        names. ``Bot.model`` itself is never changed. Without one, the body and its devices run their
        own models.

        ``observe`` is the boundary observers (``hsm.observe(...)``) whoever runs the bot opts into, for
        logs and records: the body runs on ``Bot.model`` redefined with them (under its name, when it
        has one), so they watch this body only, and they never act on it.

        ``firmware``, when given, is the bot's one firmware (``bot.Firmware``): an unstarted machine
        whose ``model`` it runs on. The body starts it once its devices and abilities are attached,
        stops it first, and starts it again, fresh, when it activates again. Cognition sees it as
        ``<firmware ref="firmware">`` and is offered its commands under the key ``firmware``.

        Raises:
            ValueError: If two devices share one id (one device given twice included), an innate,
                acquired or input ability's id is already the body's (``"bot"``), a device's or another
                ability's, or a timeout is not positive.
            TypeError: If something in ``observe`` is not an hsm observer.
        """

        super().__init__()
        for given in observe:
            if not isinstance(typing.cast(object, given), hsm.ObservationElement):
                raise TypeError(f"observe takes hsm observers (hsm.observe(...)), not {type(given).__name__}")
        if processing_timeout is not None and processing_timeout <= datetime.timedelta():
            raise ValueError("processing_timeout must be positive.")
        if deactivation_timeout is not None and deactivation_timeout <= datetime.timedelta():
            raise ValueError("deactivation_timeout must be positive.")
        if attachment_timeout is not None and attachment_timeout <= datetime.timedelta():
            raise ValueError("attachment_timeout must be positive.")
        if self._processing_timeout <= datetime.timedelta():
            raise ValueError("processing_timeout must be positive.")
        if self._deactivation_timeout <= datetime.timedelta():
            raise ValueError("deactivation_timeout must be positive.")
        self._processing_timeout_override = processing_timeout
        self._deactivation_timeout_override = deactivation_timeout
        self._attachment_timeout = (
            attachment_timeout if attachment_timeout is not None else _DEFAULT_BOT_ATTACHMENT_TIMEOUT
        )
        held: dict[str, Device] = {}
        for device in devices:
            identity = device.identity()
            if identity in held:
                raise ValueError(f"Bot devices must have distinct ids; {identity!r} is given twice.")
            held[identity] = device
        self._devices = held
        self._device_source_refs = {}
        self._cognition = cognition
        self._cognition_id = None
        self._focused_device = None
        self._processing_focus_candidates = ()
        self._held = ()
        self._innate_ability_instances = tuple(ability_type() for ability_type in self._innate_abilities)
        self._input = tuple(input)
        self._output = tuple(output)
        self._acquired_abilities = tuple(acquired_abilities)
        # Every actor cognition may address has its own key: the body is "bot", a device its id, an
        # ability its id. A clash is the composer's to resolve by naming one; the body never suffixes.
        keys = {"bot", *held}
        for ability in Bot._addressable_abilities(self):
            identity = ability.identity()
            if identity in keys:
                raise ValueError(
                    f"Bot actors must have distinct keys; ability {identity!r} is already one. Name the "
                    + f"ability with its own id: {type(ability).__name__}(id=...)."
                )
            keys.add(identity)
        self._stm_memory: memory.StmMemory | None = stm_memory
        self._firmware = firmware
        self._name = name
        self._observers = tuple(observe)
        self._attachments = attachment.Group(*Bot._lifecycle_attachment_members(self))
        self._lifecycle = None
        self._lifecycle_step = 0
        self._lifecycle_step_id = ""
        self._answering = set()

    async def attach(
        self,
        environment: Environment,
        *,
        placement: space.Placement | None = None,
        reply_to: hsm.Instance | None = None,
        operation_id: str | None = None,
    ) -> typing.Self:
        """Start the bot in ``environment`` if needed, join it, and request activation.

        ``operation_id`` (a fresh id when omitted) is the id of the one public terminal that
        answers this request: ``bot.activated`` or ``bot.activating.failed``. The terminal is
        delivered to the bot and, when given, to ``reply_to``.
        """

        require_environment_scope(environment, self, participant="Bot")
        try:
            _ = await mosfet.started(environment, self, self._model_for_instance())
        except Exception as error:
            if not lifecycle.is_already_running_error(error):
                raise
        # A Bot is not a Device: its presence is an attach-time decision, and detach ends it.
        # Unconditional: the already-running branch above swallows its error, and join is idempotent.
        environment.join(self, placement=placement)
        await self.dispatch(
            environment,
            events.ActivateEvent.with_data_and_id(
                events.ActivateEventData(reply_to=reply_to),
                operation_id or uuid.uuid4().hex,
            ),
        )
        return self

    def _model_for_instance(self) -> hsm.Model:
        """Return the topology model for this runtime bot instance: its own, or ``/<name>`` with its observers."""

        if self._name is None and not self._observers:
            return self.model
        return hsm.redefine(self.model, self._name, *self._observers)

    def _model_for_device(self, device: Device) -> hsm.Model | None:
        """Return the published model for one configured device: its own, or ``/<name><DeviceClass>``."""

        model = device.model
        if model is None or self._name is None:
            return model
        return hsm.redefine(model, f"{self._name}{type(device).__name__}")

    async def detach(
        self,
        environment: Environment,
        *,
        reply_to: hsm.Instance | None = None,
        operation_id: str | None = None,
    ) -> typing.Self:
        """Request deactivation and leave ``environment``.

        A started bot answers the request from every lifecycle state with exactly one public
        terminal carrying ``operation_id`` (a fresh id when omitted): ``bot.deactivated``, or
        ``bot.degraded`` when teardown cannot finish. It is delivered to the bot and, when given,
        to ``reply_to``. An unstarted bot is already detached and answers nothing.
        """

        require_environment_scope(environment, self, participant="Bot")
        try:
            await self.dispatch(
                environment,
                events.DeactivateEvent.with_data_and_id(
                    events.DeactivateEventData(reply_to=reply_to),
                    operation_id or uuid.uuid4().hex,
                ),
            )
        except Exception as error:
            # An unstarted or stopped bot is already detached; deactivation is idempotent.
            if not lifecycle.is_not_started_error(error):
                raise
        # Presence is attachment-scoped, not activation-scoped: a deactivated but still attached
        # bot is legitimately in the environment, so only detach ends it — reboot deactivates without
        # detaching and keeps presence. Pairs with the join in attach; no test can observe the
        # removal (a stopped or inactive bot ignores broadcasts either way), so this is ownership
        # completeness and map hygiene. Do not delete it as untested.
        environment.leave(self)
        return self

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop its firmware, the body, then every part it started, in reverse: its abilities, cognition and the register among them, then its devices.

        Whoever starts a part stops it, and each part stops what it started: a device the
        peripherals it powers, a composite ability its children (cognition its faculties and its
        memory, which closes the database it opened). So nothing the body started outlives it, and
        a body built again from the same parts starts again. Stopping a part that never started,
        or already stopped, does nothing.
        """

        if self._firmware is not None:
            await hsm.stop(self._firmware, ctx)
        await hsm.Instance.stop(self, ctx)
        await hsm.stop(self._attachments, ctx)
        for ability in reversed(Bot._lifecycle_abilities(self)):
            await hsm.stop(ability, ctx)
        # Device.stop stops the peripherals that device powers, so this owns configured devices only.
        for device in reversed(list(self._devices.values())):
            await device.stop(ctx)

    @staticmethod
    def _lifecycle_abilities(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
        # The short-term register is perception machinery the body owns: it attaches with the
        # body so the fan-out (and peers such as Learning) can reach it by message.
        register = () if instance._stm_memory is None else (instance._stm_memory,)
        configured = (
            instance._cognition,
            *register,
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
    def _cognition_abilities(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
        """Return only the bot abilities cognition is allowed to inventory."""

        return (*instance._innate_ability_instances, *instance._acquired_abilities)

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
    async def _request_attachment(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event,
        *,
        step_id: str,
        kind: typing.Literal["attach", "detach"],
    ) -> None:
        """Address one attachment request, correlated by lifecycle step, whose typed terminal returns to Bot."""

        lifetime = instance.context()
        operation = await mosfet.started(
            _private_scope(ctx),
            _AttachmentOperation(instance, step_id),
            _AttachmentOperation.model,
            hsm.Config(id=step_id),
        )
        request: hsm.Event[typing.Any]
        if kind == "attach":
            request = attachment.AttachEvent.with_data(
                attachment.AttachData(actor=instance, reply_to=operation, timeout=instance._attachment_timeout)
            )
        else:
            request = attachment.DetachEvent.with_data(
                attachment.DetachData(actor=instance, reply_to=operation, timeout=instance._attachment_timeout)
            )
        request = dataclasses.replace(
            request,
            id=step_id,
            source=hsm.id(instance),
            target=hsm.id(instance._attachments),
            metadata=dict(event.metadata),
        )
        try:
            try:
                if kind == "attach":
                    await instance._attachments.attach(ctx, typing.cast(hsm.Event[attachment.AttachData], request))
                else:
                    await instance._attachments.detach(ctx, typing.cast(hsm.Event[attachment.DetachData], request))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failed = attachment.FailedData(
                    actor=instance,
                    kind=attachment.FailureKind.DISPATCH,
                    message=f"Bot lifecycle {kind} dispatch failed: {error}",
                )
                failed_event = (
                    attachment.AttachFailedEvent.with_data(failed)
                    if kind == "attach"
                    else attachment.DetachFailedEvent.with_data(failed)
                )
                _ = hsm.dispatch(
                    lifetime,
                    instance,
                    dataclasses.replace(
                        failed_event,
                        id=step_id,
                        source=step_id,
                        target=hsm.id(instance),
                        metadata=dict(event.metadata),
                    ),
                )
            await asyncio.wrap_future(ctx.done())
        finally:
            await hsm.stop(operation, hsm.Context())

    @staticmethod
    async def _deactivate_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        # The firmware is the machinery on top of the body's parts: it stops before they detach.
        await Bot._stop_firmware(instance)
        await Bot._request_attachment(ctx, instance, event, step_id=instance._lifecycle_step_id, kind="detach")

    @staticmethod
    async def _stop_firmware(instance: "Bot") -> None:
        """Stop the body's firmware; one that never started, or already stopped, stops as nothing."""

        if instance._firmware is not None:
            await hsm.stop(instance._firmware, instance.context())

    @staticmethod
    async def _start_firmware_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        """Start the body's firmware in the body's own lifetime, now that every part is attached.

        It runs in the body's environment, where cognition's world block can show it; ``hsm.Keys.Owner``
        in its context is the body, which its reports go to. The step answers itself with the private
        started or failed terminal under its own step id.
        """

        step_id = instance._lifecycle_step_id
        firmware = instance._firmware
        assert firmware is not None, "the firmware step is entered only for a body with firmware"
        terminal = _FirmwareStartedEvent
        try:
            model = bound_model(firmware)
            if model is None:
                raise RuntimeError(f"{type(firmware).__name__} has no model to run.")
            _ = await mosfet.started(instance.context(), firmware, model)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _LOG.warning("Bot firmware start failed kind=%s", span.failure_kind(error, type(error).__name__))
            terminal = _FirmwareFailedEvent
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                terminal.with_data(_FirmwareStepData(request_id=step_id)),
                id=step_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _deactivation_cleanup(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event,
        *,
        cleanup: typing.Literal["preserve", "reset"] | None,
    ) -> None:
        step_id = instance._lifecycle_step_id
        lifetime = instance.context()
        environment = Environment.from_context(lifetime)
        # Stopped already when deactivation ran; when its bound expired first, it stops here.
        await Bot._stop_firmware(instance)
        if cleanup == "reset":
            for member in Bot._lifecycle_attachment_members(instance):
                await deliver(
                    member,
                    lifetime,
                    dataclasses.replace(
                        attachment.DetachEvent.with_data(
                            attachment.DetachData(actor=instance, timeout=instance._attachment_timeout)
                        ),
                        id=step_id,
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
        # The turn lifetime ends here: drop the turn-scoped attention allowlist with it
        # so a later turn never inherits a stale gate (set/clear lifecycle, HSM-COMPLETION-001).
        instance._processing_focus_candidates = ()
        # Ownership map is only valid while active; rebuild happens on next active entry.
        Bot._clear_owned_device_sources(ctx, instance, event)
        terminal = _CleanupData(request_id=step_id, kind="deactivation")
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _CleanupDoneEvent.with_data(terminal),
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
    def _target_device_reference(instance: "Bot", input: events.InputData, source: str = "") -> str | None:
        if isinstance(input, events.InputEventData):
            return Bot._interrupt_reference(instance, input, source)
        references = Bot._device_references_for_event(instance, input)
        return references[0] if len(references) == 1 else None

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
            if instance._stm_memory is not None:
                # Registration is perception machinery, not interpretation: record what was admitted,
                # never what it meant. The record is a one-way message to the body-owned register,
                # so register storage never blocks admitting the stimulus (body reflex floor); the
                # register logs its own failed writes.
                _ = hsm.dispatch(
                    ctx,
                    instance._stm_memory,
                    dataclasses.replace(
                        memory.stm.RecordEvent.with_data(memory.stm.event_record(event)),
                        source=hsm.id(instance),
                        target=hsm.id(instance._stm_memory),
                        metadata=dict(event.metadata),
                    ),
                )
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
        # Envelope correlation only (typed payload + source/target against the cognition
        # address recorded at activation). Peer liveness is never probed (HSM-CONTEXT-001):
        # a reboot the owner addressed to this body is accepted, including after the peer
        # stopped. Delivery already chose the recipient (HSM-DELIVERY-001).
        cognition_id = instance._cognition_id
        return (
            cognition_id is not None
            and isinstance(event.data, events.RebootEventData)
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
        """Mechanical body-side entry point for the cognition-owned attention policy.

        Cognition owns attention policy (``cognition.types.attention_selection_error``):
        this delegate applies the same verdict at body turn gates so illegal
        focus/clear selections fail the turn instead of silently dropping. The body
        never invents attention policy here — it enforces cognition's.
        """

        return cognition.types.attention_selection_error(
            selection,
            focus_candidates=focus_candidates,
            configured_device_names=configured_device_names,
            enforce_candidates=enforce_candidates,
            focused_device=focused_device,
        )

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
    def _holds_percepts(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return bool(instance._held)

    @staticmethod
    def _holds_percepts_with_focus(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        # One guard: hsm keeps a transition's last guard only, so two would not both hold.
        return Bot._holds_percepts(ctx, instance, event) and Bot._has_focused_device(ctx, instance, event)

    @staticmethod
    def _hold_percept(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        """Hold what reached the body, in arrival order, for the turn that will perceive it."""

        del ctx
        # Telemetry only (bot.turn.percept.wait): no behavior reads held_at.
        instance._held = (*instance._held, _Held(event=event, held_at=time.monotonic()))
        _ = instance.set(_HELD_PERCEPTS_ATTRIBUTE, len(instance._held))

    @staticmethod
    def _release_held(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        """Release what is still held as the body leaves active: counted and logged, never silent.

        The reason is read from the typed payload of what took the body out of active: a
        deactivation, a reboot, the processing cancellation bound that degrades it, or (any other
        exit) the body being stopped.
        """

        del ctx
        released = len(instance._held)
        if not released:
            return
        instance._held = ()
        _ = instance.set(_HELD_PERCEPTS_ATTRIBUTE, 0)
        data = event.data
        reason = (
            "deactivated"
            if isinstance(data, events.DeactivateEventData)
            else "rebooted"
            if isinstance(data, events.RebootEventData)
            else "degraded"
            if isinstance(data, _ProcessingOperationData)
            else "stopped"
        )
        _PERCEPTS_RELEASED.add(released, {"reason": reason})
        _LOG.info("Bot released held percepts count=%d reason=%s", released, reason)

    @staticmethod
    def _resolved_processing_timeout(instance: "Bot") -> datetime.timedelta:
        """Effective turn bound: constructor injection, else the class (subclass) default."""

        override = instance._processing_timeout_override
        return override if override is not None else type(instance)._processing_timeout

    @staticmethod
    def _resolved_deactivation_timeout(instance: "Bot") -> datetime.timedelta:
        """Effective deactivation bound: constructor injection, else the class default."""

        override = instance._deactivation_timeout_override
        return override if override is not None else type(instance)._deactivation_timeout

    @staticmethod
    def _bot_processing_timeout(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return Bot._resolved_processing_timeout(instance)

    @staticmethod
    def _bot_deactivation_timeout(
        ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, event
        return Bot._resolved_deactivation_timeout(instance)

    @staticmethod
    def _dispatch_actors(instance: "Bot") -> dict[str, hsm.Instance]:
        """Map cognition-visible devices, innate and acquired abilities, and input abilities.

        Input abilities are sense organs cognition may direct; what a turn may select from them
        is only what their live topology enables. The firmware is offered under ``firmware``: its
        commands while a state that reacts to them is active. Output ports and ability-owned child
        machines are composition wiring and never enter this map.
        """

        actors: dict[str, hsm.Instance] = {"bot": instance, **instance._devices}
        for ability in Bot._addressable_abilities(instance):
            actors[ability.identity()] = ability
        if instance._firmware is not None:
            # "firmware" is reserved: no device or ability may take it (bot.define's B0007).
            actors[_FIRMWARE_ATTRIBUTE] = instance._firmware
        return actors

    @staticmethod
    def _addressable_abilities(instance: "Bot") -> tuple[abilities.Ability[typing.Any, typing.Any], ...]:
        """Innate, acquired and input abilities, each once: the abilities cognition may address by id."""

        addressable: list[abilities.Ability[typing.Any, typing.Any]] = []
        for ability in (*instance._innate_ability_instances, *instance._acquired_abilities, *instance._input):
            # One actor, one key: an ability composed both as acquired and as input is offered once.
            if not any(known is ability for known in addressable):
                addressable.append(ability)
        return tuple(addressable)

    @staticmethod
    async def _dispatch_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        del event
        # The take runs first and synchronously (the activity starts eagerly on entry), so nothing
        # held can slip between it and the turn: the longest arrival-order prefix of what is held
        # within the count, byte and participant bounds, and always at least one item. Only size
        # and identity are read; nothing is ranked, filtered or dropped.
        held = instance._held
        if not held:
            raise AssertionError("a turn starts only with something held")
        arrivals: list[tuple[_Held, mosfet.InputData, tuple[mosfet.Participant, ...], str]] = []
        weight = 0
        union: list[mosfet.Participant] = []
        for item in held[: events.MAX_PERCEPTS]:
            if isinstance(item.event.data, events.InputEventData):
                # The device's stamped participants ride forward unchanged (none when it knows no one).
                arrived: mosfet.InputData = item.event.data
                stamped = item.event.data.participants
                kind = "device_stimulus"
            elif isinstance(item.event.data, cognition.InputData):
                # Sensory / contribution products hand cognition.InputEvent (Listening pattern).
                # Body admits the handoff; it does not inspect conversation or other product domains.
                # The producer's stamped participants ride forward unchanged.
                arrived = item.event.data.stimulus
                stamped = item.event.data.participants
                kind = "ability_product"
            else:
                raise AssertionError(f"unsupported body processing event data: {type(item.event.data)!r}")
            try:
                size = len(json.dumps(event_json_value(arrived)))
            except (TypeError, ValueError):
                # Unmeasurable is over the bound: it is taken, alone, when it is first.
                size = _MAX_PERCEIVED_BYTES + 1
            joined = list(union)
            for participant in stamped:
                if participant not in joined:
                    joined.append(participant)
            if arrivals and (weight + size > _MAX_PERCEIVED_BYTES or len(joined) > _MAX_PERCEIVED_PARTICIPANTS):
                break
            arrivals.append((item, arrived, stamped, kind))
            weight += size
            union = joined
        instance._held = held[len(arrivals) :]
        _ = instance.set(_HELD_PERCEPTS_ATTRIBUTE, len(instance._held))
        taken_at = time.monotonic()  # telemetry only
        for item, _arrived, _stamped, kind in arrivals:
            _PERCEPT_WAIT.record(taken_at - item.held_at, {"bot.handoff.kind": kind})
        _PERCEPT_COUNT.record(len(arrivals))
        first = arrivals[0][0].event
        # The span covers admitting the stimulus and handing it to cognition, and stops there:
        # the wait below ends by cancellation on every ordinary turn, which is not a failure of
        # the handoff.
        with span.operation(
            "bot.body.cognition_handoff",
            scope="bot.body",
            component="body",
            stage="cognition_handoff",
            context=telemetry.event_context(first),
        ) as active:
            if len(arrivals) == 1:
                # A single percept is its own perception: cognition gets exactly what arrived.
                _item, stimulus, participants, kind = arrivals[0]
                envelope_source = first.source
                active.set_attribute("bot.handoff.kind", kind)
            else:
                # Everything taken is perceived together, in arrival order, each percept unchanged
                # with the source it arrived on and its own participants. The turn's participants are
                # their order-preserving union.
                percepts = tuple(
                    events.PerceptData(stimulus=arrived, source=item.event.source, participants=stamped)
                    for item, arrived, stamped, _kind in arrivals
                )
                stimulus = dataclasses.replace(
                    events.PerceivedEvent.with_data(events.PerceivedData(percepts=percepts)),
                    id=first.id,
                    source=hsm.id(instance),
                )
                participants = tuple(union)
                envelope_source = hsm.id(instance)
                active.set_attribute("bot.handoff.kind", "perceived")
            active.set_attribute("bot.turn.percept.count", len(arrivals))
            active.set_attribute("bot.turn.percept.held.count", len(instance._held))
            # Attention bias is cognition-owned (cognition.types.attention_bias_for_turn):
            # the body stamps identity context and applies the verdict mechanically. The
            # bias reads device identity only — never what happened, never insistence —
            # so nothing here arbitrates per stimulus; a batch is biased percept by percept.
            focus, focus_candidates = cognition.types.attention_bias_for_turn(
                stimulus,
                envelope_source=envelope_source,
                device_names=instance._devices,
                device_source_refs=instance._device_source_refs,
                focused_device=instance._focused_device,
            )
            instance._processing_focus_candidates = focus_candidates
            active.set_attribute("bot.device.focus_candidate.count", len(focus_candidates))
            active.set_attribute("bot.ability.count", len(Bot._cognition_abilities(instance)))
            cognition_input = cognition.InputData(
                stimulus=stimulus,
                abilities=Bot._cognition_abilities(instance),
                actors=Bot._dispatch_actors(instance),
                focus=focus,
                focus_candidates=focus_candidates,
                participants=participants,
            )
        # Live turn capability + activity-owned timer. The cancel-token capability is minted with
        # the timer id so a cancelled confirmation can match immediately (before the cancelling
        # activity runs). Successful terminals retire both; timeout keeps only the cancel token.
        request_id = first.id
        assert request_id
        _ = await processing.start_operation(instance, request_id)
        timer: _ProcessingOperation | None = None
        try:
            with span.operation(
                "bot.body.cognition_dispatch",
                scope="bot.body",
                component="body",
                stage="cognition_dispatch",
                context=telemetry.event_context(first),
            ):
                input_event = telemetry.inject_context(
                    dataclasses.replace(
                        cognition.InputEvent.with_data_and_id(cognition_input, request_id),
                        source=hsm.id(instance),
                        target=hsm.id(instance._cognition),
                        metadata=dict(first.metadata),
                    )
                )
                _ = hsm.dispatch(ctx, instance._cognition, input_event)
            if not ctx.is_done():
                timer = _ProcessingOperation()
                _ = await _ProcessingOperation.started(
                    ctx,
                    timer,
                    owner=instance,
                    request_id=request_id,
                    delay=Bot._resolved_processing_timeout(instance),
                    event_template=_ProcessingTimedOutEvent,
                )
                cancel_id = _bot_cancel_operation_id(request_id, hsm.id(timer), instance)
                _ = await processing.start_operation(instance, cancel_id)
            if not ctx.is_done():
                await asyncio.wrap_future(ctx.Done())
        finally:
            if timer is not None:
                await hsm.stop(timer, hsm.Context())
            # Drop only the turn id if still live. Cancel-token ops are owned by the cancel path
            # (_retire_bot_turn on success, cancel confirmation/timeout, or deactivation cleanup).
            if processing.active_operation(instance, request_id) is not None:
                processing.finish_operation(ctx, instance, request_id)

    @staticmethod
    def _cancel_bot_processing(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ProcessingOperationData)
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
        # The per-turn candidate set was computed when this turn's cognition input was built
        # (attention_bias_for_turn); report exactly what cognition received, never re-derive it
        # from whatever the body happens to currently have focused.
        focus_candidates = instance._processing_focus_candidates
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
        if isinstance(data, _ProcessingOperationData):
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
        seconds = Bot._resolved_processing_timeout(instance).total_seconds()
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
            isinstance(data, _ProcessingOperationData)
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
        # Reflex floor: an unoccupied body turns to look at what arrived. Identity
        # resolution only (which device the signal arrived on) — never what happened
        # and never stimulus insistence, so this grants attention without bias.
        del ctx
        if isinstance(event.data, events.InputEventData):
            processing_data = event.data
        elif isinstance(event.data, cognition.InputData):
            processing_data = event.data.stimulus
        else:
            raise AssertionError(f"unsupported body processing event data: {type(event.data)!r}")
        instance._focused_device = Bot._target_device_reference(instance, processing_data, event.source)

    @staticmethod
    def _focus_first_held(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        # Reflex floor at a turn's end: an unoccupied body turns to look at what arrived first.
        # Arrival order only, never rank: exactly what replaying the first held stimulus did.
        del event
        Bot._focus_event_target(ctx, instance, instance._held[0].event)

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
        step_id = instance._lifecycle_step_id
        lifetime = instance.context()
        environment = Environment.from_context(lifetime)
        try:
            group_scope = _private_scope(lifetime)
            try:
                _ = await mosfet.started(group_scope, instance._attachments, instance._attachments.model)
            except Exception as error:
                if not lifecycle.is_already_running_error(error):
                    raise

            # Configured devices only: a device powers its own peripherals (Device.start), so
            # walking the tree here would start them a second time.
            for device in instance._devices.values():
                require_environment_scope(environment, device, participant="Device")
                model = instance._model_for_device(device)
                if model is None:
                    raise RuntimeError(f"{type(device).__name__} has no lifecycle model.")
                try:
                    _ = await mosfet.started(
                        environment,
                        device,
                        model,
                        hsm.Config(data=model),
                        owner=instance,
                    )
                except Exception as error:
                    if not lifecycle.is_already_running_error(error):
                        raise
            ability_scope = _private_scope(lifetime)
            for ability in Bot._lifecycle_abilities(instance):
                model = ability.model
                if model is None:
                    raise RuntimeError(f"{type(ability).__name__} has no lifecycle model.")
                try:
                    _ = await mosfet.started(ability_scope, ability, model)
                except Exception as error:
                    # Idempotent when already running under private scope; raise when the
                    # ability is already running under the environment instance map (would receive
                    # environment broadcasts directly).
                    shares_environment_instances = environment.contains(ability)
                    if not lifecycle.is_already_running_error(error) or shares_environment_instances:
                        raise
            # The abilities above are live now: record the cognition address the reboot
            # guard correlates against. Recorded before the attach request so a reboot
            # that lands mid-activation still attributes to this cognition.
            instance._cognition_id = hsm.id(instance._cognition)
            await Bot._request_attachment(ctx, instance, event, step_id=step_id, kind="attach")
        except asyncio.CancelledError:
            raise
        except Exception:
            failure = attachment.FailedData(
                actor=instance,
                kind=attachment.FailureKind.INITIALIZATION,
                message="Bot activation preparation failed.",
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    attachment.AttachFailedEvent.with_data(failure),
                    id=step_id,
                    source=step_id,
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _accept_activate(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        """Record the accepted activation as the request the lifecycle now answers."""

        del ctx
        data = event.data
        assert isinstance(data, events.ActivateEventData)
        instance._lifecycle = _LifecycleRequest(operation_id=event.id, reply_to=data.reply_to)

    @staticmethod
    def _accept_deactivate(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        """Record the accepted deactivation; a stop in progress then answers this newer request."""

        del ctx
        data = event.data
        assert isinstance(data, events.DeactivateEventData)
        instance._lifecycle = _LifecycleRequest(operation_id=event.id, reply_to=data.reply_to)

    @staticmethod
    def _enter_lifecycle_step(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        """Freeze a fresh private correlation id for the lifecycle step being entered."""

        del ctx, event
        instance._lifecycle_step += 1
        instance._lifecycle_step_id = f"{hsm.id(instance)}:lifecycle:{instance._lifecycle_step}"

    @staticmethod
    def _announce(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
        terminal: hsm.Event[typing.Any],
    ) -> None:
        """Answer the accepted lifecycle request with ``terminal``, to the bot and its reply target.

        Stamped from the durable request, never from the triggering event: a timer or private
        completion carries an id of its own, and the activator correlates by its operation id.

        Delivery is the gate for the reply target: it hears the answer only while it runs. A
        reboot re-enters activation and answers the same request again, and by then a one-shot
        activator (``bot.start``'s) has stopped, so the answer is addressed by delivery (the
        target is stamped as it is delivered) and an undelivered one is counted, never raised.
        """

        request = instance._lifecycle
        assert request is not None, "every lifecycle step is entered by accepting a request"
        answer = dataclasses.replace(
            terminal,
            id=request.operation_id,
            source=hsm.id(instance),
            target=hsm.id(instance),
            metadata=dict(event.metadata),
        )
        _ = hsm.dispatch(ctx, instance, answer)
        if request.reply_to is not None:
            delivery = asyncio.ensure_future(
                hsm.dispatch(
                    ctx, request.reply_to, dataclasses.replace(answer, target="", metadata=dict(event.metadata))
                )
            )
            instance._answering.add(delivery)
            delivery.add_done_callback(lambda done: Bot._settle_answer(instance, answer.name, done))

    @staticmethod
    def _settle_answer(instance: "Bot", terminal: str, delivery: asyncio.Future[bool]) -> None:
        """Count a lifecycle answer its reply target did not take; a target that stopped hears nothing."""

        instance._answering.discard(delivery)
        if delivery.cancelled():
            return
        error = delivery.exception()
        if error is None:
            return
        _UNDELIVERED.add(1, {"terminal": terminal})
        _LOG.info("Bot lifecycle answer undelivered terminal=%s error=%s", terminal, type(error).__name__)

    @staticmethod
    def _dispatch_activating_done(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        """Publish the public activation terminal after private attachment completion."""

        Bot._announce(ctx, instance, event, events.ActivatingDoneEvent.with_data(events.ActivatingDoneEventData()))

    @staticmethod
    def _dispatch_activating_failed(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        """Publish the public activation failure after the private failure terminal."""

        Bot._announce(ctx, instance, event, events.ActivatingFailedEvent.with_data(events.ActivatingFailedEventData()))

    @staticmethod
    def _dispatch_deactivating_done(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        """Publish the public deactivation terminal after private cleanup completion."""

        Bot._announce(ctx, instance, event, events.DeactivatingDoneEvent.with_data(events.DeactivatingDoneEventData()))

    @staticmethod
    def _answer_degraded(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        """Answer a deactivation of an already degraded bot: it stays degraded."""

        Bot._announce(ctx, instance, event, events.DegradedEvent.with_data(events.DegradedEventData(stage="degraded")))

    @staticmethod
    def _degrade(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
        stage: events.DegradedStage,
    ) -> None:
        """Announce a transition into degraded whose stage the calling effect names as a constant."""

        _DEGRADED.add(1, {"stage": stage})
        _LOG.warning("Bot degraded stage=%s", stage)
        Bot._announce(ctx, instance, event, events.DegradedEvent.with_data(events.DegradedEventData(stage=stage)))

    @staticmethod
    def _dispatch_degraded_activation_cleanup(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        Bot._degrade(ctx, instance, event, "activation_cleanup")

    @staticmethod
    def _dispatch_degraded_activation_cancel(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        Bot._degrade(ctx, instance, event, "activation_cancel")

    @staticmethod
    def _dispatch_degraded_deactivation_cleanup(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        Bot._degrade(ctx, instance, event, "deactivation_cleanup")

    @staticmethod
    def _dispatch_degraded_reboot_cleanup(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        Bot._degrade(ctx, instance, event, "reboot_cleanup")

    @staticmethod
    def _dispatch_degraded_processing_cancel(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        Bot._degrade(ctx, instance, event, "processing_cancel")

    @staticmethod
    def _matches_attachment_terminal(
        ctx: hsm.Context,
        instance: "Bot",
        event: hsm.Event[typing.Any],
        *,
        kind: typing.Literal["attach", "detach"],
    ) -> bool:
        del ctx
        data = event.data
        expected = (
            (attachment.AttachCompleteData, attachment.FailedData)
            if kind == "attach"
            else (attachment.DetachedData, attachment.FailedData)
        )
        return (
            isinstance(data, expected)
            and data.actor is instance
            and event.id == instance._lifecycle_step_id
            and event.source == event.id
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _matches_attach_terminal_with_firmware(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        # One guard: hsm keeps a transition's last guard only, so two would not both hold.
        return instance._firmware is not None and Bot._matches_attach_terminal(ctx, instance, event)

    @staticmethod
    def _matches_firmware_terminal(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _FirmwareStepData)
            and event.id == data.request_id
            and event.id == instance._lifecycle_step_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _count_firmware_started(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _FIRMWARE_STARTS.add(1, {"bot.outcome": "ok"})

    @staticmethod
    def _count_firmware_failed(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _FIRMWARE_STARTS.add(1, {"bot.outcome": "failed"})

    @staticmethod
    def _count_firmware_timeout(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _FIRMWARE_STARTS.add(1, {"bot.outcome": "timeout"})
        _LOG.warning("Bot firmware start timed out")

    @staticmethod
    def _bot_attachment_timeout(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._attachment_timeout

    @staticmethod
    def _matches_attach_terminal(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        return Bot._matches_attachment_terminal(ctx, instance, event, kind="attach")

    @staticmethod
    def _matches_detach_terminal(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> bool:
        return Bot._matches_attachment_terminal(ctx, instance, event, kind="detach")

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
            isinstance(data, _CleanupData)
            and data.kind == kind
            and event.id == data.request_id
            and event.id == instance._lifecycle_step_id
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
        step_id = instance._lifecycle_step_id
        lifetime = instance.context()
        environment = Environment.from_context(lifetime)
        # Deactivation may cancel activation before the activity can publish its
        # incrementally-started ownership set. Cleanup therefore owns the complete
        # configured lifecycle surface; stopping an actor that never started is
        # intentionally idempotent at this boundary. The firmware, on top of the parts, goes first.
        await Bot._stop_firmware(instance)
        await hsm.stop(instance._attachments, environment)
        for ability in reversed(Bot._lifecycle_abilities(instance)):
            await hsm.stop(ability, lifetime)
        # Device.stop stops the peripherals that device powers, so this owns configured devices only.
        for device in reversed(list(instance._devices.values())):
            await device.stop(environment)
        Bot._clear_owned_device_sources(ctx, instance, event)
        terminal = _CleanupData(request_id=step_id, kind="activation")
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _CleanupDoneEvent.with_data(terminal),
                id=terminal.request_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _cancelling_processing_activity(ctx: hsm.Context, instance: "Bot", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, _ProcessingOperationData)
        cancel_id = _bot_cancel_operation_id(data.request_id, data.token, instance)
        # Cancel-token capability was minted with the turn timer; only the cancel-timeout timer
        # is owned here.
        timer = _ProcessingOperation()
        _ = await _ProcessingOperation.started(
            ctx,
            timer,
            owner=instance,
            request_id=data.request_id,
            delay=min(
                max(Bot._resolved_processing_timeout(instance), _MIN_BOT_CANCELLATION_TIMEOUT),
                _MAX_BOT_CANCELLATION_TIMEOUT,
            ),
            event_template=_ProcessingCancelTimedOutEvent,
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

        The devices map is fixed at construction and keyed by each device's own construction-time
        id (``Device.identity()``), so the references are one unchanging truth;
        the runtime ids bind only once the devices are actually up, which is why this restates
        on every activation (a restarted device is a new actor with a new id) and never
        earlier — an id that does not exist yet is not stamped.

        Snapshot ``owned_devices`` is configured reference → shell device id (environment
        presence), and ``firmware`` the firmware's runtime id, which restarts with the bot too.
        ``_device_source_refs`` is the reverse hot-path map: every started actor id
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
        firmware = instance._firmware
        if firmware is not None and lifecycle.is_started(firmware):
            _ = instance.set(_FIRMWARE_ATTRIBUTE, hsm.id(firmware))

    @staticmethod
    def _clear_owned_device_sources(ctx: hsm.Context, instance: "Bot", event: hsm.Event[typing.Any]) -> None:
        """Drop start-time ownership ids when devices are no longer live under this body."""

        del ctx, event
        instance._device_source_refs = {}

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Bot",
        hsm.attribute(_OWNED_DEVICES_ATTRIBUTE),
        hsm.attribute(_FIRMWARE_ATTRIBUTE),
        hsm.attribute(_HELD_PERCEPTS_ATTRIBUTE, 0),
        hsm.initial(hsm.target("inactive")),
        # Every accepted bot.deactivate gets exactly one public terminal under its id: inactive
        # and degraded answer it at once, the stop states supersede the request they are running
        # for, and the activation-rollback and reboot states defer it to the state that can.
        hsm.state(
            "inactive",
            hsm.transition(
                hsm.on(events.ActivateEvent),
                hsm.effect(_accept_activate),
                hsm.target("../activating"),
            ),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_accept_deactivate, _dispatch_deactivating_done),
            ),
        ),
        hsm.state(
            "activating",
            hsm.entry(_enter_lifecycle_step),
            hsm.activity(_activate_activity),
            hsm.transition(
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus),
                hsm.target("../reboot_deactivating"),
            ),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_accept_deactivate, _clear_focus),
                hsm.target("../activation_cancelling"),
            ),
            # Every part attached: a body with firmware starts it next, one without is active.
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.guard(_matches_attach_terminal_with_firmware),
                hsm.target("../firmware_starting"),
            ),
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.guard(_matches_attach_terminal),
                hsm.effect(_dispatch_activating_done),
                hsm.target("../active"),
            ),
            hsm.transition(
                hsm.on(attachment.AttachFailedEvent),
                hsm.guard(_matches_attach_terminal),
                hsm.effect(_dispatch_activating_failed),
                hsm.target("../activation_cleanup"),
            ),
        ),
        # The last activation step: the firmware starts on top of the attached parts. A failure,
        # or no answer within the attachment bound, fails the activation and rolls it back. A
        # product handed over meanwhile (a firmware may report as it starts) waits for active.
        hsm.state(
            "firmware_starting",
            hsm.entry(_enter_lifecycle_step),
            hsm.activity(_start_firmware_activity),
            hsm.defer(cognition.InputEvent),
            hsm.transition(
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus),
                hsm.target("../reboot_deactivating"),
            ),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_accept_deactivate, _clear_focus),
                hsm.target("../activation_cancelling"),
            ),
            hsm.transition(
                hsm.on(_FirmwareStartedEvent),
                hsm.guard(_matches_firmware_terminal),
                hsm.effect(_count_firmware_started, _dispatch_activating_done),
                hsm.target("../active"),
            ),
            hsm.transition(
                hsm.on(_FirmwareFailedEvent),
                hsm.guard(_matches_firmware_terminal),
                hsm.effect(_count_firmware_failed, _dispatch_activating_failed),
                hsm.target("../activation_cleanup"),
            ),
            hsm.transition(
                hsm.after(_bot_attachment_timeout),
                hsm.effect(_count_firmware_timeout, _dispatch_activating_failed),
                hsm.target("../activation_cleanup"),
            ),
        ),
        # A stop accepted during activation: roll back what activation started, then answer it.
        # No bot.reboot transition: a cognition reboot never turns an accepted stop back into an
        # activation.
        hsm.state(
            "activation_cancelling",
            hsm.entry(_enter_lifecycle_step),
            hsm.activity(_activation_cleanup_activity),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_accept_deactivate),
            ),
            hsm.transition(
                hsm.on(_CleanupDoneEvent),
                hsm.guard(_matches_activation_cleanup_done),
                hsm.effect(_dispatch_deactivating_done),
                hsm.target("../inactive"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.effect(_dispatch_degraded_activation_cancel),
                hsm.target("../degraded"),
            ),
        ),
        # Rollback after a failed activation; no stop is pending, so cognition may still reboot.
        hsm.state(
            "activation_cleanup",
            hsm.entry(_enter_lifecycle_step),
            hsm.activity(_activation_cleanup_activity),
            hsm.defer(events.DeactivateEvent),
            hsm.transition(
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus),
                hsm.target("../reboot_deactivating"),
            ),
            hsm.transition(
                hsm.on(_CleanupDoneEvent),
                hsm.guard(_matches_activation_cleanup_done),
                hsm.target("../inactive"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.effect(_dispatch_degraded_activation_cleanup),
                hsm.target("../degraded"),
            ),
        ),
        hsm.state(
            "deactivating",
            hsm.entry(_enter_lifecycle_step),
            hsm.activity(_deactivate_activity),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_accept_deactivate),
            ),
            hsm.transition(
                hsm.on(attachment.DetachedEvent, attachment.DetachFailedEvent),
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
            hsm.entry(_enter_lifecycle_step),
            hsm.activity(_deactivation_cleanup_activity),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_accept_deactivate),
            ),
            hsm.transition(
                hsm.on(_CleanupDoneEvent),
                hsm.guard(_matches_deactivation_cleanup_done),
                hsm.effect(_dispatch_deactivating_done),
                hsm.target("../inactive"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.effect(_dispatch_degraded_deactivation_cleanup),
                hsm.target("../degraded"),
            ),
        ),
        hsm.state(
            "reboot_deactivating",
            hsm.entry(_enter_lifecycle_step),
            hsm.activity(_deactivate_activity),
            hsm.defer(events.DeactivateEvent),
            hsm.transition(
                hsm.on(attachment.DetachedEvent),
                hsm.guard(_matches_detach_terminal),
                hsm.target("../reboot_cleanup_preserve"),
            ),
            hsm.transition(
                hsm.on(attachment.DetachFailedEvent),
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
            hsm.entry(_enter_lifecycle_step),
            hsm.activity(_reboot_cleanup_preserve_activity),
            hsm.defer(events.DeactivateEvent),
            hsm.transition(
                hsm.on(_CleanupDoneEvent),
                hsm.guard(_matches_deactivation_cleanup_done),
                hsm.target("../activating"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.effect(_dispatch_degraded_reboot_cleanup),
                hsm.target("../degraded"),
            ),
        ),
        hsm.state(
            "reboot_cleanup_reset",
            hsm.entry(_enter_lifecycle_step),
            hsm.activity(_reboot_cleanup_reset_activity),
            hsm.defer(events.DeactivateEvent),
            hsm.transition(
                hsm.on(_CleanupDoneEvent),
                hsm.guard(_matches_deactivation_cleanup_done),
                hsm.target("../activating"),
            ),
            hsm.transition(
                hsm.after(_bot_deactivation_timeout),
                hsm.effect(_dispatch_degraded_reboot_cleanup),
                hsm.target("../degraded"),
            ),
        ),
        hsm.state(
            "degraded",
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_accept_deactivate, _answer_degraded),
            ),
        ),
        hsm.state(
            "active",
            hsm.initial(hsm.target("unfocused")),
            # Devices are up by now (started during activation), so their runtime ids are real.
            hsm.entry(_describe_owned_devices),
            # Whatever is still held when the body leaves active is released, counted, not replayed.
            hsm.exit(_release_held),
            hsm.transition(
                hsm.on(events.RebootEvent),
                hsm.guard(_reboot_requested_by_cognition),
                hsm.effect(_clear_focus),
                hsm.target("../reboot_deactivating"),
            ),
            # Explicit environment input events fan out in parallel to input abilities (never cognition;
            # cognition is only told, uniformly, that a stimulus arrived).
            # Efference is Speaking→Listening only (speaking-owned product), not body fan-out.
            hsm.transition(
                hsm.on(SoundEvent, VisualEvent),
                hsm.effect(_fan_out_input),
            ),
            hsm.transition(
                hsm.on(events.DeactivateEvent),
                hsm.effect(_accept_deactivate, _clear_focus),
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
                # Reflex floor: something happening to this bot is the occasion. The guard
                # reads device identity and nothing else — never what happened, never
                # insistence — so the turn is granted blind to content and the body turns
                # to look. What (if anything) to do with it is cognition's, and ignoring
                # is a real answer. Per-stimulus bias lives in cognition, never here.
                # The turn takes what it perceives from what the body holds, so the occasion is held
                # first, like anything that arrives while the bot is busy.
                hsm.transition(
                    hsm.on(events.InputEvent),
                    hsm.guard(_input_targets_configured_device),
                    hsm.effect(_hold_percept, _focus_event_target),
                    hsm.target("../processing"),
                ),
                # Sensory / contribution products: explicit cognition.InputEvent handoff
                # (never raw environment media; never conversation.OutputEvent special-case).
                # Bot grants Cognition the turn; it does not inspect or route the stimulus.
                hsm.transition(
                    hsm.on(cognition.InputEvent),
                    hsm.effect(_hold_percept, _focus_event_target),
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
                    hsm.effect(_hold_percept),
                    hsm.target("../processing"),
                ),
                hsm.transition(
                    hsm.on(cognition.InputEvent),
                    hsm.effect(_hold_percept),
                    hsm.target("../processing"),
                ),
            ),
            # Every turn ends here. Something held starts the next turn at once, which perceives it
            # with whatever else is held; an unoccupied body first turns to what arrived first.
            # Otherwise the body idles where its attention is.
            hsm.choice(
                "turn_ended",
                hsm.transition(hsm.guard(_holds_percepts_with_focus), hsm.target("processing")),
                hsm.transition(hsm.guard(_holds_percepts), hsm.effect(_focus_first_held), hsm.target("processing")),
                hsm.transition(hsm.guard(_has_focused_device), hsm.target("focused")),
                hsm.transition(hsm.target("unfocused")),
            ),
            hsm.state(
                "processing",
                hsm.activity(_dispatch_bot_processing),
                # A happening survives a busy turn: what arrives is held, and the next turn
                # perceives it together with everything else held, in arrival order.
                hsm.transition(
                    hsm.on(events.InputEvent),
                    hsm.guard(_input_targets_configured_device),
                    hsm.effect(_hold_percept),
                ),
                hsm.transition(hsm.on(cognition.InputEvent), hsm.effect(_hold_percept)),
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
                    hsm.guard(_processing_completed),
                    hsm.effect(_retire_bot_turn),
                    hsm.target("../turn_ended"),
                ),
                hsm.transition(
                    hsm.on(events.ProcessingFailedEvent),
                    hsm.guard(_processing_failed),
                    hsm.effect(_retire_bot_turn),
                    hsm.target("../turn_ended"),
                ),
                hsm.transition(
                    hsm.on(_ProcessingTimedOutEvent),
                    hsm.guard(_matches_processing_timeout),
                    hsm.effect(_cancel_bot_processing),
                    hsm.target("../cancelling_processing"),
                ),
            ),
            hsm.state(
                "cancelling_processing",
                hsm.activity(_cancelling_processing_activity),
                hsm.transition(
                    hsm.on(events.InputEvent),
                    hsm.guard(_input_targets_configured_device),
                    hsm.effect(_hold_percept),
                ),
                hsm.transition(hsm.on(cognition.InputEvent), hsm.effect(_hold_percept)),
                hsm.transition(
                    hsm.on(cognition.CancelledEvent, processing.CancelledEvent),
                    hsm.guard(_matches_cognition_cancelled),
                    hsm.effect(_dispatch_processing_timeout_failure),
                    hsm.target("../turn_ended"),
                ),
                hsm.transition(
                    hsm.on(_ProcessingCancelTimedOutEvent),
                    hsm.guard(_matches_processing_timeout),
                    hsm.effect(_dispatch_processing_timeout_failure, _dispatch_degraded_processing_cancel),
                    # Relative so redefines (a body named for its bot) keep a valid target.
                    hsm.target("../../degraded"),
                ),
            ),
        ),
        hsm.observe(observer),
    )
