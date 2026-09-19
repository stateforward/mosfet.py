import asyncio
import collections.abc
import dataclasses
import datetime
import typing
import uuid

import hsm
import mosfet
import pydantic

from mosfet import abilities
from mosfet import address
from mosfet import lifecycle
from mosfet.protocols import attachment

from mosfet.device.events import (
    FirmwareInitializingFailedEvent,
    FirmwareInitializingDoneEvent,
    FirmwareInitializingDoneEventData,
    FirmwareInitializingFailedEventData,
)
from mosfet.telemetry import observer
from mosfet.environment import Environment, require_environment_scope, space
from mosfet.telemetry.hsm import Traced, deliver

_DEFAULT_FIRMWARE = mosfet.define(
    "DeviceFirmware",
    hsm.initial(hsm.target("initialized")),
    hsm.state("initialized"),
    hsm.observe(observer),
)
_DEFAULT_FIRMWARE_INITIALIZING_TIMEOUT = datetime.timedelta(minutes=5)


class _FirmwareInitializingCleanupData(pydantic.BaseModel):
    """Result details for rollback after firmware initialization fails."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    message: str = pydantic.Field(
        description="Human-readable reason firmware initialization or its rollback failed.",
        examples=["Device firmware initialization timed out after 300 seconds."],
    )
    operation_id: str = pydantic.Field(
        min_length=1,
        description="Live firmware-cleanup capability identity for this device activity.",
        examples=["firmware-cleanup-1"],
    )


_FirmwareInitializingCleanedUpEvent = hsm.Event[_FirmwareInitializingCleanupData](
    name="device.firmware.initializing.cleaned_up",
    kind=hsm.CompletionEventKind,
    schema=_FirmwareInitializingCleanupData,
)
_FirmwareInitializingCleanupFailedEvent = hsm.Event[_FirmwareInitializingCleanupData](
    name="device.firmware.initializing.cleanup_failed",
    kind=hsm.ErrorEventKind,
    schema=_FirmwareInitializingCleanupData,
)


def _require_attach_environment_scope(environment: Environment, instance: "Device") -> None:
    if environment.contains(instance):
        return
    if not lifecycle.is_started(instance):
        raise RuntimeError("Device is not started in this environment.")
    raise RuntimeError("Device is already started in another environment.")


class Device(Traced, attachment.Attachment):
    """Environment interaction surface that records attached external actors."""

    firmware_model: typing.ClassVar[hsm.Model] = _DEFAULT_FIRMWARE
    observation_name: typing.ClassVar[str | None] = None
    """Snapshot attribute key this device contributes when owned as another device's peripheral.

    ``None`` is the default and means this peripheral declares no owner-visible observation —
    right for a transducer like a microphone or speaker, which has no snapshot state of its own
    to show. A peripheral with owner-visible state (a display showing caller ID) names itself
    here; the owning device never invents that name on the peripheral's behalf. Set once per
    subclass, never per instance: the owner folds each started peripheral's own attributes,
    keyed by this leaf name, into its own :meth:`take_snapshot` the same way it already folds
    firmware attributes. One name per owner is the contract, enforced at :meth:`Device.__init__`:
    two owned peripherals declaring the same non-``None`` name raise there rather than silently
    resolving by write order in a later snapshot.
    """
    _attachment_limit: typing.ClassVar[int | None] = None
    _firmware_initializing_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_FIRMWARE_INITIALIZING_TIMEOUT
    required_bot_abilities: typing.ClassVar[tuple[type[abilities.Ability[typing.Any, typing.Any]], ...]] = ()
    _peripherals: tuple["Device", ...]
    _placement: space.Placement | None
    _firmware: hsm.Instance | None
    # Live capability tokens for firmware init/cleanup activities (typed event data carries the same id).
    _firmware_init_operation_id: str | None
    _firmware_cleanup_operation_id: str | None

    def __init_subclass__(
        cls,
        required_bot_abilities: tuple[type[abilities.Ability[typing.Any, typing.Any]], ...] = (),
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.required_bot_abilities = required_bot_abilities

    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str

    def __init__(
        self,
        *,
        peripherals: collections.abc.Iterable["Device"] = (),
        placement: space.Placement | None = None,
    ) -> None:
        super().__init__()
        # Where this device is, and how quiet a sound can get before it stops hearing. No class
        # default: a device cannot know where the robot holding it stands, and defaulting would
        # silently co-locate every device at the origin — perfect cross-talk with no error.
        self._placement = placement
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self._peripherals = tuple(peripherals)
        # One observation name per owner is the documented contract (see ``observation_name``):
        # a silent last-write-wins fold would corrupt a real device's snapshot with no
        # diagnostic pointing at the modeling mistake. Catch it here, at construction, rather
        # than inside live telemetry observation during a later transition.
        seen_observation_names: set[str] = set()
        for peripheral in self._peripherals:
            name = type(peripheral).observation_name
            if name is None:
                continue
            if name in seen_observation_names:
                raise ValueError(
                    f"Multiple peripherals declare the observation name {name!r}; an owner can fold "
                    + "only one peripheral's observation under each name."
                )
            seen_observation_names.add(name)
        self._firmware = None
        self._firmware_init_operation_id = None
        self._firmware_cleanup_operation_id = None

    @staticmethod
    def device_tree(*roots: "Device") -> tuple["Device", ...]:
        """Depth-first pre-order walk of devices and privately owned peripherals."""

        ordered: list[Device] = []
        seen: set[int] = set()

        def visit(device: Device) -> None:
            identifier = id(device)
            if identifier in seen:
                return
            seen.add(identifier)
            ordered.append(device)
            for peripheral in device._peripherals:
                visit(peripheral)

        for root in roots:
            visit(root)
        return tuple(ordered)

    @staticmethod
    def _powered_peripherals(device: "Device") -> tuple["Device", ...]:
        """Transitive peripherals a device powers, deduplicated, rejecting a cycle into the owner.

        ``device_tree`` already dedupes a diamond; the cycle check is what keeps ``start`` and
        ``stop`` — which recurse through the peripherals' own lifecycle methods — from recursing
        without bound. Failing loudly beats a ``RecursionError`` raised from inside bring-up.
        """

        ordered = Device.device_tree(*device._peripherals)
        if any(peripheral is device for peripheral in ordered):
            raise RuntimeError(f"{type(device).__name__} peripherals form a cycle back into the device.")
        return ordered

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        # A device powers its own peripherals, the way a handset powers its own transducers, and
        # powers them first: firmware wires itself to them during bring-up, so they have to be
        # live before this device starts. Ownership decides this, never the caller's walk order.
        # One resolved scope for the whole set, so a bare ``ctx`` cannot leave each machine in an
        # instance map of its own — attaching across those would read as a foreign environment.
        scope = Environment.from_context(ctx)
        owner_model = data if isinstance(data, hsm.Model) else type(self).model
        owner_name = owner_model.qualified_name if owner_model is not None else None
        used_names: set[str] = set()
        for peripheral in Device._powered_peripherals(self):
            base_model = type(peripheral).model
            if base_model is None:
                raise RuntimeError(f"{type(peripheral).__name__} has no lifecycle model.")
            model_name = f"{owner_model.qualified_name.removeprefix('/') if owner_model is not None else ''}{type(peripheral).__name__}"
            unique_name = model_name
            suffix = 2
            while unique_name in used_names:
                unique_name = f"{model_name}{suffix}"
                suffix += 1
            used_names.add(unique_name)
            model = hsm.redefine(base_model, unique_name)
            if lifecycle.is_started(peripheral):
                # A peripheral powered by this device must already live in this device's scope.
                # Re-registering a machine that is started in another environment would leave it
                # broadcast- and teardown-invisible to the owning device (HSM-CONTEXT-001).
                if not scope.contains(peripheral):
                    message_start = f"{type(peripheral).__name__} is already started in another environment"
                    raise RuntimeError(f"{message_start}; a device cannot power a peripheral from a different scope.")
                _ = mosfet.register(peripheral, model, owner=owner_name)
                continue
            _ = await mosfet.started(scope, peripheral, model, owner=owner_name)
        instance = await super().start(scope, data)
        # Presence is a property of being started in an environment, not of being owned by a bot.
        # Whoever puts a device into the scope takes it out: start joins, stop leaves, and both
        # resolve that scope from self.context(), which is the authoritative one. A passed ctx may
        # be a bare context, or a different scope than the device actually started in; either
        # resolves to a throwaway Environment whose participant set is empty, so the operation would
        # silently do nothing.
        Environment.from_context(self.context()).join(self, placement=self._placement)
        return instance

    @typing.override
    async def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> bool:
        environment = Environment.from_context(ctx)
        _require_attach_environment_scope(environment, self)
        return await deliver(self, ctx, event)

    @typing.override
    async def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> bool:
        # Stopped/unstarted device: surface typed failure when a reply sink exists.
        if not lifecycle.is_started(self):
            data = event.data
            if not isinstance(data, attachment.DetachData):
                return False
            reply_to: hsm.Instance = data.reply_to if data.reply_to is not None else data.actor
            reply_id = hsm.id(reply_to) if lifecycle.is_started(reply_to) else ""
            await deliver(
                reply_to,
                ctx,
                dataclasses.replace(
                    attachment.DetachFailedEvent.with_data(
                        attachment.FailedData(
                            actor=data.actor,
                            kind=attachment.FailureKind.DISPATCH,
                            message=f"{type(self).__name__} is stopped or not started; detach refused.",
                        )
                    ),
                    id=event.id,
                    source=event.source or "",
                    target=reply_id,
                    metadata=dict(event.metadata),
                ),
            )
            return False
        require_environment_scope(Environment.from_context(ctx), self, participant="Device")
        return await deliver(self, ctx, event)

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        # Pairs with the join in start and resolves the same way, for the reason given there.
        # Still valid on an already-stopped device: Instance.context() keeps the last started
        # context, and a never-started one yields a bare context, so this is a no-op rather than
        # an error — Bot activation cleanup stops the whole configured device set regardless.
        # No test can observe this: hsm.Started re-news the machine with a fresh id and
        # dispatch_to de-dupes on snapshot ID, so a stale entry can never mis-deliver. It is
        environment = Environment.from_context(self.context())
        environment.leave(self)
        device_path = f"{environment.scope_path}/{hsm.id(self)}" if lifecycle.is_started(self) else None
        address.unregister_instance(self, device_path)
        await hsm.Instance.stop(self, ctx)
        if self.model is not None:
            _ = mosfet.register(self, self.model, clear_owner=True)

        firmware = self._firmware
        if firmware is not None:
            # Do not raise into Bot multi-device teardown by calling hsm.stop on a dead machine.
            if not lifecycle.is_started(firmware):
                if self._firmware is firmware:
                    self._firmware = None
            else:
                firmware_path = f"{environment.scope_path}/{hsm.id(firmware)}"
                address.unregister_instance(firmware, firmware_path)
                await hsm.stop(firmware)
                if lifecycle.is_started(firmware):
                    raise RuntimeError("Device firmware remained started after Device stop.")
                if self._firmware is firmware:
                    self._firmware = None
            _ = mosfet.register(firmware, self.firmware_model, clear_owner=True)

        # Genuine reverse of start: firmware goes down before the transducers it holds
        # attachments to, so it can never emit at an already-stopped peripheral, and the
        # peripherals go last. Stopping one that never started is idempotent, which the Bot
        # cleanup boundary relies on. This is the only teardown path peripherals have.
        for peripheral in Device._powered_peripherals(self):
            model = type(peripheral).model
            peripheral_path = (
                f"{environment.scope_path}/{hsm.id(peripheral)}" if lifecycle.is_started(peripheral) else None
            )
            address.unregister_instance(peripheral, peripheral_path)
            await peripheral.stop(ctx)
            if model is not None:
                _ = mosfet.register(peripheral, model, clear_owner=True)

    @typing.override
    async def restart(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        """Stop and start again under ``ctx``, the scope the device comes back up in.

        ``ctx`` MUST outlive this device's own context, which ``stop`` cancels — pass the environment,
        or another durable scope. The device does not synthesize a replacement context: a
        synthesized one carries the addressing map but no environment scope, so the device would come
        back addressable yet absent from environment presence, and every later broadcast from it would
        silently reach nobody.

        This deliberately diverges from ``hsm.HSM.restart``, which still synthesizes a parentless
        context when handed the machine's own. Devices require the caller to name the scope, so
        ``hsm.restart(device)`` — which defaults ``ctx`` to ``sm.context()`` — raises here.
        """

        if ctx is self.context():
            raise ValueError(
                "Device.restart needs a scope that outlives the device; self.context() is canceled by stop."
            )
        if ctx.is_done():
            raise ValueError("Device.restart needs a live scope; the given context is already canceled.")
        await self.stop(ctx)
        # Device.start, not hsm.Instance.start: restart must re-join the presence stop left.
        _ = await self.start(ctx, data)
        # Restart does not go through bot.started, so restore the studio owner that stop cleared.
        if self.model is not None:
            _ = mosfet.register(self, self.model, owner=self.model.qualified_name)
        return self

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        """Deliver into device shell and firmware. No event-name routing table.

        Shell-only lifecycle payloads (attach/detach) stay on the shell.
        Other events also reach firmware when present; unmatched triggers are ignored by
        normal HSM semantics. Dual delivery is intentional for shared device/firmware
        observations; subclasses (e.g. Phone) may override with narrower routing.
        Resolves to whether the shell or the firmware accepted the event.
        """

        async def _deliver() -> bool:
            shell_delivered = await deliver(self, ctx, event)
            firmware = self._firmware
            if firmware is None:
                return shell_delivered
            # Shell-only lifecycle (typed payload, not event.name).
            if isinstance(event.data, (attachment.AttachData, attachment.DetachData)):
                return shell_delivered
            firmware_delivered = await firmware.dispatch(ctx, event)
            return shell_delivered or firmware_delivered

        # Match hsm.Instance.dispatch: eager Task so fire-and-forget effect paths still run.
        return asyncio.Task(_deliver(), loop=asyncio.get_running_loop(), eager_start=True)

    @typing.override
    def take_snapshot(self) -> hsm.Snapshot:
        snapshot = super().take_snapshot()
        # What the firmware declares for observation is the device's to show: attributes merge
        # the same way transitions do, so one snapshot of the device tells the whole story.
        # Behaviorally the device IS its firmware — observers need the behavioral state; the
        # shell's attach-lifecycle state stays observable live via ``instance.state()``.
        attributes: dict[str, typing.Any] = {}
        if snapshot.Attributes:
            attributes.update(snapshot.Attributes)
        state = snapshot.State
        transitions = snapshot.Transitions
        firmware = self._firmware
        # Stopped machines cannot take_snapshot. Firmware may already be stopped during
        # initialization_failing cleanup while still referenced here.
        if firmware is not None and lifecycle.is_started(firmware):
            firmware_snapshot = firmware.take_snapshot()
            if firmware_snapshot.Attributes:
                attributes.update(firmware_snapshot.Attributes)
            state = firmware_snapshot.State or snapshot.State
            transitions = (*snapshot.Transitions, *firmware_snapshot.Transitions)
        # A started owned peripheral's own observation folds up under its declared
        # ``observation_name``, the same way firmware attributes fold up under their own leaf
        # names: one snapshot of the owner tells the whole story, including what a display shows.
        # A peripheral that declares no observation name (the default) contributes nothing.
        # A peripheral with nothing at all set on its own snapshot (no ``hsm.attribute`` ever
        # written) also contributes nothing — but one whose attributes are all ``None`` still
        # does: a display showing nobody is a fact about that display, not the absence of one,
        # and folding it away would make "blank" indistinguishable from "not there". ``Device``
        # guarantees at construction that at most one peripheral claims a given name, so this
        # never has to arbitrate a collision.
        for peripheral in self._peripherals:
            name = type(peripheral).observation_name
            if name is None:
                continue
            peripheral_snapshot = lifecycle.snapshot_if_started(peripheral)
            if peripheral_snapshot is None or not peripheral_snapshot.Attributes:
                continue
            peripheral_attributes = typing.cast(collections.abc.Mapping[str, object], peripheral_snapshot.Attributes)
            observation: dict[str, object] = {
                key.rpartition("/")[2]: value for key, value in peripheral_attributes.items()
            }
            attributes[name] = observation
        return dataclasses.replace(
            snapshot,
            State=state,
            Attributes=attributes or None,
            Transitions=transitions,
        )

    def _create_firmware_instance(self, ctx: hsm.Context, event: hsm.Event) -> hsm.Instance:
        del ctx, event
        return hsm.Instance()

    async def _initialize_firmware(self, ctx: hsm.Context, event: hsm.Event) -> None:
        # Firmware outlives this activity: parent under the device machine context, not activity ctx.
        # Activity cancel on state exit would otherwise mark firmware/service contexts done (HSM-CONTEXT-001).
        lifetime = self.context()
        owner_model = type(self).model
        owner_name = owner_model.qualified_name if owner_model is not None else None
        self._firmware = await mosfet.started(
            lifetime,
            self._create_firmware_instance(ctx, event),
            self.firmware_model,
            hsm.Config(Data=event.data),
            owner=owner_name,
        )
        self._on_firmware_started(ctx, event)
        await self._after_firmware_started(lifetime, event)

    @staticmethod
    async def _initialize_firmware_activity(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        device_id = hsm.id(instance)
        operation_id = event.id or uuid.uuid4().hex
        instance._firmware_init_operation_id = operation_id
        try:
            try:
                await instance._initialize_firmware(ctx, event)
            except Exception as error:
                result = FirmwareInitializingFailedEvent.with_data(
                    FirmwareInitializingFailedEventData(message=str(error), operation_id=operation_id)
                )
            else:
                result = FirmwareInitializingDoneEvent.with_data(
                    FirmwareInitializingDoneEventData(operation_id=operation_id)
                )
            # Await RTC so the correlated guard can observe the live capability before teardown.
            await instance.dispatch(
                ctx,
                dataclasses.replace(
                    result,
                    id=operation_id,
                    source=device_id,
                    target=device_id,
                    metadata=dict(event.metadata),
                ),
            )
            await asyncio.wrap_future(ctx.done())
        finally:
            if instance._firmware_init_operation_id == operation_id:
                instance._firmware_init_operation_id = None

    @staticmethod
    def _is_current_firmware_initialization_result(
        ctx: hsm.Context,
        instance: "Device",
        event: hsm.Event,
    ) -> bool:
        del ctx
        data = event.data
        device_id = hsm.id(instance)
        if not isinstance(data, (FirmwareInitializingDoneEventData, FirmwareInitializingFailedEventData)):
            return False
        return (
            instance._firmware_init_operation_id == data.operation_id
            and event.id == data.operation_id
            and event.source == device_id
            and event.target == device_id
        )

    @staticmethod
    def _firmware_initializing_timeout_delay(
        ctx: hsm.Context,
        instance: "Device",
        event: hsm.Event,
    ) -> datetime.timedelta:
        del ctx, event
        return instance._firmware_initializing_timeout

    @staticmethod
    async def _cleanup_failed_firmware_activity(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        device_id = hsm.id(instance)
        operation_id = event.id or uuid.uuid4().hex
        instance._firmware_cleanup_operation_id = operation_id
        try:
            data = event.data
            if isinstance(data, FirmwareInitializingFailedEventData):
                failure_message = data.message
            else:
                seconds = instance._firmware_initializing_timeout.total_seconds()
                failure_message = f"Device firmware initialization timed out after {seconds:g} seconds."
            firmware = instance._firmware
            if firmware is not None:
                try:
                    # HSM-CONTEXT-001: do not probe firmware.state() for readiness before stop.
                    await hsm.stop(firmware)
                except Exception as error:
                    result = _FirmwareInitializingCleanupFailedEvent.with_data(
                        _FirmwareInitializingCleanupData(
                            message=f"Device firmware initialization rollback failed: {error}",
                            operation_id=operation_id,
                        )
                    )
                    await hsm.dispatch(
                        ctx,
                        instance,
                        dataclasses.replace(
                            result,
                            id=operation_id,
                            source=device_id,
                            target=device_id,
                            metadata=dict(event.metadata),
                        ),
                    )
                    return
            if instance._firmware is firmware:
                instance._firmware = None
            result = _FirmwareInitializingCleanedUpEvent.with_data(
                _FirmwareInitializingCleanupData(message=failure_message, operation_id=operation_id)
            )
            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    result,
                    id=operation_id,
                    source=device_id,
                    target=device_id,
                    metadata=dict(event.metadata),
                ),
            )
            await asyncio.wrap_future(ctx.done())
        finally:
            if instance._firmware_cleanup_operation_id == operation_id:
                instance._firmware_cleanup_operation_id = None

    @staticmethod
    def _is_current_firmware_cleanup_completion(
        ctx: hsm.Context,
        instance: "Device",
        event: hsm.Event,
    ) -> bool:
        del ctx
        data = event.data
        device_id = hsm.id(instance)
        return (
            isinstance(data, _FirmwareInitializingCleanupData)
            and instance._firmware_cleanup_operation_id == data.operation_id
            and event.id == data.operation_id
            and event.source == device_id
            and event.target == device_id
            and instance._firmware is None
        )

    @staticmethod
    def _dispatch_failed_firmware_attachment(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, attachment.AttachData)
        target = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                attachment.AttachFailedEvent.with_data(
                    attachment.FailedData(
                        actor=data.actor,
                        kind=attachment.FailureKind.INITIALIZATION,
                        message="Device firmware initialization failed.",
                    )
                ),
                id=event.id,
                source=hsm.id(instance),
                target=attachment.Attachment._actor_id(target),
                metadata=dict(event.metadata),
            ),
        )

    def _on_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event

    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Device",
        hsm.initial(hsm.target("initializing")),
        hsm.state(
            "initializing",
            hsm.activity(_initialize_firmware_activity),
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.transition(
                hsm.after(_firmware_initializing_timeout_delay),
                hsm.target("../initialization_failing"),
            ),
            hsm.transition(
                hsm.on(FirmwareInitializingDoneEvent),
                hsm.guard(_is_current_firmware_initialization_result),
                hsm.target("../detached"),
            ),
            hsm.transition(
                hsm.on(FirmwareInitializingFailedEvent),
                hsm.guard(_is_current_firmware_initialization_result),
                hsm.target("../initialization_failing"),
            ),
        ),
        hsm.state(
            "initialization_failing",
            hsm.activity(_cleanup_failed_firmware_activity),
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.transition(
                hsm.on(_FirmwareInitializingCleanedUpEvent),
                hsm.guard(_is_current_firmware_cleanup_completion),
                hsm.target("../failed"),
            ),
            hsm.transition(hsm.on(_FirmwareInitializingCleanupFailedEvent)),
        ),
        hsm.state(
            "failed",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._can_attach),
                hsm.effect(_dispatch_failed_firmware_attachment),
            ),
        ),
        hsm.state(
            "detached",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._can_attach),
                hsm.effect(
                    attachment.Attachment._attach,
                    attachment.Attachment._remember_attachment_request,
                    attachment.Attachment._queue_attach_complete,
                ),
                hsm.target("../attaching"),
            ),
        ),
        hsm.state(
            "attaching",
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.effect(attachment.Attachment._deliver_attach_complete),
                hsm.target("../attached"),
            ),
            hsm.transition(
                hsm.after(attachment.Attachment._attachment_timeout_delay),
                hsm.effect(attachment.Attachment._timeout_attachment),
                hsm.target("../detached"),
            ),
        ),
        # Attached is ownership only (who owns the device). Domain "in use" / call state
        # lives on firmware (e.g. Phone ringing); bot attention is focus — not a shell mode.
        hsm.state(
            "attached",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._is_attached),
                hsm.effect(attachment.Attachment._dispatch_attach_complete_existing),
            ),
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._can_attach),
                hsm.effect(
                    attachment.Attachment._attach,
                    attachment.Attachment._dispatch_attach_complete_created,
                ),
            ),
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._is_attach_request),
                hsm.effect(attachment.Attachment._dispatch_attach_failed),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._detach_would_leave_attachments),
                hsm.effect(attachment.Attachment._detach, attachment.Attachment._dispatch_detach_complete_removed),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._detach_last_attachment),
                hsm.effect(attachment.Attachment._detach, attachment.Attachment._dispatch_detach_complete_removed),
                hsm.target("../detached"),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._is_detach_request),
                hsm.effect(attachment.Attachment._dispatch_detach_failed),
            ),
        ),
        hsm.observe(observer),
    )
