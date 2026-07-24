import asyncio
import collections.abc
import dataclasses
import datetime
import typing
import uuid

import hsm
import pydantic

from bot import abilities
from bot import lifecycle
from bot.protocols import attachment

from bot.device.events import (
    FirmwareInitializingFailedEvent,
    FirmwareInitializingDoneEvent,
    FirmwareInitializingDoneEventData,
    FirmwareInitializingFailedEventData,
)
from bot.telemetry import observer
from bot.world import World, require_world_scope

_DEFAULT_FIRMWARE = hsm.define(
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


def _require_attach_world_scope(world: World, instance: "Device") -> None:
    instance_scope = instance.context().value(hsm.Keys.Instances)
    world_scope = world.value(hsm.Keys.Instances)
    if instance_scope is world_scope:
        return
    if instance_scope is None:
        raise RuntimeError("Device is not started in this world.")
    raise RuntimeError("Device is already started in another world.")


class Device(hsm.Instance, attachment.Attachment):
    """Environment interaction surface that records attached external actors."""

    firmware_model: typing.ClassVar[hsm.Model] = _DEFAULT_FIRMWARE
    _attachment_limit: typing.ClassVar[int | None] = None
    _firmware_initializing_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_FIRMWARE_INITIALIZING_TIMEOUT
    required_bot_abilities: typing.ClassVar[tuple[type[abilities.Ability[typing.Any, typing.Any]], ...]] = ()
    _peripherals: tuple["Device", ...]
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

    def __init__(
        self,
        *,
        peripherals: collections.abc.Iterable["Device"] = (),
    ) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._peripherals = tuple(peripherals)
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

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        return await super().start(ctx, data)

    @typing.override
    async def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> None:
        world = World.from_context(ctx)
        _require_attach_world_scope(world, self)
        await hsm.Instance.dispatch(self, ctx, event)

    @typing.override
    async def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> None:
        # Stopped/unstarted device: surface typed failure when a reply sink exists.
        if not lifecycle.is_started(self):
            data = event.data
            if not isinstance(data, attachment.DetachData):
                return
            reply_to: hsm.Instance = data.reply_to if data.reply_to is not None else data.actor
            reply_id = hsm.id(reply_to) if lifecycle.is_started(reply_to) else ""
            await hsm.Instance.dispatch(
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
            return
        require_world_scope(World.from_context(ctx), self, participant="Device")
        await hsm.Instance.dispatch(self, ctx, event)

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        await hsm.Instance.stop(self, ctx)

        firmware = self._firmware
        if firmware is None:
            return
        # Do not raise into Bot multi-device teardown by calling hsm.stop on a dead machine.
        if not lifecycle.is_started(firmware):
            if self._firmware is firmware:
                self._firmware = None
            return
        firmware_id = hsm.id(firmware)
        firmware_instances: object | None = firmware.context().value(hsm.Keys.Instances)
        await hsm.stop(firmware)
        if lifecycle.is_started(firmware):
            raise RuntimeError("Device firmware remained started after Device stop.")
        if firmware_id and isinstance(firmware_instances, collections.abc.MutableMapping):
            typed_map = typing.cast(collections.abc.MutableMapping[str, object], firmware_instances)
            if typed_map.get(firmware_id) is firmware:
                _ = typed_map.pop(firmware_id, None)
        if self._firmware is firmware:
            self._firmware = None

    @typing.override
    async def restart(self, ctx: hsm.Context, data: object = None) -> typing.Self | None:
        restart_ctx = ctx
        if ctx is self.context():
            values: dict[typing.Hashable, object] = {}
            instances = ctx.value(hsm.Keys.Instances)
            owner = ctx.value(hsm.Keys.Owner)
            if instances is not None:
                values[hsm.Keys.Instances] = instances
            if owner is not None:
                values[hsm.Keys.HSM] = owner
            restart_ctx = hsm.Context(values=values)
        if restart_ctx.is_done():
            return None
        await self.stop(restart_ctx)
        _ = await super().start(restart_ctx, data)
        return self

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        """Deliver into device shell and firmware. No event-name routing table.

        Shell-only lifecycle payloads (attach/detach) stay on the shell.
        Other events also reach firmware when present; unmatched triggers are ignored by
        normal HSM semantics. Dual delivery is intentional for shared device/firmware
        observations; subclasses (e.g. Phone) may override with narrower routing.
        """

        async def _deliver() -> None:
            await hsm.Instance.dispatch(self, ctx, event)
            firmware = self._firmware
            if firmware is None:
                return
            # Shell-only lifecycle (typed payload, not event.name).
            if isinstance(event.data, (attachment.AttachData, attachment.DetachData)):
                return
            await firmware.dispatch(ctx, event)

        # Match hsm.Instance.dispatch: eager Task so fire-and-forget effect paths still run.
        return asyncio.Task(_deliver(), loop=asyncio.get_running_loop(), eager_start=True)

    @typing.override
    def take_snapshot(self) -> hsm.Snapshot:
        snapshot = super().take_snapshot()
        firmware = self._firmware
        if firmware is None:
            return snapshot
        # Stopped machines cannot take_snapshot. Firmware may already be stopped during
        # initialization_failing cleanup while still referenced here.
        if not lifecycle.is_started(firmware):
            return snapshot
        firmware_snapshot = firmware.take_snapshot()
        return dataclasses.replace(
            snapshot,
            Transitions=(*snapshot.Transitions, *firmware_snapshot.Transitions),
        )

    def _create_firmware_instance(self, ctx: hsm.Context, event: hsm.Event) -> hsm.Instance:
        del ctx, event
        return hsm.Instance()

    async def _initialize_firmware(self, ctx: hsm.Context, event: hsm.Event) -> None:
        # Firmware outlives this activity: parent under the device machine context, not activity ctx.
        # Activity cancel on state exit would otherwise mark firmware/service contexts done (HSM-CONTEXT-001).
        lifetime = self.context()
        self._firmware = await hsm.started(
            lifetime,
            self._create_firmware_instance(ctx, event),
            self.firmware_model,
            hsm.Config(Data=event.data),
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

    model: typing.ClassVar[hsm.Model | None] = hsm.define(
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
                    attachment.Attachment._remember_attachment_timeout,
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
