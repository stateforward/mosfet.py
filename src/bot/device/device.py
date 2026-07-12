import collections.abc
import dataclasses
import datetime
import typing

import hsm
import pydantic

from bot import abilities


from bot.device.events import (
    ActivateEvent,
    ATTACH_CREATED_METADATA_KEY,
    AttachEvent,
    AttachEventData,
    DeactivateEvent,
    DetachEvent,
    DetachEventData,
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
_FirmwareInitializingCleanedUpEvent = hsm.Event[str](
    name="device.firmware.initializing.cleaned_up",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(str),
)


def _bot_identifiers(bot: hsm.Instance) -> tuple[str, ...]:
    identifiers: list[str] = []
    try:
        runtime_id = hsm.id(bot)
    except hsm.ErrorValidatingModel:
        runtime_id = ""
    if runtime_id:
        identifiers.append(runtime_id)
    schema_id = getattr(bot, "id", None)
    if isinstance(schema_id, str) and schema_id and schema_id not in identifiers:
        identifiers.append(schema_id)
    return tuple(identifiers)


def _same_bot(left: hsm.Instance, right: hsm.Instance) -> bool:
    if left is right:
        return True
    return not set(_bot_identifiers(left)).isdisjoint(_bot_identifiers(right))


def _require_attach_world_scope(world: World, instance: "Device") -> None:
    instance_scope = instance.context().value(hsm.Keys.Instances)
    world_scope = world.context.value(hsm.Keys.Instances)
    if instance_scope is world_scope:
        return
    if instance_scope is None:
        raise RuntimeError("Device is not started in this world.")
    raise RuntimeError("Device is already started in another world.")


class Device(hsm.Instance):
    """Environment interaction surface that records attached HSM bot references."""

    firmware_model: typing.ClassVar[hsm.Model] = _DEFAULT_FIRMWARE
    _firmware_initializing_timeout: typing.ClassVar[datetime.timedelta] = _DEFAULT_FIRMWARE_INITIALIZING_TIMEOUT
    required_bot_abilities: typing.ClassVar[tuple[type[abilities.Ability[typing.Any, typing.Any]], ...]] = ()
    _peripherals: tuple["Device", ...]
    _bots: list[hsm.Instance]
    _pending_bots: list[hsm.Instance]
    _firmware: hsm.Instance | None

    def __init_subclass__(
        cls,
        required_bot_abilities: tuple[type[abilities.Ability[typing.Any, typing.Any]], ...] = (),
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.required_bot_abilities = required_bot_abilities

    def __init__(
        self,
        bots: collections.abc.Iterable[hsm.Instance] = (),
        peripherals: collections.abc.Iterable["Device"] = (),
    ) -> None:
        super().__init__()
        self._bots = []
        self._pending_bots = list(bots)
        self._peripherals = tuple(peripherals)
        self._firmware = None

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

    async def attach(self, world: World, bot: hsm.Instance) -> None:
        _require_attach_world_scope(world, self)
        await hsm.Instance.dispatch(self, world.context, AttachEvent.with_data(AttachEventData(bot=bot)))

    async def detach(self, world: World, bot: hsm.Instance) -> None:
        if not self.state() or self.state() == self.model.qualified_name:
            return
        require_world_scope(world, self, participant="Device")
        await hsm.Instance.dispatch(self, world.context, DetachEvent.with_data(DetachEventData(bot=bot)))

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name in self.model.events:
            return super().dispatch(ctx, event)
        firmware_events = getattr(self.firmware_model, "events", None)
        if (
            self._firmware is not None
            and isinstance(firmware_events, collections.abc.Mapping)
            and event.name in firmware_events
        ):
            return self._firmware.dispatch(ctx, event)
        return super().dispatch(ctx, event)

    @typing.override
    def take_snapshot(self) -> hsm.Snapshot:
        snapshot = super().take_snapshot()
        if self._firmware is None:
            return snapshot
        firmware_snapshot = self._firmware.take_snapshot()
        return dataclasses.replace(
            snapshot,
            Transitions=(*snapshot.Transitions, *firmware_snapshot.Transitions),
        )

    def _create_firmware_instance(self, ctx: hsm.Context, event: hsm.Event) -> hsm.Instance:
        del ctx, event
        return hsm.Instance()

    def _can_activate(self, ctx: hsm.Context, event: hsm.Event) -> bool:
        del ctx, event
        return False

    def _on_inactive_entry(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event

    def _on_inactive_exit(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event

    def _attached_bot_index(self, bot: hsm.Instance) -> int | None:
        for index, attached_bot in enumerate(self._bots):
            if _same_bot(attached_bot, bot):
                return index
        return None

    def _pending_bot_index(self, bot: hsm.Instance) -> int | None:
        for index, pending_bot in enumerate(self._pending_bots):
            if _same_bot(pending_bot, bot):
                return index
        return None

    @staticmethod
    def _is_attached_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, AttachEventData) and instance._attached_bot_index(data.bot) is not None

    @staticmethod
    def _can_attach_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, AttachEventData) and instance._attached_bot_index(data.bot) is None

    @staticmethod
    def _is_pending_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, AttachEventData) and instance._pending_bot_index(data.bot) is not None

    @staticmethod
    def _can_queue_pending_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, AttachEventData)
            and instance._attached_bot_index(data.bot) is None
            and instance._pending_bot_index(data.bot) is None
        )

    @staticmethod
    def _can_remove_pending_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, DetachEventData) and instance._pending_bot_index(data.bot) is not None

    @staticmethod
    def _detach_would_leave_bots(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, DetachEventData):
            return False
        return instance._attached_bot_index(data.bot) is not None and len(instance._bots) > 1

    @staticmethod
    def _detach_last_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, DetachEventData):
            return False
        return instance._attached_bot_index(data.bot) is not None and len(instance._bots) == 1

    @staticmethod
    def _attach_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, AttachEventData)
        instance._bots.append(data.bot)
        if data.bot.state():
            bot_identifiers = _bot_identifiers(data.bot)
            _ = data.bot.dispatch(
                ctx,
                dataclasses.replace(
                    AttachEvent.with_data(data),
                    source=hsm.id(instance),
                    target=bot_identifiers[0] if bot_identifiers else "",
                    metadata={**event.metadata, ATTACH_CREATED_METADATA_KEY: True},
                ),
            )

    @staticmethod
    def _notify_attached_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, AttachEventData)
        if not data.bot.state():
            return
        bot_identifiers = _bot_identifiers(data.bot)
        _ = data.bot.dispatch(
            ctx,
            dataclasses.replace(
                AttachEvent.with_data(data),
                source=hsm.id(instance),
                target=bot_identifiers[0] if bot_identifiers else "",
                metadata={**event.metadata, ATTACH_CREATED_METADATA_KEY: False},
            ),
        )

    @staticmethod
    def _queue_pending_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        del ctx
        data = event.data
        assert isinstance(data, AttachEventData)
        instance._pending_bots.append(data.bot)

    @staticmethod
    def _remove_pending_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        del ctx
        data = event.data
        assert isinstance(data, DetachEventData)
        index = typing.cast(int, instance._pending_bot_index(data.bot))
        del instance._pending_bots[index]

    @staticmethod
    def _detach_bot(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        del ctx
        data = event.data
        assert isinstance(data, DetachEventData)
        index = typing.cast(int, instance._attached_bot_index(data.bot))
        del instance._bots[index]

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
        _ = self.dispatch(
            ctx,
            FirmwareInitializingDoneEvent.with_data(FirmwareInitializingDoneEventData()),
        )
        pending_bots = tuple(self._pending_bots)
        self._pending_bots.clear()
        for bot in pending_bots:
            _ = self.dispatch(ctx, AttachEvent.with_data(AttachEventData(bot=bot)))

    @staticmethod
    async def _initialize_firmware_activity(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        try:
            await instance._initialize_firmware(ctx, event)
        except Exception as error:
            _ = instance.dispatch(
                ctx,
                FirmwareInitializingFailedEvent.with_data(FirmwareInitializingFailedEventData(message=str(error))),
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
        data = event.data
        if isinstance(data, FirmwareInitializingFailedEventData):
            failure_message = data.message
        else:
            seconds = instance._firmware_initializing_timeout.total_seconds()
            failure_message = f"Device firmware initialization timed out after {seconds:g} seconds."
        firmware = instance._firmware
        instance._firmware = None
        if firmware is not None and firmware.state():
            try:
                # Stop against the firmware's own context so cancel completes hsm.stop's wait.
                await hsm.stop(firmware)
            except Exception as error:
                failure_message = f"Device firmware initialization rollback failed: {error}"
        _ = hsm.dispatch(ctx, instance, _FirmwareInitializingCleanedUpEvent.with_data(failure_message))

    @staticmethod
    def _clear_failed_firmware_reference(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        del ctx, event
        instance._firmware = None

    @staticmethod
    def _dispatch_firmware_initializing_failure(
        ctx: hsm.Context,
        instance: "Device",
        event: hsm.Event,
    ) -> None:
        data = event.data
        if isinstance(data, str):
            message = data
        elif isinstance(data, FirmwareInitializingFailedEventData):
            message = data.message
        else:
            seconds = instance._firmware_initializing_timeout.total_seconds()
            message = f"Device firmware initialization timed out after {seconds:g} seconds."
        failure = FirmwareInitializingFailedEvent.with_data(FirmwareInitializingFailedEventData(message=message))
        Device._dispatch_pending_firmware_initializing_failure(ctx, instance, failure)

    @staticmethod
    def _dispatch_pending_firmware_initializing_failure(
        ctx: hsm.Context,
        instance: "Device",
        event: hsm.Event[typing.Any],
    ) -> None:
        pending_bots = tuple(instance._pending_bots)
        instance._pending_bots.clear()
        for bot in pending_bots:
            if not bot.state():
                continue
            bot_identifiers = _bot_identifiers(bot)
            _ = bot.dispatch(
                ctx,
                dataclasses.replace(
                    event,
                    source=hsm.id(instance),
                    target=bot_identifiers[0] if bot_identifiers else "",
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _dispatch_failed_firmware_attachment(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, AttachEventData)
        if not data.bot.state():
            return
        bot_identifiers = _bot_identifiers(data.bot)
        _ = data.bot.dispatch(
            ctx,
            dataclasses.replace(
                FirmwareInitializingFailedEvent.with_data(
                    FirmwareInitializingFailedEventData(message="Device firmware initialization failed.")
                ),
                source=hsm.id(instance),
                target=bot_identifiers[0] if bot_identifiers else "",
                metadata=dict(event.metadata),
            ),
        )

    def _on_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event

    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event

    async def _do_inactive_activity(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event

    def _on_active_entry(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event

    def _on_active_exit(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event

    async def _do_active_activity(self, ctx: hsm.Context, event: hsm.Event) -> None:
        del ctx, event

    @staticmethod
    def _on_inactive_entry_effect(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        instance._on_inactive_entry(ctx, event)

    @staticmethod
    def _on_inactive_exit_effect(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        instance._on_inactive_exit(ctx, event)

    @staticmethod
    async def _do_inactive_activity_effect(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        await instance._do_inactive_activity(ctx, event)

    @staticmethod
    def _can_activate_guard(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> bool:
        return instance._can_activate(ctx, event)

    @staticmethod
    def _on_active_entry_effect(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        instance._on_active_entry(ctx, event)

    @staticmethod
    def _on_active_exit_effect(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        instance._on_active_exit(ctx, event)

    @staticmethod
    async def _do_active_activity_effect(ctx: hsm.Context, instance: "Device", event: hsm.Event) -> None:
        await instance._do_active_activity(ctx, event)

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "Device",
        hsm.initial(hsm.target("initializing")),
        hsm.state(
            "initializing",
            hsm.activity(_initialize_firmware_activity),
            hsm.transition(
                hsm.on(AttachEvent),
                hsm.guard(_is_pending_bot),
            ),
            hsm.transition(
                hsm.on(AttachEvent),
                hsm.guard(_can_queue_pending_bot),
                hsm.effect(_queue_pending_bot),
            ),
            hsm.transition(
                hsm.on(DetachEvent),
                hsm.guard(_can_remove_pending_bot),
                hsm.effect(_remove_pending_bot),
            ),
            hsm.transition(
                hsm.after(_firmware_initializing_timeout_delay),
                hsm.target("../initialization_failing"),
            ),
            hsm.transition(
                hsm.on(FirmwareInitializingDoneEvent),
                hsm.target("../detached"),
            ),
            hsm.transition(
                hsm.on(FirmwareInitializingFailedEvent),
                hsm.target("../initialization_failing"),
            ),
        ),
        hsm.state(
            "initialization_failing",
            hsm.activity(_cleanup_failed_firmware_activity),
            # Attach may arrive while cleanup is still running (durable firmware stop is not free).
            # Queue the bot so failure dispatch after cleanup can notify them (HSM-CONTEXT-001 timing).
            hsm.transition(
                hsm.on(AttachEvent),
                hsm.guard(_is_pending_bot),
            ),
            hsm.transition(
                hsm.on(AttachEvent),
                hsm.guard(_can_queue_pending_bot),
                hsm.effect(_queue_pending_bot),
            ),
            hsm.transition(
                hsm.on(DetachEvent),
                hsm.guard(_can_remove_pending_bot),
                hsm.effect(_remove_pending_bot),
            ),
            hsm.transition(
                hsm.on(_FirmwareInitializingCleanedUpEvent),
                hsm.effect(_dispatch_firmware_initializing_failure),
                hsm.target("../failed"),
            ),
            hsm.transition(
                hsm.after(_firmware_initializing_timeout_delay),
                hsm.effect(_clear_failed_firmware_reference, _dispatch_firmware_initializing_failure),
                hsm.target("../failed"),
            ),
        ),
        hsm.state(
            "failed",
            hsm.transition(
                hsm.on(AttachEvent),
                hsm.guard(_can_attach_bot),
                hsm.effect(_dispatch_failed_firmware_attachment),
            ),
        ),
        hsm.state(
            "detached",
            hsm.transition(
                hsm.on(AttachEvent),
                hsm.guard(_can_attach_bot),
                hsm.effect(_attach_bot),
                hsm.target("../attached/inactive"),
            ),
        ),
        hsm.state(
            "attached",
            hsm.initial(hsm.target("inactive")),
            hsm.transition(
                hsm.on(AttachEvent),
                hsm.guard(_is_attached_bot),
                hsm.effect(_notify_attached_bot),
            ),
            hsm.transition(
                hsm.on(AttachEvent),
                hsm.guard(_can_attach_bot),
                hsm.effect(_attach_bot),
            ),
            hsm.transition(
                hsm.on(DetachEvent),
                hsm.guard(_detach_would_leave_bots),
                hsm.effect(_detach_bot),
            ),
            hsm.transition(
                hsm.on(DetachEvent),
                hsm.guard(_detach_last_bot),
                hsm.effect(_detach_bot),
                hsm.target("../detached"),
            ),
            hsm.state(
                "inactive",
                hsm.entry(_on_inactive_entry_effect),
                hsm.exit(_on_inactive_exit_effect),
                hsm.activity(_do_inactive_activity_effect),
                hsm.transition(
                    hsm.on(ActivateEvent),
                    hsm.guard(_can_activate_guard),
                    hsm.target("../active"),
                ),
            ),
            hsm.state(
                "active",
                hsm.entry(_on_active_entry_effect),
                hsm.exit(_on_active_exit_effect),
                hsm.activity(_do_active_activity_effect),
                hsm.transition(
                    hsm.on(DeactivateEvent),
                    hsm.target("../inactive"),
                ),
            ),
        ),
        hsm.observe(observer),
    )
