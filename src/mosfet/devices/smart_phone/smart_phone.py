from mosfet.devices import phone

import dataclasses
import importlib.resources
import typing
import uuid

import hsm
import mosfet
import pydantic

from mosfet.environment import SoundEvent, Environment
from mosfet.telemetry import observer

# Flat symbol imports (not `from . import display`), as in the phone package: SmartPhone accepts
# a `display` constructor parameter, which would shadow a `display` module import.
from .display import Display, NotificationsData, NotificationsEvent
from .events import (
    DismissNotificationData,
    DismissNotificationEvent,
    NotificationData,
    NotificationEvent,
    NotificationNotFoundData,
    NotificationNotFoundEvent,
    ReadNotificationData,
    ReadNotificationEvent,
    SendTextMessageData,
    SendTextMessageEvent,
    ServiceTextMessageSendFailedEvent,
    ServiceTextMessageSendRequestedEvent,
    ServiceTextMessageSentEvent,
    SmsTextData,
    SoundData,
    TextMessageSendFailedData,
    TextMessageSendFailedEvent,
    TextMessageSentData,
    TextMessageSentEvent,
)

NOTIFICATION_DB = 70.0
"""How loud a handset's notification ding is, in dB SPL measured one metre away.

Device-intrinsic like :data:`mosfet.devices.phone.RINGER_DB`, and deliberately 10 dB below it. Both come out of the same
loudspeaker and both are for the room — a text alert has to reach someone who is not holding the
phone, which is why it sits well above :data:`mosfet.devices.phone.CALL_PROGRESS_DB`. But a ring must keep fetching you
until you answer, while a notification will wait, so handsets play one short, quieter ding.
Across a room it is audible and easy to miss, which is what a notification is.
"""

NOTIFICATION_WAV = (importlib.resources.files(__package__) / "assets" / "notification.wav").read_bytes()
"""Notification ding: the one short chime a handset plays for any notification."""


class Firmware(phone.Firmware):
    """Smart-phone firmware: the phone's call firmware plus messaging and the lock screen.

    Every call transition is the phone's own, defined once in :class:`mosfet.devices.phone.Firmware`;
    this firmware adds only what a smart phone does besides calls.
    """

    # The lock screen's pending list, oldest first. Firmware owns it and drives the display from
    # it; the display only shows what it is sent. Rebound, never mutated, so the empty default is
    # every new handset's own empty lock screen.
    _notifications: tuple[NotificationData, ...] = ()

    @staticmethod
    def _publish_text_message_send_requested(ctx: hsm.Context, instance: "Firmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, SendTextMessageData)
        Firmware._publish(ctx, instance, event, ServiceTextMessageSendRequestedEvent.with_data(data))

    @staticmethod
    def _publish_text_message_sent(ctx: hsm.Context, instance: "Firmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, TextMessageSentData)
        Firmware._publish(ctx, instance, event, TextMessageSentEvent.with_data(data))

    @staticmethod
    def _publish_text_message_send_failed(ctx: hsm.Context, instance: "Firmware", event: hsm.Event) -> None:
        # The service's verdict travels with the fact it explains, as for a failed dial.
        data = event.data
        assert isinstance(data, TextMessageSendFailedData)
        Firmware._publish(ctx, instance, event, TextMessageSendFailedEvent.with_data(data))

    @staticmethod
    def _show_notifications(ctx: hsm.Context, instance: "Firmware", trigger: hsm.Event) -> None:
        Firmware._drive_display(
            ctx, instance, trigger, NotificationsEvent.with_data(NotificationsData(pending=instance._notifications))
        )

    @staticmethod
    def _add_notification(ctx: hsm.Context, instance: "Firmware", event: hsm.Event) -> None:
        """A notification stays on the lock screen until it is read or dismissed, held or not."""

        data = event.data
        assert isinstance(data, NotificationData)
        instance._notifications = (*instance._notifications, data)
        Firmware._show_notifications(ctx, instance, event)

    @staticmethod
    def _is_pending_notification(ctx: hsm.Context, instance: "Firmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        assert isinstance(data, ReadNotificationData | DismissNotificationData)
        return any(notification.id == data.id for notification in instance._notifications)

    @staticmethod
    def _is_unknown_notification(ctx: hsm.Context, instance: "Firmware", event: hsm.Event) -> bool:
        return not Firmware._is_pending_notification(ctx, instance, event)

    @staticmethod
    def _remove_notification(ctx: hsm.Context, instance: "Firmware", event: hsm.Event) -> None:
        """Read and dismiss both take it off the lock screen; neither is announced to anyone."""

        data = event.data
        assert isinstance(data, ReadNotificationData | DismissNotificationData)
        instance._notifications = tuple(
            notification for notification in instance._notifications if notification.id != data.id
        )
        Firmware._show_notifications(ctx, instance, event)

    @staticmethod
    def _publish_notification_not_found(ctx: hsm.Context, instance: "Firmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, ReadNotificationData | DismissNotificationData)
        # The command is the trigger itself, so its canonical name is what the failure names.
        not_found = NotificationNotFoundData.model_validate({"id": data.id, "command": event.name})
        Firmware._publish(ctx, instance, event, NotificationNotFoundEvent.with_data(not_found))

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Phone",
        *phone.Firmware._topology,
        # Messaging is independent of the call: a handset sends texts whatever state its call is
        # in, so these live at the root rather than under any call state.
        hsm.transition(hsm.on(SendTextMessageEvent), hsm.effect(_publish_text_message_send_requested)),
        hsm.transition(hsm.on(ServiceTextMessageSentEvent), hsm.effect(_publish_text_message_sent)),
        hsm.transition(hsm.on(ServiceTextMessageSendFailedEvent), hsm.effect(_publish_text_message_send_failed)),
        # The lock screen is independent of the call too. Reading and dismissing are the holder's
        # own acts on it; replying to a message is not reading it.
        hsm.transition(hsm.on(NotificationEvent), hsm.effect(_add_notification)),
        hsm.transition(
            hsm.on(ReadNotificationEvent), hsm.guard(_is_pending_notification), hsm.effect(_remove_notification)
        ),
        hsm.transition(
            hsm.on(DismissNotificationEvent), hsm.guard(_is_pending_notification), hsm.effect(_remove_notification)
        ),
        hsm.transition(
            hsm.on(ReadNotificationEvent),
            hsm.guard(_is_unknown_notification),
            hsm.effect(_publish_notification_not_found),
        ),
        hsm.transition(
            hsm.on(DismissNotificationEvent),
            hsm.guard(_is_unknown_notification),
            hsm.effect(_publish_notification_not_found),
        ),
        hsm.observe(observer),
    )


class SmartPhone(phone.Phone):
    """A phone that also texts: every call behavior of :class:`mosfet.devices.phone.Phone`, plus messaging.

    Incoming texts ding into the room, reach the holder as a notification, and wait on the lock
    screen until read or dismissed; outgoing texts go through the same service as calls.
    """

    firmware_model: typing.ClassVar[hsm.Model] = Firmware.model
    _firmware_type: typing.ClassVar[type[phone.Firmware]] = Firmware
    _display_type: typing.ClassVar[type[phone.Display]] = Display
    _command_schemas: typing.ClassVar[tuple[type[pydantic.BaseModel], ...]] = (
        *phone.Phone._command_schemas,
        SendTextMessageData,
        ReadNotificationData,
        DismissNotificationData,
    )

    @typing.override
    def _observe(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        """The phone's two paths, plus what messaging makes plain to the hand.

        A text that did not go is silent to the room and plain to the hand (the "not delivered"
        mark on the message you just sent), so it reaches the holder. A text that went is not
        surfaced at all: the holder pressed send and already knows what it sent, a handset marks
        success with nothing worth noticing, and telling the bot would buy it a fresh turn for
        every message it sends — an invitation to answer itself. :data:`TextMessageSentEvent`
        stays a published fact for the service side and observers.

        A read or dismiss that named no pending notification is plain to the hand in the same way;
        one that worked is visible on the display the holder is looking at, and nothing more.
        """

        if isinstance(event.data, TextMessageSendFailedData | NotificationNotFoundData):
            self._tell_holders(ctx, event)
            return
        super()._observe(ctx, event)

    @typing.override
    async def _route(self, ctx: hsm.Context, event: hsm.Event) -> bool:
        """An incoming text becomes a notification; everything else routes as on a phone.

        One notification, stamped once with its handset id, drives all three paths: the room
        hears the ding, whoever holds the phone gets the banner, and firmware keeps it on the lock
        screen until it is read or dismissed — held or not. Resolves to whether firmware accepted
        it (``False`` without firmware).

        Notifications are the one observation with both paths, unlike a call's: the ding is
        heard as the handset's generic ding, ``kind`` ``phone.notification``, and sounds the same
        whatever caused it, so the banner in your hand is how you learn what it was.
        """

        if not isinstance(event.data, SmsTextData):
            return await super()._route(ctx, event)
        notification = dataclasses.replace(
            NotificationEvent.with_data(
                # Validated, not asserted: the event's own name must be a notifying one.
                NotificationData.model_validate(
                    {
                        "id": event.data.id or event.id or uuid.uuid4().hex,
                        "name": event.name,
                        "data": event.data,
                    }
                )
            ),
            id=event.id,
            metadata=dict(event.metadata),
        )
        ding = dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=NOTIFICATION_WAV,
                    media_type="audio/wav",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="phone.notification",
                    notification=notification.data,
                    amplitude_db=NOTIFICATION_DB,
                )
            ),
            source=hsm.id(self),
            metadata=dict(notification.metadata),
        )
        placement = self._placement
        _ = Environment.from_context(ctx).broadcast(ding, origin=None if placement is None else placement.position)
        self._tell_holders(ctx, notification)
        firmware = self._firmware
        if firmware is None:
            return False
        return await firmware.dispatch(ctx, notification)
