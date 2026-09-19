"""Smart-phone messaging contracts: texts in, texts out, and the lock screen.

A smart phone is a phone: every call contract lives in :mod:`mosfet.devices.phone` and is reused
unchanged. What a smart phone adds is messaging, so only messaging is defined here.

Event names keep the ``phone.`` domain. A text arrives at a phone number and is sent from one;
these are facts about the phone line, not about which handset model is holding it, and a smart
phone offers them alongside the phone's own call commands under one domain.
"""

from mosfet.devices import phone

import typing

import hsm
from mosfet import event
import pydantic

SmsText = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description=(
            "Text body of one phone-domain message. This is the local handset property, not a model payload: "
            "the phone stores the last incoming text under its display."
        ),
        examples=["Book me the 10:15."],
    ),
]


class SmsTextData(pydantic.BaseModel):
    """Phone-domain text message.

    SMS event payload carries no call identity because it is independent
    transport on the same phone. The phone is the wired object; a service
    provider may need its own source identity, but the phone does not.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "message-id",
                    "sender": "+15555550101",
                    "text": "Book me the 10:15.",
                }
            ],
        },
    )

    id: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Optional provider-neutral message identifier. This has no handset-global meaning and is never "
            "written onto the phone as a default."
        ),
        examples=["message-id"],
    )
    sender: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Who sent this message. Null when the service did not report a sender. A routed reply needs "
            "the phone's own sender policy outside this payload."
        ),
        examples=["+15555550101"],
    )
    text: SmsText


SmsTextEvent = hsm.Event[SmsTextData](
    name="phone.sms.text",
    schema=SmsTextData,
)


SmsAddress = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description=(
            "Who a text message goes to, written exactly the way the phone reported the sender of a message "
            "you received: replying to a text means sending to its sender. It is an address on the messaging "
            "service, not a call handle, and it is the same for every message to or from that party."
        ),
    ),
]


class SendTextMessageData(pydantic.BaseModel):
    """Command from an operator or owner asking the phone to send one text message.

    Like dialling, this is pressing send on a handset: the phone hands the message to its
    messaging service and later learns whether it went. A text is not a call, so this is legal
    whether or not a call is up. There is no example recipient, deliberately: a concrete address
    in a tool schema acts as a default, and a bot unsure who it was talking to would text it.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    to: SmsAddress
    text: str = pydantic.Field(
        min_length=1,
        description=(
            "The message body, verbatim: exactly this text is delivered to the recipient's phone as written and "
            "cannot be taken back once sent. Plain written language, not markup and not a description of a "
            "message. Required and non-empty: choosing not to text is done by not sending one."
        ),
    )


class TextMessageSentData(pydantic.BaseModel):
    """The messaging service accepted one outgoing text message."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"to": "+15555550101", "text": "See you at 10:15."}]},
    )

    to: SmsAddress
    text: SmsText


class TextMessageSendFailedData(pydantic.BaseModel):
    """The messaging service did not send one outgoing text message."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"to": "+15555550101", "text": "See you at 10:15.", "failure_kind": "remote_unavailable"}]
        },
    )

    to: SmsAddress
    text: SmsText
    failure_kind: phone.FailureKind = pydantic.Field(
        description="Normalized low-cardinality reason the message was not sent.",
        examples=["provider_unavailable", "remote_unavailable"],
    )


NotificationId = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description=(
            "The id of one notification pending on this phone, exactly as the phone's display lists it. "
            "Only a notification still pending can be named."
        ),
    ),
]


class ReadNotificationData(pydantic.BaseModel):
    """Command from the holder: open one pending notification, which is reading it.

    Like tapping a banner on a lock screen: the notification leaves the pending list because it
    has been seen. A text notification already carries the whole message, so opening it reveals
    nothing further; what changes is only that it no longer waits to be looked at. Replying to a
    message does not read its notification — reading is its own act, and only this command does it.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    id: NotificationId


class DismissNotificationData(pydantic.BaseModel):
    """Command from the holder: clear one pending notification without opening it.

    Like swiping a banner away on a lock screen: the notification leaves the pending list
    unread. Nothing is sent anywhere and the sender is never told.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    id: NotificationId


class NotificationNotFoundData(pydantic.BaseModel):
    """A read or dismiss named a notification that is not pending on this phone.

    Nothing changed on the handset: the id was never shown, or that notification was already
    read or dismissed. The phone says so to whoever pressed it rather than doing nothing silently.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"id": "message-id", "command": "phone.read_notification"}]},
    )

    id: str = pydantic.Field(
        min_length=1,
        description="The notification id the command named, which is not pending on this phone.",
        examples=["message-id"],
    )
    command: typing.Literal["phone.read_notification", "phone.dismiss_notification"] = pydantic.Field(
        description="Which command named it: reading or dismissing.",
        examples=["phone.read_notification"],
    )


class NotificationData(pydantic.BaseModel):
    """The phone event a notification ding announces, carried whole on the ding.

    A handset dings for many reasons and the ding sounds the same for all of them; what tells you
    why is the banner, and the banner is a view of the event that caused it. So the ding carries
    that event — its ``id`` names this notification on the handset, its canonical ``name`` says what kind of notification this is and ``data`` is
    its own typed payload, unchanged. Today the only notifying event is a new text message
    (``phone.sms.text``); another notification kind extends ``name`` and ``data`` together.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "The phone event that caused a notification ding: its canonical event name and its "
                "own payload, as the phone received it. The name identifies the notification kind."
            ),
            "examples": [
                {
                    "id": "message-id",
                    "name": "phone.sms.text",
                    "data": {"id": "message-id", "sender": "+15555550101", "text": "Book me the 10:15."},
                }
            ],
        },
    )

    id: str = pydantic.Field(
        min_length=1,
        description=(
            "Which notification this is on the handset: what phone.read_notification and "
            "phone.dismiss_notification name. For a text message it is the message's own id when the "
            "service reported one."
        ),
        examples=["message-id"],
    )
    name: typing.Literal["phone.sms.text"] = pydantic.Field(
        description=(
            "Canonical name of the phone event that caused the notification, which is what kind of "
            "notification it is. phone.sms.text: a new text message arrived."
        ),
        examples=["phone.sms.text"],
    )
    data: SmsTextData = pydantic.Field(
        description=(
            "The causing event's own payload, unchanged. For phone.sms.text this is the new "
            "message — sender and text — which is what the handset's banner previews."
        ),
    )


NotificationEvent = hsm.Event[NotificationData](
    name="phone.notification",
    schema=NotificationData,
)
"""A notification on the handset, felt by whoever holds it: the banner, carrying the event that caused it."""


class SoundData(phone.SoundData):
    """``environment.sound`` payload of a smart phone: the phone's sounds plus the notification ding.

    Adds one ``kind`` to the phone's vocabulary:

    * ``phone.notification`` — the notification ding. Carries the event that caused it.

    Only the ding carries a notification.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "Smart-phone-elevated acoustic stimulus for environment.sound. Either the ringer "
                "(phone.ringing, carrying the caller ID the phone reports, or null when the caller "
                "is unknown or withheld), a call-progress tone the exchange put in the caller's "
                "ear (phone.busy, phone.reorder), which carries no caller, or the generic "
                "notification ding (phone.notification), carrying the phone event that caused it."
            ),
            "examples": [
                {
                    "audio": "YXVkaW8tY2h1bms=",
                    "media_type": "audio/wav",
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "kind": "phone.notification",
                    "notification": {
                        "id": "message-id",
                        "name": "phone.sms.text",
                        "data": {"id": "message-id", "sender": "+15555550101", "text": "Book me the 10:15."},
                    },
                },
            ],
        },
    )

    notification: NotificationData | None = pydantic.Field(
        default=None,
        description=(
            "On a phone.notification ding, the phone event that caused it — its name says what "
            "kind of notification it is and its data is what the banner shows. Always null on the "
            "ringer and on call-progress tones, which are not notifications."
        ),
    )


SendTextMessageEvent = hsm.Event[SendTextMessageData](
    name="phone.send_text_message",
    kind=event.EventKind,
    schema=SendTextMessageData,
)

ReadNotificationEvent = hsm.Event[ReadNotificationData](
    name="phone.read_notification",
    kind=event.EventKind,
    schema=ReadNotificationData,
)
DismissNotificationEvent = hsm.Event[DismissNotificationData](
    name="phone.dismiss_notification",
    kind=event.EventKind,
    schema=DismissNotificationData,
)
NotificationNotFoundEvent = hsm.Event[NotificationNotFoundData](
    name="phone.notification_not_found",
    schema=NotificationNotFoundData,
)


# Messaging is independent of calls: request out to the service, the service's verdict back in,
# and the committed public outcome. The service echoes the request envelope id on its verdict.
ServiceTextMessageSendRequestedEvent = hsm.Event[SendTextMessageData](
    name="phone.service.text_message_send_requested",
    schema=SendTextMessageData,
)
ServiceTextMessageSentEvent = hsm.Event[TextMessageSentData](
    name="phone.service.text_message_sent",
    schema=TextMessageSentData,
)
ServiceTextMessageSendFailedEvent = hsm.Event[TextMessageSendFailedData](
    name="phone.service.text_message_send_failed",
    schema=TextMessageSendFailedData,
)
TextMessageSentEvent = hsm.Event[TextMessageSentData](
    name="phone.text_message.sent",
    schema=TextMessageSentData,
)
TextMessageSendFailedEvent = hsm.Event[TextMessageSendFailedData](
    name="phone.text_message.send_failed",
    schema=TextMessageSendFailedData,
)
