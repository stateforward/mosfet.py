"""Smart-phone messaging contracts: texts in, texts out, and the lock screen.

A smart phone is a phone: every call contract lives in :mod:`mosfet.devices.phone` and is reused
unchanged. What a smart phone adds is messaging, so only messaging is defined here.

Event names keep the ``phone.`` domain. A text arrives at a phone number and is sent from one;
these are facts about the phone line, not about which handset model is holding it, and a smart
phone offers them alongside the phone's own call commands under one domain.
"""

from mosfet.devices import phone

import datetime
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


MessageId = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description=(
            "The id of one message in a thread on this phone, exactly as a phone.thread page listed it: the "
            "messaging service's own id when it reported one, otherwise the number the handset gave it."
        ),
        examples=["message-3"],
    ),
]


class ReadThreadData(pydantic.BaseModel):
    """Command from the holder: open the conversation with one correspondent and read a page of it.

    Like opening a thread in a phone's messages app: the handset keeps every text it received from
    and sent to each correspondent, and nothing about a thread reaches the holder until the holder
    opens it. The phone answers with one page, ``phone.thread`` — the newest messages first time,
    and older ones by naming the oldest message already read as ``before``, which is scrolling back.
    Reading a thread changes nothing on the handset: it does not read or dismiss a notification
    and the correspondent is never told. There is no example correspondent, deliberately, as for a
    recipient: a concrete address in a tool schema acts as a default.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )

    with_: SmsAddress = pydantic.Field(
        alias="with",
        description=(
            "Whose conversation to read: the address of the other party, written exactly as the phone reported "
            "the sender of a text you received or the recipient of one you sent. A correspondent this phone has "
            "never texted with has an empty thread."
        ),
    )
    before: MessageId | None = pydantic.Field(
        default=None,
        description=(
            "Read the messages older than this one. Null (the default) reads the newest messages. To scroll "
            "back, name the oldest message on the page you already have; a message the phone no longer keeps "
            "has nothing older on the handset."
        ),
        examples=[None, "message-3"],
    )
    limit: int = pydantic.Field(
        default=20,
        ge=1,
        le=50,
        description="Most messages to read on this page, from 1 to 50. Defaults to 20.",
        examples=[20],
    )


class ThreadMessageData(pydantic.BaseModel):
    """One message in a thread: who wrote it is ``direction``, from the handset's side."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {"id": "message-1", "at": "2026-10-20T09:00:00Z", "direction": "received", "text": "Book me the 10:15."}
            ]
        },
    )

    id: MessageId
    at: datetime.datetime = pydantic.Field(
        description="When the handset received the message, or when the service accepted it for sending.",
        examples=["2026-10-20T09:00:00Z"],
    )
    direction: typing.Literal["received", "sent"] = pydantic.Field(
        description="received: the correspondent sent it to this phone. sent: this phone sent it to them.",
        examples=["received", "sent"],
    )
    text: SmsText


class ThreadData(pydantic.BaseModel):
    """One page of the conversation with one correspondent, as the phone keeps it.

    The answer to ``phone.read_thread``: the newest ``limit`` messages before the one it named,
    oldest first, the way a thread reads down the screen. ``more`` says whether older messages
    remain on the handset to scroll back to. An empty page is an answer, not a failure: nobody
    has texted with that correspondent on this phone, or nothing older is kept.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "correspondent": "+15555550101",
                    "messages": [
                        {
                            "id": "message-1",
                            "at": "2026-10-20T09:00:00Z",
                            "direction": "received",
                            "text": "Book me the 10:15.",
                        },
                        {"id": "message-2", "at": "2026-10-20T09:01:00Z", "direction": "sent", "text": "Done."},
                    ],
                    "more": False,
                }
            ]
        },
    )

    correspondent: SmsAddress
    messages: tuple[ThreadMessageData, ...] = pydantic.Field(
        description="The page's messages, oldest first; empty when there is nothing to read.",
    )
    more: bool = pydantic.Field(
        description=(
            "Whether older messages remain on the handset. Read them with phone.read_thread, naming the first "
            "message on this page as before."
        ),
        examples=[False],
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
ReadThreadEvent = hsm.Event[ReadThreadData](
    name="phone.read_thread",
    kind=event.EventKind,
    schema=ReadThreadData,
)
ThreadEvent = hsm.Event[ThreadData](
    name="phone.thread",
    schema=ThreadData,
)
"""One page of a thread, in the hand of whoever read it: the answer to :data:`ReadThreadEvent`."""


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
