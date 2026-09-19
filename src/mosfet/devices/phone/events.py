from mosfet.devices import audio
from mosfet import environment

import typing

import hsm
from mosfet import event
import pydantic

CallId = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description=(
            "Provider-neutral session handle for one phone call. Firmware and the phone service use it to reject "
            "stale service callbacks belonging to an earlier call. It is a transport handle with no counterpart on "
            "a handset, so it never appears on an operator command: firmware already knows which call it has."
        ),
        examples=["livekit:caller", "sip:session-9f3a"],
    ),
]
_WRITTEN_SEPARATORS = str.maketrans("", "", " \t -‐‑‒–—―−.()/")
"""Every character a number is *printed* with and none it is dialled with.

Typographic dashes and the non-breaking space are in here beside the ASCII ones because a number
that has been through a text round trip is not written the way it was typed, and an en dash is
still somebody writing a hyphen.
"""


def _dialed_digits(value: object) -> object:
    """Read a written number the way a hand does: the separators stay on the paper.

    The digits are the number and the formatting is not. A number is printed with spaces, dashes
    and brackets and dialled without them, because a keypad has no key for any of them — and a
    number that has been said out loud, crossed a room and come back through speech recognition
    has whatever punctuation the transcriber chose, none of which the caller controlled. So
    ``555-0142``, ``5550142``, ``555 0142`` and ``(555) 0142`` are one number here, exactly as
    they are to a person.

    Anything else is left exactly as it came in, so what is not a number is rejected as one rather
    than quietly turned into one.
    """

    if isinstance(value, str):
        return value.translate(_WRITTEN_SEPARATORS)
    return value


Number = typing.Annotated[
    str,
    # Field before BeforeValidator, and not the other way round: a validator listed first hides
    # the pattern from the generated JSON schema, which is the one thing a model reads to learn
    # what a number looks like. The validator still runs first at validation time.
    pydantic.Field(
        pattern=r"^\+?[0-9]{2,15}$",
        description=(
            "The number to dial: digits, optionally with a leading + for an international number. That is what "
            "makes a number a number — it can be said out loud, written down, and pressed on a keypad. A name, an "
            "address, or a handle is none of those and reaches nobody, however confidently it is dialled. Write it "
            "however it was given to you: spaces, dashes, dots and brackets are how a number is printed rather than "
            "part of it, so a number written with them and the same number written without them are one number and "
            "reach one phone. Unlike call_id a number belongs to the phone rather than to a call: it is the same "
            "before, during, and after every call, and dialling it twice reaches the same phone twice. Dialling a "
            "number nothing answers is a real outcome, not an error to avoid — the attempt comes back unreachable, "
            "the way it does for a person."
        ),
    ),
    pydantic.BeforeValidator(_dialed_digits),
]
Caller = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description=(
            "Who the call is with, as the phone service reports it — the caller ID a handset would show. Not a "
            "session handle and not unique: the same caller may ring repeatedly. Absent when the caller withheld "
            "identification."
        ),
        examples=["Front desk", "+15555550123", "Support queue"],
    ),
]
TransferId = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description=(
            "Provider-neutral identifier for one transfer attempt within a call. Firmware uses this value with "
            "call_id and target to reject stale callbacks from earlier attempts to the same destination."
        ),
        examples=["transfer-123"],
    ),
]
TransferTargetKind = typing.Literal["address", "device", "service"]
FailureKind = typing.Literal[
    "provider_unavailable",
    "remote_unavailable",
    "media_unavailable",
    "signaling_failed",
    "call_declined",
    "transfer_rejected",
    "timeout",
    "unknown",
]
HangUpOutcome = typing.Literal["local_hang_up", "declined", "remote_hang_up", "failed", "transferred"]
NoCallReason = typing.Literal["nothing_to_answer", "dial_not_answered", "dial_abandoned", "dial_failed"]

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
    failure_kind: FailureKind = pydantic.Field(
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


class CallIdData(pydantic.BaseModel):
    """Payload scoped to one provider-neutral call identifier."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"call_id": "livekit:caller"}],
        },
    )

    call_id: CallId


class TransferTarget(pydantic.BaseModel):
    """Provider-neutral destination for a call transfer request."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"kind": "address", "value": "helpdesk@example.com"}],
        },
    )

    kind: TransferTargetKind = pydantic.Field(
        description=(
            "Category of transfer destination. Firmware records the intent without knowing how a service provider "
            "will resolve the target."
        ),
        examples=["address", "device", "service"],
    )
    value: str = pydantic.Field(
        min_length=1,
        description=(
            "Opaque destination identifier understood by the selected phone service provider, such as an address, "
            "device handle, or service route."
        ),
        examples=["helpdesk@example.com", "operator-desk", "customer-support"],
    )


class IncomingCallData(CallIdData):
    """Signal from a phone service provider that a call is ringing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"call_id": "livekit:caller", "caller": "Front desk"}],
        },
    )

    caller: Caller | None = None


class DialData(pydantic.BaseModel):
    """Command from an operator or owner asking firmware to dial a number.

    Carries no call id. You dial a number and the exchange assigns the call, so the session
    handle arrives from the provider on connect and there is nothing for a caller to supply.

    No example number, deliberately. Every number is deployment-specific, and a concrete one in a
    tool schema acts as a default: a bot that is unsure what it was told dials the number printed
    here instead of the one it heard.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    number: Number


class AnswerCallData(pydantic.BaseModel):
    """Command from an operator or owner asking firmware to answer the call that is ringing.

    Fieldless, like the answer button on a handset: it answers whatever is ringing. Firmware
    already holds the call, so naming one could only name the wrong one.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{}]},
    )


class DeclineCallData(pydantic.BaseModel):
    """Command from an operator or owner asking firmware to decline the call that is ringing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{}]},
    )


class HangUpCallData(pydantic.BaseModel):
    """Command from an operator or owner asking firmware to hang up the call it is on."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{}]},
    )


class TransferCallData(pydantic.BaseModel):
    """Command from an operator or owner asking firmware to transfer the call it is on."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "transfer_id": "transfer-123",
                    "target": {"kind": "address", "value": "helpdesk@example.com"},
                }
            ],
        },
    )

    transfer_id: TransferId
    target: TransferTarget = pydantic.Field(
        description=(
            "Provider-neutral transfer destination. The provider owns the transport mechanics; phone firmware owns "
            "the user-visible transfer state."
        ),
        examples=[{"kind": "address", "value": "helpdesk@example.com"}],
    )


class DialFailedData(pydantic.BaseModel):
    """Signal from a phone service provider that an outbound dial never became a call.

    Distinct from :class:`CallFailedData` because there is no call to name: the attempt failed
    before the exchange assigned one.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"failure_kind": "remote_unavailable"}],
        },
    )

    failure_kind: FailureKind = pydantic.Field(
        description=(
            "Normalized low-cardinality reason the outbound attempt did not become a call. "
            "call_declined is not remote_unavailable: somebody answered and said no, which is a "
            "different fact about the far end than nobody being there."
        ),
        examples=["remote_unavailable", "call_declined", "provider_unavailable", "signaling_failed"],
    )


class AnswerRequestData(CallIdData):
    """Firmware asking the phone service to answer one identified call.

    The operator command is fieldless; firmware stamps the call it already holds. Two acts by two
    parties: pressing answer, and telling the exchange which line is being answered.
    """


class DeclineRequestData(CallIdData):
    """Firmware asking the phone service to decline one identified call."""


class HangUpRequestData(CallIdData):
    """Firmware asking the phone service to hang up one identified call."""


class TransferRequestData(CallIdData):
    """Firmware asking the phone service to transfer one identified call."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "call_id": "livekit:caller",
                    "transfer_id": "transfer-123",
                    "target": {"kind": "address", "value": "helpdesk@example.com"},
                }
            ],
        },
    )

    transfer_id: TransferId
    target: TransferTarget = pydantic.Field(
        description="Provider-neutral transfer destination for the requested transfer.",
        examples=[{"kind": "address", "value": "helpdesk@example.com"}],
    )


class CallConnectedData(CallIdData):
    """Signal from a phone service provider that the current call is connected."""

    party: Caller | None = pydantic.Field(
        default=None,
        description=(
            "Who the connected call is with, as the provider knows them. Null when the provider "
            "never learned the far end's identity — a withheld caller ID or an unresolvable number "
            "connects exactly like an identified call, and the provider invents nobody."
        ),
        examples=["Front desk", "+15555550123"],
    )


class MediaReadyData(CallIdData):
    """Signal from a phone service provider that call media is ready for audio routing."""


class ServiceAudioData(audio.OutputData):
    """Signal from a phone service provider carrying audio that should play from the phone speaker."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {
                    "call_id": "call-123",
                    "audio": "YXVkaW8tY2h1bms=",
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 48000,
                    "channels": 1,
                }
            ],
        },
    )

    call_id: CallId


class RemoteHangUpData(CallIdData):
    """Signal from a phone service provider that the remote party ended the current call."""


class CallFailedData(CallIdData):
    """Signal from a phone service provider that the current call failed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"call_id": "call-123", "failure_kind": "signaling_failed"}],
        },
    )

    failure_kind: FailureKind = pydantic.Field(
        description=(
            "Normalized low-cardinality failure category. Provider-specific error details should remain in provider "
            "diagnostics, not the core phone contract."
        ),
        examples=["signaling_failed", "provider_unavailable"],
    )


class TransferAcceptedData(CallIdData):
    """Signal from a phone service provider that it accepted a transfer request."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "call_id": "call-123",
                    "transfer_id": "transfer-123",
                    "target": {"kind": "address", "value": "helpdesk@example.com"},
                }
            ],
        },
    )

    transfer_id: TransferId
    target: TransferTarget = pydantic.Field(
        description="Echoed transfer target for the accepted transfer operation.",
        examples=[{"kind": "address", "value": "helpdesk@example.com"}],
    )


class TransferCompletedData(CallIdData):
    """Signal from a phone service provider that the current call transfer completed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "call_id": "call-123",
                    "transfer_id": "transfer-123",
                    "target": {"kind": "address", "value": "helpdesk@example.com"},
                }
            ],
        },
    )

    transfer_id: TransferId
    target: TransferTarget = pydantic.Field(
        description="Echoed transfer target for the completed transfer operation.",
        examples=[{"kind": "address", "value": "helpdesk@example.com"}],
    )


class TransferFailedData(CallIdData):
    """Signal from a phone service provider that the current call transfer failed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "call_id": "call-123",
                    "transfer_id": "transfer-123",
                    "target": {"kind": "address", "value": "helpdesk@example.com"},
                    "failure_kind": "transfer_rejected",
                }
            ],
        },
    )

    transfer_id: TransferId
    failure_kind: FailureKind = pydantic.Field(
        description=(
            "Normalized low-cardinality transfer failure category. Provider-specific error details should remain in "
            "provider diagnostics."
        ),
        examples=["transfer_rejected", "timeout"],
    )
    target: TransferTarget = pydantic.Field(
        description="Echoed transfer target for the failed transfer operation.",
        examples=[{"kind": "address", "value": "helpdesk@example.com"}],
    )


class CallData(CallIdData):
    """Committed public phone state for a specific call."""


class RingingData(pydantic.BaseModel):
    """Committed public phone state that a call is ringing.

    Carries who is calling, not which session is ringing: a ringing handset shows caller ID, and
    the session handle has no counterpart on it. Distinct from :class:`CallData` so
    environment elevation can select ring acoustics by payload type, never by ``event.name``.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"caller": "Front desk"}, {}],
        },
    )

    caller: Caller | None = None


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


class SoundData(environment.SoundData):
    """``environment.sound`` payload elevated from phone ringing (or call-scoped acoustic energy).

    Subclasses :class:`~mosfet.environment.SoundData` with the caller ID, so what a bot perceives
    when a phone rings is who is calling. ``None`` is a real ring: a withheld caller still rings.

    The phone's ``kind`` vocabulary, all of it sound a handset genuinely makes:

    * ``phone.ringing`` — the ringer, for an incoming call. Carries the caller.
    * ``phone.busy`` — busy tone. A dialled line refused the call or is engaged.
    * ``phone.reorder`` — reorder tone (fast busy). The network could not complete the call.
    * ``phone.notification`` — the notification ding. Carries the event that caused it.

    Only the ringer carries a caller. A call-progress tone is put on the line by the exchange and
    says nothing about who was dialled, which is exactly why a caller learns so little from one.
    Only the ding carries a notification.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "Phone-elevated acoustic stimulus for environment.sound. Either the ringer "
                "(phone.ringing, carrying the caller ID the phone reports, or null when the caller "
                "is unknown or withheld) or a call-progress tone the exchange put in the caller's "
                "ear (phone.busy, phone.reorder), which carries no caller, or the generic "
                "notification ding (phone.notification), carrying the phone event that caused it."
            ),
            "examples": [
                {
                    "audio": "YXVkaW8tY2h1bms=",
                    "media_type": "audio/wav",
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "kind": "phone.ringing",
                    "caller": "Front desk",
                },
                {
                    "audio": "YXVkaW8tY2h1bms=",
                    "media_type": "audio/wav",
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "kind": "phone.busy",
                },
                {
                    "audio": "YXVkaW8tY2h1bms=",
                    "media_type": "audio/wav",
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "kind": "phone.notification",
                    "notification": {
                        "name": "phone.sms.text",
                        "data": {"id": "message-id", "sender": "+15555550101", "text": "Book me the 10:15."},
                    },
                },
            ],
        },
    )

    caller: Caller | None = pydantic.Field(
        default=None,
        description=(
            "Who this call is with, as the phone reports it. Null when the caller is unknown or "
            "withheld — an anonymous call rings exactly like an identified one — and always null "
            "on a call-progress tone, which identifies nobody."
        ),
        examples=["Front desk", "+15555550123"],
    )
    notification: NotificationData | None = pydantic.Field(
        default=None,
        description=(
            "On a phone.notification ding, the phone event that caused it — its name says what "
            "kind of notification it is and its data is what the banner shows. Always null on the "
            "ringer and on call-progress tones, which are not notifications."
        ),
    )


class HungUpData(CallIdData):
    """Committed public phone state that the call ended."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"call_id": "call-123", "outcome": "remote_hang_up"}],
        },
    )

    outcome: HangUpOutcome = pydantic.Field(
        description="Provider-neutral reason the phone firmware now considers the call ended.",
        examples=["local_hang_up", "declined", "remote_hang_up", "failed", "transferred"],
    )


class NoCallData(pydantic.BaseModel):
    """Committed public phone state that a requested call action left the phone with no call.

    The thing that was asked for did not happen, and no call exists to report it against — so
    there is no call id here, and inventing one would fabricate the call that never was.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {"reason": "nothing_to_answer"},
                {"reason": "dial_not_answered"},
                {"reason": "dial_failed", "failure_kind": "call_declined"},
            ],
        },
    )

    reason: NoCallReason = pydantic.Field(
        description=(
            "Why no call resulted: nothing_to_answer when answering with nothing ringing, "
            "dial_not_answered when an outbound attempt timed out unconnected, dial_abandoned "
            "when the attempt was hung up before it connected, dial_failed when the service "
            "reported the attempt could not be completed."
        ),
        examples=["nothing_to_answer", "dial_not_answered", "dial_abandoned", "dial_failed"],
    )
    failure_kind: FailureKind | None = pydantic.Field(
        default=None,
        description=(
            "The service's normalized verdict on a dial_failed attempt, carried through unchanged. "
            "Null for every other reason, because nothing outside the phone decided those: nobody "
            "answered, the operator hung up, or there was nothing ringing to answer. This is what "
            "separates a refused line from an unreachable one — the same distinction a real caller "
            "hears as busy tone against reorder — so flattening it away leaves the phone unable to "
            "say which sound the exchange would have made."
        ),
        examples=["call_declined", "remote_unavailable", "provider_unavailable"],
    )


class TransferData(CallIdData):
    """Committed public phone state for an in-progress or completed transfer."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "call_id": "call-123",
                    "transfer_id": "transfer-123",
                    "target": {"kind": "address", "value": "helpdesk@example.com"},
                }
            ],
        },
    )

    transfer_id: TransferId
    target: TransferTarget = pydantic.Field(
        description="Provider-neutral destination that firmware is transferring or has transferred the call to.",
        examples=[{"kind": "address", "value": "helpdesk@example.com"}],
    )


class CallTransferFailedData(TransferData):
    """Committed public phone state that an attempted transfer failed and the original call remains active."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "call_id": "call-123",
                    "transfer_id": "transfer-123",
                    "target": {"kind": "address", "value": "helpdesk@example.com"},
                    "failure_kind": "timeout",
                }
            ],
        },
    )

    failure_kind: FailureKind = pydantic.Field(
        description="Normalized low-cardinality failure category for the committed transfer failure.",
        examples=["timeout", "transfer_rejected"],
    )


IncomingCallEvent = hsm.Event[IncomingCallData](
    name="phone.service.incoming_call",
    schema=IncomingCallData,
)
CallConnectedEvent = hsm.Event[CallConnectedData](
    name="phone.service.call_connected",
    schema=CallConnectedData,
)
ServiceMediaReadyEvent = hsm.Event[MediaReadyData](
    name="phone.service.media_ready",
    schema=MediaReadyData,
)
ServiceAudioReceivedEvent = hsm.Event[ServiceAudioData](
    name="phone.service.audio_received",
    schema=ServiceAudioData,
)
RemoteHangUpEvent = hsm.Event[RemoteHangUpData](
    name="phone.service.remote_hang_up",
    schema=RemoteHangUpData,
)
CallFailedEvent = hsm.Event[CallFailedData](
    name="phone.service.call_failed",
    schema=CallFailedData,
)
TransferAcceptedEvent = hsm.Event[TransferAcceptedData](
    name="phone.service.transfer_accepted",
    schema=TransferAcceptedData,
)
ServiceTransferCompletedEvent = hsm.Event[TransferCompletedData](
    name="phone.service.transfer_completed",
    schema=TransferCompletedData,
)
ServiceTransferFailedEvent = hsm.Event[TransferFailedData](
    name="phone.service.transfer_failed",
    schema=TransferFailedData,
)
AnswerCallEvent = hsm.Event[AnswerCallData](
    name="phone.answer_call",
    kind=event.EventKind,
    schema=AnswerCallData,
)
DialEvent = hsm.Event[DialData](
    name="phone.dial",
    kind=event.EventKind,
    schema=DialData,
)
DeclineCallEvent = hsm.Event[DeclineCallData](
    name="phone.decline_call",
    kind=event.EventKind,
    schema=DeclineCallData,
)
HangUpCallEvent = hsm.Event[HangUpCallData](
    name="phone.hang_up_call",
    kind=event.EventKind,
    schema=HangUpCallData,
)
TransferCallEvent = hsm.Event[TransferCallData](
    name="phone.transfer_call",
    kind=event.EventKind,
    schema=TransferCallData,
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

ServiceAnswerRequestedEvent = hsm.Event[AnswerRequestData](
    name="phone.service.answer_requested",
    schema=AnswerRequestData,
)
# Dial alone carries no call id: the exchange assigns the call, so the handle comes back on
# connect rather than going out with the request.
ServiceDialRequestedEvent = hsm.Event[DialData](
    name="phone.service.dial_requested",
    schema=DialData,
)
ServiceDialFailedEvent = hsm.Event[DialFailedData](
    name="phone.service.dial_failed",
    schema=DialFailedData,
)
ServiceDeclineRequestedEvent = hsm.Event[DeclineRequestData](
    name="phone.service.decline_requested",
    schema=DeclineRequestData,
)
ServiceHangUpRequestedEvent = hsm.Event[HangUpRequestData](
    name="phone.service.hang_up_requested",
    schema=HangUpRequestData,
)
ServiceTransferRequestedEvent = hsm.Event[TransferRequestData](
    name="phone.service.transfer_requested",
    schema=TransferRequestData,
)

RingingEvent = hsm.Event[RingingData](
    name="phone.ringing",
    schema=RingingData,
)
AnsweredEvent = hsm.Event[CallData](
    name="phone.answered",
    schema=CallData,
)
MediaReadyEvent = hsm.Event[CallData](
    name="phone.media_ready",
    schema=CallData,
)
HungUpEvent = hsm.Event[HungUpData](
    name="phone.hung_up",
    schema=HungUpData,
)
NoCallEvent = hsm.Event[NoCallData](
    name="phone.no_call",
    schema=NoCallData,
)
TransferStartedEvent = hsm.Event[TransferData](
    name="phone.transfer_started",
    schema=TransferData,
)
CallTransferCompletedEvent = hsm.Event[TransferData](
    name="phone.transfer_completed",
    schema=TransferData,
)
CallTransferFailedEvent = hsm.Event[CallTransferFailedData](
    name="phone.transfer_failed",
    schema=CallTransferFailedData,
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
