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
NoCallReason = typing.Literal["dial_not_answered", "dial_abandoned", "dial_failed"]


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


class SoundData(environment.SoundData):
    """``environment.sound`` payload elevated from phone ringing (or call-scoped acoustic energy).

    Subclasses :class:`~mosfet.environment.SoundData` with the caller ID, so what a bot perceives
    when a phone rings is who is calling. ``None`` is a real ring: a withheld caller still rings.

    The phone's ``kind`` vocabulary, all of it sound a handset genuinely makes:

    * ``phone.ringing`` — the ringer, for an incoming call. Carries the caller.
    * ``phone.busy`` — busy tone. A dialled line refused the call or is engaged.
    * ``phone.reorder`` — reorder tone (fast busy). The network could not complete the call.

    Only the ringer carries a caller. A call-progress tone is put on the line by the exchange and
    says nothing about who was dialled, which is exactly why a caller learns so little from one.
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
                "ear (phone.busy, phone.reorder), which carries no caller."
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
                {"reason": "dial_not_answered"},
                {"reason": "dial_failed", "failure_kind": "call_declined"},
            ],
        },
    )

    reason: NoCallReason = pydantic.Field(
        description=(
            "Why no call resulted: dial_not_answered when an outbound attempt timed out unconnected, dial_abandoned "
            "when the attempt was hung up before it connected, dial_failed when the service "
            "reported the attempt could not be completed."
        ),
        examples=["dial_not_answered", "dial_abandoned", "dial_failed"],
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
