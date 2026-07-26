from bot.devices import audio
from bot.environment import SoundData

import typing

import hsm
from bot import event_schema
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
    """Command from an operator or owner asking firmware to dial an outbound call.

    Carries no call id. You dial a destination and the exchange assigns the call, so the session
    handle arrives from the provider on connect and there is nothing for a caller to supply.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"target": {"kind": "address", "value": "sip:helpdesk@example.com"}}],
        },
    )

    target: TransferTarget = pydantic.Field(
        description=(
            "Provider-neutral outbound destination. The provider owns address resolution and transport call setup; "
            "phone firmware owns the call lifecycle."
        ),
        examples=[{"kind": "address", "value": "sip:helpdesk@example.com"}],
    )

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

class MediaReadyData(CallIdData):
    """Signal from a phone service provider that call media is ready for audio routing."""

class ServiceAudioData(audio.AudioOutputData):
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

class PhoneCallData(CallIdData):
    """Committed public phone state for a specific call."""


class RingingData(pydantic.BaseModel):
    """Committed public phone state that a call is ringing.

    Carries who is calling, not which session is ringing: a ringing handset shows caller ID, and
    the session handle has no counterpart on it. Distinct from :class:`PhoneCallData` so
    environment elevation can select ring acoustics by payload type, never by ``event.name``.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"caller": "Front desk"}, {}],
        },
    )

    caller: Caller | None = None


class PhoneSoundData(SoundData):
    """``environment.sound`` payload elevated from phone ringing (or call-scoped acoustic energy).

    Subclasses :class:`~bot.environment.SoundData` with the caller ID, so what a bot perceives
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


class PhoneHungUpData(CallIdData):
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

class PhoneTransferData(CallIdData):
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

class PhoneTransferFailedData(PhoneTransferData):
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
    kind=event_schema.EventKind,
    schema=AnswerCallData,
)
DialEvent = hsm.Event[DialData](
    name="phone.dial",
    kind=event_schema.EventKind,
    schema=DialData,
)
DeclineCallEvent = hsm.Event[DeclineCallData](
    name="phone.decline_call",
    kind=event_schema.EventKind,
    schema=DeclineCallData,
)
HangUpCallEvent = hsm.Event[HangUpCallData](
    name="phone.hang_up_call",
    kind=event_schema.EventKind,
    schema=HangUpCallData,
)
TransferCallEvent = hsm.Event[TransferCallData](
    name="phone.transfer_call",
    kind=event_schema.EventKind,
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
AnsweredEvent = hsm.Event[PhoneCallData](
    name="phone.answered",
    schema=PhoneCallData,
)
MediaReadyEvent = hsm.Event[PhoneCallData](
    name="phone.media_ready",
    schema=PhoneCallData,
)
HungUpEvent = hsm.Event[PhoneHungUpData](
    name="phone.hung_up",
    schema=PhoneHungUpData,
)
NoCallEvent = hsm.Event[NoCallData](
    name="phone.no_call",
    schema=NoCallData,
)
TransferStartedEvent = hsm.Event[PhoneTransferData](
    name="phone.transfer_started",
    schema=PhoneTransferData,
)
CallTransferCompletedEvent = hsm.Event[PhoneTransferData](
    name="phone.transfer_completed",
    schema=PhoneTransferData,
)
CallTransferFailedEvent = hsm.Event[PhoneTransferFailedData](
    name="phone.transfer_failed",
    schema=PhoneTransferFailedData,
)
