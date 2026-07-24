from bot.devices import audio
from bot.world import SoundData

import typing

import hsm
import pydantic

CallId = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description=(
            "Provider-neutral identifier for one phone call. Firmware uses this value to reject stale service "
            "callbacks and commands for a previous call. "
            "When answering or declining a ring elevated as world.sound, copy this from the stimulus "
            "event.data.call_id (also mirrored on event.id for correlation) — never a documentation "
            "example such as call-123."
        ),
        examples=["livekit:caller", "sip:session-9f3a"],
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
    "transfer_rejected",
    "timeout",
    "unknown",
]
HangUpOutcome = typing.Literal["local_hang_up", "declined", "remote_hang_up", "failed", "transferred"]

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
            "examples": [{"call_id": "call-123", "display_hint": "Front desk"}],
        },
    )

    display_hint: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Optional provider-neutral text that may help an operator decide whether to answer. Firmware must not "
            "treat this as stable caller identity."
        ),
        examples=["Front desk", "Support queue"],
    )

class DialData(CallIdData):
    """Command from an operator or owner asking firmware to dial an outbound call."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "call_id": "call-123",
                    "target": {"kind": "address", "value": "sip:helpdesk@example.com"},
                }
            ],
        },
    )

    target: TransferTarget = pydantic.Field(
        description=(
            "Provider-neutral outbound destination. The provider owns address resolution and transport call setup; "
            "phone firmware owns the call lifecycle."
        ),
        examples=[{"kind": "address", "value": "sip:helpdesk@example.com"}],
    )

class AnswerCallData(CallIdData):
    """Command from an operator or owner asking firmware to answer the current ringing call."""

class DeclineCallData(CallIdData):
    """Command from an operator or owner asking firmware to decline the current ringing call."""

class HangUpCallData(CallIdData):
    """Command from an operator or owner asking firmware to hang up the current active call."""

class TransferCallData(CallIdData):
    """Command from an operator or owner asking firmware to transfer the current active call."""

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
        description=(
            "Provider-neutral transfer destination. The provider owns the transport mechanics; phone firmware owns "
            "the user-visible transfer state."
        ),
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


class RingingData(PhoneCallData):
    """Committed public phone state that a call is ringing.

    Distinct from :class:`PhoneCallData` used by answered/media-ready so world elevation
    can select ring acoustics by payload type without ``event.name`` discrimination.
    """


class PhoneSoundData(SoundData):
    """``world.sound`` payload elevated from phone ringing (or call-scoped acoustic energy).

    Subclasses :class:`~bot.world.SoundData` with a typed ``call_id`` so models copy
    ``event.data.call_id`` into answer/decline tools instead of inferring from ``event.id``.
    Elevation also mirrors ``call_id`` onto ``event.id`` for HSM correlation.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "Phone-elevated acoustic stimulus for world.sound. Includes call_id so tool "
                "arguments can copy event.data.call_id directly."
            ),
            "examples": [
                {
                    "audio": "YXVkaW8tY2h1bms=",
                    "media_type": "audio/wav",
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "kind": "phone.ringing",
                    "call_id": "livekit:caller",
                }
            ],
        },
    )

    call_id: CallId = pydantic.Field(
        description=(
            "Live provider-neutral call identifier for this elevated phone sound. When kind is "
            "phone.ringing or phone.call, answer/decline tools must use this value as call_id."
        ),
        examples=["livekit:caller", "sip:session-9f3a"],
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
    kind=hsm.CallEventKind,
    schema=AnswerCallData,
)
DialEvent = hsm.Event[DialData](
    name="phone.dial",
    kind=hsm.CallEventKind,
    schema=DialData,
)
DeclineCallEvent = hsm.Event[DeclineCallData](
    name="phone.decline_call",
    kind=hsm.CallEventKind,
    schema=DeclineCallData,
)
HangUpCallEvent = hsm.Event[HangUpCallData](
    name="phone.hang_up_call",
    kind=hsm.CallEventKind,
    schema=HangUpCallData,
)
TransferCallEvent = hsm.Event[TransferCallData](
    name="phone.transfer_call",
    kind=hsm.CallEventKind,
    schema=TransferCallData,
)

ServiceAnswerRequestedEvent = hsm.Event[AnswerCallData](
    name="phone.service.answer_requested",
    schema=AnswerCallData,
)
ServiceDialRequestedEvent = hsm.Event[DialData](
    name="phone.service.dial_requested",
    schema=DialData,
)
ServiceDeclineRequestedEvent = hsm.Event[DeclineCallData](
    name="phone.service.decline_requested",
    schema=DeclineCallData,
)
ServiceHangUpRequestedEvent = hsm.Event[HangUpCallData](
    name="phone.service.hang_up_requested",
    schema=HangUpCallData,
)
ServiceTransferRequestedEvent = hsm.Event[TransferCallData](
    name="phone.service.transfer_requested",
    schema=TransferCallData,
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
