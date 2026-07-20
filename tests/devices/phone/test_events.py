from bot.devices import phone

import hsm

from tests.type_helpers import object_dict

PHONE_EVENTS = (
    phone.IncomingCallEvent,
    phone.CallConnectedEvent,
    phone.ServiceMediaReadyEvent,
    phone.ServiceAudioReceivedEvent,
    phone.RemoteHangUpEvent,
    phone.CallFailedEvent,
    phone.TransferAcceptedEvent,
    phone.ServiceTransferCompletedEvent,
    phone.ServiceTransferFailedEvent,
    phone.DialEvent,
    phone.AnswerCallEvent,
    phone.DeclineCallEvent,
    phone.HangUpCallEvent,
    phone.TransferCallEvent,
    phone.ServiceAnswerRequestedEvent,
    phone.ServiceDialRequestedEvent,
    phone.ServiceDeclineRequestedEvent,
    phone.ServiceHangUpRequestedEvent,
    phone.ServiceTransferRequestedEvent,
    phone.RingingEvent,
    phone.AnsweredEvent,
    phone.MediaReadyEvent,
    phone.HungUpEvent,
    phone.TransferStartedEvent,
    phone.CallTransferCompletedEvent,
    phone.CallTransferFailedEvent,
)

def test_phone_service_events_use_transport_neutral_pydantic_schemas() -> None:
    assert phone.IncomingCallEvent.name == "phone.service.incoming_call"
    assert object_dict(phone.IncomingCallEvent.schema) == phone.IncomingCallData.model_json_schema()
    assert object_dict(phone.IncomingCallEvent.schema)["required"] == ["call_id"]
    assert phone.CallConnectedEvent.name == "phone.service.call_connected"
    assert object_dict(phone.CallConnectedEvent.schema) == phone.CallConnectedData.model_json_schema()
    assert phone.ServiceMediaReadyEvent.name == "phone.service.media_ready"
    assert object_dict(phone.ServiceMediaReadyEvent.schema) == phone.MediaReadyData.model_json_schema()
    assert phone.ServiceAudioReceivedEvent.name == "phone.service.audio_received"
    assert object_dict(phone.ServiceAudioReceivedEvent.schema) == phone.ServiceAudioData.model_json_schema()
    assert object_dict(phone.ServiceAudioReceivedEvent.schema)["required"] == ["audio", "call_id"]
    assert phone.RemoteHangUpEvent.name == "phone.service.remote_hang_up"
    assert object_dict(phone.RemoteHangUpEvent.schema) == phone.RemoteHangUpData.model_json_schema()
    assert phone.CallFailedEvent.name == "phone.service.call_failed"
    assert object_dict(phone.CallFailedEvent.schema) == phone.CallFailedData.model_json_schema()

def test_phone_transfer_events_use_call_identity_and_target_schemas() -> None:
    transfer_target = phone.TransferTarget(kind="address", value="helpdesk@example.com")
    transfer_command = phone.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=transfer_target)
    transfer_accepted = phone.TransferAcceptedData(call_id="call-123", transfer_id="transfer-123", target=transfer_target)
    transfer_completed = phone.TransferCompletedData(call_id="call-123", transfer_id="transfer-123", target=transfer_target)
    transfer_failed = phone.TransferFailedData(
        call_id="call-123",
        transfer_id="transfer-123",
        target=transfer_target,
        failure_kind="transfer_rejected",
    )

    assert transfer_command.target == transfer_target
    assert transfer_command.transfer_id == "transfer-123"
    assert transfer_accepted.target == transfer_target
    assert transfer_completed.target == transfer_target
    assert transfer_failed.failure_kind == "transfer_rejected"
    assert phone.TransferAcceptedEvent.name == "phone.service.transfer_accepted"
    assert object_dict(phone.TransferAcceptedEvent.schema) == phone.TransferAcceptedData.model_json_schema()
    assert phone.ServiceTransferCompletedEvent.name == "phone.service.transfer_completed"
    assert object_dict(phone.ServiceTransferCompletedEvent.schema) == phone.TransferCompletedData.model_json_schema()
    assert phone.ServiceTransferFailedEvent.name == "phone.service.transfer_failed"
    assert object_dict(phone.ServiceTransferFailedEvent.schema) == phone.TransferFailedData.model_json_schema()
    assert object_dict(phone.TransferCallEvent.schema)["required"] == ["call_id", "transfer_id", "target"]
    assert object_dict(phone.TransferAcceptedEvent.schema)["required"] == ["call_id", "transfer_id", "target"]
    assert object_dict(phone.ServiceTransferCompletedEvent.schema)["required"] == ["call_id", "transfer_id", "target"]
    assert object_dict(phone.ServiceTransferFailedEvent.schema)["required"] == ["call_id", "transfer_id", "failure_kind", "target"]

def test_phone_service_audio_data_describes_speaker_output_for_current_call() -> None:
    data = phone.ServiceAudioData(audio=b"playback-audio", call_id="call-123", media_type="audio/pcm", channels=1)

    assert data.call_id == "call-123"
    assert data.audio == b"playback-audio"
    assert data.media_type == "audio/pcm"
    assert data.channels == 1

def test_phone_command_events_are_separate_from_provider_request_contracts() -> None:
    target = phone.TransferTarget(kind="address", value="sip:helpdesk@example.com")
    dial_data = phone.DialData(call_id="call-123", target=target)
    answer_data = phone.AnswerCallData(call_id="call-123")
    decline_data = phone.DeclineCallData(call_id="call-123")
    hang_up_data = phone.HangUpCallData(call_id="call-123")

    assert dial_data.call_id == "call-123"
    assert dial_data.target == target
    assert answer_data.call_id == "call-123"
    assert decline_data.call_id == "call-123"
    assert hang_up_data.call_id == "call-123"
    assert phone.DialEvent.name == "phone.dial"
    assert object_dict(phone.DialEvent.schema) == phone.DialData.model_json_schema()
    assert object_dict(phone.DialEvent.schema)["required"] == ["call_id", "target"]
    assert phone.AnswerCallEvent.name == "phone.answer_call"
    assert object_dict(phone.AnswerCallEvent.schema) == phone.AnswerCallData.model_json_schema()
    assert phone.DeclineCallEvent.name == "phone.decline_call"
    assert object_dict(phone.DeclineCallEvent.schema) == phone.DeclineCallData.model_json_schema()
    assert phone.HangUpCallEvent.name == "phone.hang_up_call"
    assert object_dict(phone.HangUpCallEvent.schema) == phone.HangUpCallData.model_json_schema()
    assert phone.TransferCallEvent.name == "phone.transfer_call"
    assert object_dict(phone.TransferCallEvent.schema) == phone.TransferCallData.model_json_schema()

    assert phone.ServiceAnswerRequestedEvent.name == "phone.service.answer_requested"
    assert phone.ServiceDialRequestedEvent.name == "phone.service.dial_requested"
    assert object_dict(phone.ServiceDialRequestedEvent.schema) == phone.DialData.model_json_schema()
    assert phone.ServiceDeclineRequestedEvent.name == "phone.service.decline_requested"
    assert phone.ServiceHangUpRequestedEvent.name == "phone.service.hang_up_requested"
    assert phone.ServiceTransferRequestedEvent.name == "phone.service.transfer_requested"

def test_phone_public_events_describe_committed_firmware_state() -> None:
    ringing = phone.RingingData(call_id="call-123")
    hung_up = phone.PhoneHungUpData(call_id="call-123", outcome="remote_hang_up")
    transfer = phone.PhoneTransferData(
        call_id="call-123",
        transfer_id="transfer-123",
        target=phone.TransferTarget(kind="address", value="helpdesk@example.com"),
    )
    transfer_failed = phone.PhoneTransferFailedData(
        call_id="call-123",
        transfer_id="transfer-123",
        target=phone.TransferTarget(kind="address", value="helpdesk@example.com"),
        failure_kind="timeout",
    )

    assert ringing.call_id == "call-123"
    assert isinstance(ringing, phone.PhoneCallData)
    assert hung_up.outcome == "remote_hang_up"
    assert transfer.target.kind == "address"
    assert transfer_failed.failure_kind == "timeout"
    assert phone.RingingEvent.name == "phone.ringing"
    assert object_dict(phone.RingingEvent.schema) == phone.RingingData.model_json_schema()
    assert phone.AnsweredEvent.name == "phone.answered"
    assert object_dict(phone.AnsweredEvent.schema) == phone.PhoneCallData.model_json_schema()
    assert phone.MediaReadyEvent.name == "phone.media_ready"
    assert object_dict(phone.MediaReadyEvent.schema) == phone.PhoneCallData.model_json_schema()
    assert phone.HungUpEvent.name == "phone.hung_up"
    assert object_dict(phone.HungUpEvent.schema) == phone.PhoneHungUpData.model_json_schema()
    assert phone.TransferStartedEvent.name == "phone.transfer_started"
    assert object_dict(phone.TransferStartedEvent.schema) == phone.PhoneTransferData.model_json_schema()
    assert phone.CallTransferCompletedEvent.name == "phone.transfer_completed"
    assert object_dict(phone.CallTransferCompletedEvent.schema) == phone.PhoneTransferData.model_json_schema()
    assert phone.CallTransferFailedEvent.name == "phone.transfer_failed"
    assert object_dict(phone.CallTransferFailedEvent.schema) == phone.PhoneTransferFailedData.model_json_schema()

def test_phone_event_names_do_not_use_observed_suffix() -> None:
    event_names = [event.name for event in PHONE_EVENTS]

    assert all("observed" not in name for name in event_names)

def test_phone_events_are_hsm_events() -> None:
    for event in PHONE_EVENTS:
        assert isinstance(event, hsm.Event)
