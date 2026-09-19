from mosfet.devices import phone, smart_phone

import typing

import hsm
import pydantic
import pytest

from tests.type_helpers import object_dict

MESSAGING_EVENTS = (
    smart_phone.SmsTextEvent,
    smart_phone.NotificationEvent,
    smart_phone.SendTextMessageEvent,
    smart_phone.ReadNotificationEvent,
    smart_phone.DismissNotificationEvent,
    smart_phone.NotificationNotFoundEvent,
    smart_phone.ServiceTextMessageSendRequestedEvent,
    smart_phone.ServiceTextMessageSentEvent,
    smart_phone.ServiceTextMessageSendFailedEvent,
    smart_phone.TextMessageSentEvent,
    smart_phone.TextMessageSendFailedEvent,
)


def _schema_properties(schema: object) -> dict[str, object]:
    return typing.cast(dict[str, object], object_dict(schema).get("properties", {}))


def test_sms_event_schema_is_phone_local() -> None:
    """SMS is its own phone transport, not a second call schema."""

    sms = smart_phone.SmsTextData(id="message-id", sender="+15555550101", text="Book me the 10:15.")

    assert smart_phone.SmsTextEvent.name == "phone.sms.text"
    assert object_dict(smart_phone.SmsTextEvent.schema) == smart_phone.SmsTextData.model_json_schema()
    assert sms.id == "message-id"
    assert sms.sender == "+15555550101"
    assert "call_id" not in _schema_properties(smart_phone.SmsTextEvent.schema)
    assert smart_phone.SmsTextData(text="Book me the 10:15.").id is None
    assert smart_phone.SmsTextData(text="Book me the 10:15.").sender is None
    with pytest.raises(pydantic.ValidationError):
        smart_phone.SmsTextData(id="", text="Book me the 10:15.")
    with pytest.raises(pydantic.ValidationError):
        smart_phone.SmsTextData(sender="", text="Book me the 10:15.")
    with pytest.raises(pydantic.ValidationError):
        smart_phone.SmsTextData(text="")


def test_messaging_events_are_phone_domain_hsm_events() -> None:
    """A text arrives at a phone number: messaging facts keep the phone domain on a smart phone."""

    for event in MESSAGING_EVENTS:
        assert isinstance(event, hsm.Event)
        assert event.name.startswith("phone."), event.name
        assert object_dict(event.schema) == typing.cast(type[pydantic.BaseModel], event.schema).model_json_schema()


def test_messaging_moved_off_the_basic_phone() -> None:
    """A basic phone exports calls only; messaging contracts live on the smart phone."""

    for name in (
        "SmsTextEvent",
        "SmsTextData",
        "NotificationData",
        "NotificationEvent",
        "SendTextMessageEvent",
        "ReadNotificationEvent",
        "DismissNotificationEvent",
        "NotificationsEvent",
        "NOTIFICATION_WAV",
        "NOTIFICATION_DB",
    ):
        assert not hasattr(phone, name), name
        assert hasattr(smart_phone, name), name


def test_a_smart_phone_sound_is_a_phone_sound_that_can_carry_a_notification() -> None:
    assert issubclass(smart_phone.SoundData, phone.SoundData)
    assert "notification" in smart_phone.SoundData.model_fields
    assert "notification" not in phone.SoundData.model_fields


def test_a_notification_names_what_caused_it() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = smart_phone.NotificationData.model_validate(
            {"id": "x", "name": "phone.ringing", "data": {"text": "Book me the 10:15."}}
        )
