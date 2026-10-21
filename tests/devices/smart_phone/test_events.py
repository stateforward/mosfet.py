from mosfet.devices import phone, smart_phone

import datetime
import typing

import hsm
import pydantic
import pytest

from mosfet.abilities import processing

from tests.type_helpers import object_dict

MESSAGING_EVENTS = (
    smart_phone.SmsTextEvent,
    smart_phone.NotificationEvent,
    smart_phone.SendTextMessageEvent,
    smart_phone.ReadNotificationEvent,
    smart_phone.DismissNotificationEvent,
    smart_phone.NotificationNotFoundEvent,
    smart_phone.ReadThreadEvent,
    smart_phone.ThreadEvent,
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


def test_reading_a_thread_is_a_model_callable_command_named_by_correspondent() -> None:
    """The command is offered as a tool; ``with`` is its wire name, the way a model writes it."""

    schema = smart_phone.ReadThreadData.model_json_schema()
    properties = _schema_properties(schema)
    read = smart_phone.ReadThreadData.model_validate({"with": "+15555550101"})

    assert smart_phone.ReadThreadEvent.name == "phone.read_thread"
    assert smart_phone.ReadThreadEvent.kind == processing.EventKind
    assert smart_phone.ThreadEvent.name == "phone.thread"
    assert smart_phone.ThreadEvent.kind != processing.EventKind
    assert schema["required"] == ["with"]
    assert set(properties) == {"with", "before", "limit"}
    for name in ("with", "before", "limit"):
        assert object_dict(properties[name]).get("description"), name
    # No example correspondent, as for a recipient: a concrete address would act as a default.
    assert "examples" not in object_dict(properties["with"])
    assert (read.with_, read.before, read.limit) == ("+15555550101", None, 20)
    assert (
        smart_phone.ReadThreadData.model_validate({"with": "+15555550101", "before": "message-3", "limit": 50}).limit
        == 50
    )
    for invalid in (
        {"with": ""},
        {"with": "+15555550101", "limit": 0},
        {"with": "+15555550101", "limit": 51},
        {"with": "+15555550101", "before": ""},
        {"with": "+15555550101", "unread": True},
        # One spelling only: the wire name a model writes.
        {"with_": "+15555550101"},
    ):
        with pytest.raises(pydantic.ValidationError):
            _ = smart_phone.ReadThreadData.model_validate(invalid)


def test_a_thread_page_carries_its_messages_oldest_first() -> None:
    page = smart_phone.ThreadData.model_validate(
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
    )

    assert [message.direction for message in page.messages] == ["received", "sent"]
    assert page.messages[0].at == datetime.datetime(2026, 10, 20, 9, tzinfo=datetime.UTC)
    with pytest.raises(pydantic.ValidationError):
        _ = smart_phone.ThreadMessageData.model_validate(
            {"id": "message-1", "at": "2026-10-20T09:00:00Z", "direction": "draft", "text": "Hi."}
        )
