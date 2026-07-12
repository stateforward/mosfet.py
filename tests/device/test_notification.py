import typing

import hsm

from bot.device import (
    Notification,
    DeviceNotificationData,
    NotificationMarkUnreadEventData,
    NotificationReadEventData,
    NotificationMarkUnreadEvent,
    NotificationReadEvent,
)
from tests.hsm_model import transition_map
from tests.type_helpers import object_dict


def test_device_notification_data_schema_describes_durable_references_and_hint() -> None:
    data = DeviceNotificationData(
        device_qualified_name="device/operator-phone",
        firmware_qualified_name="phone.caller",
        service_qualified_name="phone.webrtc",
        hint="answer_call",
    )
    schema = DeviceNotificationData.model_json_schema()

    assert data.device_qualified_name == "device/operator-phone"
    assert data.firmware_qualified_name == "phone.caller"
    assert data.service_qualified_name == "phone.webrtc"
    assert data.hint == "answer_call"
    assert schema["required"] == [
        "device_qualified_name",
        "firmware_qualified_name",
        "service_qualified_name",
        "hint",
    ]

    properties = object_dict(typing.cast(object, schema["properties"]))
    device_qualified_name_schema = object_dict(properties["device_qualified_name"])
    firmware_qualified_name_schema = object_dict(properties["firmware_qualified_name"])
    service_qualified_name_schema = object_dict(properties["service_qualified_name"])
    hint_schema = object_dict(properties["hint"])
    assert device_qualified_name_schema["examples"] == ["operator-phone"]
    assert device_qualified_name_schema["description"]
    assert firmware_qualified_name_schema["examples"] == ["phone.caller"]
    assert firmware_qualified_name_schema["description"]
    assert service_qualified_name_schema["examples"] == ["phone.webrtc"]
    assert service_qualified_name_schema["description"]
    assert "message_id" not in properties
    assert hint_schema["examples"] == ["answer_call", "review_message", "open_service"]
    assert hint_schema["description"]


def test_device_notification_is_an_hsm_instance_with_data() -> None:
    data = DeviceNotificationData(
        device_qualified_name="device/operator-phone",
        firmware_qualified_name="phone.caller",
        service_qualified_name="phone.webrtc",
        hint="answer_call",
    )
    notification = Notification(data=data)

    assert isinstance(notification, hsm.Instance)
    assert not isinstance(notification, hsm.Event)
    assert notification.data == data
    assert notification.model.qualified_name == "/DeviceNotification"


def test_notification_read_and_unread_events_use_pydantic_schemas() -> None:
    read_schema = object_dict(NotificationReadEvent.schema)
    mark_unread_schema = object_dict(NotificationMarkUnreadEvent.schema)

    assert isinstance(NotificationReadEvent, hsm.Event)
    assert NotificationReadEvent.name == "device.notification.read"
    assert read_schema == NotificationReadEventData.model_json_schema()
    assert read_schema["description"]
    assert read_schema["examples"] == [{}]
    assert isinstance(NotificationMarkUnreadEvent, hsm.Event)
    assert NotificationMarkUnreadEvent.name == "device.notification.mark_unread"
    assert mark_unread_schema == NotificationMarkUnreadEventData.model_json_schema()
    assert mark_unread_schema["description"]
    assert mark_unread_schema["examples"] == [{}]


def test_device_notification_model_tracks_read_and_unread_state() -> None:
    model = Notification.model

    assert model.qualified_name == "/DeviceNotification"
    assert "/DeviceNotification/unread" in model.members
    assert "/DeviceNotification/read" in model.members
    assert model.initial == "/DeviceNotification/.initial"
    transitions = transition_map(model)
    assert "device.notification.read" in transitions["/DeviceNotification/unread"]
    assert "device.notification.read" in transitions["/DeviceNotification/read"]
    assert "device.notification.mark_unread" in transitions["/DeviceNotification/read"]
    assert getattr(model.members["/DeviceNotification/read"], "activity") == []
