from typing import ClassVar

import hsm
from pydantic import BaseModel, ConfigDict, Field

from bot.telemetry import observer


class DeviceNotificationData(BaseModel):
    """Durable reference data for a device notification."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "device_qualified_name": "device/operator-phone",
                    "service_qualified_name": "phone",
                    "hint": "answer_call",
                }
            ],
        },
    )

    device_qualified_name: str = Field(
        description="Human-readable name of the device that dispatched the notification.",
        examples=["operator-phone"],
    )
    firmware_qualified_name: str = Field(
        description="Human-readable name of the firmware that produced the notification inside the device sandbox.",
        examples=["phone.caller"],
    )
    service_qualified_name: str = Field(
        description="Human-readable name of the service that produced the notification inside the device sandbox.",
        examples=["phone.webrtc"],
    )
    hint: str = Field(
        description=(
            "Decision-making hint supplied by the service or device that created the notification. Use this to "
            "suggest the likely next device action without encoding read/unread state or notification content."
        ),
        examples=["answer_call", "review_message", "open_service"],
    )


class NotificationReadEventData(BaseModel):
    """No-payload command that marks a device notification as read."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


class NotificationMarkUnreadEventData(BaseModel):
    """No-payload command that marks a device notification as unread."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{}],
        },
    )


NotificationReadEvent = hsm.Event(
    name="device.notification.read",
    schema=NotificationReadEventData,
)
NotificationMarkUnreadEvent = hsm.Event(
    name="device.notification.mark_unread",
    schema=NotificationMarkUnreadEventData,
)


class Notification(hsm.Instance):
    """Stateful notification instance owned by a device."""

    def __init__(self, data: DeviceNotificationData | None = None) -> None:
        super().__init__()
        self.data: DeviceNotificationData | None = data

    model: ClassVar[hsm.Model] = hsm.define(
        "DeviceNotification",
        hsm.initial(hsm.target("unread")),
        hsm.state(
            "unread",
            hsm.transition(
                hsm.on(NotificationReadEvent),
                hsm.target("../read"),
            ),
        ),
        hsm.state(
            "read",
            hsm.transition(
                hsm.on(NotificationReadEvent),
                hsm.target("../read"),
            ),
            hsm.transition(
                hsm.on(NotificationMarkUnreadEvent),
                hsm.target("../unread"),
            ),
        ),
        hsm.observe(observer),
    )
