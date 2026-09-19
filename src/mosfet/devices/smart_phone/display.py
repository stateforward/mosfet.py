from . import events

import typing

import hsm
import pydantic

from mosfet.devices import phone
from xml.sax import saxutils


class NotificationsData(pydantic.BaseModel):
    """Drive signal for a phone's lock screen: every notification still pending, oldest first.

    Firmware owns the list — it adds a notification when one arrives and takes it off when the
    holder reads or dismisses it — and puts the whole list on this wire each time it changes, the
    way it puts a caller ID there. The display only shows it. An empty list clears the screen of
    notifications.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "pending": [
                        {
                            "id": "message-id",
                            "name": "phone.sms.text",
                            "data": {"id": "message-id", "sender": "+15555550101", "text": "Book me the 10:15."},
                        }
                    ]
                },
                {"pending": []},
            ],
        },
    )

    pending: tuple[events.NotificationData, ...] = pydantic.Field(
        default=(),
        description=(
            "Notifications on the screen that nobody has read or dismissed yet, oldest first, each "
            "carrying the event that caused it."
        ),
    )

    def __model_repr__(self) -> str:
        """Lock screen as the holder sees it: one element per pending notification, ids first."""

        rendered: list[str] = []
        for notification in self.pending:
            message = notification.data
            sender = "" if message.sender is None else f"<sender>{saxutils.escape(message.sender)}</sender>"
            attributes = f"id={saxutils.quoteattr(notification.id)} name={saxutils.quoteattr(notification.name)}"
            text = f"<text>{saxutils.escape(message.text)}</text>"
            rendered.append(f"<notification {attributes}>{sender}{text}</notification>")
        return "".join(rendered)


NotificationsEvent = hsm.Event[NotificationsData](
    name="phone.display.notifications",
    schema=NotificationsData,
)


class Display(phone.Display):
    """A smart phone's screen: the phone's caller ID, plus the lock screen of pending notifications.

    Extends :class:`mosfet.devices.phone.Display` rather than redefining it, so caller ID behaves
    exactly as on a basic phone. It stays a passive transducer: firmware owns the pending list and
    puts it on the wire each time it changes; the screen only shows it.
    """

    @staticmethod
    def _show_notifications(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> None:
        """Show the pending notifications firmware put on the wire; none pending shows nothing."""

        del ctx
        data = event.data
        assert isinstance(data, NotificationsData)
        _ = instance.set("notifications", data if data.pending else None)

    model: typing.ClassVar[hsm.Model | None] = hsm.redefine(
        typing.cast(hsm.Model, phone.Display.model),
        hsm.attribute("notifications"),
        hsm.transition(hsm.on(NotificationsEvent), hsm.effect(_show_notifications)),
    )
