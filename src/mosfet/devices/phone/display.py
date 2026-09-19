from . import events

import typing

import hsm
import pydantic

from mosfet.device import Device
from xml.sax import saxutils


class CallerIdData(pydantic.BaseModel):
    """Caller ID drive signal for a phone's display: who to show, or null to clear the screen.

    Firmware is the controller that decides what a call's caller ID means — the caller reported
    on a ring, the party the provider stamped on a connect. The display makes no such decision;
    it shows exactly what firmware puts on this wire, the way an unwired display shows nothing.

    ``caller_id=None`` clears the screen. That is a real, distinct instruction, not the absence
    of one: a handset's display goes blank once a call ends, and firmware sends this to say so
    rather than leaving whatever was last shown lingering on the glass.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [{"caller_id": "Front desk"}, {}],
        },
    )

    caller_id: events.Caller | None = pydantic.Field(
        default=None,
        description=(
            "Who the display should show as the party on the line, exactly as a handset's screen presents "
            "caller ID while ringing or connected. Null clears the screen: nobody is shown, whether because "
            "the call ended or because the caller withheld identification."
        ),
        examples=["Front desk", "+15555550123"],
    )


CallerIdEvent = hsm.Event[CallerIdData](
    name="phone.display.caller_id",
    schema=CallerIdData,
)


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


class Display(Device):
    """Passive output peripheral that shows caller ID and pending notifications on a phone handset's screen.

    Mirrors :class:`~mosfet.devices.audio.microphone.Microphone` and
    :class:`~mosfet.devices.audio.speaker.Speaker`: no firmware of its own, because a handset screen
    is a transducer for light, not a computer. It shows exactly what its controller puts on the
    wire and nothing else — it never decides who is calling or what a caller ID means; that
    judgment belongs to whatever attached to it (phone firmware, for a handset).

    Unlike the audio transducers it has no environment-facing side: a screen neither hears
    ``environment.sound`` nor puts anything into it, so there is no acoustic counterpart here.
    It is still attached at bring-up the same way the audio peripherals are — firmware is the
    controller that wires every peripheral it drives, and a display wired in uniformly with the
    others is what keeps bring-up from special-casing one transducer over another. But unlike the
    mouthpiece, which transduces only while attached, showing a caller ID is not gated on that
    attachment: firmware clears the display on its own first entry into ``hung_up``, before
    bring-up has finished wiring anything, and a display that only listened while attached would
    silently drop that first clear. No behavior gates on the attachment today; it exists for the
    same ownership reason the other transducers are attached, not because caller ID needs it.
    """

    observation_name: typing.ClassVar[str | None] = "display"
    """Owner-visible key: an owning device folds this display's ``caller_id`` under ``display``."""

    @staticmethod
    def _show(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> None:
        """Show whatever the controller put on the wire, including a screen-clearing null."""

        del ctx
        data = event.data
        assert isinstance(data, CallerIdData)
        _ = instance.set("caller_id", data.caller_id)

    @staticmethod
    def _show_notifications(ctx: hsm.Context, instance: "Display", event: hsm.Event[typing.Any]) -> None:
        """Show the pending notifications firmware put on the wire; none pending shows nothing."""

        del ctx
        data = event.data
        assert isinstance(data, NotificationsData)
        _ = instance.set("notifications", data if data.pending else None)

    model: typing.ClassVar[hsm.Model | None] = hsm.redefine(
        typing.cast(hsm.Model, Device.model),
        hsm.attribute("caller_id"),
        hsm.transition(hsm.on(CallerIdEvent), hsm.effect(_show)),
        hsm.attribute("notifications"),
        hsm.transition(hsm.on(NotificationsEvent), hsm.effect(_show_notifications)),
    )
