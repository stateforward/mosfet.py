"""Tests for the smart phone Display peripheral (mosfet.devices.smart_phone.display)."""

import asyncio
import typing

import hsm
import mosfet

from mosfet.devices import phone, smart_phone
from tests.devices.phone.handset import wait_until

_NOTIFICATION = smart_phone.NotificationData(
    id="message-id",
    name="phone.sms.text",
    data=smart_phone.SmsTextData(id="message-id", sender="+15555550101", text="Book me the 10:15."),
)


def _attributes(display: phone.Display) -> dict[str, object]:
    return dict(display.take_snapshot().Attributes or {})


def test_a_smart_phone_display_is_a_phone_display() -> None:
    assert issubclass(smart_phone.Display, phone.Display)
    assert smart_phone.Display.observation_name == "display"


def test_a_smart_phone_display_shows_caller_id_and_the_lock_screen() -> None:
    """Caller ID works as on a phone; the pending list shows until firmware sends an empty one."""

    async def run() -> tuple[object, object, object]:
        display = smart_phone.Display()
        _ = await mosfet.started(None, display, typing.cast(hsm.Model, display.model))
        await wait_until(lambda: display.state() == "/Device/detached")
        await display.dispatch(
            display.context(), phone.CallerIdEvent.with_data(phone.CallerIdData(caller_id="Front desk"))
        )
        await display.dispatch(
            display.context(),
            smart_phone.NotificationsEvent.with_data(smart_phone.NotificationsData(pending=(_NOTIFICATION,))),
        )
        await asyncio.sleep(0)
        shown = _attributes(display)
        await display.dispatch(
            display.context(), smart_phone.NotificationsEvent.with_data(smart_phone.NotificationsData(pending=()))
        )
        await asyncio.sleep(0)
        cleared = _attributes(display)
        return shown.get("/Device/caller_id"), shown.get("/Device/notifications"), cleared.get("/Device/notifications")

    caller_id, notifications, cleared = asyncio.run(run())

    assert caller_id == "Front desk"
    assert notifications == smart_phone.NotificationsData(pending=(_NOTIFICATION,))
    assert cleared is None


def test_a_phone_display_has_no_lock_screen() -> None:
    """A basic phone shows who is calling and nothing else."""

    model = typing.cast(hsm.Model, phone.Display.model)

    assert smart_phone.NotificationsEvent.name not in model.events
    assert "/Device/notifications" not in model.members
    assert "/Device/notifications" in typing.cast(hsm.Model, smart_phone.Display.model).members


def test_notifications_event_name_carries_the_phone_domain() -> None:
    assert smart_phone.NotificationsEvent.name == "phone.display.notifications"


def test_the_lock_screen_renders_each_pending_notification_for_the_holder() -> None:
    rendered = smart_phone.NotificationsData(pending=(_NOTIFICATION,)).__model_repr__()

    assert rendered == (
        '<notification id="message-id" name="phone.sms.text">'
        "<sender>+15555550101</sender><text>Book me the 10:15.</text></notification>"
    )
