from mosfet.devices import phone as phone_device

import asyncio
import dataclasses
import datetime
import typing
import xml.etree.ElementTree

import hsm
import pydantic
import pytest

import mosfet
from mosfet.abilities import processing
from mosfet.devices import smart_phone
from mosfet.environment import Environment
from mosfet.protocols import attachment
from tests.devices.phone.handset import (
    PhoneHolder,
    PhoneObservationRecorder,
    RoomOccupant,
    answer_call,
    emit_service_event,
    event_names,
    firmware_of,
    occasion_sources,
    phone_in_a_hand,
    wait_until,
)
from tests.hsm_instance_state import device_bots, phone_display
from tests.hsm_model import transition_map


async def _smart_phone_in_a_hand(
    environment: Environment,
) -> tuple[phone_device.Phone, phone_device.Firmware, PhoneHolder, RoomOccupant]:
    return await phone_in_a_hand(environment, smart_phone.SmartPhone(answer_timeout=datetime.timedelta(milliseconds=1)))


def test_smart_phone_sms_text_routes_to_display() -> None:
    """A phone message lands on the display as a pending notification, not a conversation surface."""

    async def run() -> tuple[smart_phone.NotificationsData | None, str | None]:
        phone = smart_phone.SmartPhone()
        _ = await mosfet.started(None, phone, typing.cast(hsm.Model, phone.model))
        await wait_until(lambda: phone.state() == "/Device/detached")

        await phone.dispatch(
            phone.context(),
            smart_phone.SmsTextEvent.with_data(
                smart_phone.SmsTextData(id="message-id", sender="+15555550101", text="Book me the 10:15.")
            ),
        )
        display = phone_display(phone)
        await wait_until(lambda: _pending_notification_ids(phone) == ["message-id"])
        attributes = display.take_snapshot().Attributes or {}
        return (
            typing.cast(smart_phone.NotificationsData | None, attributes.get("/DeviceDisplay/notifications")),
            typing.cast(str | None, attributes.get("/DeviceDisplay/caller_id")),
        )

    notifications, caller_id = asyncio.run(run())
    assert notifications is not None
    assert [notification.id for notification in notifications.pending] == ["message-id"]
    assert notifications.pending[0].data == smart_phone.SmsTextData(
        id="message-id", sender="+15555550101", text="Book me the 10:15."
    )
    assert caller_id is None


def test_smart_phone_broadcasts_a_notification_ding_carrying_the_new_message_event() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], str, object]:
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}
        environment = Environment()
        phone = smart_phone.SmartPhone()
        inside = PhoneObservationRecorder()
        outside = PhoneObservationRecorder()

        _ = await mosfet.started(environment, phone, typing.cast(hsm.Model, phone.model))
        _ = await mosfet.started(environment, inside, inside.model, hsm.Config(id="inside"))
        environment.join(inside)
        _ = await mosfet.started(None, outside, outside.model, hsm.Config(id="outside"))

        await phone.dispatch(
            phone.context(),
            dataclasses.replace(
                smart_phone.SmsTextEvent.with_data(
                    smart_phone.SmsTextData(id="message-id", sender="+15555550101", text="Book me the 10:15.")
                ),
                metadata=metadata,
            ),
        )
        display = phone_display(phone)
        await wait_until(
            lambda: bool(inside.events)
            and bool((display.take_snapshot().Attributes or {}).get("/DeviceDisplay/notifications"))
        )

        attributes = display.take_snapshot().Attributes or {}
        return inside.events, outside.events, hsm.id(phone), attributes.get("/DeviceDisplay/notifications")

    inside_events, outside_events, phone_id, notifications = asyncio.run(run())

    assert len(inside_events) == 1
    assert inside_events[0].name == "environment.sound"
    assert inside_events[0].source == phone_id
    assert inside_events[0].target == "inside"
    sound = inside_events[0].data
    assert isinstance(sound, smart_phone.SoundData)
    assert sound.kind == "phone.notification"
    assert sound.media_type == "audio/wav"
    assert sound.sample_rate_hz == 16_000
    assert sound.channels == 1
    assert sound.audio.startswith(b"RIFF")
    assert sound.audio == smart_phone.NOTIFICATION_WAV
    assert sound.amplitude_db == smart_phone.NOTIFICATION_DB
    assert sound.caller is None
    assert sound.notification is not None
    assert sound.notification.id == "message-id"
    assert sound.notification.name == smart_phone.SmsTextEvent.name == "phone.sms.text"
    assert sound.notification.data == smart_phone.SmsTextData(
        id="message-id", sender="+15555550101", text="Book me the 10:15."
    )
    assert inside_events[0].metadata.get("traceparent") == "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"
    assert outside_events == []
    # The screen shows the same notification, pending until it is read or dismissed.
    assert isinstance(notifications, smart_phone.NotificationsData)
    assert notifications.pending == (sound.notification,)


def test_a_phone_hands_whoever_holds_it_the_new_message_notification() -> None:
    """A text dings the room and puts the notification, carrying the new message, in the holder's hand."""

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], str]:
        environment = Environment()
        phone, _firmware, holder, bystander = await _smart_phone_in_a_hand(environment)
        await phone.dispatch(
            phone.context(),
            smart_phone.SmsTextEvent.with_data(
                smart_phone.SmsTextData(id="message-id", sender="+15555550101", text="Book me the 10:15.")
            ),
        )
        await wait_until(lambda: bool(holder.occasions))
        return holder.occasions, bystander.received, hsm.id(phone)

    occasions, received, phone_id = asyncio.run(run())

    assert len(occasions) == 1
    occasion = occasions[0]
    assert occasion.source == phone_id
    assert occasion.target == "holder"
    assert isinstance(occasion.data, mosfet.InputEventData)
    observation = occasion.data.observation
    assert observation is not None
    assert observation.event == smart_phone.NotificationEvent.name == "phone.notification"
    notification = observation.data
    assert isinstance(notification, smart_phone.NotificationData)
    assert notification.id == "message-id"
    assert notification.name == "phone.sms.text"
    assert notification.data == smart_phone.SmsTextData(
        id="message-id", sender="+15555550101", text="Book me the 10:15."
    )
    # The room hears only the ding; the notification itself is not broadcast.
    assert [event.name for event in received] == ["environment.sound"]


def test_smart_phone_send_text_message_asks_the_service_and_commits_its_verdict() -> None:
    """Pressing send hands the text to the service; the service's verdict is the outcome.

    The verdict is correlated to its request by envelope id. A failed send reaches the holder;
    a successful one is published but is not an occasion for the holder.
    """

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[str], list[smart_phone.TextMessageSendFailedData]]:
        environment = Environment()
        phone, firmware, holder, bystander = await _smart_phone_in_a_hand(environment)
        recorder = firmware.event_recorder()
        for request_id, text in (("send-1", "See you at 10:15."), ("send-2", "Still there?")):
            await phone.dispatch(
                phone.context(),
                dataclasses.replace(
                    smart_phone.SendTextMessageEvent.with_data(
                        smart_phone.SendTextMessageData(to="+15555550101", text=text)
                    ),
                    id=request_id,
                ),
            )
        await wait_until(
            lambda: event_names(recorder).count(smart_phone.ServiceTextMessageSendRequestedEvent.name) == 2
        )
        await recorder.receive(
            phone.context(),
            dataclasses.replace(
                smart_phone.ServiceTextMessageSentEvent.with_data(
                    smart_phone.TextMessageSentData(to="+15555550101", text="See you at 10:15.")
                ),
                id="send-1",
            ),
        )
        await recorder.receive(
            phone.context(),
            dataclasses.replace(
                smart_phone.ServiceTextMessageSendFailedEvent.with_data(
                    smart_phone.TextMessageSendFailedData(
                        to="+15555550101", text="Still there?", failure_kind="remote_unavailable"
                    )
                ),
                id="send-2",
            ),
        )
        await wait_until(lambda: smart_phone.TextMessageSendFailedEvent.name in event_names(recorder))
        for _ in range(10):
            await asyncio.sleep(0)
        observed = [
            occasion.data.observation.event
            for occasion in holder.occasions
            if isinstance(occasion.data, mosfet.InputEventData) and occasion.data.observation is not None
        ]
        failures = [
            typing.cast(smart_phone.TextMessageSendFailedData, occasion.data.observation.data)
            for occasion in holder.occasions
            if isinstance(occasion.data, mosfet.InputEventData)
            and occasion.data.observation is not None
            and occasion.data.observation.event == smart_phone.TextMessageSendFailedEvent.name
        ]
        assert bystander.received == []
        return list(recorder.events), observed, failures

    published, observed, failures = asyncio.run(run())

    def payloads(name: str) -> list[tuple[str | None, typing.Any]]:
        return [(event.id, event.data) for event in published if event.name == name]

    assert payloads(smart_phone.ServiceTextMessageSendRequestedEvent.name) == [
        ("send-1", smart_phone.SendTextMessageData(to="+15555550101", text="See you at 10:15.")),
        ("send-2", smart_phone.SendTextMessageData(to="+15555550101", text="Still there?")),
    ]
    assert payloads(smart_phone.TextMessageSentEvent.name) == [
        ("send-1", smart_phone.TextMessageSentData(to="+15555550101", text="See you at 10:15.")),
    ]
    assert payloads(smart_phone.TextMessageSendFailedEvent.name) == [
        (
            "send-2",
            smart_phone.TextMessageSendFailedData(
                to="+15555550101", text="Still there?", failure_kind="remote_unavailable"
            ),
        ),
    ]
    assert [failure.text for failure in failures] == ["Still there?"]
    assert smart_phone.TextMessageSentEvent.name not in observed


def test_send_text_message_schema_offers_no_default_recipient() -> None:
    """The command's schema carries no example address a model could copy as a default."""

    schema = smart_phone.SendTextMessageData.model_json_schema()
    assert schema["required"] == ["to", "text"]
    assert "examples" not in schema["properties"]["to"]
    with pytest.raises(pydantic.ValidationError):
        _ = smart_phone.SendTextMessageData(to="+15555550101", text="")


class PhoneOwner(hsm.Instance):
    """A perspective that owns a phone the way a bot does: by naming it in ``owned_devices``."""

    _owned: dict[str, str]

    def __init__(self, owned: dict[str, str]) -> None:
        super().__init__()
        self._owned = dict(owned)

    @staticmethod
    def _note_owned_devices(ctx: hsm.Context, instance: "PhoneOwner", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _ = instance.set("owned_devices", dict(instance._owned))

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "PhoneOwner",
        hsm.attribute("owned_devices"),
        hsm.initial(hsm.target("holding")),
        hsm.state("holding", hsm.entry(_note_owned_devices)),
    )


def _pending_notification_ids(phone: phone_device.Phone) -> list[str]:
    notifications = (phone_display(phone).take_snapshot().Attributes or {}).get("/DeviceDisplay/notifications")
    if notifications is None:
        return []
    assert isinstance(notifications, smart_phone.NotificationsData)
    return [notification.id for notification in notifications.pending]


async def _text_phone(phone: phone_device.Phone, message_id: str, text: str) -> None:
    await phone.dispatch(
        phone.context(),
        smart_phone.SmsTextEvent.with_data(smart_phone.SmsTextData(id=message_id, sender="+15555550101", text=text)),
    )


def test_a_notification_waits_on_the_lock_screen_for_whoever_picks_the_phone_up_later() -> None:
    """Nobody holds the phone when the text lands; it is still there, in context, once somebody does."""

    async def run() -> tuple[list[str], str | None, list[hsm.Event[typing.Any]]]:
        environment = Environment()
        phone = smart_phone.SmartPhone()
        _ = await mosfet.started(environment, phone, typing.cast(hsm.Model, phone.model))
        await wait_until(lambda: phone.state() == "/Device/detached")
        await _text_phone(phone, "message-id", "Book me the 10:15.")
        await wait_until(lambda: _pending_notification_ids(phone) == ["message-id"])

        holder = PhoneHolder()
        _ = await mosfet.started(environment, holder, holder.model, hsm.Config(id="holder"))
        await phone.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=holder)))
        await wait_until(lambda: holder in device_bots(phone))
        owner = PhoneOwner({"phone": hsm.id(phone)})
        _ = await mosfet.started(environment, owner, owner.model)
        return _pending_notification_ids(phone), environment.model_snapshot(owner), holder.occasions

    pending, context, occasions = asyncio.run(run())

    assert pending == ["message-id"]
    # Picking the phone up later is not a new notification: nothing is replayed into the hand.
    assert occasions == []
    assert context is not None
    root = xml.etree.ElementTree.fromstring(context)
    notification = root.find("self/owned_devices/device[@ref='phone']/display/notifications/notification")
    assert notification is not None
    assert notification.get("id") == "message-id"
    assert notification.get("name") == "phone.sms.text"
    assert notification.findtext("sender") == "+15555550101"
    assert notification.findtext("text") == "Book me the 10:15."


@pytest.mark.parametrize("command", [smart_phone.ReadNotificationEvent, smart_phone.DismissNotificationEvent])
def test_reading_or_dismissing_takes_only_that_notification_off_the_lock_screen(
    command: hsm.Event[typing.Any],
) -> None:
    """Both leave the pending list; neither is announced to the holder, who did it."""

    async def run() -> tuple[list[str], list[str | None]]:
        environment = Environment()
        phone, _firmware, holder, _bystander = await _smart_phone_in_a_hand(environment)
        await _text_phone(phone, "first", "Book me the 10:15.")
        await _text_phone(phone, "second", "Actually 10:30.")
        await wait_until(lambda: _pending_notification_ids(phone) == ["first", "second"])
        # JSON ingress, the way a model's tool call arrives.
        await phone.dispatch(phone.context(), dataclasses.replace(command, data={"id": "first"}))
        await wait_until(lambda: _pending_notification_ids(phone) == ["second"])
        for _ in range(10):
            await asyncio.sleep(0)
        return _pending_notification_ids(phone), occasion_sources(holder)

    pending, told = asyncio.run(run())

    assert pending == ["second"]
    assert told == [smart_phone.NotificationEvent.name, smart_phone.NotificationEvent.name]


def test_the_last_notification_read_clears_the_lock_screen() -> None:
    async def run() -> object:
        phone, _firmware, _holder, _bystander = await _smart_phone_in_a_hand(Environment())
        await _text_phone(phone, "message-id", "Book me the 10:15.")
        await wait_until(lambda: _pending_notification_ids(phone) == ["message-id"])
        await phone.dispatch(
            phone.context(),
            smart_phone.ReadNotificationEvent.with_data(smart_phone.ReadNotificationData(id="message-id")),
        )
        await wait_until(lambda: _pending_notification_ids(phone) == [])
        return (phone.take_snapshot().Attributes or {}).get("display")

    display = asyncio.run(run())

    assert isinstance(display, dict)
    assert typing.cast(dict[str, object], display).get("notifications") is None


@pytest.mark.parametrize("command", [smart_phone.ReadNotificationEvent, smart_phone.DismissNotificationEvent])
def test_naming_a_notification_that_is_not_pending_tells_the_holder(command: hsm.Event[typing.Any]) -> None:
    """An unknown or already-cleared id is a typed failure in the hand, never a silent no-op."""

    async def run() -> tuple[list[str], list[hsm.Event[typing.Any]]]:
        phone, _firmware, holder, _bystander = await _smart_phone_in_a_hand(Environment())
        await _text_phone(phone, "message-id", "Book me the 10:15.")
        await wait_until(lambda: _pending_notification_ids(phone) == ["message-id"])
        await phone.dispatch(phone.context(), dataclasses.replace(command, data={"id": "no-such-id"}))
        await wait_until(lambda: len(holder.occasions) == 2)
        return _pending_notification_ids(phone), holder.occasions

    pending, occasions = asyncio.run(run())

    assert pending == ["message-id"]
    failure = occasions[-1].data
    assert isinstance(failure, mosfet.InputEventData)
    assert failure.observation is not None
    assert failure.observation.event == smart_phone.NotificationNotFoundEvent.name
    assert failure.observation.data == smart_phone.NotificationNotFoundData.model_validate(
        {"id": "no-such-id", "command": command.name}
    )


def test_replying_to_a_text_does_not_read_its_notification() -> None:
    """Reading is its own act: sending a text to the sender leaves the notification pending."""

    async def run() -> list[str]:
        phone, _firmware, _holder, _bystander = await _smart_phone_in_a_hand(Environment())
        await _text_phone(phone, "message-id", "Book me the 10:15.")
        await wait_until(lambda: _pending_notification_ids(phone) == ["message-id"])
        await phone.dispatch(
            phone.context(),
            smart_phone.SendTextMessageEvent.with_data(
                smart_phone.SendTextMessageData(to="+15555550101", text="Done.")
            ),
        )
        for _ in range(10):
            await asyncio.sleep(0)
        return _pending_notification_ids(phone)

    assert asyncio.run(run()) == ["message-id"]


def test_the_smart_phone_offers_its_holder_read_and_dismiss_whatever_its_call_is_doing() -> None:
    transitions = transition_map(smart_phone.SmartPhone.firmware_model)

    for command in (smart_phone.ReadNotificationEvent.name, smart_phone.DismissNotificationEvent.name):
        for state in ("/Phone/hung_up", "/Phone/ringing", "/Phone/answered/media_ready"):
            assert command in transitions[state], (command, state)


def _transition_signatures(model: hsm.Model, state: str, event_name: str) -> list[tuple[object, ...]]:
    return [
        (transition.source, transition.target, tuple(transition.effect), transition.guard)
        for transition in transition_map(model)[state][event_name]
    ]


def test_a_smart_phone_handles_every_call_exactly_as_a_phone_does() -> None:
    """Every call transition of the phone is present, unchanged, in the smart phone's firmware.

    The smart phone defines its firmware from the phone's own call topology, so a call on either
    handset walks the same states through the same guards, effects, and targets.
    """

    phone_model = phone_device.Phone.firmware_model
    smart_model = smart_phone.SmartPhone.firmware_model

    assert smart_model is not phone_model
    assert issubclass(smart_phone.SmartPhone, phone_device.Phone)
    assert issubclass(smart_phone.Firmware, phone_device.Firmware)
    for state, events in transition_map(phone_model).items():
        for event_name in events:
            assert _transition_signatures(smart_model, state, event_name) == _transition_signatures(
                phone_model, state, event_name
            ), (state, event_name)


def test_a_smart_phone_rings_answers_and_hangs_up_like_a_phone() -> None:
    """The call path end to end on a smart phone: the ring is heard, the holder is told, the call ends."""

    async def run() -> tuple[list[str], list[str | None], list[hsm.Event[typing.Any]]]:
        environment = Environment()
        listener = PhoneObservationRecorder()
        _ = await mosfet.started(environment, listener, listener.model, hsm.Config(id="listener"))
        environment.join(listener)
        phone, firmware, holder, _bystander = await _smart_phone_in_a_hand(environment)
        states: list[str] = []
        await emit_service_event(
            phone,
            phone_device.IncomingCallEvent.with_data(
                phone_device.IncomingCallData(call_id="call-1", caller="Front desk")
            ),
        )
        await wait_until(lambda: firmware.state() == "/Phone/ringing")
        states.append(firmware.state())
        await answer_call(phone, "call-1")
        await wait_until(lambda: firmware.state().startswith("/Phone/answered"))
        states.append(firmware.state())
        await phone.dispatch(phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()))
        await wait_until(lambda: firmware.state() == "/Phone/hung_up")
        states.append(firmware.state())
        await wait_until(lambda: phone_device.HungUpEvent.name in occasion_sources(holder))
        return states, occasion_sources(holder), listener.events

    states, told, heard = asyncio.run(run())

    assert states == ["/Phone/ringing", "/Phone/answered/media_connecting", "/Phone/hung_up"]
    assert [event.data.kind for event in heard if isinstance(event.data, phone_device.SoundData)] == ["phone.ringing"]
    assert told == [phone_device.AnsweredEvent.name, phone_device.HungUpEvent.name]


def test_a_smart_phone_hears_a_failed_dial_as_the_exchange_tone() -> None:
    async def run() -> list[str | None]:
        environment = Environment()
        listener = PhoneObservationRecorder()
        _ = await mosfet.started(environment, listener, listener.model, hsm.Config(id="listener"))
        environment.join(listener)
        phone, firmware, _holder, _bystander = await _smart_phone_in_a_hand(environment)
        await phone.dispatch(phone.context(), phone_device.DialEvent.with_data(phone_device.DialData(number="5550142")))
        await wait_until(lambda: firmware.state() == "/Phone/dialing")
        await emit_service_event(
            phone,
            phone_device.ServiceDialFailedEvent.with_data(phone_device.DialFailedData(failure_kind="call_declined")),
        )
        await wait_until(lambda: bool(listener.events))
        return [event.data.kind for event in listener.events if isinstance(event.data, phone_device.SoundData)]

    assert asyncio.run(run()) == ["phone.busy"]


def _operation_names(phone: phone_device.Phone) -> list[str]:
    event_map = dict(typing.cast(hsm.Model, phone.model).events)
    event_map.update(phone.firmware_model.events)
    names: list[str] = []
    for transition in hsm.take_snapshot(phone.context(), phone).Transitions:
        for event_name in transition.events:
            event = event_map.get(event_name)
            if event is not None and event.kind == processing.EventKind and event.name not in names:
                names.append(event.name)
    return names


def test_smart_phone_offers_the_phone_call_menu_plus_messaging_in_every_state() -> None:
    """Each state offers exactly the phone's call commands for that state, then texting and the lock screen."""

    messaging = [
        smart_phone.SendTextMessageEvent.name,
        smart_phone.ReadNotificationEvent.name,
        smart_phone.DismissNotificationEvent.name,
    ]

    async def run() -> list[list[str]]:
        phone = smart_phone.SmartPhone()
        _ = await mosfet.started(None, phone, typing.cast(hsm.Model, phone.model))
        await wait_until(lambda: phone.state() == "/Device/detached")
        menus = [_operation_names(phone)]
        await emit_service_event(
            phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123"))
        )
        menus.append(_operation_names(phone))
        await answer_call(phone, "call-123")
        menus.append(_operation_names(phone))
        await emit_service_event(
            phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123"))
        )
        menus.append(_operation_names(phone))
        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(
                    transfer_id="transfer-123",
                    target=phone_device.TransferTarget(kind="address", value="helpdesk@example.com"),
                )
            ),
        )
        menus.append(_operation_names(phone))
        return menus

    idle, ringing, answered, media_ready, transferring = asyncio.run(run())

    # Answer exists only while something is ringing, on a smart phone as on a phone.
    assert idle == [phone_device.DialEvent.name, *messaging]
    assert ringing == [phone_device.AnswerCallEvent.name, phone_device.DeclineCallEvent.name, *messaging]
    assert answered == [phone_device.HangUpCallEvent.name, *messaging]
    assert media_ready == [phone_device.TransferCallEvent.name, phone_device.HangUpCallEvent.name, *messaging]
    assert transferring == [phone_device.HangUpCallEvent.name, *messaging]


def test_a_smart_phone_answers_only_while_ringing() -> None:
    transitions = transition_map(smart_phone.SmartPhone.firmware_model)

    assert [state for state, events in transitions.items() if phone_device.AnswerCallEvent.name in events] == [
        "/Phone/ringing"
    ]


def test_a_smart_phone_takes_messaging_commands_and_a_phone_does_not() -> None:
    """A JSON send-text command enters a smart phone's firmware; a basic phone has nowhere to put it."""

    async def run(phone: phone_device.Phone) -> tuple[bool, list[str]]:
        _ = await mosfet.started(None, phone, typing.cast(hsm.Model, phone.model))
        await wait_until(lambda: phone.state() == "/Device/detached")
        accepted = await phone.dispatch(
            phone.context(),
            dataclasses.replace(smart_phone.SendTextMessageEvent, data={"to": "+15555550101", "text": "Hi."}),
        )
        for _ in range(10):
            await asyncio.sleep(0)
        return accepted, event_names(firmware_of(phone).event_recorder())

    smart_accepted, smart_published = asyncio.run(run(smart_phone.SmartPhone()))
    _, basic_published = asyncio.run(run(phone_device.Phone()))

    assert smart_accepted is True
    assert smart_published == [smart_phone.ServiceTextMessageSendRequestedEvent.name]
    # The shell takes delivery and ignores it, as any unmatched event: nothing reaches firmware.
    assert basic_published == []
    assert smart_phone.SendTextMessageEvent.name not in phone_device.Phone.firmware_model.events
