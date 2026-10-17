from mosfet.devices import audio, computer, hid

import asyncio
import collections.abc
import datetime
import pathlib
import typing

import hsm
import mosfet
import mosfet.lifecycle
import pydantic
import pytest

from mosfet import media
from mosfet.abilities import speaking, vocal
from mosfet.environment import Environment, SoundEvent, VisualEvent, space
from mosfet.protocols import attachment, transport
from tests import metric_reader
from tests.devices.computer.remote import (
    DEVICE_ID,
    FakeStream,
    FakeTransport,
    Holder,
    connected,
    disconnected,
    json_message,
    open_streams,
    pcm_message,
    peer_send,
    settle,
    still_message,
    wait_until,
)
from tests.hsm_instance_state import device_firmware, device_peripherals, start_ability_tree

PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, media_type="image/png")
LINK_TIMEOUT = datetime.timedelta(seconds=10)
FRAME = pcm_message(b"remote voice")


class Ears(hsm.Instance):
    """Someone in the computer's room: keeps every sound and sight that reaches them."""

    received: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.received = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Ears", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.received.append(event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Ears",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(SoundEvent, VisualEvent), hsm.effect(_record))),
    )


class Remote(typing.NamedTuple):
    device: computer.Computer
    transport: FakeTransport
    streams: dict[str, FakeStream]


async def started_computer(
    environment: Environment,
    *,
    transport: FakeTransport | None = None,
    device_id: str = DEVICE_ID,
    link_timeout: datetime.timedelta = LINK_TIMEOUT,
) -> tuple[computer.Computer, FakeTransport]:
    resolved = transport if transport is not None else FakeTransport()
    if not mosfet.lifecycle.is_started(resolved):
        _ = await mosfet.started(environment, resolved, FakeTransport.model, hsm.Config(id=f"transport-{device_id}"))
    device = computer.Computer(id=device_id, transport=resolved, person_id="p-1", link_timeout=link_timeout)
    _ = await mosfet.started(environment, device, typing.cast(hsm.Model, device.model))
    return device, resolved


async def linked_computer(environment: Environment, *, device_id: str = DEVICE_ID) -> Remote:
    device, transport = await started_computer(environment, device_id=device_id)
    await wait_until(lambda: device.state() == "/Device/detached")
    _ = await transport.dispatch(hsm.Context(), connected(transport))
    streams = await open_streams(environment, transport)
    await wait_until(lambda: firmware_state(device) == "/Computer/linked")
    return Remote(device, transport, streams)


def firmware_state(device: computer.Computer) -> str:
    firmware = device_firmware(device)
    return "" if firmware is None else firmware.state()


async def held(environment: Environment, device: computer.Computer) -> Holder:
    holder = Holder()
    _ = await mosfet.started(environment, holder, holder.model)
    _ = await hsm.dispatch(environment, device, attachment.AttachEvent.with_data(attachment.AttachData(actor=holder)))
    await wait_until(lambda: device.state() == "/Device/attached")
    return holder


async def joined_ears(environment: Environment) -> Ears:
    ears = Ears()
    _ = await mosfet.started(environment, ears, Ears.model, hsm.Config(id="ears"))
    environment.join(ears)
    return ears


def test_computer_rejects_invalid_construction() -> None:
    fake = FakeTransport()
    with pytest.raises(ValueError, match="link_timeout"):
        _ = computer.Computer(id=DEVICE_ID, transport=fake, link_timeout=datetime.timedelta(0))
    with pytest.raises(pydantic.ValidationError):
        _ = computer.Computer(id="Not A Device", transport=fake)
    with pytest.raises(pydantic.ValidationError):
        _ = computer.Computer(id=DEVICE_ID, transport=fake, person_id="")
    with pytest.raises(pydantic.ValidationError):
        _ = computer.Computer(id=DEVICE_ID, transport=fake, peer_id="not a peer")
    # A valid device id is not always a valid peer id: the default peer id is the id itself.
    with pytest.raises(pydantic.ValidationError):
        _ = computer.Computer(id="my_desk", transport=fake)


def test_a_computer_is_known_by_its_id_and_its_remote_machine_by_its_peer_id() -> None:
    """DSL-30: ``id`` is the body's reference, ``"computer"`` by default; ``peer_id`` names the remote machine."""

    assert computer.Computer(transport=FakeTransport()).identity() == "computer"
    assert computer.Computer(id="desk", transport=FakeTransport()).identity() == "desk"
    assert computer.Computer(id="my_desk", transport=FakeTransport(), peer_id="my-desk").identity() == "my_desk"
    peer_ids: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(computer.PeerId)
    assert peer_ids.validate_python("studio-laptop") == "studio-laptop"
    with pytest.raises(pydantic.ValidationError):
        _ = peer_ids.validate_python("my_desk")


def test_computer_owns_seven_ports_and_its_mouthpiece_sits_with_it() -> None:
    placement = space.Placement(position=space.Position(x=1.0, y=2.0))
    device = computer.Computer(id=DEVICE_ID, transport=FakeTransport(), placement=placement)

    kinds = [type(port) for port in device_peripherals(device)]

    assert kinds == [
        hid.Keyboard,
        hid.Pointer,
        computer.Camera,
        computer.Microphone,
        computer.Display,
        computer.Speaker,
        computer.Mouthpiece,
    ]
    assert computer.Computer.firmware_model is computer.App.model


@pytest.mark.parametrize("peer_id", [None, "laptop-peer"])
def test_bring_up_registers_the_link_with_the_ports_channels_then_wires_every_port(peer_id: str | None) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], tuple[hsm.Instance, ...], list[str]]:
        environment = Environment()
        fake = FakeTransport()
        _ = await mosfet.started(environment, fake, FakeTransport.model, hsm.Config(id="transport"))
        device = computer.Computer(id=DEVICE_ID, transport=fake, peer_id=peer_id, person_id="p-1")
        _ = await mosfet.started(environment, device, typing.cast(hsm.Model, device.model))
        await wait_until(lambda: device.state() == "/Device/detached")
        ports = device_peripherals(device)
        await wait_until(lambda: all(port.state().endswith("/attached") for port in ports))
        return fake.requests, ports, [port.state().rpartition("/")[2] for port in ports]

    requests, ports, port_states = asyncio.run(run())

    (registration,) = requests
    data = registration.data
    assert isinstance(data, transport.PeerAttachData)
    assert data.peer_id == (DEVICE_ID if peer_id is None else peer_id)
    assert [(channel.name, channel.owner) for channel in data.channels] == [
        ("keyboard", ports[0]),
        ("pointer", ports[1]),
        ("display", ports[4]),
        ("camera", ports[2]),
        ("microphone", ports[3]),
        ("speaker", ports[5]),
    ]
    assert port_states == ["attached"] * 7


@pytest.mark.parametrize("register", ["refuse", "silent"])
def test_bring_up_fails_when_the_transport_will_not_carry_the_computer(register: str) -> None:
    async def run() -> str:
        device, _ = await started_computer(
            Environment(), transport=FakeTransport(register=register), link_timeout=datetime.timedelta(milliseconds=20)
        )
        await wait_until(lambda: device.state() == "/Device/failed")
        return device.state()

    assert asyncio.run(run()) == "/Device/failed"


def test_stop_releases_the_link_before_the_firmware_goes_then_unregisters_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[list[str], list[tuple[str, str]], bool]:
        environment = Environment()
        remote = await linked_computer(environment)
        firmware = device_firmware(remote.device)
        assert firmware is not None
        await remote.device.stop(environment)
        return (
            [request.name for request in remote.transport.requests],
            remote.transport.closed(),
            mosfet.lifecycle.is_started(firmware),
        )

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _detach_outcomes(metrics)
        names, closed, firmware_started = asyncio.run(run())
        after = _detach_outcomes(metrics)

    assert names == [attachment.AttachEvent.name, attachment.DetachEvent.name]
    assert closed == [("c1", "released")]
    assert firmware_started is False
    assert after.get("detached", 0) - before.get("detached", 0) == 1


def _detach_outcomes(metrics: metric_reader.Reader) -> dict[object, float]:
    return {
        point.attributes.get("bot.outcome"): point.value
        for point in metrics.points("bot.devices.computer.link.detach.count")
    }


def test_the_link_coming_up_and_going_down_is_told_to_the_holder() -> None:
    async def run() -> list[tuple[str, object]]:
        environment = Environment()
        device, fake = await started_computer(environment)
        await wait_until(lambda: device.state() == "/Device/detached")
        holder = await held(environment, device)
        _ = await fake.dispatch(hsm.Context(), connected(fake))
        _ = await fake.dispatch(hsm.Context(), disconnected(fake, reason="transport_lost"))
        await wait_until(lambda: len(holder.told) == 2)
        return holder.observations()

    observations = asyncio.run(run())

    assert observations == [
        (computer.LinkUpEvent.name, computer.LinkUpData(device_id=DEVICE_ID, transport="websocket")),
        (computer.LinkDownEvent.name, computer.LinkDownData(device_id=DEVICE_ID, reason="transport_lost")),
    ]


def test_what_a_person_does_is_told_to_the_holder_and_never_put_in_the_room() -> None:
    async def run() -> tuple[list[tuple[str, object]], list[hsm.Event[typing.Any]]]:
        environment = Environment()
        remote = await linked_computer(environment)
        holder = await held(environment, remote.device)
        ears = await joined_ears(environment)
        show = computer.ShowEvent.with_data(
            computer.ShowData(elements=(computer.ButtonElement(id="send", label="Send"),))
        )
        _ = await remote.device.dispatch(remote.device.context(), show)
        layout = computer.LayoutData(boxes=(computer.BoxData(id="send", x=0.0, y=0.0, width=1.0, height=1.0),))
        _ = await remote.streams["display"].dispatch(
            hsm.Context(), peer_send(json_message(computer.LayoutEvent.with_data(layout)))
        )
        for action in ("down", "up"):
            pointer = hid.PointerData(action=action, pointer_type="touch", pointer_id=0, x=0.5, y=0.5)
            _ = await remote.streams["pointer"].dispatch(
                hsm.Context(), peer_send(json_message(hid.PointerEvent.with_data(pointer)))
            )
        await wait_until(lambda: bool(holder.observations()))
        await settle()
        return holder.observations(), ears.received

    observations, room = asyncio.run(run())

    assert observations == [
        (computer.ActivatedEvent.name, computer.ActivatedData(device_id=DEVICE_ID, person_id="p-1", element_id="send"))
    ]
    assert room == []


def test_keys_typed_on_the_remote_keyboard_are_told_to_the_holder() -> None:
    async def run() -> list[tuple[str, object]]:
        environment = Environment()
        remote = await linked_computer(environment)
        holder = await held(environment, remote.device)
        show = computer.ShowEvent.with_data(
            computer.ShowData(elements=(computer.FieldElement(id="name", focused=True),))
        )
        _ = await remote.device.dispatch(remote.device.context(), show)
        for key, text in (("h", "h"), ("i", "i"), ("Enter", None)):
            typed = hid.KeyEvent.with_data(hid.KeyData(action="down", key=key, code="", text=text))
            _ = await remote.streams["keyboard"].dispatch(hsm.Context(), peer_send(json_message(typed)))
        await wait_until(lambda: bool(holder.observations()))
        return holder.observations()

    assert asyncio.run(run()) == [
        (
            computer.TextEnteredEvent.name,
            computer.TextEnteredData(device_id=DEVICE_ID, person_id="p-1", field_id="name", text="hi"),
        )
    ]


def test_a_still_and_remote_audio_are_put_into_the_room_from_the_computer() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[tuple[str, object]], str]:
        environment = Environment()
        remote = await linked_computer(environment)
        holder = await held(environment, remote.device)
        ears = await joined_ears(environment)
        _ = await remote.streams["camera"].dispatch(hsm.Context(), peer_send(still_message(PNG)))
        _ = await remote.streams["microphone"].dispatch(hsm.Context(), peer_send(FRAME))
        await wait_until(lambda: len(ears.received) == 2)
        await settle()
        return ears.received, holder.observations(), hsm.id(remote.device)

    room, told, runtime_id = asyncio.run(run())

    assert sorted(event.name for event in room) == sorted([VisualEvent.name, SoundEvent.name])
    assert all(event.source == runtime_id for event in room)
    assert told == []


def test_show_enters_firmware_only_and_speaker_audio_is_not_an_owner_command() -> None:
    async def run() -> tuple[computer.ContentData | None, list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]]]:
        environment = Environment()
        remote = await linked_computer(environment)
        show = computer.ShowEvent.with_data(computer.ShowData(elements=(computer.TextElement(id="hi", text="Hi"),)))
        _ = await remote.device.dispatch(remote.device.context(), show)
        frame = audio.OutputData(audio=b"pcm!", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
        _ = await remote.device.dispatch(remote.device.context(), audio.OutputEvent.with_data(frame))
        shown = computer.ContentData(elements=(computer.TextElement(id="hi", text="Hi"),))
        await wait_until(lambda: any(event.data == shown for event in remote.streams["display"].sent))
        await settle()
        attributes = remote.device.take_snapshot().Attributes or {}
        display = typing.cast(dict[str, object], attributes.get("display", {}))
        return (
            typing.cast(computer.ContentData | None, display.get("content")),
            remote.streams["display"].sent,
            remote.streams["speaker"].sent,
        )

    content, display, speaker = asyncio.run(run())

    assert content == computer.ContentData(elements=(computer.TextElement(id="hi", text="Hi"),))
    assert display[-1].data == content
    assert speaker == []


def test_one_microphone_frame_is_heard_once_in_the_room_and_never_goes_back_up() -> None:
    """The echo loop regression: remote audio enters the room once and no stream carries it back."""

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]]]:
        environment = Environment()
        remote = await linked_computer(environment)
        ears = await joined_ears(environment)
        _ = await remote.streams["microphone"].dispatch(hsm.Context(), peer_send(FRAME))
        await wait_until(lambda: bool(ears.received))
        await settle(30)
        return ears.received, remote.streams["speaker"].sent

    room, speaker = asyncio.run(run())

    (heard,) = room
    assert isinstance(heard.data, computer.MicrophoneSoundData)
    assert speaker == []


def test_two_computers_never_carry_one_remote_voice_to_either_speaker() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]]]:
        environment = Environment()
        first = await linked_computer(environment, device_id="computer-a")
        second = await linked_computer(environment, device_id="computer-b")
        ears = await joined_ears(environment)
        _ = await first.streams["microphone"].dispatch(hsm.Context(), peer_send(FRAME))
        await wait_until(lambda: bool(ears.received))
        await settle(30)
        return first.streams["speaker"].sent, second.streams["speaker"].sent

    assert asyncio.run(run()) == ([], [])


class _Utterance(vocal.speech.SpeechEncoder):
    """A vocal tract: words in, 16-bit sound out. Local, so the chain can be checked without a network."""

    def __init__(self) -> None:
        self.format = vocal.speech.Format(sample_rate_hz=16_000, channels=1)

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        spoken = b"spoken:" + said.encode()
        yield vocal.speech.Audio(pcm=spoken + b"\x00" * (len(spoken) % 2), text=said)


async def _voice(environment: Environment) -> speaking.Speaking:
    """A bot's voice in the computer's room: a speaking ability and the mouth it attaches to and plays through."""

    mouth = audio.Speaker()
    _ = await mosfet.started(environment, mouth, typing.cast(hsm.Model, audio.Speaker.model))
    voice = speaking.Speaking(encoder=_Utterance(), speaker=mouth)
    await start_ability_tree(environment, voice)
    await wait_until(lambda: (voice.state() or "").endswith("/wired/voiced/idle"))
    return voice


def test_a_bot_that_speaks_is_heard_on_the_remote_speaker_only_while_linked() -> None:
    """The bot's voice path, end to end: speaking is enough, and only a linked computer plays it."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        device, fake = await started_computer(environment)
        await wait_until(lambda: device.state() == "/Device/detached")
        ports = device_peripherals(device)
        await wait_until(lambda: all(port.state().endswith("/attached") for port in ports))
        voice = await _voice(environment)

        _ = await voice.apply(speaking.InputData(parts=(vocal.speech.Part(words="Hello."),)), ctx=environment)
        await settle(50)

        _ = await fake.dispatch(hsm.Context(), connected(fake))
        streams = await open_streams(environment, fake)
        await wait_until(lambda: firmware_state(device) == "/Computer/linked")
        await settle()
        _ = await voice.apply(speaking.InputData(parts=(vocal.speech.Part(words="Hello again."),)), ctx=environment)
        await wait_until(lambda: bool(streams["speaker"].sent), timeout=5.0)
        await settle(50)
        return streams["speaker"].sent

    (played,) = asyncio.run(run())

    assert played.name == audio.OutputEvent.name
    assert played.target == "stream-speaker"
    assert isinstance(played.data, audio.OutputData) and played.data.audio.startswith(b"spoken:")
