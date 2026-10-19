from mosfet.devices import audio, computer, hid

import asyncio
import dataclasses
import pathlib
import typing

import hsm
import pytest

from mosfet import media
from mosfet.environment import Environment, SoundData, SoundEvent, VisualData, VisualEvent
from tests import metric_reader
from tests.devices.computer.remote import (
    DEVICE_ID,
    PERSON,
    FakeStream,
    Owner,
    ports,
    settle,
    start,
    stream,
    wait_until,
    wire,
)

PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, media_type="image/png")
SCREEN = computer.ShowData(
    elements=(
        computer.TextElement(id="question", text="What should we build?", role="heading"),
        computer.FieldElement(id="answer", placeholder="Type here"),
        computer.ButtonElement(id="send", label="Send"),
    )
)
LAYOUT = computer.LayoutData(
    boxes=(
        computer.BoxData(id="question", x=0.0, y=0.0, width=1.0, height=0.2),
        computer.BoxData(id="answer", x=0.1, y=0.3, width=0.8, height=0.1),
        computer.BoxData(id="send", x=0.4, y=0.6, width=0.2, height=0.1),
    )
)
UP = computer.LinkUpData(device_id=DEVICE_ID, transport="websocket")


class Remote(typing.NamedTuple):
    app: computer.App
    owner: Owner
    ports: computer.Ports
    speaker_stream: FakeStream


async def started_app(environment: Environment, *, person_id: str | None = "p-1") -> Remote:
    owner = Owner()
    await start(environment, owner, "computer")
    wired = ports()
    await start(environment, wired.display, "display")
    await start(environment, wired.speaker, "speaker")
    speaker_stream = await stream(environment, wired.speaker, audio.PcmCodec(), "speaker")
    app = computer.App(device=owner, ports=wired, person_id=person_id)
    await start(environment, app, "app")
    await wire(environment, wired.speaker, app)
    await settle()
    return Remote(app, owner, wired, speaker_stream)


async def send(app: computer.App, event: hsm.Event[typing.Any]) -> None:
    _ = await app.dispatch(hsm.Context(), event)


def key(
    key_name: str, *, text: str | None = None, action: typing.Literal["down", "up"] = "down"
) -> hsm.Event[typing.Any]:
    return hid.KeyEvent.with_data(hid.KeyData(action=action, key=key_name, code="", text=text))


def pointer(
    action: typing.Literal["down", "move", "up", "cancel"], x: float, y: float, pointer_id: int = 0
) -> hsm.Event[typing.Any]:
    return hid.PointerEvent.with_data(
        hid.PointerData(action=action, pointer_type="mouse", pointer_id=pointer_id, x=x, y=y)
    )


def shown(remote: Remote) -> computer.ContentData | None:
    attributes = remote.ports.display.take_snapshot().Attributes or {}
    return typing.cast(computer.ContentData | None, attributes.get("/Device/content"))


async def on_screen(remote: Remote) -> None:
    await send(remote.app, computer.ShowEvent.with_data(SCREEN))
    await send(remote.app, computer.LayoutEvent.with_data(LAYOUT))
    await wait_until(lambda: shown(remote) is not None)


async def click(remote: Remote, x: float, y: float) -> None:
    await send(remote.app, pointer("down", x, y))
    await send(remote.app, pointer("up", x, y))


def test_show_drives_the_display_with_exactly_what_the_bot_chose() -> None:
    async def run() -> computer.ContentData | None:
        remote = await started_app(Environment())
        await on_screen(remote)
        return shown(remote)

    assert asyncio.run(run()) == computer.ContentData(elements=SCREEN.elements)


def test_typing_into_a_focused_field_then_enter_tells_the_holder_once_and_clears_it() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], computer.ContentData | None, computer.ContentData | None]:
        remote = await started_app(Environment())
        await on_screen(remote)
        await click(remote, 0.5, 0.35)
        for character in "hi!":
            await send(remote.app, key(character, text=character))
        await send(remote.app, key("Backspace"))
        await send(remote.app, key("a", action="up"))
        await settle()
        typed = shown(remote)
        await send(remote.app, key("Enter"))
        await wait_until(lambda: bool(remote.owner.observations()))
        await settle()
        return remote.owner.observations(), typed, shown(remote)

    observations, typed, after = asyncio.run(run())

    assert typed is not None
    (field,) = [element for element in typed.elements if isinstance(element, computer.FieldElement)]
    assert (field.value, field.focused) == ("hi", True)
    (entered,) = observations
    assert entered.name == computer.TextEnteredEvent.name
    assert entered.data == computer.TextEnteredData(
        device_id=DEVICE_ID, person_id="p-1", participants=PERSON, field_id="answer", text="hi"
    )
    assert after is not None
    (cleared,) = [element for element in after.elements if isinstance(element, computer.FieldElement)]
    assert cleared.value == ""


def test_typing_on_a_blank_screen_then_enter_tells_the_holder_the_text_with_no_field() -> None:
    # Live (builder check): keys typed on the blank screen of a just-linked computer reached no one.
    async def run() -> list[hsm.Event[typing.Any]]:
        remote = await started_app(Environment())
        for character in "hi!":
            await send(remote.app, key(character, text=character))
        await send(remote.app, key("Backspace"))
        await send(remote.app, key("Enter"))
        await wait_until(lambda: bool(remote.owner.observations()))
        await send(remote.app, key("Enter"))
        await settle()
        return remote.owner.observations()

    (entered,) = asyncio.run(run())

    assert entered.name == computer.TextEnteredEvent.name
    assert entered.data == computer.TextEnteredData(
        device_id=DEVICE_ID, person_id="p-1", participants=PERSON, field_id=None, text="hi"
    )


def test_typing_with_no_field_focused_then_enter_tells_the_holder_and_focusing_a_field_drops_the_line() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        remote = await started_app(Environment())
        await on_screen(remote)
        await send(remote.app, key("h", text="h"))
        await send(remote.app, key("Enter"))
        await wait_until(lambda: bool(remote.owner.observations()))
        await send(remote.app, key("x", text="x"))
        await click(remote, 0.5, 0.35)
        await send(remote.app, key("y", text="y"))
        await send(remote.app, key("Enter"))
        await wait_until(lambda: len(remote.owner.observations()) == 2)
        # The bot shows the screen again with nothing focused: the "x" typed before the click is gone.
        await on_screen(remote)
        await send(remote.app, key("Enter"))
        await settle()
        return remote.owner.observations()

    unfocused, focused = asyncio.run(run())

    assert unfocused.data == computer.TextEnteredData(
        device_id=DEVICE_ID, person_id="p-1", participants=PERSON, field_id=None, text="h"
    )
    assert focused.data == computer.TextEnteredData(
        device_id=DEVICE_ID, person_id="p-1", participants=PERSON, field_id="answer", text="y"
    )


def test_enter_on_an_empty_line_or_with_shift_tells_nobody() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        remote = await started_app(Environment())
        await on_screen(remote)
        await send(remote.app, key("Enter"))
        await click(remote, 0.5, 0.35)
        await send(remote.app, key("Enter"))
        await send(remote.app, key("x", text="x"))
        await send(
            remote.app,
            dataclasses.replace(
                key("Enter"), data=hid.KeyData(action="down", key="Enter", code="", modifiers=frozenset({"shift"}))
            ),
        )
        await settle()
        return remote.owner.observations()

    assert asyncio.run(run()) == []


def test_pressing_and_releasing_on_a_button_activates_it() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        remote = await started_app(Environment())
        await on_screen(remote)
        await click(remote, 0.5, 0.65)
        await wait_until(lambda: bool(remote.owner.observations()))
        return remote.owner.observations()

    (activated,) = asyncio.run(run())

    assert activated.name == computer.ActivatedEvent.name
    assert activated.data == computer.ActivatedData(
        device_id=DEVICE_ID, person_id="p-1", participants=PERSON, element_id="send"
    )


@pytest.mark.parametrize(
    "gesture",
    [
        (("down", 0.5, 0.65), ("up", 0.5, 0.35)),
        (("down", 0.5, 0.35), ("up", 0.5, 0.65)),
        (("down", 0.5, 0.65), ("cancel", 0.5, 0.65), ("up", 0.5, 0.65)),
        (("up", 0.5, 0.65),),
    ],
)
def test_a_release_elsewhere_or_a_cancelled_press_activates_nothing(
    gesture: tuple[tuple[typing.Literal["down", "up", "cancel"], float, float], ...],
) -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        remote = await started_app(Environment())
        await on_screen(remote)
        for action, x, y in gesture:
            await send(remote.app, pointer(action, x, y))
        await settle()
        return [event for event in remote.owner.observations() if event.name == computer.ActivatedEvent.name]

    assert asyncio.run(run()) == []


def test_a_still_is_put_into_the_room_stamped_with_who_uses_the_computer() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        remote = await started_app(Environment())
        await send(remote.app, computer.StillEvent.with_data(computer.StillData(image=PNG)))
        await wait_until(lambda: bool(remote.owner.observations()))
        return remote.owner.observations()

    (seen,) = asyncio.run(run())

    assert seen.name == VisualEvent.name
    assert isinstance(seen.data, VisualData)
    assert seen.data.image is PNG
    assert seen.data.kind == "computer.camera"
    assert seen.data.participants == PERSON
    assert "device_id" not in VisualData.model_fields


def test_microphone_audio_is_put_into_the_room_as_the_type_never_relayed_back() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        remote = await started_app(Environment())
        frame = audio.InputData(audio=b"remote voice", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
        await send(remote.app, audio.InputEvent.with_data(frame))
        await wait_until(lambda: bool(remote.owner.observations()))
        return remote.owner.observations()

    (heard,) = asyncio.run(run())

    assert heard.name == SoundEvent.name
    assert isinstance(heard.data, computer.MicrophoneSoundData)
    assert (heard.data.audio, heard.data.kind, heard.data.amplitude_db) == (
        b"remote voice",
        "computer.microphone",
        None,
    )
    assert heard.data.participants == PERSON


def test_a_typed_line_and_a_clicked_button_are_stamped_with_who_uses_the_computer() -> None:
    # Live (builder check round 3, G1): what the person typed named no one, so recall had no person to follow.
    async def run() -> list[hsm.Event[typing.Any]]:
        remote = await started_app(Environment())
        await on_screen(remote)
        await click(remote, 0.5, 0.35)
        await send(remote.app, key("y", text="y"))
        await send(remote.app, key("Enter"))
        await click(remote, 0.5, 0.65)
        await wait_until(lambda: len(remote.owner.observations()) == 2)
        return remote.owner.observations()

    entered, activated = asyncio.run(run())

    assert isinstance(entered.data, computer.TextEnteredData)
    assert isinstance(activated.data, computer.ActivatedData)
    assert entered.data.participants == activated.data.participants == PERSON


def test_without_a_person_the_computer_stamps_no_participant() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        remote = await started_app(Environment(), person_id=None)
        await send(remote.app, computer.StillEvent.with_data(computer.StillData(image=PNG)))
        await send(remote.app, audio.InputEvent.with_data(audio.InputData(audio=b"remote voice")))
        await send(remote.app, key("h", text="h"))
        await send(remote.app, key("Enter"))
        await wait_until(lambda: len(remote.owner.observations()) == 3)
        return remote.owner.observations()

    for observation in asyncio.run(run()):
        assert isinstance(observation.data, VisualData | SoundData | computer.TextEnteredData)
        assert observation.data.participants == ()


def _echo_outcomes(metrics: metric_reader.Reader) -> dict[object, float]:
    return {
        point.attributes.get("outcome"): point.value
        for point in metrics.points("bot.devices.computer.peripheral.event.count")
        if point.attributes.get("peripheral") == "mouthpiece"
    }


def test_room_sound_goes_up_the_link_only_while_linked_and_the_remote_voice_never_does(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = SoundData(audio=b"bot speech", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
    echo = computer.MicrophoneSoundData(audio=b"remote voice")

    async def run() -> tuple[int, list[hsm.Event[typing.Any]]]:
        remote = await started_app(Environment())
        await send(remote.app, computer.HeardEvent.with_data(computer.HeardData(sound=room)))
        await settle()
        unlinked = len(remote.speaker_stream.named(audio.OutputEvent))
        await send(remote.app, computer.LinkUpEvent.with_data(UP))
        await wait_until(lambda: remote.app.state() == "/Computer/linked")
        await send(remote.app, computer.HeardEvent.with_data(computer.HeardData(sound=echo)))
        await send(remote.app, computer.HeardEvent.with_data(computer.HeardData(sound=room)))
        await wait_until(lambda: bool(remote.speaker_stream.named(audio.OutputEvent)))
        await settle()
        return unlinked, remote.speaker_stream.named(audio.OutputEvent)

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _echo_outcomes(metrics)
        unlinked, relayed = asyncio.run(run())
        after = _echo_outcomes(metrics)

    assert unlinked == 0
    (played,) = relayed
    assert played.data == audio.OutputData(
        audio=b"bot speech", media_type="audio/pcm", sample_rate_hz=16_000, channels=1
    )
    assert played.source == "speaker"
    for outcome in ("delivered", "dropped", "echo_suppressed"):
        assert after.get(outcome, 0) - before.get(outcome, 0) == 1, outcome


def test_nothing_happens_that_nobody_caused() -> None:
    """Coming up linked and receiving a layout changes no screen and tells nobody anything else."""

    async def run() -> tuple[list[str], computer.ContentData | None]:
        remote = await started_app(Environment())
        await send(remote.app, computer.LinkUpEvent.with_data(UP))
        await send(remote.app, computer.LayoutEvent.with_data(LAYOUT))
        await send(remote.app, pointer("move", 0.5, 0.5))
        await settle()
        return [event.name for event in remote.owner.observations()], shown(remote)

    names, content = asyncio.run(run())

    assert names == [computer.LinkUpEvent.name]
    assert content == computer.ContentData()
