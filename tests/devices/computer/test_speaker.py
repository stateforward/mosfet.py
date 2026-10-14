from mosfet.devices import audio, computer

import asyncio
import dataclasses
import pathlib
import typing

import hsm
import pytest

from mosfet.environment import Environment, SoundEvent
from tests import metric_reader
from tests.devices.computer.remote import (
    Controller,
    EndData,
    EndStreamEvent,
    FakeStream,
    settle,
    start,
    stream,
    wait_until,
    wire,
)

_TRACE = {"traceparent": "00-00000000000000000000000000000000-0000000000000000-00"}
FRAME = audio.OutputData(audio=b"bot speech", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
PCM = audio.PcmCodec()


class Ear(Controller):
    """Something in the room that would hear the speaker if it played into the room."""


class Room(typing.NamedTuple):
    speaker: computer.Speaker
    controller: Controller
    ear: Ear


async def started_speaker(environment: Environment) -> Room:
    speaker = computer.Speaker()
    await start(environment, speaker, "speaker")
    controller, ear = Controller(), Ear()
    await start(environment, controller, "controller")
    await start(environment, ear, "ear")
    environment.join(ear)
    return Room(speaker, controller, ear)


def output(controller_id: str) -> hsm.Event[typing.Any]:
    return dataclasses.replace(
        audio.OutputEvent.with_data(FRAME), id="frame-1", source=controller_id, metadata=dict(_TRACE)
    )


def test_attached_speaker_sends_its_controllers_audio_to_its_stream_and_not_into_the_room() -> None:
    async def run() -> tuple[FakeStream, list[hsm.Event[typing.Any]]]:
        environment = Environment()
        room = await started_speaker(environment)
        owned = await stream(environment, room.speaker, PCM, "speaker")
        await wire(environment, room.speaker, room.controller)
        await wait_until(room.controller.attached)

        _ = await room.speaker.dispatch(hsm.Context(), output("controller"))
        await wait_until(lambda: bool(owned.sent))
        await settle()
        return owned, room.ear.named(SoundEvent)

    owned, heard = asyncio.run(run())

    (forwarded,) = owned.sent
    assert forwarded.data is FRAME
    assert (forwarded.id, forwarded.source, forwarded.target) == ("frame-1", "speaker", "stream-speaker")
    assert forwarded.metadata == _TRACE
    assert owned.messages == [b"\x00\x00\x3e\x80\x00\x01" + b"bot speech"]
    assert heard == []


def test_unattached_speaker_sends_nothing() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        room = await started_speaker(environment)
        owned = await stream(environment, room.speaker, PCM, "speaker")

        _ = await room.speaker.dispatch(hsm.Context(), output("controller"))
        await settle()
        return owned.sent

    assert asyncio.run(run()) == []


def _unowned(metrics: metric_reader.Reader) -> float:
    return sum(
        point.value
        for point in metrics.points("bot.transform.dropped.count")
        if point.attributes.get("reason") == "unowned" and point.attributes.get("codec") == "pcm"
    )


@pytest.mark.parametrize("ended", ["never_owned", "own_stream_ended"])
def test_an_attached_speaker_with_no_stream_drops_and_counts_the_frame(
    ended: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        room = await started_speaker(environment)
        await wire(environment, room.speaker, room.controller)
        await wait_until(room.controller.attached)
        sent: list[hsm.Event[typing.Any]] = []
        if ended == "own_stream_ended":
            owned = await stream(environment, room.speaker, PCM, "speaker")
            await settle()
            _ = await owned.dispatch(hsm.Context(), EndStreamEvent.with_data(EndData(reason="finished")))
            await settle()
            sent = owned.sent
        _ = await room.speaker.dispatch(hsm.Context(), output("controller"))
        await settle()
        return sent

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _unowned(metrics)
        sent = asyncio.run(run())
        after = _unowned(metrics)

    assert sent == []
    assert after - before == 1
