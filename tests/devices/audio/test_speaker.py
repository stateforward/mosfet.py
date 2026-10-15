from mosfet.devices import audio

import asyncio
import collections.abc
import dataclasses
import datetime
import pathlib
import typing

import hsm
import mosfet
import pytest

from mosfet.device import Device
from mosfet.protocols import attachment
from mosfet.environment import SoundData, SoundEvent, Environment
from tests import metric_reader
from tests.hsm_instance_state import device_peripherals

_TRACE = {"traceparent": "00-11111111111111111111111111111111-1111111111111111-01"}
# 8 kHz mono s16le: 10 ms is 80 samples, 160 bytes.
_RATE = 8_000
_SLICE = 160
_FRAME = datetime.timedelta(milliseconds=10)
_BUFFER = datetime.timedelta(seconds=30)


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for condition.")
        await asyncio.sleep(0)


async def settle(turns: int = 10) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


class RecordingDevice(Device):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[hsm.Event[typing.Any]] = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        del ctx
        self.events.append(event)

        done: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        done.set_result(True)
        return done

    def named(self, event: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        return [received for received in self.events if received.name == event.name]


class FakeTime:
    """One clock for the speaker's schedule and its ``hsm.after`` timers, stepped by the test.

    ``now`` is what the injected clock reads. Every timer the speaker arms is kept with the duration it
    asked for, and fires only when the test releases it.
    """

    now: float

    def __init__(self, now: float) -> None:
        self.now = now
        self.timers: list[tuple[datetime.timedelta, asyncio.Future[datetime.datetime]]] = []

    def clock(self) -> float:
        return self.now

    def after(self, duration: datetime.timedelta) -> asyncio.Future[datetime.datetime]:
        timer: asyncio.Future[datetime.datetime] = asyncio.get_running_loop().create_future()
        self.timers.append((duration, timer))
        return timer

    def pacing(self) -> list[datetime.timedelta]:
        """Durations of the playout timers (the speaker's lifecycle timers are minutes long)."""

        return [duration for duration, _ in self.timers if duration < datetime.timedelta(seconds=1)]

    def release(self, now: float) -> None:
        """Move the clock to ``now`` and fire the newest pending playout timer."""

        self.now = now
        pending = [
            timer
            for duration, timer in self.timers
            if duration < datetime.timedelta(seconds=1) and not timer.done() and not timer.cancelled()
        ]
        pending[-1].set_result(datetime.datetime.now())


class Room(typing.NamedTuple):
    environment: Environment
    speaker: audio.Speaker
    controller: RecordingDevice
    ear: RecordingDevice


async def room(
    *,
    frame: datetime.timedelta = _FRAME,
    max_buffered: datetime.timedelta = _BUFFER,
    time: FakeTime | None = None,
    controllers: int = 1,
) -> tuple[Room, tuple[RecordingDevice, ...]]:
    environment = Environment()
    speaker = (
        audio.Speaker(frame=frame, max_buffered=max_buffered)
        if time is None
        else audio.Speaker(frame=frame, max_buffered=max_buffered, clock=time.clock)
    )
    config = (
        hsm.Config(id="room-speaker")
        if time is None
        else hsm.Config(id="room-speaker", clock=hsm.Clock(after=time.after))
    )
    _ = await mosfet.started(environment, speaker, typing.cast(hsm.Model, speaker.model), config)
    ear = RecordingDevice()
    _ = await mosfet.started(environment, ear, typing.cast(hsm.Model, ear.model), hsm.Config(id="ear"))
    attached: list[RecordingDevice] = []
    for index in range(controllers):
        controller = RecordingDevice()
        _ = await mosfet.started(
            environment, controller, typing.cast(hsm.Model, controller.model), hsm.Config(id=f"controller-{index}")
        )
        _ = await speaker.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller)))
        await wait_until(lambda controller=controller: bool(controller.named(attachment.AttachCompleteEvent)))
        attached.append(controller)
    return Room(environment, speaker, attached[0], ear), tuple(attached)


def pcm(slices: int, *, fill: int = 1) -> bytes:
    return bytes([fill, 0]) * (_SLICE // 2) * slices


def output(
    commander: hsm.Instance | str,
    audio_bytes: bytes,
    *,
    mark: str | None = None,
    event_id: str = "frame",
    media_type: str = "audio/pcm",
    sample_rate_hz: int | None = _RATE,
    channels: int | None = 1,
) -> hsm.Event[typing.Any]:
    return dataclasses.replace(
        audio.OutputEvent.with_data(
            audio.OutputData(
                audio=audio_bytes,
                media_type=media_type,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                mark=mark,
            )
        ),
        id=event_id,
        source=commander if isinstance(commander, str) else hsm.id(commander),
        metadata=dict(_TRACE),
    )


def flush(commander: hsm.Instance, *, event_id: str = "flush") -> hsm.Event[typing.Any]:
    return dataclasses.replace(audio.FlushEvent.with_data(audio.FlushData()), id=event_id, source=hsm.id(commander))


def _dropped(metrics: metric_reader.Reader, reason: str) -> float:
    return sum(
        point.value
        for point in metrics.points("bot.devices.audio.speaker.dropped.count")
        if point.attributes.get("reason") == reason
    )


def test_speaker_is_generic_audio_output_peripheral() -> None:
    speaker = audio.Speaker()

    assert isinstance(speaker, Device)
    assert device_peripherals(speaker) == ()
    assert audio.Speaker.required_bot_abilities == ()


def test_speaker_dispatches_audio_output_to_target_device() -> None:
    async def run() -> None:
        speaker = audio.Speaker()
        target = RecordingDevice()
        _ = await mosfet.started(None, speaker, typing.cast(hsm.Model, speaker.model), hsm.Config(id="phone-speaker"))
        _ = await mosfet.started(None, target, typing.cast(hsm.Model, target.model), hsm.Config(id="physical-speaker"))
        target.events.clear()
        data = audio.OutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=44_100, channels=2)

        _ = await speaker.dispatch_audio_output(hsm.Context(), target, data, metadata=_TRACE)

        assert len(target.events) == 1
        event = target.events[0]
        assert event.name == audio.OutputEvent.name
        assert event.data == data
        assert event.source == "phone-speaker"
        assert event.target == "physical-speaker"
        assert event.metadata == _TRACE

    asyncio.run(run())


def test_speaker_transduces_attached_controller_signal_into_environment_sound() -> None:
    """A speaker converts signal into acoustic energy for the environment, the mirror of a microphone.

    ``devices.audio.output`` from the controller that attached becomes ``environment.sound`` sourced at
    the speaker, so bots hear playout as environment stimulus rather than as raw device product. A short
    chunk is one slice and plays at once.
    """

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], str]:
        (setup, _) = await room(frame=datetime.timedelta(milliseconds=100))
        outside = RecordingDevice()
        _ = await mosfet.started(None, outside, typing.cast(hsm.Model, outside.model), hsm.Config(id="outside"))
        _ = await setup.speaker.dispatch(setup.environment, output(setup.controller, pcm(1)))
        await wait_until(lambda: bool(setup.ear.named(SoundEvent)))
        return setup.ear.named(SoundEvent), outside.named(SoundEvent), hsm.id(setup.speaker)

    sounds, outside_events, speaker_id = asyncio.run(run())

    (event,) = sounds
    assert event.data == SoundData(audio=pcm(1), media_type="audio/pcm", sample_rate_hz=_RATE, channels=1)
    assert event.source == speaker_id
    assert event.target == "ear"
    assert event.metadata == _TRACE
    assert outside_events == []


def test_unattached_speaker_transduces_nothing() -> None:
    """An unwired speaker is silent, exactly as its physical counterpart is."""

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        speaker = audio.Speaker()
        ear = RecordingDevice()
        _ = await mosfet.started(environment, speaker, typing.cast(hsm.Model, speaker.model), hsm.Config(id="speaker"))
        _ = await mosfet.started(environment, ear, typing.cast(hsm.Model, ear.model), hsm.Config(id="ear"))

        _ = await speaker.dispatch(environment, output("controller", pcm(1)))
        await settle()

        return ear.named(SoundEvent)

    assert asyncio.run(run()) == []


def test_speaker_paces_a_chunk_as_frame_slices_against_its_schedule() -> None:
    """Each slice is one ``environment.sound`` at the moment it starts playing, never ahead of schedule.

    The timer to the next slice is measured against the speaker's own schedule, not from when the last
    timer happened to fire: a slice released 3 ms late shortens the next wait by 3 ms, so jitter never
    accumulates.
    """

    async def run() -> tuple[list[bytes], list[datetime.timedelta], list[int]]:
        time = FakeTime(100.0)
        (setup, _) = await room(time=time)
        chunk = pcm(1, fill=1) + pcm(1, fill=2) + pcm(1, fill=3)
        _ = await setup.speaker.dispatch(setup.environment, output(setup.controller, chunk))
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 1)
        heard_after: list[int] = [len(setup.ear.named(SoundEvent))]
        time.release(100.013)
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 2)
        heard_after.append(len(setup.ear.named(SoundEvent)))
        time.release(100.020)
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 3)
        heard_after.append(len(setup.ear.named(SoundEvent)))
        time.release(100.030)
        await settle()
        sounds = [typing.cast(SoundData, event.data).audio for event in setup.ear.named(SoundEvent)]
        return sounds, time.pacing(), heard_after

    sounds, pacing, heard_after = asyncio.run(run())

    assert sounds == [pcm(1, fill=1), pcm(1, fill=2), pcm(1, fill=3)]
    assert heard_after == [1, 2, 3]
    assert [round(duration.total_seconds(), 6) for duration in pacing] == [0.01, 0.007, 0.01]


def test_speaker_plays_in_real_time() -> None:
    async def run() -> tuple[list[float], float]:
        (setup, _) = await room()
        loop = asyncio.get_running_loop()
        heard: list[float] = []
        started = loop.time()
        _ = await setup.speaker.dispatch(setup.environment, output(setup.controller, pcm(4)))
        while len(heard) < 4:
            await wait_until(lambda: len(setup.ear.named(SoundEvent)) > len(heard))
            heard.append(loop.time())
        return heard, started

    heard, started = asyncio.run(run())

    for index, at in enumerate(heard):
        assert at - started >= index * _FRAME.total_seconds() - 0.001


def test_speaker_reports_a_mark_to_its_commander_once_its_last_slice_has_played() -> None:
    async def run() -> tuple[list[int], list[hsm.Event[typing.Any]], str]:
        time = FakeTime(0.0)
        (setup, _) = await room(time=time)
        _ = await setup.speaker.dispatch(setup.environment, output(setup.controller, pcm(2), mark="u1.1", event_id="a"))
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 1)
        played_after: list[int] = [len(setup.controller.named(audio.PlayedEvent))]
        time.release(0.010)
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 2)
        played_after.append(len(setup.controller.named(audio.PlayedEvent)))
        time.release(0.020)
        await wait_until(lambda: bool(setup.controller.named(audio.PlayedEvent)))
        await settle()
        return played_after, setup.controller.named(audio.PlayedEvent), hsm.id(setup.speaker)

    played_after, played, speaker_id = asyncio.run(run())

    assert played_after == [0, 0]
    (event,) = played
    assert event.data == audio.PlayedData(mark="u1.1")
    assert (event.source, event.target) == (speaker_id, "controller-0")


def test_flush_drops_only_the_flushers_unplayed_slices_and_reports_their_marks() -> None:
    """A shared speaker keeps a stream per commander: one controller cannot silence another."""

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], list[bytes]]:
        time = FakeTime(0.0)
        (setup, (first, second)) = await room(time=time, controllers=2)
        _ = await setup.speaker.dispatch(setup.environment, output(first, pcm(1, fill=1), mark="a1", event_id="a1"))
        _ = await setup.speaker.dispatch(setup.environment, output(second, pcm(1, fill=2), mark="b1", event_id="b1"))
        _ = await setup.speaker.dispatch(setup.environment, output(first, pcm(2, fill=3), mark="a2", event_id="a2"))
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 1)
        _ = await setup.speaker.dispatch(setup.environment, flush(first))
        await wait_until(lambda: bool(first.named(audio.FlushedEvent)))
        time.release(0.010)
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 2)
        time.release(0.020)
        await wait_until(lambda: bool(second.named(audio.PlayedEvent)))
        await settle()
        sounds = [typing.cast(SoundData, event.data).audio for event in setup.ear.named(SoundEvent)]
        return first.named(audio.FlushedEvent) + first.named(audio.PlayedEvent), second.named(audio.PlayedEvent), sounds

    first_told, second_told, sounds = asyncio.run(run())

    played, flushed = (
        [event for event in first_told if event.name == audio.PlayedEvent.name],
        [event for event in first_told if event.name == audio.FlushedEvent.name],
    )
    # The slice already playing when the flush came finishes; only what was queued is dropped.
    assert [event.data for event in played] == [audio.PlayedData(mark="a1")]
    (event,) = flushed
    assert event.data == audio.FlushedData(dropped_seconds=0.02, marks=("a2",))
    assert (event.id, event.target) == ("flush", "controller-0")
    assert [event.data for event in second_told] == [audio.PlayedData(mark="b1")]
    assert sounds == [pcm(1, fill=1), pcm(1, fill=2)]


def test_flush_of_a_silent_speaker_reports_nothing_dropped() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        (setup, _) = await room()
        _ = await setup.speaker.dispatch(setup.environment, flush(setup.controller))
        await wait_until(lambda: bool(setup.controller.named(audio.FlushedEvent)))
        return setup.controller.named(audio.FlushedEvent)

    (event,) = asyncio.run(run())

    assert event.data == audio.FlushedData(dropped_seconds=0.0, marks=())
    assert event.id == "flush"


def test_a_chunk_past_the_buffer_bound_is_dropped_whole_and_reported(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[bytes]]:
        time = FakeTime(0.0)
        (setup, _) = await room(time=time, max_buffered=datetime.timedelta(milliseconds=30))
        _ = await setup.speaker.dispatch(setup.environment, output(setup.controller, pcm(3, fill=1), event_id="fits"))
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 1)
        _ = await setup.speaker.dispatch(
            setup.environment, output(setup.controller, pcm(2, fill=2), mark="late", event_id="over")
        )
        await wait_until(lambda: bool(setup.controller.named(audio.FlushedEvent)))
        sounds = [typing.cast(SoundData, event.data).audio for event in setup.ear.named(SoundEvent)]
        return setup.controller.named(audio.FlushedEvent), sounds

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _dropped(metrics, "overflow")
        flushed, sounds = asyncio.run(run())
        after = _dropped(metrics, "overflow")

    (event,) = flushed
    assert event.data == audio.FlushedData(dropped_seconds=0.02, marks=("late",))
    assert event.id == "over"
    assert sounds == [pcm(1, fill=1)]
    assert after - before == 1


_REFUSED: dict[str, tuple[collections.abc.Callable[[hsm.Instance], hsm.Event[typing.Any]], str]] = {
    "compressed": (lambda controller: output(controller, pcm(1), media_type="audio/opus"), "media_type"),
    "no_rate": (lambda controller: output(controller, pcm(1), sample_rate_hz=None), "media_type"),
    "no_channels": (lambda controller: output(controller, pcm(1), channels=None), "media_type"),
    "partial_sample": (lambda controller: output(controller, b"\x01\x00\x02"), "media_type"),
    "stranger": (lambda controller: output("stranger", pcm(1)), "unattached"),
}


@pytest.mark.parametrize("case", sorted(_REFUSED))
def test_a_frame_the_speaker_cannot_pace_or_did_not_take_is_dropped_and_counted(
    case: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, reason = _REFUSED[case]

    async def run() -> list[hsm.Event[typing.Any]]:
        (setup, _) = await room()
        _ = await setup.speaker.dispatch(setup.environment, frame(setup.controller))
        await settle()
        return setup.ear.named(SoundEvent)

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _dropped(metrics, reason)
        heard = asyncio.run(run())
        after = _dropped(metrics, reason)

    assert heard == []
    assert after - before == 1


def test_detaching_the_last_controller_drops_what_is_still_queued() -> None:
    async def run() -> tuple[int, str]:
        time = FakeTime(0.0)
        (setup, _) = await room(time=time)
        _ = await setup.speaker.dispatch(setup.environment, output(setup.controller, pcm(3)))
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 1)
        _ = await setup.speaker.detach(
            setup.environment, attachment.DetachEvent.with_data(attachment.DetachData(actor=setup.controller))
        )
        await wait_until(lambda: setup.speaker.state() == "/Device/detached")
        _ = await setup.speaker.attach(
            setup.environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=setup.controller))
        )
        await wait_until(lambda: setup.speaker.state() == "/Device/attached/silent")
        await settle()
        return len(setup.ear.named(SoundEvent)), setup.speaker.state()

    heard, state = asyncio.run(run())

    assert heard == 1
    assert state == "/Device/attached/silent"


def test_a_controller_that_detaches_is_told_nothing_and_its_queued_audio_is_dropped(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], list[bytes]]:
        time = FakeTime(0.0)
        (setup, (first, second)) = await room(time=time, controllers=2)
        _ = await setup.speaker.dispatch(setup.environment, output(second, pcm(1, fill=2), mark="b1", event_id="b1"))
        _ = await setup.speaker.dispatch(setup.environment, output(second, pcm(1, fill=3), mark="b2", event_id="b2"))
        _ = await setup.speaker.dispatch(setup.environment, output(first, pcm(1, fill=1), mark="a1", event_id="a1"))
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 1)
        _ = await setup.speaker.detach(
            setup.environment, attachment.DetachEvent.with_data(attachment.DetachData(actor=second))
        )
        await wait_until(lambda: bool(second.named(attachment.DetachedEvent)))
        time.release(0.010)
        await wait_until(lambda: len(setup.ear.named(SoundEvent)) == 2)
        time.release(0.020)
        await wait_until(lambda: bool(first.named(audio.PlayedEvent)))
        await settle()
        sounds = [typing.cast(SoundData, event.data).audio for event in setup.ear.named(SoundEvent)]
        return second.named(audio.PlayedEvent), first.named(audio.PlayedEvent), sounds

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _dropped(metrics, "unattached")
        second_played, first_played, sounds = asyncio.run(run())
        after = _dropped(metrics, "unattached")

    assert second_played == []
    assert [event.data for event in first_played] == [audio.PlayedData(mark="a1")]
    assert sounds == [pcm(1, fill=2), pcm(1, fill=1)]
    assert after - before == 1


def test_speaker_does_not_override_firmware_model() -> None:
    """Playout lives on the shell, so a speaker declares no firmware model of its own.

    It still *has* firmware — the inherited default really starts. What is pinned here is only
    that this class does not declare its own, which is what keeps transduction on the shell.
    """

    assert audio.Speaker.firmware_model is Device.firmware_model
