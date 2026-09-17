from __future__ import annotations

from mosfet.devices import audio as audio_device

import asyncio
import collections.abc
import dataclasses
import struct
import typing

import pytest

from mosfet.providers.livekit import pcm_batch


@dataclasses.dataclass
class ManualTimerHandle:
    """Stand-in for ``asyncio.TimerHandle`` whose cancellation is inspectable."""

    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


@dataclasses.dataclass
class ManualLoop:
    """Deterministic stand-in for the batcher's injected ``loop``.

    Captures the idle callback ``RemotePcmBatcher`` schedules via ``call_later`` instead of
    letting it run on a real timer, so tests fire idle flushes on demand rather than sleeping.
    ``create_task`` still delegates to the real running loop so the resulting flush is a real,
    awaitable ``asyncio.Task``.
    """

    real_loop: asyncio.AbstractEventLoop
    pending: list[tuple[collections.abc.Callable[[], None], ManualTimerHandle]] = dataclasses.field(
        default_factory=list
    )
    last_task: asyncio.Task[None] | None = dataclasses.field(default=None, init=False)

    def call_later(self, _delay: float, callback: collections.abc.Callable[[], None]) -> ManualTimerHandle:
        handle = ManualTimerHandle()
        self.pending.append((callback, handle))
        return handle

    def create_task(self, coro: collections.abc.Coroutine[object, object, None]) -> asyncio.Task[None]:
        self.last_task = self.real_loop.create_task(coro)
        return self.last_task

    def fire_idle(self) -> None:
        callback, handle = self.pending.pop(0)
        if not handle.cancelled:
            callback()


@dataclasses.dataclass
class RecordingEmit:
    chunks: list[audio_device.InputData] = dataclasses.field(default_factory=list)

    async def __call__(self, chunk: audio_device.InputData) -> None:
        self.chunks.append(chunk)


def pcm_16bit_le(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def manual_loop() -> ManualLoop:
    return ManualLoop(real_loop=asyncio.get_running_loop())


def as_loop(loop: ManualLoop) -> asyncio.AbstractEventLoop:
    # ManualLoop only duck-types the subset of AbstractEventLoop the batcher actually calls
    # (call_later, create_task); route the cast through object since the two types otherwise
    # share no structural overlap for the type checker.
    return typing.cast(asyncio.AbstractEventLoop, typing.cast(object, loop))


# -- pcm_duration_ms -----------------------------------------------------------------


def test_livekit_pcm_batch_duration_ms_computes_from_16_bit_mono_samples() -> None:
    # 4 mono 16-bit samples = 8 bytes; at 8000 Hz that is 4 / 8000 s = 0.5 ms.
    pcm = pcm_16bit_le(1, 2, 3, 4)

    duration_ms = pcm_batch.pcm_duration_ms(pcm, sample_rate_hz=8000, channels=1)

    # 4000 / 8000 has an exact binary-float representation, so plain equality is safe here.
    assert duration_ms == 0.5


def test_livekit_pcm_batch_duration_ms_divides_by_channel_count() -> None:
    # Same 8 bytes as the mono case, but stereo halves the sample count (2 frames of 2 ch),
    # so duration halves too: 2 / 8000 s = 0.25 ms.
    pcm = pcm_16bit_le(1, 2, 3, 4)

    duration_ms = pcm_batch.pcm_duration_ms(pcm, sample_rate_hz=8000, channels=2)

    # 2000 / 8000 has an exact binary-float representation, so plain equality is safe here.
    assert duration_ms == 0.25


@pytest.mark.parametrize("sample_rate_hz", [0, -1])
def test_livekit_pcm_batch_duration_ms_guards_non_positive_sample_rate(sample_rate_hz: int) -> None:
    assert pcm_batch.pcm_duration_ms(pcm_16bit_le(1, 2), sample_rate_hz=sample_rate_hz, channels=1) == 0.0


@pytest.mark.parametrize("channels", [0, -1])
def test_livekit_pcm_batch_duration_ms_guards_non_positive_channels(channels: int) -> None:
    assert pcm_batch.pcm_duration_ms(pcm_16bit_le(1, 2), sample_rate_hz=8000, channels=channels) == 0.0


def test_livekit_pcm_batch_duration_ms_guards_empty_pcm() -> None:
    assert pcm_batch.pcm_duration_ms(b"", sample_rate_hz=8000, channels=1) == 0.0


# -- RemotePcmBatcher --------------------------------------------------------------------


def test_livekit_pcm_batch_batcher_flushes_at_max_utterance_threshold() -> None:
    async def scenario() -> RecordingEmit:
        emit = RecordingEmit()
        # 1000 Hz mono => 1 sample == 1 ms, so 1200 samples hits the default 1200.0 ms
        # max_utterance_ms threshold exactly (the ">=" boundary).
        batcher = pcm_batch.RemotePcmBatcher(emit=emit)
        pcm = pcm_16bit_le(*range(1200))

        await batcher.push(audio_device.InputData(audio=pcm, sample_rate_hz=1000, channels=1))
        return emit

    emit = asyncio.run(scenario())

    assert len(emit.chunks) == 1
    assert emit.chunks[0].audio == pcm_16bit_le(*range(1200))


def test_livekit_pcm_batch_batcher_does_not_flush_below_max_utterance_threshold() -> None:
    async def scenario() -> tuple[RecordingEmit, ManualLoop]:
        emit = RecordingEmit()
        loop = manual_loop()
        batcher = pcm_batch.RemotePcmBatcher(emit=emit, loop=as_loop(loop))
        pcm = pcm_16bit_le(*range(100))

        await batcher.push(audio_device.InputData(audio=pcm, sample_rate_hz=1000, channels=1))
        return emit, loop

    emit, loop = asyncio.run(scenario())

    assert emit.chunks == []
    # A buffer below threshold reschedules the idle flush instead of emitting immediately.
    assert len(loop.pending) == 1


def test_livekit_pcm_batch_batcher_flushes_on_idle_timeout_with_partial_buffer() -> None:
    async def scenario() -> RecordingEmit:
        emit = RecordingEmit()
        loop = manual_loop()
        batcher = pcm_batch.RemotePcmBatcher(emit=emit, idle_ms=250.0, loop=as_loop(loop))
        pcm = pcm_16bit_le(*range(50))

        await batcher.push(audio_device.InputData(audio=pcm, sample_rate_hz=1000, channels=1))
        assert emit.chunks == []

        loop.fire_idle()
        assert loop.last_task is not None
        await loop.last_task

        # The idle flush must drain the buffer, not just read it: an explicit flush() right
        # after should find nothing left to emit. If the buffer weren't cleared, this second
        # flush would re-emit the same PCM (double-transcribing an utterance).
        await batcher.flush()
        return emit

    emit = asyncio.run(scenario())

    assert len(emit.chunks) == 1
    assert emit.chunks[0].audio == pcm_16bit_le(*range(50))
    assert emit.chunks[0].sample_rate_hz == 1000
    assert emit.chunks[0].channels == 1


def test_livekit_pcm_batch_batcher_emits_chunk_with_accumulated_bytes_rate_channels_and_media_type() -> None:
    async def scenario() -> RecordingEmit:
        emit = RecordingEmit()
        batcher = pcm_batch.RemotePcmBatcher(emit=emit)
        first = pcm_16bit_le(*range(700))
        second = pcm_16bit_le(*range(700, 1200))

        await batcher.push(audio_device.InputData(audio=first, media_type="audio/pcm", sample_rate_hz=1000, channels=1))
        await batcher.push(
            audio_device.InputData(audio=second, media_type="audio/pcm", sample_rate_hz=1000, channels=1)
        )
        return emit

    emit = asyncio.run(scenario())

    assert len(emit.chunks) == 1
    chunk = emit.chunks[0]
    assert chunk.audio == pcm_16bit_le(*range(1200))
    assert chunk.sample_rate_hz == 1000
    assert chunk.channels == 1
    assert chunk.media_type == "audio/pcm"


def test_livekit_pcm_batch_batcher_flush_is_noop_on_empty_buffer() -> None:
    async def scenario() -> RecordingEmit:
        emit = RecordingEmit()
        batcher = pcm_batch.RemotePcmBatcher(emit=emit)

        await batcher.flush()
        return emit

    emit = asyncio.run(scenario())

    assert emit.chunks == []


def test_livekit_pcm_batch_batcher_flush_emits_partial_buffer_and_cancels_idle_timer() -> None:
    async def scenario() -> tuple[RecordingEmit, ManualLoop]:
        emit = RecordingEmit()
        loop = manual_loop()
        batcher = pcm_batch.RemotePcmBatcher(emit=emit, loop=as_loop(loop))
        pcm = pcm_16bit_le(*range(10))

        await batcher.push(audio_device.InputData(audio=pcm, sample_rate_hz=1000, channels=1))
        assert emit.chunks == []

        await batcher.flush()
        return emit, loop

    emit, loop = asyncio.run(scenario())

    assert len(emit.chunks) == 1
    assert emit.chunks[0].audio == pcm_16bit_le(*range(10))
    # The pending idle handle from the push is cancelled so it cannot fire a second, stale flush.
    _, handle = loop.pending[0]
    assert handle.cancelled is True


def test_livekit_pcm_batch_batcher_aclose_emits_pending_buffer_and_marks_closed() -> None:
    async def scenario() -> RecordingEmit:
        emit = RecordingEmit()
        batcher = pcm_batch.RemotePcmBatcher(emit=emit)
        pcm = pcm_16bit_le(*range(10))

        await batcher.push(audio_device.InputData(audio=pcm, sample_rate_hz=1000, channels=1))
        await batcher.aclose()
        return emit

    emit = asyncio.run(scenario())

    assert len(emit.chunks) == 1
    assert emit.chunks[0].audio == pcm_16bit_le(*range(10))


def test_livekit_pcm_batch_batcher_drops_pushes_after_close() -> None:
    async def scenario() -> RecordingEmit:
        emit = RecordingEmit()
        batcher = pcm_batch.RemotePcmBatcher(emit=emit)

        await batcher.aclose()
        # Push enough audio to cross the default 1200.0 ms max_utterance_ms threshold on its
        # own. If the closed flag were not actually honored, this single push would flush
        # immediately (same as test_..._flushes_at_max_utterance_threshold) and emit a chunk;
        # a working closed check must drop it before it ever reaches that accounting.
        await batcher.push(audio_device.InputData(audio=pcm_16bit_le(*range(1200)), sample_rate_hz=1000, channels=1))
        return emit

    emit = asyncio.run(scenario())

    assert emit.chunks == []


def test_livekit_pcm_batch_batcher_flushes_buffered_audio_when_format_changes_before_accumulating_new_format() -> None:
    async def scenario() -> RecordingEmit:
        emit = RecordingEmit()
        batcher = pcm_batch.RemotePcmBatcher(emit=emit)
        mono = pcm_16bit_le(*range(10))
        stereo = pcm_16bit_le(*range(10, 14))

        await batcher.push(audio_device.InputData(audio=mono, sample_rate_hz=1000, channels=1))
        await batcher.push(audio_device.InputData(audio=stereo, sample_rate_hz=1000, channels=2))
        return emit

    emit = asyncio.run(scenario())

    # The channel-count change flushes the buffered mono audio under the mono format before the
    # stereo bytes start a new buffer.
    assert len(emit.chunks) == 1
    assert emit.chunks[0].audio == pcm_16bit_le(*range(10))
    assert emit.chunks[0].channels == 1


def test_livekit_pcm_batch_batcher_forwards_every_sample_in_order_across_chunk_seams() -> None:
    """Chunk boundaries may fall anywhere, but no audio may be lost or reordered at one.

    Where the seams land stopped mattering once voice detection became streaming: the detector
    carries speech state across them, so a talkspurt split between two chunks is still heard in
    both. What still matters -- and is this assembler's whole remaining job -- is that the
    stream it hands on is the stream it was given. A dropped or reordered sample at a seam is
    audio the detector never gets to carry anything across.
    """

    async def scenario() -> RecordingEmit:
        emit = RecordingEmit()
        loop = manual_loop()
        # 1000 Hz mono => 1 sample == 1 ms, so the default cap flushes every 1200 samples.
        # Frames arrive back to back as LiveKit delivers them, so the idle flush never fires.
        batcher = pcm_batch.RemotePcmBatcher(emit=emit, loop=as_loop(loop))
        for start in range(0, 3000, 10):
            await batcher.push(
                audio_device.InputData(
                    audio=pcm_16bit_le(*range(start, start + 10)),
                    sample_rate_hz=1000,
                    channels=1,
                )
            )
        await batcher.flush()
        return emit

    emit = asyncio.run(scenario())

    forwarded = b"".join(chunk.audio for chunk in emit.chunks)
    assert forwarded == pcm_16bit_le(*range(3000))
    # More than one chunk, so the concatenation above actually crossed seams.
    assert len(emit.chunks) > 1
