"""Accumulate remote LiveKit PCM frames into chunks worth elevating as environment sound.

LiveKit delivers ~10 ms frames. Running the whole perception pipeline once per frame is the
cost this exists to avoid; it is a transport-shaped concern, which is why it lives at the
provider boundary (ingress-only, no shared Environment / direct wiring).

Chunk boundaries are deliberately *not* meaningful. Voice detection streams, so it carries
speech state across whichever seams this assembler happens to produce, and a talkspurt split
between two chunks stays audible in both. Nothing downstream may go back to treating a chunk
as a self-contained clip to be judged on its own.
"""

from __future__ import annotations

from mosfet.devices import audio

import asyncio
import logging
import typing
import collections.abc
import dataclasses

from opentelemetry import metrics

from mosfet.telemetry import span


_LOG = logging.getLogger(__name__)
_SCOPE = "mosfet.providers.livekit"
_COMPONENT = "livekit.pcm_batch"

_METER = metrics.get_meter(_SCOPE)
_DROPS = _METER.create_counter(
    "bot.provider.livekit.pcm_batch.dropped.count",
    unit="{chunk}",
    description="Count of remote PCM chunks dropped by the LiveKit batcher before assembly.",
)
_IDLE_FLUSH_FAILURES = _METER.create_counter(
    "bot.provider.livekit.pcm_batch.idle_flush.failure.count",
    unit="{failure}",
    description="Count of LiveKit batcher idle flushes whose emit failed.",
)


def pcm_duration_ms(pcm: bytes, *, sample_rate_hz: int, channels: int) -> float:
    if sample_rate_hz <= 0 or channels <= 0 or not pcm:
        return 0.0
    samples = len(pcm) // (2 * channels)
    return 1000.0 * samples / float(sample_rate_hz)


@dataclasses.dataclass(slots=True)
class RemotePcmBatcher:
    """Accumulate remote PCM and emit utterance-sized ``InputData`` chunks.

    Flush rules (first match):
    - buffered duration >= ``max_utterance_ms``
    - idle for ``idle_ms`` with a non-empty buffer (end of talkspurts / single frames)
    """

    emit: collections.abc.Callable[[audio.InputData], collections.abc.Awaitable[None]]
    max_utterance_ms: float = 1_200.0
    idle_ms: float = 250.0
    loop: asyncio.AbstractEventLoop | None = None

    _buffer: bytearray = dataclasses.field(default_factory=bytearray, init=False, repr=False)
    _sample_rate_hz: int | None = dataclasses.field(default=None, init=False)
    _channels: int | None = dataclasses.field(default=None, init=False)
    _media_type: str | None = dataclasses.field(default="audio/pcm", init=False)
    _idle_handle: asyncio.TimerHandle | None = dataclasses.field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock, init=False, repr=False)
    _closed: bool = dataclasses.field(default=False, init=False)

    async def _emit_traced(self, chunk: audio.InputData, reason: str) -> None:
        """Emit one assembled chunk under a span naming why the seam fell here.

        This is the unit that reaches perception, so it is the span a sound is followed by:
        how long it was, how many samples, and which flush rule cut it. Never the PCM.
        ``reason`` is a closed vocabulary chosen by the call sites below.
        """

        with span.operation(
            "bot.provider.livekit.pcm_batch.flush",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="flush",
            attributes={"bot.pcm_batch.reason": reason},
        ) as active:
            sample_rate_hz = chunk.sample_rate_hz
            channels = chunk.channels
            if sample_rate_hz is not None and channels is not None:
                active.set_attribute("bot.audio.samples", len(chunk.audio) // (2 * channels))
                active.set_attribute(
                    "bot.audio.duration_ms",
                    round(pcm_duration_ms(chunk.audio, sample_rate_hz=sample_rate_hz, channels=channels)),
                )
            await self.emit(chunk)

    def _note_drop(self, data: audio.InputData, *, reason: str) -> None:
        """Record one chunk dropped before assembly as metric + span (never silent).

        A drop is data the detector never hears, so the counter answers "how
        much" and the span answers "why". ``reason`` is a closed vocabulary;
        byte counts ride the span as numbers, never as attribute values.
        """

        try:
            _DROPS.add(1, attributes={"bot.component.name": _COMPONENT, "bot.pcm_batch.drop_reason": reason})
        except Exception:
            pass
        with span.operation(
            "bot.provider.livekit.pcm_batch.drop",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="push",
            attributes={"bot.pcm_batch.drop_reason": reason},
        ) as active:
            active.set_attribute("bot.audio.bytes", len(data.audio))

    async def push(self, data: audio.InputData) -> None:
        if not data.audio:
            return
        if self._closed:
            # Closed with data is loss, not a no-op: the producer outlived the
            # batcher and these bytes go nowhere. Count them where an operator
            # can see them instead of returning silently.
            self._note_drop(data, reason="closed")
            return
        to_emit: list[tuple[audio.InputData, str]] = []
        async with self._lock:
            if self._closed:
                dropped = True
            else:
                dropped = False
                format_changed = self._sample_rate_hz is not None and (
                    data.sample_rate_hz != self._sample_rate_hz
                    or data.channels != self._channels
                    or data.media_type != self._media_type
                )
                if format_changed and self._buffer:
                    to_emit.append((self._snapshot_unlocked(), "format_changed"))
                    self._clear_unlocked()
                self._sample_rate_hz = data.sample_rate_hz
                self._channels = data.channels
                self._media_type = data.media_type
                self._buffer.extend(data.audio)
                duration_ms = pcm_duration_ms(
                    bytes(self._buffer),
                    # Same invariant `_snapshot_unlocked` asserts: pushed frames always carry
                    # a concrete format, so the optional fields are populated here.
                    sample_rate_hz=typing.cast(int, data.sample_rate_hz),
                    channels=typing.cast(int, data.channels),
                )
                if duration_ms >= self.max_utterance_ms:
                    to_emit.append((self._snapshot_unlocked(), "max_utterance"))
                    self._clear_unlocked()
                    self._cancel_idle_unlocked()
                else:
                    self._reschedule_idle_unlocked()
        if dropped:
            # Lost the race with aclose after the pre-check: same loss, same
            # record, kept outside the lock so telemetry stays off it.
            self._note_drop(data, reason="closed")
            return
        for chunk, reason in to_emit:
            await self._emit_traced(chunk, reason)

    async def flush(self) -> None:
        async with self._lock:
            pending = self._snapshot_unlocked() if self._buffer else None
            self._clear_unlocked()
            self._cancel_idle_unlocked()
        if pending is not None:
            await self._emit_traced(pending, "flush")

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            pending = self._snapshot_unlocked() if self._buffer else None
            self._clear_unlocked()
            self._cancel_idle_unlocked()
        if pending is not None:
            await self._emit_traced(pending, "closed")

    def _snapshot_unlocked(self) -> audio.InputData:
        assert self._sample_rate_hz is not None
        assert self._channels is not None
        return audio.InputData(
            audio=bytes(self._buffer),
            media_type=self._media_type,
            sample_rate_hz=self._sample_rate_hz,
            channels=self._channels,
        )

    def _clear_unlocked(self) -> None:
        self._buffer.clear()

    def _cancel_idle_unlocked(self) -> None:
        if self._idle_handle is not None:
            self._idle_handle.cancel()
            self._idle_handle = None

    def _reschedule_idle_unlocked(self) -> None:
        self._cancel_idle_unlocked()
        loop = self.loop or asyncio.get_running_loop()
        delay = max(self.idle_ms, 1.0) / 1000.0

        def _on_idle() -> None:
            self._idle_handle = None
            flushed = asyncio.ensure_future(self._idle_flush(), loop=loop)

            def _surface_failure(done: asyncio.Future[None]) -> None:
                # Same hazard as the uplink publish: nothing awaits this flush, so a failing
                # emit would lose the buffered audio and leave only an unretrieved-task warning.
                # This batcher has no machine to report to, so the normalized kind on a
                # counter, a span, and one log line is the floor. The raw error text is
                # never logged: it is unbounded-cardinality, the kind is not.
                if done.cancelled() or done.exception() is None:
                    return
                error = done.exception()
                assert error is not None
                kind = span.failure_kind(error, "publish_failed")
                try:
                    _IDLE_FLUSH_FAILURES.add(
                        1,
                        attributes={"bot.component.name": _COMPONENT, "bot.failure.kind": kind},
                    )
                except Exception:
                    pass
                with span.operation(
                    "bot.provider.livekit.pcm_batch.idle_flush",
                    scope=_SCOPE,
                    component=_COMPONENT,
                    stage="flush",
                ) as failed:
                    span.record_failure(failed, kind)
                _LOG.error("livekit remote pcm idle flush failed kind=%s", kind)

            flushed.add_done_callback(_surface_failure)

        self._idle_handle = loop.call_later(delay, _on_idle)

    async def _idle_flush(self) -> None:
        async with self._lock:
            if self._closed or not self._buffer:
                return
            pending = self._snapshot_unlocked()
            self._clear_unlocked()
        await self._emit_traced(pending, "idle")


__all__ = [
    "RemotePcmBatcher",
    "pcm_duration_ms",
]
