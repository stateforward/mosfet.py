from .events import (
    FlushData,
    FlushedData,
    FlushedEvent,
    FlushEvent,
    OutputData,
    OutputEvent,
    PlayedData,
    PlayedEvent,
    routed_audio_event,
)

import collections
import collections.abc
import dataclasses
import datetime
import time
import typing

import hsm
import mosfet
from opentelemetry import metrics

from mosfet.device import Device
from mosfet.environment import SoundData, SoundEvent, Environment, space

_MEDIA_TYPE = "audio/pcm"
_SAMPLE_BYTES = 2
"""Linear PCM here is always s16le: two bytes per sample per channel."""

_DEFAULT_FRAME = datetime.timedelta(milliseconds=100)
_DEFAULT_MAX_BUFFERED = datetime.timedelta(seconds=30)

type _DropReason = typing.Literal["flushed", "overflow", "media_type", "unattached"]

_METER = metrics.get_meter("mosfet.devices.audio")
_BUFFERED = _METER.create_histogram(
    "bot.devices.audio.speaker.buffered",
    unit="s",
    description="Playout queued in a speaker and not yet started, sampled each time a chunk is queued.",
)
_DROPPED = _METER.create_counter(
    "bot.devices.audio.speaker.dropped.count",
    unit="{chunk}",
    description=(
        "Commanded chunks a speaker dropped without playing them whole, by reason: flushed by their commander "
        "(flushed), past the buffer bound (overflow), not pace-able linear PCM (media_type), or from a source "
        "that is not an attached controller (unattached)."
    ),
)


def _dropped(reason: _DropReason, chunks: int = 1) -> None:
    if chunks:
        _DROPPED.add(chunks, {"reason": reason})


def _nested(model: hsm.Model, state: str, *elements: hsm.Element) -> list[hsm.Element]:
    """``model``'s own elements, with ``elements`` nested inside its top-level state named ``state``."""

    return [
        dataclasses.replace(element, owned_elements=[*element.owned_elements, *elements])
        if element.qualified_name == state
        else element
        for element in model.owned_elements
    ]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Slice:
    """One frame-sized piece of a commanded chunk, queued until it plays.

    ``commander`` is the attached controller that sent the chunk: an address, dispatched to and never read.
    Only a chunk's last slice carries its ``mark`` (and the ``event_id`` of the output that carried it).
    """

    audio: bytes
    sample_rate_hz: int
    channels: int
    seconds: float
    chunk: int
    mark: str | None
    event_id: str
    commander: hsm.Instance
    metadata: dict[str, object]


class Speaker(Device):
    """Transducer that converts audio output signal into acoustic energy in the environment, in real time.

    The mirror of :class:`~mosfet.devices.audio.microphone.Microphone`: a microphone converts what
    it hears into signal on the way in, a speaker converts signal into sound on the way out.
    Both are transducers with no firmware and no routing decision of their own; the difference is
    only which way the attachment points. The controller that attached decides what to play.

    Like a real one, the speaker drains a bounded buffer in real time. A commanded chunk of linear PCM
    (``audio/pcm``, s16le, with a rate and channel count) is cut into slices of at most ``frame``, and
    each slice is one ``environment.sound`` broadcast at the moment it starts playing. The next slice
    is due when the last one has played, measured against a schedule kept on the injected ``clock``,
    so timer jitter never accumulates and playout is never ahead of schedule.

    A speaker attached by several controllers plays one stream per controller on one timeline, in
    arrival order, the way an audio server keeps per-client streams. What it says back goes only to
    the controller that commanded the audio:

    * ``devices.audio.played`` with a chunk's ``mark`` once that chunk's last slice has played;
    * ``devices.audio.flushed`` for ``devices.audio.flush``: the flusher's own unplayed slices are
      dropped and their marks listed (the slice already playing finishes, so a flush takes effect
      within one ``frame``); another controller's audio is untouched;
    * ``devices.audio.flushed`` for a chunk that would push the buffer past ``max_buffered``, which is
      dropped whole.

    A frame that is not pace-able PCM, or that comes from a source that is not an attached
    controller, is dropped and counted on ``bot.devices.audio.speaker.dropped.count``. Detaching the
    last controller drops whatever is still queued. Transduction happens only while attached, exactly
    as an unwired speaker is silent.
    """

    output_event: typing.ClassVar[hsm.Event[OutputData]] = OutputEvent

    _amplitude_db: float | None
    _frame_seconds: float
    _max_buffered_seconds: float
    _clock: collections.abc.Callable[[], float]
    # Durable playout state, owned by this machine: the buffer, the slice now playing, when it is due to
    # end, how much is queued, and a counter that tells chunks apart. Cleared when the speaker is detached.
    _queue: collections.deque[_Slice]
    _current: _Slice | None
    _next_at: float
    _queued_seconds: float
    _chunks: int

    def __init__(
        self,
        *,
        peripherals: collections.abc.Iterable[Device] = (),
        placement: space.Placement | None = None,
        amplitude_db: float | None = None,
        frame: datetime.timedelta = _DEFAULT_FRAME,
        max_buffered: datetime.timedelta = _DEFAULT_MAX_BUFFERED,
        clock: collections.abc.Callable[[], float] = time.monotonic,
    ) -> None:
        """Build a speaker.

        Args:
            peripherals: Devices this speaker powers.
            placement: Where this transducer is in the environment, if known.
            amplitude_db: How loud it plays at the reference distance; ``None`` carries everywhere.
            frame: The longest slice broadcast as one ``environment.sound``.
            max_buffered: The most playout the buffer holds; a chunk past it is dropped whole.
            clock: Monotonic seconds the playout schedule is kept against.

        Raises:
            ValueError: ``frame`` is not positive or ``max_buffered`` is shorter than ``frame``.
        """

        if frame <= datetime.timedelta(0):
            raise ValueError(f"Speaker frame must be positive, got {frame}.")
        if max_buffered < frame:
            raise ValueError(f"Speaker max_buffered must hold at least one frame, got {max_buffered}.")
        super().__init__(peripherals=peripherals, placement=placement)
        # How loud this transducer plays, measured at the reference distance. A handset earpiece
        # and a room speaker are different hardware; without a value the sound carries everywhere,
        # which is what keeps geometry opt-in.
        self._amplitude_db = amplitude_db
        self._frame_seconds = frame.total_seconds()
        self._max_buffered_seconds = max_buffered.total_seconds()
        self._clock = clock
        self._queue = collections.deque()
        self._current = None
        self._next_at = 0.0
        self._queued_seconds = 0.0
        self._chunks = 0

    @property
    def amplitude_db(self) -> float | None:
        """How loud this mouth plays at the reference distance, or None if unconstrained."""

        return self._amplitude_db

    @property
    def placement(self) -> space.Placement | None:
        """Where this transducer is in the environment, if known."""

        return self._placement

    @staticmethod
    def _commander(instance: "Speaker", event: hsm.Event[typing.Any]) -> hsm.Instance | None:
        """The attached controller that stamped itself as ``event.source`` (post-delivery correlation)."""

        return next((actor for actor in instance._attachments if hsm.id(actor) == event.source), None)

    @staticmethod
    def _seconds(data: OutputData) -> float | None:
        """Playout time of a pace-able linear PCM chunk, or None when it has no duration to pace."""

        rate, channels = data.sample_rate_hz, data.channels
        if data.media_type != _MEDIA_TYPE or rate is None or channels is None:
            return None
        frame_bytes = _SAMPLE_BYTES * channels
        if len(data.audio) % frame_bytes:
            return None
        return len(data.audio) // frame_bytes / rate

    @staticmethod
    def _is_unattached_frame(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, OutputData) and Speaker._commander(instance, event) is None

    @staticmethod
    def _is_frame(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, OutputData)

    @staticmethod
    def _overflows(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, OutputData) or Speaker._commander(instance, event) is None:
            return False
        seconds = Speaker._seconds(data)
        return seconds is not None and instance._queued_seconds + seconds > instance._max_buffered_seconds

    @staticmethod
    def _fits(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, OutputData) or Speaker._commander(instance, event) is None:
            return False
        seconds = Speaker._seconds(data)
        return seconds is not None and instance._queued_seconds + seconds <= instance._max_buffered_seconds

    @staticmethod
    def _is_attached_flush(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, FlushData) and Speaker._commander(instance, event) is not None

    @staticmethod
    def _has_queued(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return bool(instance._queue)

    @staticmethod
    def _until_next(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        """Time left on the schedule until the slice now playing has played; zero when already due."""

        del ctx, event
        return datetime.timedelta(seconds=max(0.0, instance._next_at - instance._clock()))

    @staticmethod
    def _report_flushed(
        ctx: hsm.Context,
        instance: "Speaker",
        commander: hsm.Instance,
        cause: hsm.Event[typing.Any],
        report: FlushedData,
    ) -> None:
        """Tell ``commander`` which of its audio was dropped unplayed, correlated by ``cause``'s id."""

        _ = hsm.dispatch(
            ctx,
            commander,
            dataclasses.replace(
                FlushedEvent.with_data(report),
                id=cause.id,
                source=hsm.id(instance),
                target=hsm.id(commander),
                metadata=dict(cause.metadata),
            ),
        )

    @staticmethod
    def _start_schedule(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """A silent speaker starts playing now: the schedule restarts at the clock."""

        del ctx, event
        instance._next_at = instance._clock()

    @staticmethod
    def _enqueue(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """Cut a commanded chunk into frame slices and queue them behind what is already queued."""

        del ctx
        data = event.data
        assert isinstance(data, OutputData) and data.sample_rate_hz is not None and data.channels is not None
        commander = Speaker._commander(instance, event)
        assert commander is not None
        rate, channels = data.sample_rate_hz, data.channels
        frame_bytes = _SAMPLE_BYTES * channels
        step = max(1, int(instance._frame_seconds * rate)) * frame_bytes
        instance._chunks += 1
        starts = range(0, len(data.audio), step)
        last = starts[-1]
        for start in starts:
            piece = data.audio[start : start + step]
            seconds = len(piece) // frame_bytes / rate
            instance._queue.append(
                _Slice(
                    audio=piece,
                    sample_rate_hz=rate,
                    channels=channels,
                    seconds=seconds,
                    chunk=instance._chunks,
                    mark=data.mark if start == last else None,
                    event_id=event.id if start == last else "",
                    commander=commander,
                    metadata=dict(event.metadata),
                )
            )
            instance._queued_seconds += seconds
        _BUFFERED.record(instance._queued_seconds)

    @staticmethod
    def _reject_overflow(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, OutputData)
        commander = Speaker._commander(instance, event)
        assert commander is not None
        seconds = Speaker._seconds(data)
        assert seconds is not None
        marks = () if data.mark is None else (data.mark,)
        Speaker._report_flushed(ctx, instance, commander, event, FlushedData(dropped_seconds=seconds, marks=marks))
        _dropped("overflow")

    @staticmethod
    def _drop_unattached(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _dropped("unattached")

    @staticmethod
    def _drop_unpaceable(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _dropped("media_type")

    @staticmethod
    def _flush_nothing(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """A silent speaker has nothing of anyone's left to drop; it says so."""

        commander = Speaker._commander(instance, event)
        assert commander is not None
        Speaker._report_flushed(ctx, instance, commander, event, FlushedData(dropped_seconds=0.0, marks=()))

    @staticmethod
    def _flush(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """Drop the flusher's queued slices, keep everyone else's, and report what was dropped."""

        commander = Speaker._commander(instance, event)
        assert commander is not None
        dropped = [piece for piece in instance._queue if piece.commander is commander]
        instance._queue = collections.deque(piece for piece in instance._queue if piece.commander is not commander)
        instance._queued_seconds = sum(piece.seconds for piece in instance._queue)
        report = FlushedData(
            dropped_seconds=sum(piece.seconds for piece in dropped),
            marks=tuple(piece.mark for piece in dropped if piece.mark is not None),
        )
        Speaker._report_flushed(ctx, instance, commander, event, report)
        _dropped("flushed", len({piece.chunk for piece in dropped}))

    @staticmethod
    def _play_head(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """Start the next slice: broadcast it as ``environment.sound`` and schedule its end."""

        del ctx, event
        piece = instance._queue.popleft()
        instance._queued_seconds -= piece.seconds
        instance._current = piece
        instance._next_at += piece.seconds
        placement = instance._placement
        sound = dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=piece.audio,
                    media_type=_MEDIA_TYPE,
                    sample_rate_hz=piece.sample_rate_hz,
                    channels=piece.channels,
                    amplitude_db=instance._amplitude_db,
                )
            ),
            source=hsm.id(instance),
            metadata=dict(piece.metadata),
        )
        # The environment works out what reaches whom; this only says how loud, and from where. The
        # speaker's own lifetime context resolves the scope it was started in, whoever dispatched.
        _ = Environment.from_context(instance.context()).broadcast(
            sound, origin=None if placement is None else placement.position
        )

    @staticmethod
    def _finish_slice(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """The playing slice has played: report its mark, then let go of queued audio nobody commands now.

        Both go by the controllers attached now: a controller that detached is told nothing, and its
        audio still queued is dropped as it reaches the head of the queue.
        """

        del event
        piece = instance._current
        instance._current = None
        attached = tuple(instance._attachments)
        if piece is not None and piece.mark is not None and any(piece.commander is actor for actor in attached):
            _ = hsm.dispatch(
                ctx,
                piece.commander,
                dataclasses.replace(
                    PlayedEvent.with_data(PlayedData(mark=piece.mark)),
                    id=piece.event_id,
                    source=hsm.id(instance),
                    target=hsm.id(piece.commander),
                    metadata=dict(piece.metadata),
                ),
            )
        stale: set[int] = set()
        while instance._queue and not any(instance._queue[0].commander is actor for actor in attached):
            stale.add(instance._queue[0].chunk)
            instance._queued_seconds -= instance._queue.popleft().seconds
        _dropped("unattached", len(stale))

    @staticmethod
    def _abandon_playout(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """Leaving ``attached`` (the last controller detached, or the speaker stopped) drops what is queued."""

        del ctx, event
        _dropped("unattached", len({piece.chunk for piece in instance._queue}))
        instance._queue.clear()
        instance._current = None
        instance._queued_seconds = 0.0

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Device",
        # The device lifecycle is Device's; playout nests inside its ``attached`` state, which a
        # redefinition cannot reopen, so the speaker rebuilds the model from Device's own elements.
        *_nested(
            typing.cast(hsm.Model, Device.model),
            "attached",
            hsm.initial(hsm.target("silent")),
            hsm.exit(_abandon_playout),
            hsm.transition(hsm.on(OutputEvent), hsm.guard(_is_unattached_frame), hsm.effect(_drop_unattached)),
            hsm.transition(hsm.on(OutputEvent), hsm.guard(_overflows), hsm.effect(_reject_overflow)),
            hsm.transition(hsm.on(OutputEvent), hsm.guard(_is_frame), hsm.effect(_drop_unpaceable)),
            hsm.state(
                "silent",
                hsm.transition(
                    hsm.on(OutputEvent),
                    hsm.guard(_fits),
                    hsm.effect(_start_schedule, _enqueue),
                    hsm.target("../playing"),
                ),
                hsm.transition(hsm.on(FlushEvent), hsm.guard(_is_attached_flush), hsm.effect(_flush_nothing)),
            ),
            hsm.state(
                "playing",
                hsm.entry(_play_head),
                hsm.transition(hsm.on(OutputEvent), hsm.guard(_fits), hsm.effect(_enqueue)),
                hsm.transition(hsm.on(FlushEvent), hsm.guard(_is_attached_flush), hsm.effect(_flush)),
                hsm.transition(hsm.after(_until_next), hsm.effect(_finish_slice), hsm.target("../next")),
            ),
            hsm.choice(
                "next",
                hsm.transition(hsm.guard(_has_queued), hsm.target("playing")),
                hsm.transition(hsm.target("silent")),
            ),
        ),
    )

    def dispatch_audio_output(
        self,
        ctx: hsm.Context,
        target: hsm.Instance,
        data: OutputData,
        *,
        metadata: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.Awaitable[bool]:
        event = routed_audio_event(self.output_event, data, source=self, target=target, metadata=metadata)
        return target.dispatch(ctx, event)
