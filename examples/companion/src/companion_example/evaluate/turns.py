"""Taking turns with the companion the way a person in the room does, and measuring each exchange.

The evaluator is a room's far side (``room.Room``) whose devices are this module's :class:`Microphone` and
:class:`Speaker`. The speaker models playback: the companion's voice reaches the room faster than it is spoken, so each
chunk plays after the one before it, for as long as it lasts, and the speaker tells whoever listens
(:class:`ChunkData`) when each chunk starts and ends playing. The microphone streams in real time, as a sound
card does: what the person says, then silence.

:class:`Turns` is the person's turn-taking, one exchange at a time. Given a line (:data:`SayEvent`) it waits until
the companion's voice has died away (the speaker quiet) and the echo window after it has passed
(``recognition.ECHO_SECONDS`` by default: the companion's ear hears a voice nobody knows within it as possibly its own),
and until the line's ``not_before``; then it says the line. Then it waits for the companion: no sound within
``reply_timeout_seconds`` is no reply; otherwise the reply lasts until the companion has been quiet for
``pause_seconds``. A line with a ``cut_in`` is a person who talks over the companion: ``cut_in_after_seconds`` into
the companion's reply, they say it while the companion is still playing. Each exchange is told once, as
:class:`ExchangeData`, to ``on_exchange``; :func:`measure` reads it.

Telemetry (meter scope ``companion_example.evaluate``): ``bot.example.evaluate.exchanges.count{outcome}``
(``replied``, ``no_reply``). Nothing said, heard or played is an attribute.
"""

import asyncio
import collections.abc
import datetime
import typing

import hsm
import mosfet
import pydantic
from mosfet.abilities.listening import recognition
from mosfet.telemetry.hsm import Traced
from opentelemetry import metrics

from companion_example import room

SCOPE: typing.Final = "companion_example.evaluate"

type Clock = collections.abc.Callable[[], float]
"""Monotonic time in seconds; ``time.monotonic`` outside tests."""
type Outcome = typing.Literal["replied", "no_reply"]

_EXCHANGES = metrics.get_meter(SCOPE).create_counter(
    "bot.example.evaluate.exchanges.count",
    unit="{exchange}",
    description="Exchanges with the companion by outcome: replied, or no_reply within the reply timeout.",
)
_SAMPLE_BYTES = 2


class ChunkData(pydantic.BaseModel):
    """One chunk of the companion's voice the room's speaker played, and when (clock seconds)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    arrived: float = pydantic.Field(description="When the chunk reached the speaker.")
    start: float = pydantic.Field(description="When it starts playing: once it arrived and what came before ended.")
    end: float = pydantic.Field(description="When it has finished playing.")
    pcm: bytes = pydantic.Field(repr=False, description="The chunk, s16le.")
    sample_rate_hz: int = pydantic.Field(gt=0, examples=[24_000])
    channels: int = pydantic.Field(gt=0, examples=[1])


class SpokenData(pydantic.BaseModel):
    """When a line said into the room began and ended (clock seconds)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    start: float
    end: float


class SayData(pydantic.BaseModel):
    """A line for the person to say once it is their turn, and what they cut in with while the companion answers."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    exchange: int = pydantic.Field(ge=0, description="Which exchange this is; the one ``on_exchange`` reports.")
    pcm: bytes = pydantic.Field(repr=False, description="The line, 16 kHz mono s16le, as the microphone hears it.")
    cut_in: bytes | None = pydantic.Field(
        default=None, repr=False, description="What the person says over the companion's reply, 16 kHz mono s16le."
    )
    not_before: float = pydantic.Field(
        default=0.0, description="The earliest the line is said (clock seconds): the end of a break."
    )


class ExchangeData(pydantic.BaseModel):
    """One exchange: the person's line, what the companion played from then until it paused, and when."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    exchange: int
    said_from: float = pydantic.Field(description="When the person began their line.")
    said_until: float = pydantic.Field(description="When the person finished it.")
    cut_in_from: float | None = pydantic.Field(description="When the person cut in over the companion, if they did.")
    cut_in_until: float | None
    chunks: tuple[ChunkData, ...] = pydantic.Field(description="the companion's voice from the line on, in order.")
    outcome: Outcome


class Measures(pydantic.BaseModel):
    """What one exchange measures."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    replied: bool = pydantic.Field(description="the companion played anything from the person's line on.")
    latency_seconds: float | None = pydantic.Field(
        description="From the end of the person's line to the first of the companion's voice that began after it."
    )
    reply_seconds: float = pydantic.Field(description="How long the companion's voice played in the exchange.")
    talk_over: bool = pydantic.Field(description="the companion's voice played while the person was saying their line.")
    cut_in_seconds: float | None = pydantic.Field(
        description="How long the companion kept playing after the person cut in; none when nobody cut in."
    )


def measure(exchange: ExchangeData) -> Measures:
    """Measure ``exchange``: a pure reading of its times."""

    chunks = exchange.chunks
    after = [chunk for chunk in chunks if chunk.start >= exchange.said_until]
    cut_in = exchange.cut_in_from
    return Measures(
        replied=bool(chunks),
        latency_seconds=after[0].start - exchange.said_until if after else None,
        reply_seconds=sum(chunk.end - chunk.start for chunk in chunks),
        talk_over=any(chunk.start < exchange.said_until and chunk.end > exchange.said_from for chunk in chunks),
        cut_in_seconds=None if cut_in is None or not chunks else max(0.0, chunks[-1].end - cut_in),
    )


class Timing(pydantic.BaseModel):
    """How the person times their turns."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    echo_seconds: float = pydantic.Field(
        default=recognition.ECHO_SECONDS,
        ge=0,
        description="How long after the companion's voice has died away the person waits before speaking.",
    )
    pause_seconds: float = pydantic.Field(
        default=1.5, gt=0, description="How long the companion is quiet before its reply counts as over."
    )
    reply_timeout_seconds: float = pydantic.Field(
        default=30.0, gt=0, description="How long the person waits for the companion to begin before it is no reply."
    )
    cut_in_after_seconds: float = pydantic.Field(
        default=2.0, gt=0, description="How far into the companion's reply a person who cuts in begins talking over it."
    )


class Mouth(typing.Protocol):
    """The person's voice into the room: says a line through the microphone, returning when it began and ended."""

    def say(self, pcm: bytes, /) -> collections.abc.Awaitable[SpokenData]: ...


class Speaker:
    """The room's speaker as the evaluator hears it: it tells ``heard`` when each chunk of the companion's voice plays.

    A ``room.Speaker``. Each chunk starts playing when it arrives or when the one before it ends, whichever is
    later, and plays for as long as it lasts; nothing is played aloud.
    """

    _clock: Clock
    _heard: collections.abc.Callable[[ChunkData], object]
    _quiet_at: float

    def __init__(self, *, clock: Clock, heard: collections.abc.Callable[[ChunkData], object]) -> None:
        self._clock = clock
        self._heard = heard
        self._quiet_at = 0.0

    def play(self, pcm: bytes, *, sample_rate_hz: int, channels: int) -> None:
        """Play ``pcm`` after what is already playing, and tell when it does."""

        now = self._clock()
        start = max(now, self._quiet_at)
        end = start + len(pcm) / (_SAMPLE_BYTES * channels * sample_rate_hz)
        self._quiet_at = end
        _ = self._heard(
            ChunkData(arrived=now, start=start, end=end, pcm=pcm, sample_rate_hz=sample_rate_hz, channels=channels)
        )

    def stop(self) -> None:
        """Stop playing: the speaker is quiet from now on."""

        self._quiet_at = min(self._quiet_at, self._clock())


class Microphone:
    """The room's microphone, scripted: in real time, as a sound card does, what it is told to say, then silence.

    A ``room.Microphone`` and a :class:`Mouth`. Every ``block_seconds`` it hands the room a block of 16 kHz mono
    s16le, from a task of its own that ``start`` begins and ``stop`` ends.
    """

    streaming: asyncio.Event
    _clock: Clock
    _block: int
    _block_seconds: float
    _pending: bytearray
    _stream: asyncio.Task[None] | None
    # The line being said: when its first block was handed, and who waits for its last.
    _began: float | None
    _said: asyncio.Future[SpokenData] | None

    def __init__(self, *, clock: Clock, block_seconds: float = room.MICROPHONE_BLOCK_SECONDS) -> None:
        self.streaming = asyncio.Event()
        self._clock = clock
        self._block_seconds = block_seconds
        self._block = _SAMPLE_BYTES * round(block_seconds * room.MICROPHONE_RATE_HZ)
        self._pending = bytearray()
        self._stream = None
        self._began = None
        self._said = None

    def start(self, heard: collections.abc.Callable[[bytes], None], /) -> None:
        """Start streaming to ``heard``: silence until something is said."""

        self._stream = asyncio.get_running_loop().create_task(
            self._capture(heard), name="companion-evaluate-microphone"
        )
        self.streaming.set()

    def stop(self) -> None:
        """Stop streaming; a line being said is never finished."""

        stream, self._stream = self._stream, None
        if stream is not None:
            _ = stream.cancel()
        self.streaming.clear()

    async def say(self, pcm: bytes, /) -> SpokenData:
        """Say ``pcm`` (16 kHz mono s16le) into the room; return when its first and last blocks were handed on."""

        said: asyncio.Future[SpokenData] = asyncio.get_running_loop().create_future()
        self._pending += pcm
        self._began = None
        self._said = said
        return await said

    async def _capture(self, heard: collections.abc.Callable[[bytes], None]) -> None:
        loop = asyncio.get_running_loop()
        due = loop.time()
        while True:
            chunk = bytes(self._pending[: self._block])
            del self._pending[: self._block]
            now = self._clock()
            if chunk and self._began is None:
                self._began = now
            heard(chunk + bytes(self._block - len(chunk)))
            said = self._said
            if chunk and not self._pending and said is not None:
                self._said = None
                if not said.done():
                    said.set_result(SpokenData(start=self._began or now, end=now))
            due += self._block_seconds
            await asyncio.sleep(max(0.0, due - loop.time()))


PlayedEvent = hsm.Event[ChunkData](name="companion_example.evaluate.played", schema=ChunkData)
"""A chunk of the companion's voice the room's speaker plays: the speaker tells the turn-taking machine."""
SayEvent = hsm.Event[SayData](name="companion_example.evaluate.say", schema=SayData)
"""A line for the person to say once it is their turn."""


class _SpokenData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    exchange: int
    spoken: SpokenData


# Private: each is dispatched by this class's own activities to itself only.
_SpokenEvent = hsm.Event[_SpokenData](
    name="companion_example.evaluate.spoken", kind=hsm.CompletionEventKind, schema=_SpokenData
)
_CutInSpokenEvent = hsm.Event[_SpokenData](
    name="companion_example.evaluate.cut_in_spoken", kind=hsm.CompletionEventKind, schema=_SpokenData
)


class Turns(Traced):
    """The person's turn-taking with the companion, one exchange at a time.

    ``idle`` waits for a line. ``busy/settling`` waits for the companion's voice to die away, the echo window after it
    and the line's ``not_before``; every chunk the companion plays meanwhile starts the wait again. ``busy/speaking``
    says the line. ``busy/awaiting`` waits ``reply_timeout_seconds`` for the companion's voice: none is no reply.
    ``busy/replying/hearing`` hears the companion until it has been quiet for ``pause_seconds``; with a cut-in,
    ``busy/replying/before_cut_in`` waits ``cut_in_after_seconds`` into the reply and, if the companion is still playing
    then, ``busy/replying/cutting_in`` says it over the companion. A line given while busy waits its turn. Each exchange
    is told to ``on_exchange``, once.
    """

    _mouth: Mouth
    _clock: Clock
    _timing: Timing
    _on_exchange: collections.abc.Callable[[ExchangeData], None]
    # When the speaker is quiet: written by every chunk it plays.
    _quiet_at: float
    # The exchange in hand, from its line to its end; written only by the transitions that take, say and end it.
    _line: SayData | None
    _spoken: SpokenData | None
    _cut_in: SpokenData | None
    _chunks: list[ChunkData]

    def __init__(
        self,
        *,
        mouth: Mouth,
        clock: Clock,
        on_exchange: collections.abc.Callable[[ExchangeData], None],
        timing: Timing | None = None,
    ) -> None:
        super().__init__()
        self._mouth = mouth
        self._clock = clock
        self._timing = Timing() if timing is None else timing
        self._on_exchange = on_exchange
        self._quiet_at = 0.0
        self._line = None
        self._spoken = None
        self._cut_in = None
        self._chunks = []

    # -- guards and timers ------------------------------------------------------------------------

    @staticmethod
    def _until_settled(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        line = instance._line
        assert line is not None
        due = max(instance._quiet_at + instance._timing.echo_seconds, line.not_before)
        return datetime.timedelta(seconds=max(0.0, due - instance._clock()))

    @staticmethod
    def _until_paused(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        due = instance._quiet_at + instance._timing.pause_seconds
        return datetime.timedelta(seconds=max(0.0, due - instance._clock()))

    @staticmethod
    def _until_cut_in(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        due = instance._chunks[0].start + instance._timing.cut_in_after_seconds
        return datetime.timedelta(seconds=max(0.0, due - instance._clock()))

    @staticmethod
    def _reply_timeout(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return datetime.timedelta(seconds=instance._timing.reply_timeout_seconds)

    @staticmethod
    def _is_current(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, _SpokenData) and instance._line is not None and data.exchange == instance._line.exchange

    @staticmethod
    def _is_current_while_companion_plays(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> bool:
        return Turns._is_current(ctx, instance, event) and bool(instance._chunks)

    @staticmethod
    def _companion_plays(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._quiet_at > instance._clock()

    @staticmethod
    def _cuts_in(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._line is not None and instance._line.cut_in is not None

    # -- effects ----------------------------------------------------------------------------------

    @staticmethod
    def _keep(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> None:
        """Keep when the speaker is quiet."""

        del ctx
        data = event.data
        assert isinstance(data, ChunkData)
        instance._quiet_at = max(instance._quiet_at, data.end)

    @staticmethod
    def _collect(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> None:
        """Keep when the speaker is quiet, and the chunk as the exchange's."""

        Turns._keep(ctx, instance, event)
        data = event.data
        assert isinstance(data, ChunkData)
        instance._chunks.append(data)

    @staticmethod
    def _take(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, SayData)
        instance._line = data
        instance._spoken = None
        instance._cut_in = None
        instance._chunks = []

    @staticmethod
    def _spoke(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _SpokenData)
        instance._spoken = data.spoken

    @staticmethod
    def _cut(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _SpokenData)
        instance._cut_in = data.spoken

    @staticmethod
    def _finish(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> None:
        """Tell the exchange, once, and let go of it."""

        del ctx, event
        line, spoken, cut_in = instance._line, instance._spoken, instance._cut_in
        assert line is not None and spoken is not None
        outcome: Outcome = "replied" if instance._chunks else "no_reply"
        exchange = ExchangeData(
            exchange=line.exchange,
            said_from=spoken.start,
            said_until=spoken.end,
            cut_in_from=None if cut_in is None else cut_in.start,
            cut_in_until=None if cut_in is None else cut_in.end,
            chunks=tuple(instance._chunks),
            outcome=outcome,
        )
        instance._line = None
        instance._chunks = []
        _EXCHANGES.add(1, {"outcome": outcome})
        instance._on_exchange(exchange)

    # -- activities -------------------------------------------------------------------------------

    @staticmethod
    async def _run_say(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> None:
        del event
        line = instance._line
        assert line is not None
        spoken = await instance._mouth.say(line.pcm)
        _ = hsm.dispatch(ctx, instance, _SpokenEvent.with_data(_SpokenData(exchange=line.exchange, spoken=spoken)))

    @staticmethod
    async def _run_cut_in(ctx: hsm.Context, instance: "Turns", event: hsm.Event[typing.Any]) -> None:
        del event
        line = instance._line
        assert line is not None and line.cut_in is not None
        spoken = await instance._mouth.say(line.cut_in)
        _ = hsm.dispatch(ctx, instance, _CutInSpokenEvent.with_data(_SpokenData(exchange=line.exchange, spoken=spoken)))

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Turns",
        hsm.initial(hsm.target("/Turns/idle")),
        hsm.state(
            "idle",
            # The companion speaking unasked still tells the person when the room will be quiet.
            hsm.transition(hsm.on(PlayedEvent), hsm.effect(_keep)),
            hsm.transition(hsm.on(SayEvent), hsm.effect(_take), hsm.target("/Turns/busy/settling")),
        ),
        hsm.state(
            "busy",
            hsm.defer(SayEvent),
            hsm.initial(hsm.target("/Turns/busy/settling")),
            hsm.state(
                "settling",
                hsm.transition(hsm.on(PlayedEvent), hsm.effect(_keep), hsm.target("/Turns/busy/settling")),
                hsm.transition(hsm.after(_until_settled), hsm.target("/Turns/busy/speaking")),
            ),
            hsm.state(
                "speaking",
                hsm.activity(_run_say),
                hsm.transition(hsm.on(PlayedEvent), hsm.effect(_collect)),
                # The companion began while the person spoke: what it plays is its reply, heard to its pause.
                hsm.transition(
                    hsm.on(_SpokenEvent),
                    hsm.guard(_is_current_while_companion_plays),
                    hsm.effect(_spoke),
                    hsm.target("/Turns/busy/replying/hearing"),
                ),
                hsm.transition(
                    hsm.on(_SpokenEvent),
                    hsm.guard(_is_current),
                    hsm.effect(_spoke),
                    hsm.target("/Turns/busy/awaiting"),
                ),
            ),
            hsm.state(
                "awaiting",
                hsm.transition(
                    hsm.on(PlayedEvent),
                    hsm.guard(_cuts_in),
                    hsm.effect(_collect),
                    hsm.target("/Turns/busy/replying/before_cut_in"),
                ),
                hsm.transition(hsm.on(PlayedEvent), hsm.effect(_collect), hsm.target("/Turns/busy/replying/hearing")),
                hsm.transition(hsm.after(_reply_timeout), hsm.effect(_finish), hsm.target("/Turns/idle")),
            ),
            hsm.state(
                "replying",
                hsm.initial(hsm.target("/Turns/busy/replying/hearing")),
                hsm.state(
                    "hearing",
                    hsm.transition(
                        hsm.on(PlayedEvent), hsm.effect(_collect), hsm.target("/Turns/busy/replying/hearing")
                    ),
                    hsm.transition(hsm.after(_until_paused), hsm.effect(_finish), hsm.target("/Turns/idle")),
                ),
                hsm.state(
                    "before_cut_in",
                    hsm.transition(
                        hsm.on(PlayedEvent), hsm.effect(_collect), hsm.target("/Turns/busy/replying/before_cut_in")
                    ),
                    hsm.transition(hsm.after(_until_cut_in), hsm.target("/Turns/busy/replying/cut_in_due")),
                    hsm.transition(hsm.after(_until_paused), hsm.effect(_finish), hsm.target("/Turns/idle")),
                ),
                hsm.choice(
                    "cut_in_due",
                    hsm.transition(hsm.guard(_companion_plays), hsm.target("/Turns/busy/replying/cutting_in")),
                    # The companion is quiet when the person meant to cut in: there is nothing to talk over.
                    hsm.transition(hsm.target("/Turns/busy/replying/hearing")),
                ),
                hsm.state(
                    "cutting_in",
                    hsm.activity(_run_cut_in),
                    hsm.transition(hsm.on(PlayedEvent), hsm.effect(_collect)),
                    hsm.transition(
                        hsm.on(_CutInSpokenEvent),
                        hsm.guard(_is_current),
                        hsm.effect(_cut),
                        hsm.target("/Turns/busy/replying/hearing"),
                    ),
                ),
            ),
        ),
    )


__all__ = [
    "SCOPE",
    "ChunkData",
    "Clock",
    "ExchangeData",
    "Measures",
    "Microphone",
    "Mouth",
    "Outcome",
    "PlayedEvent",
    "SayData",
    "SayEvent",
    "Speaker",
    "SpokenData",
    "Timing",
    "Turns",
    "measure",
]
