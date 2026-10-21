"""The live check: Willy on the lab model server is born, hears a stranger, learns them, and forgets them.

    cd examples/willy && set -a && . ./.env.example && set +a && WILLY_LIVE=1 uv run pytest -m live tests/test_live.py -q

Two gates, and both must hold: every test here is ``live``, so ``-m "not live"`` (CI, and every other gate)
deselects it, and it is skipped unless ``WILLY_LIVE=1``. With ``WILLY_LIVE=1`` nothing is skipped: a missing or
malformed ``WILLY_*`` setting is a failure naming it, and so is a lab service that does not answer, so a down
server never passes silently. The lab is Willy's own stack on CT 108 (``.env.example``; its runbook is
``lab-ct108.md``), speech and the decision server included. If a unit is down, step L1 fails naming it and where it
runs.

The test writes its own ``bot.define`` of Willy's shape, as the bot file does, on the real parts: the lab's
language model (``openai_compat.Processor``), memory on its embeddings, the voice registry on an
``IdentityStore`` sharing memory's file database in the test's data directory, ``listening.Listening`` on the
streaming speech server with voiceprints, diarization and tone, ``speaking.Speaking`` in the voice Willy is born
with (``bot.voice`` on the lab's catalog and designer), ``identity.Identity``, ``communication.Communication``,
and the voice registry given to the cognition, whose memory knows people. The room is one ``computer.Computer`` served by a loopback ``websocket.Server``, and its
far side is ``willy_example.room`` itself. The far side's sound card is the one seam: a scripted microphone that
streams in real time (what the test says, and silence between), and a speaker that keeps what it played. Nothing
else is faked, and no audio is committed: the strangers' speech is made by the lab's speech engine at test time,
in two voices from its catalog ("A" and "B") that are not Willy's own.

What it asserts is perception and capacity, never a choice. The model is the lab's, recorded (every call it is
given, and what it chose) and never steered: whether Willy asks who A is, what it says, and which tools it calls
are its own, and they are written to the report (printed as they happen, so pytest shows them on a failure, and
kept in ``live-report.jsonl``, readable by its owner alone) and to OTEL JSONL in the data directory. Where the test
needs a capacity used, it dispatches the event itself, standing in for a selection, or drives Speaking with a fixed
reply, so whether Willy speaks is never the test's question. A turn speech woke is one a conversation's speech woke,
or speech heard while Willy was speaking (``listening.overlap``). Each step is bounded, and a failure says which step
and its bound:

* L1 preflight (10 s): the intuition and reasoning model servers, asked with the ``openai_api_key`` their clients
  send, list ``WILLY_INTUITION_MODEL`` and ``WILLY_REASONING_MODEL``, and ``/healthz`` answers 200 on speech to text, speech, voiceprints, tone, diarization and
  embeddings.
* L2 birth (300 s): Willy is born with a voice. Speaking constructs its encoder from the voice inception gave
  it, and memory keeps that voice.
* L3 unknown voice (60 s): A's utterance, through the room's far side, reaches the model as a ``source`` that is
  a ``voice:`` ref nobody is known by (``known`` false), and ``memory.know`` is offered. What Willy perceived is
  read from the calls of the minds that took the line up in thought (intuition, reasoning): recall comes after
  reflex. A line Willy does not take up within 20 s (reflex settled it on its own) is said again, as in L4, L5
  and L7.
* L4 knowing is a capacity (60 s): the test has Willy remember that A's ref is called Sam. A's next utterance is
  the same ref (a real voiceprint match through the persistent store), ``known`` true, and Sam; or, when that
  sentence's print falls below the match threshold, a new voice shown as resembling Sam's person (R-13).
* L5 two voices, one utterance (180 s): A then B in one utterance are two distinct ``source`` voice refs across
  the turns Willy takes on it, one turn when the speech server hears it as one (diarization and voiceprints).
  Talked over by Willy, they say it again once it has finished.
* L6 own voice (90 s): Speaking says a fixed reply; its echo, played by the room's speaker into the room's
  microphone, resolves to ``you``, and ``you.refs`` lists that ``voice:`` ref. When no turn shows it within
  20 s, A speaks in the room again, which gives Willy one.
* L7 forget (60 s): forgetting A's person erases A's print, so A's next utterance is a new ref and not that
  person: unknown, unless Willy itself has come to know the new voice again.

The run takes a few minutes, most of it Willy's birth, while its model conceives a voice. The
owner's ``WILLY_HOST``, ``WILLY_PORT`` and ``WILLY_CAMERA`` are not used: the room is served on loopback, and
Willy does not see.
"""

import array
import asyncio
import collections.abc
import contextlib
import dataclasses
import datetime
import functools
import json
import os
import pathlib
import secrets
import socket as sockets
import sys
import time
import typing
import urllib.parse
import uuid

import httpx
import hsm
import mosfet
import pydantic
import pytest
import willy_example
from mosfet import bot, environment, media, runtime, telemetry
from mosfet.abilities import (
    cognition,
    communication,
    embedding,
    identification,
    identity,
    listening,
    memory,
    processing,
    speaking,
    vocal,
)
from mosfet.abilities.communication import conversation
from mosfet.abilities.listening import recognition
from mosfet.bot import testing
from mosfet.devices import audio, computer
from mosfet.providers import openai_compat
from mosfet.providers import speech_server
from mosfet.providers import sqlite_memory
from mosfet.providers import websocket
from willy_example import room
from willy_example.tiers import Tiers

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("WILLY_LIVE") != "1",
        reason="the live check runs against the lab model server only with WILLY_LIVE=1",
    ),
]

_PREFLIGHT = 10.0
_BIRTH = 300.0
_HEARD_BOUND = 60.0
_OWN_VOICE = 90.0
# A step said again, as often as Willy talks over it, within this long.
_TALKED_OVER_BOUND = 180.0
# How long the room's speaker is quiet before what it played counts as all of the reply.
_QUIET_SECONDS = 1.5
# How long the room is silent after an utterance, so the speech server hears the turn end.
_TRAILING_SILENCE_SECONDS = 1.5
# How long a person waits for Willy to take up what was said in thought before saying it again, as people do when
# nobody answers; and, after the echo, for a turn before A speaks again to give Willy one.
_TURN_PATIENCE_SECONDS = 20.0
_POLL_SECONDS = 0.1
# A turn heard speech woke: a conversation's speech, or speech heard over Willy's own (an overlap), which wakes
# listening.overlap instead.
_HEARD = frozenset({conversation.OutputEvent.name, listening.OverlapEvent.name})
_SECRETS_DIRECTORY = pathlib.Path("/run/secrets")
# Where each unit runs: Willy's lab stack, whose runbook says how to start and stop it.
_CT108 = "lab-ct108.md"
# Each of the lab's services, by the setting that names it, the unit that serves it and where that unit runs.
_HEALTH = (
    ("WILLY_STT_URL", "mosfet-stt", _CT108),
    ("WILLY_TTS_URL", "mosfet-tts", _CT108),
    ("WILLY_VOICEPRINT_URL", "mosfet-voiceprint", _CT108),
    ("WILLY_EMOTION_URL", "mosfet-emotion", _CT108),
    ("WILLY_DIAR_URL", "mosfet-diar", _CT108),
    ("WILLY_EMBED_URL", "mosfet-embed", _CT108),
)
# What A and B say: the world's speech, which never names a tool, a person or a reply.
_A_ARRIVES = "Hi, is anybody home? I just got back from the store."
_A_AGAIN = "I left the groceries on the kitchen counter, by the window."
_A_ASKS = "Has anyone seen my keys this morning?"
_B_ANSWERS = "They are hanging on the hook next to the front door."
_A_LEAVES = "Okay, I am heading out again for a little while."
_A_PROMPTS = "It is getting a bit cold in here, isn't it?"
_OWN_WORDS = "I can hear you in the living room. The groceries are on the counter."


class _Models(pydantic.BaseModel):
    """What ``GET /v1/models`` answers, as far as the preflight reads it."""

    class _Model(pydantic.BaseModel):
        id: str

    data: tuple[_Model, ...]


@dataclasses.dataclass
class Report:
    """What the live run saw and what Willy chose: printed as it happens, and kept as JSON lines."""

    lines: list[str] = dataclasses.field(default_factory=list[str])

    def write(self, kind: str, **fields: object) -> None:
        line = json.dumps({"kind": kind, **fields}, default=str)
        print(line)
        self.lines.append(line)

    def save(self, path: pathlib.Path) -> None:
        """Keep the report at ``path``, readable by its owner alone (0600), as the OTEL files beside it are.

        It holds what the household said and what Willy chose, so it is never left world-readable.
        """

        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            _ = file.write("".join(f"{line}\n" for line in self.lines))


@dataclasses.dataclass
class Minds:
    """What Willy's two minds were given and chose, in order, whichever took the call."""

    calls: list[processing.InputData] = dataclasses.field(default_factory=list[processing.InputData])
    chosen: list[processing.SelectedEvent] = dataclasses.field(default_factory=list[processing.SelectedEvent])


class Recorded(processing.Processor):
    """Willy's model, the lab's, with every call it is given and everything it chose recorded; it never steers it.

    Reflex's decision model (``Decided``) and intuition are recorded into the same ``calls``: a heard turn reaches
    whichever mind takes it, so what Willy perceived is read from the first call of the turn, on any of them.
    """

    image_capacity: media.ImageCapacity
    calls: list[processing.InputData]
    chosen: list[processing.SelectedEvent]

    def __init__(self, model: processing.Processor, report: Report, log: Minds) -> None:
        self.image_capacity = model.image_capacity
        # The log both minds' recorders share: one list of calls, and one of what either chose.
        self.calls = log.calls
        self.chosen = log.chosen
        self._model: processing.Processor = model
        self._report: Report = report

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        stimulus = _stimulus(input)
        chosen = await self._model.process(input, drafts=drafts)
        if isinstance(chosen, tuple):
            self.chosen.extend(chosen)
        self._report.write(
            "chose",
            woke=None if stimulus is None else stimulus.name,
            woke_id=None if stimulus is None else stimulus.id,
            woke_source=None if stimulus is None else stimulus.source,
            # A decision model that passes answers the decision layer's own envelope, not events: it is reported by kind.
            events=[
                {
                    "event": selected.event,
                    # A model's selection is JSON; anything else (a rebuilt stimulus) is not reported.
                    "data": typing.cast(dict[str, object], selected.data) if isinstance(selected.data, dict) else None,
                }
                for selected in chosen
            ]
            if isinstance(chosen, tuple)
            else type(typing.cast(object, chosen)).__name__,
        )
        return chosen

    def turns(self, after: int) -> collections.abc.Iterator[processing.InputData]:
        """Every call of the turns heard speech woke that Willy took up in thought, from call ``after`` on: what
        Willy perceived of each.

        A heard turn reaches reflex's decision model first and then, as each hands it on, intuition and reasoning.
        Recall comes after reflex (D-R-52), so only intuition's and reasoning's calls carry what Willy perceived of
        the people (``recalled.associations``; the recent turns ride beside them) and offer what takes thought
        (``memory.know``); reflex's call is not read. A turn reflex settled on its own (a behavior took it, or the
        decision model chose) has no such call, which is Willy's choice: the person waits, then says it again
        (``House.heard_saying``).
        """

        for index, call in enumerate(self.calls):
            stimulus = _stimulus(call)
            recalled = call.recalled
            if (
                stimulus is not None
                and index >= after
                and _heard(stimulus)
                and isinstance(recalled, cognition.RecalledData)
                and recalled.associations is not None
            ):
                yield call


class Decided(Recorded, processing.DecisionProcessor):
    """Willy's reflex decision model, on the decision server, recorded into the model's ``calls`` the same way."""


class Microphone:
    """The room's microphone, scripted: it streams in real time, as a sound card does, what it is told, then silence."""

    def __init__(self) -> None:
        self.streaming: asyncio.Event = asyncio.Event()
        self._pending: bytearray = bytearray()
        self._heard_all: asyncio.Event = asyncio.Event()
        self._stream: asyncio.Task[None] | None = None

    def start(self, heard: collections.abc.Callable[[bytes], None]) -> None:
        self._stream = asyncio.get_running_loop().create_task(self._capture(heard), name="willy-live-microphone")
        self.streaming.set()

    def stop(self) -> None:
        if self._stream is not None:
            _ = self._stream.cancel()
            self._stream = None
        self.streaming.clear()

    async def say(self, pcm: bytes) -> None:
        """Say ``pcm`` (16 kHz mono s16le) into the room; return once it, and the silence after it, were heard."""

        silence = bytes(2 * round(_TRAILING_SILENCE_SECONDS * room.MICROPHONE_RATE_HZ))
        self._heard_all.clear()
        self._pending += pcm + silence
        _ = await self._heard_all.wait()

    async def _capture(self, heard: collections.abc.Callable[[bytes], None]) -> None:
        loop = asyncio.get_running_loop()
        block = 2 * round(room.MICROPHONE_BLOCK_SECONDS * room.MICROPHONE_RATE_HZ)
        due = loop.time()
        while True:
            chunk = bytes(self._pending[:block])
            del self._pending[:block]
            heard(chunk + bytes(block - len(chunk)))
            if not self._pending:
                self._heard_all.set()
            due += room.MICROPHONE_BLOCK_SECONDS
            await asyncio.sleep(max(0.0, due - loop.time()))


@dataclasses.dataclass(frozen=True)
class Played:
    at: float
    pcm: bytes
    sample_rate_hz: int
    channels: int


class Told:
    """What reached Willy, by the id it arrived under: observed at its body, never steered.

    Everything Willy takes in is held by its body first, so an answer to something done is told to Willy when its
    body holds it, whenever Willy then turns to it.
    """

    def __init__(self) -> None:
        self.told: dict[str, hsm.Event[typing.Any]] = {}
        # When each piece of heard speech reached Willy, by its id.
        self.heard_at: dict[str, float] = {}

    def observed(self, ctx: hsm.Context, instance: hsm.Instance, observed: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        data = typing.cast(dict[str, object], observed.data)
        if data.get("occurrence") != "behavior" or not observed.source.endswith("/_hold_percept"):
            return
        arrived = data.get("event")
        if isinstance(arrived, hsm.Event) and isinstance(arrived.data, cognition.InputData):
            stimulus = arrived.data.stimulus
            if isinstance(stimulus, hsm.Event) and stimulus.id:
                self.told[stimulus.id] = typing.cast(hsm.Event[typing.Any], stimulus)
                if _heard(typing.cast(hsm.Event[typing.Any], stimulus)):
                    self.heard_at[stimulus.id] = time.monotonic()


@dataclasses.dataclass(frozen=True)
class Mark:
    """Where Willy's calls were, and when, as something is said: what it heard after is what comes next."""

    calls: int
    at: float


class Speaker:
    """The room's speaker, scripted: it keeps everything it played, and when.

    Speech reaches it faster than it is spoken, so ``quiet_at`` is when a real speaker would have finished: each
    chunk plays after the one before it, for as long as it lasts.
    """

    def __init__(self) -> None:
        self.played: list[Played] = []
        self.quiet_at: float = 0.0

    def play(self, pcm: bytes, *, sample_rate_hz: int, channels: int) -> None:
        now = time.monotonic()
        self.played.append(Played(at=now, pcm=pcm, sample_rate_hz=sample_rate_hz, channels=channels))
        self.quiet_at = max(now, self.quiet_at) + len(pcm) / (2 * channels * sample_rate_hz)

    def stop(self) -> None:
        return None


def _stimulus(call: processing.InputData) -> hsm.Event[typing.Any] | None:
    given = call.input
    if isinstance(given, cognition.reasoning.ProcessorInput):
        given = given.host_input.input
    return typing.cast(hsm.Event[typing.Any], given) if isinstance(given, hsm.Event) else None


def _percepts(stimulus: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
    """What the turn perceived: its stimulus, or each percept of a batch that queued behind a busy turn."""

    batch = stimulus.data
    if isinstance(batch, mosfet.PerceivedData):
        return [
            typing.cast(hsm.Event[typing.Any], percept.stimulus)
            for percept in batch.percepts
            if isinstance(percept.stimulus, hsm.Event)
        ]
    return [stimulus]


def _heard(stimulus: hsm.Event[typing.Any]) -> bool:
    """Whether heard speech woke the turn, alone or with what else queued."""

    return any(percept.name in _HEARD for percept in _percepts(stimulus))


def _known_by_willy(model: Recorded) -> set[str]:
    """Every voice or person Willy itself chose to remember something about (``memory.know``)."""

    known: set[str] = set()
    for selected in model.chosen:
        if selected.event == memory.KnowEvent.name and isinstance(selected.data, dict):
            about = typing.cast(dict[str, object], selected.data).get("about")
            if isinstance(about, str):
                known.add(about)
    return known


def _names_chosen(model: Recorded) -> list[str]:
    """Every name Willy itself chose to remember someone by (``memory.know`` with a name fact)."""

    names: list[str] = []
    for selected in model.chosen:
        if selected.event == memory.KnowEvent.name and isinstance(selected.data, dict):
            fact = typing.cast(dict[str, object], selected.data).get("fact")
            if isinstance(fact, dict):
                name = typing.cast(dict[str, object], fact).get("name")
                if isinstance(name, str):
                    names.append(name)
    return names


def _recalled(call: processing.InputData) -> memory.AssociationsData:
    recalled = call.recalled
    assert isinstance(recalled, cognition.RecalledData), "the turn was given nothing recalled"
    assert isinstance(recalled.associations, memory.AssociationsData), "the turn was given no associations"
    return recalled.associations


def _others(call: processing.InputData) -> list[memory.Present]:
    """The voices speaking on the turn that are not Willy's own."""

    return [
        present
        for present in _recalled(call).people
        if present.part is mosfet.ParticipantRole.SOURCE
        and present.ref.startswith("voice:")
        and present.person != memory.SELF
    ]


def _itself(call: processing.InputData) -> list[memory.Present]:
    """Who on the turn is Willy itself: a ref ``you`` lists, resolved to ``self``."""

    you = _recalled(call).you
    return [present for present in _recalled(call).people if present.ref in you.refs and present.person == memory.SELF]


def _person(call: processing.InputData, person: str) -> memory.Person:
    (found,) = [known for known in _recalled(call).persons if known.id == person]
    return found


def _offered(call: processing.InputData) -> frozenset[str]:
    return frozenset(schema.name for schema in call.schemas)


def _root(url: str) -> str:
    """A service's root over HTTP, from the URL its setting gives (``ws://`` and ``/v1`` included)."""

    parts = urllib.parse.urlsplit(url)
    return f"{'https' if parts.scheme in ('https', 'wss') else 'http'}://{parts.netloc}"


def _free_port() -> int:
    with sockets.socket(sockets.AF_INET, sockets.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return typing.cast(tuple[str, int], probe.getsockname())[1]


def _resampled(pcm: bytes, sample_rate_hz: int) -> bytes:
    """Mono s16le PCM at ``sample_rate_hz`` as the room's microphone captures it (16 kHz), by linear interpolation."""

    source = array.array("h")
    source.frombytes(pcm)
    if sys.byteorder == "big":
        source.byteswap()
    if not source:
        return b""
    step = sample_rate_hz / room.MICROPHONE_RATE_HZ
    last = len(source) - 1
    resampled = array.array("h")
    for index in range(len(source) * room.MICROPHONE_RATE_HZ // sample_rate_hz):
        at = index * step
        before = min(int(at), last)
        after = min(before + 1, last)
        resampled.append(round(source[before] + (source[after] - source[before]) * (at - before)))
    if sys.byteorder == "big":
        resampled.byteswap()
    return resampled.tobytes()


async def _spoken(encoder: openai_compat.SpeechEncoder, words: str) -> bytes:
    """``words`` said by the lab's speech engine in the encoder's voice, as the room's microphone would hear it."""

    text = vocal.speech.Text()
    text.write(vocal.speech.Piece(part=0, words=words))
    text.close()
    pcm = b"".join([chunk.pcm async for chunk in encoder.encode(text)])
    assert encoder.format.channels == 1, "the lab's speech engine answered more than one channel"
    return _resampled(pcm, encoder.format.sample_rate_hz)


@contextlib.asynccontextmanager
async def _step(name: str, bound: float) -> collections.abc.AsyncGenerator[None]:
    try:
        async with asyncio.timeout(bound):
            yield
    except TimeoutError:
        raise AssertionError(f"{name}: not done within {bound:.0f} s (see the report above)") from None


async def _answers(
    client: httpx.AsyncClient,
    variable: str,
    unit: str,
    where: str,
    url: str,
    api_key: pydantic.SecretStr | None = None,
) -> httpx.Response:
    """GET ``url``, with ``api_key`` as its bearer when given (a keyed server answers 401 without it), else none."""

    headers = {} if api_key is None else {"Authorization": f"Bearer {api_key.get_secret_value()}"}
    try:
        response = await client.get(url, headers=headers)
    except httpx.HTTPError as error:
        raise AssertionError(
            f"L1 preflight: {unit} ({variable}) does not answer at {url}: {type(error).__name__}; see {where}"
        ) from None
    assert response.status_code == 200, (
        f"L1 preflight: {unit} ({variable}) answered {response.status_code} at {url}; see {where}"
    )
    return response


async def _preflight(settings: willy_example.Settings, api_key: pydantic.SecretStr) -> None:
    async with httpx.AsyncClient(timeout=_PREFLIGHT) as client:
        # The model server is asked with the key Willy's model client sends; the speech units take none.
        for tier, url, model in (
            ("INTUITION", settings.intuition_url, settings.intuition_model),
            ("REASONING", settings.reasoning_url, settings.reasoning_model),
        ):
            listed = await _answers(
                client, f"WILLY_{tier}_URL", "mosfet-llm", _CT108, f"{url.rstrip('/')}/models", api_key
            )
            models = {served.id for served in _Models.model_validate_json(listed.content).data}
            assert model in models, f"L1 preflight: mosfet-llm does not serve WILLY_{tier}_MODEL; it serves {models}"
        urls = {
            "WILLY_STT_URL": settings.stt_url,
            "WILLY_TTS_URL": settings.tts_url,
            "WILLY_VOICEPRINT_URL": settings.voiceprint_url,
            "WILLY_EMOTION_URL": settings.emotion_url,
            "WILLY_DIAR_URL": settings.diar_url,
            "WILLY_EMBED_URL": settings.embed_url,
        }
        _ = await asyncio.gather(
            *(
                _answers(client, variable, unit, where, f"{_root(urls[variable])}/healthz")
                for variable, unit, where in _HEALTH
            )
        )


@dataclasses.dataclass
class House:
    """Willy running, the room's far side connected to its computer, and what the test does in the house."""

    run: testing.Run
    model: Recorded
    store: memory.Memory
    mouth: speaking.Speaking
    microphone: Microphone
    speaker: Speaker
    told: Told
    report: Report

    async def settled(self) -> None:
        """Wait, as people in the room do, until Willy's voice has died away: its speaker quiet past the echo
        window (D-R-37), in which a voice nobody knows is heard match-only, as it may be Willy's own."""

        async with _step("Willy's voice dies away", _HEARD_BOUND):
            while time.monotonic() - self.speaker.quiet_at < recognition.ECHO_SECONDS:
                await asyncio.sleep(_POLL_SECONDS)

    async def heard_saying(
        self, pcm: bytes, step: str, enough: collections.abc.Callable[[list[processing.InputData]], bool]
    ) -> list[processing.InputData]:
        """Say ``pcm`` once Willy's voice has died away, and the turns Willy took on it once they are ``enough``.

        The speech server may hear one utterance as one turn or several; what was heard is every turn on it.
        Heard over Willy's voice, a voice nobody knows is never a new voice (D-R-37), so an utterance Willy talked
        over is said again, as people do, once a turn on it shows it was not enough. So is one Willy did not take
        up in thought within ``_TURN_PATIENCE_SECONDS`` (reflex settled it on its own, ignoring it, say): whether
        Willy takes a line up is its own, and the person only says it again.
        """

        while True:
            await self.settled()
            after = self.mark()
            await self.microphone.say(pcm)
            talked_over = self.speaker.quiet_at > after.at
            try:
                heard = await asyncio.wait_for(
                    self.heard_since(after, step, lambda calls: enough(calls) or (talked_over and bool(calls))),
                    _TURN_PATIENCE_SECONDS,
                )
            except TimeoutError:
                self.report.write("not_taken_up", step=step)
                continue
            if enough(heard):
                return heard
            self.report.write("talked_over", step=step)

    async def heard_since(
        self, after: Mark, step: str, done: collections.abc.Callable[[list[processing.InputData]], bool]
    ) -> list[processing.InputData]:
        """Every turn with someone speaking, on speech that reached Willy after ``after``, once they are ``done``."""

        seen: list[processing.InputData] = []
        while not done(seen):
            seen.append(await self.turn(after, step, lambda call: bool(_others(call)) and call not in seen))
        return seen

    def mark(self) -> Mark:
        """Where the model's calls are now, and when: the turns after it on speech heard after it come next."""

        return Mark(calls=len(self.model.calls), at=time.monotonic())

    async def turn(
        self, after: Mark, step: str, wanted: collections.abc.Callable[[processing.InputData], bool]
    ) -> processing.InputData:
        """The first turn on speech that reached Willy after ``after`` that is ``wanted``; each one seen is reported.

        A turn on speech heard earlier, which Willy turns to only now, is not what was said after the mark.
        """

        reported: set[int] = set()
        while True:
            for call in self.model.turns(after.calls):
                stimulus = _stimulus(call)
                percepts = () if stimulus is None else _percepts(stimulus)
                if not any(self.told.heard_at.get(percept.id, 0.0) >= after.at for percept in percepts):
                    continue
                if id(call) not in reported:
                    reported.add(id(call))
                    sources = [
                        {"ref": present.ref, "person": present.person, "known": present.known}
                        for present in _recalled(call).people
                        if present.part is mosfet.ParticipantRole.SOURCE
                    ]
                    self.report.write("heard", step=step, sources=sources, you=_recalled(call).you.refs)
                if wanted(call):
                    return call
            await asyncio.sleep(_POLL_SECONDS)

    async def select(self, request: hsm.Event[typing.Any], data: pydantic.BaseModel) -> object:
        """Stand in for Willy selecting a memory tool: dispatch it, and return what Willy was told came of it."""

        # The memory a turn offered the model is the actor cognition dispatches the selection to.
        offered = next(call.actors["memory"] for call in self.model.calls if "memory" in call.actors)
        request_id = uuid.uuid4().hex
        _ = await hsm.dispatch(self.run.environment, offered, request.with_data_and_id(data, request_id))
        # Willy is told what came of it when the answer reaches it, whenever it then turns to it.
        while (told := self.told.told.get(request_id)) is None:
            await asyncio.sleep(_POLL_SECONDS)
        self.report.write("selected", event=request.name, outcome=told.name)
        return typing.cast(object, told.data)


def test_willy_is_born_hears_a_stranger_learns_them_and_forgets_them(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        settings = willy_example.Settings.model_validate(os.environ)
    except pydantic.ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}" for problem in error.errors()
        )
        pytest.fail(f"WILLY_LIVE=1 needs every Willy setting (see .env.example): {problems}")
    try:
        api_key = runtime.Secrets(directory=_SECRETS_DIRECTORY, environ=os.environ).get("openai_api_key")
    except runtime.MissingSecretError:
        pytest.fail("WILLY_LIVE=1 needs the openai_api_key secret: OPENAI_API_KEY=local in .env.example does")
    # The local decision server takes no key; the decision_api_key secret is sent only when it is given.
    try:
        decision_key: pydantic.SecretStr | None = runtime.Secrets(directory=_SECRETS_DIRECTORY, environ=os.environ).get(
            "decision_api_key"
        )
    except runtime.MissingSecretError:
        decision_key = None
    # The OTEL JSONL files are written under the working directory: the test's data directory.
    monkeypatch.chdir(tmp_path)
    _ = telemetry.configure(service_name="willy-live")
    report = Report()
    try:
        asyncio.run(_live(settings, api_key, decision_key, tmp_path, report))
    finally:
        telemetry.reset()
        report.save(tmp_path / "live-report.jsonl")


async def _live(
    settings: willy_example.Settings,
    api_key: pydantic.SecretStr,
    decision_key: pydantic.SecretStr | None,
    data_dir: pathlib.Path,
    report: Report,
) -> None:
    async with _step("L1 preflight", _PREFLIGHT):
        await _preflight(settings, api_key)

    (place,) = settings.rooms[:1]
    port = _free_port()
    tokens = websocket.HmacTokens(pydantic.SecretStr(secrets.token_urlsafe(32)))
    server = websocket.Server(host="127.0.0.1", port=port, verifier=tokens)
    engine = sqlite_memory.open_engine(database_path=data_dir / "memory.sqlite3")
    try:
        store = memory.Memory(
            engine=engine,
            embedder=openai_compat.Embedder(
                client=openai_compat.EmbeddingClient(
                    model=settings.embed_model, base_url=settings.embed_url, api_key=api_key.get_secret_value()
                ),
                capacity=embedding.Capacity(
                    space=willy_example.EMBEDDING_SPACE,
                    max_items=willy_example.EMBEDDING_MAX_ITEMS,
                    max_input_tokens=None,
                ),
                instructions=embedding.EMBEDDING_GEMMA_2,
            ),
        )
        voices = identification.Registry(
            identifier=identification.Matcher(
                store=sqlite_memory.IdentityStore(engine=engine), policy=willy_example.VOICE_POLICY
            ),
            kind="voice",
            max_unlinked_age=willy_example.VOICE_RETENTION,
            max_ids=willy_example.MAX_VOICE_IDS,
        )
        hearing = listening.Listening(
            transcription=lambda: speech_server.Transcription(url=settings.stt_url),
            voice_classifier=speech_server.VoiceEmbedder(base_url=settings.voiceprint_url),
            registry=voices,
            voice_diarizer=speech_server.Diarizer(base_url=settings.diar_url),
            affect_classifier=speech_server.AffectClassifier(base_url=settings.emotion_url),
        )
        talk = conversation.Conversation()
        speech_url = f"{settings.tts_url.rstrip('/')}/v1"
        voiced: list[str] = []

        def born_voice(key: str, description: str | None) -> openai_compat.SpeechEncoder:
            # Speaking constructs its encoder once, from the voice inception gave it: that is Willy being born.
            voiced.append(key)
            return openai_compat.SpeechEncoder(
                model=willy_example.SPEECH_MODEL,
                base_url=speech_url,
                voice=key,
                voice_description=description,
                expressiveness=willy_example.EXPRESSIVENESS,
                markup=openai_compat.ANGLE_TAGS,
            )

        mouth = speaking.Speaking(
            encoder=born_voice,
            speaker=audio.Speaker(),
            listening=(hearing,),
            conversation=(talk,),
            voicing_timeout=willy_example.VOICING_TIMEOUT,
        )
        minds = Minds()
        # The bot file's own tiers (willy_example.tiers), each recorded into one shared log.
        tiers = Tiers.of(
            settings,
            api_key=api_key.get_secret_value(),
            decision_key=None if decision_key is None else decision_key.get_secret_value() or None,
        )
        model = Recorded(tiers.reasoning, report, minds)
        willy = bot.define(
            "willy",
            bot.traits(*settings.traits.traits),
            bot.voice(
                speech_server.Catalog(
                    settings.tts_url, source=willy_example.VOICE_SOURCE, languages=willy_example.LANGUAGES
                ),
                designer=speech_server.Designer(settings.tts_url, source=willy_example.VOICE_SOURCE),
                languages=willy_example.LANGUAGES,
            ),
            bot.cognition(
                reasoning=model,
                intuition=Recorded(tiers.intuition, report, minds),
                reflex=Decided(tiers.reflex, report, minds),
                memory=store,
                registries=(voices,),
                associative=True,
                consolidation=True,
                autonomy=True,
                seeds=(communication.speech_heard_seed(),),
                routines=functools.partial(datetime.datetime.now, datetime.UTC),
            ),
            Tiers.timeouts(),
            bot.input(hearing),
            bot.input(
                identity.Identity(
                    recognizer=identity.SpeechNameRecognizer(
                        decoder=openai_compat.SpeechDecoder(
                            model=willy_example.TRANSCRIPTION_MODEL, base_url=settings.transcribe_url
                        )
                    ),
                    memory=store,
                )
            ),
            bot.output(mouth),
            bot.attach(communication.Communication(conversations=(talk,), speaking=mouth)),
            bot.attach(
                computer.Computer(
                    id=place.id,
                    transport=server,
                    peer_id=place.id,
                    placement=environment.Object(id=place.on, placement=place.placement).surface(),
                )
            ),
        )
        told = Told()
        observe = bot.observe(told.observed)
        async with testing.running(willy, actors=(server,), observe=(observe,)) as run:
            microphone, speaker = Microphone(), Speaker()
            house = House(
                run=run,
                model=model,
                store=store,
                mouth=mouth,
                microphone=microphone,
                speaker=speaker,
                told=told,
                report=report,
            )
            await _born(house, voiced, report)
            closed: asyncio.Future[room.ClosedData] = asyncio.get_running_loop().create_future()
            far_side = room.Room(
                url=f"ws://127.0.0.1:{port}",
                peer=place.id,
                token=tokens.issue(place.id),
                microphone=microphone,
                speaker=speaker,
                on_closed=closed.set_result,
            )
            _ = await mosfet.started(None, far_side, room.Room.model)
            try:
                await _knows_people(house, settings, speech_url, voiced[0])
            finally:
                _ = await far_side.dispatch(far_side.context(), room.LeaveEvent.with_data(room.LeaveData()))
                with contextlib.suppress(TimeoutError):
                    report.write("room", closed=(await asyncio.wait_for(asyncio.shield(closed), 10)).reason)
                await hsm.stop(far_side)
    finally:
        engine.dispose()


async def _born(house: House, voiced: list[str], report: Report) -> None:
    async with _step("L2 birth", _BIRTH):
        while not voiced:
            await asyncio.sleep(_POLL_SECONDS)
        (key,) = voiced
        kept = vocal.voice.recalled_given(
            await memory.transact(house.run.environment, house.store, vocal.voice.given_select_input())
        )
        report.write("born", voice=key, kept=[given.ref.key for given in kept])
        assert key in {given.ref.key for given in kept}, "L2 birth: memory does not keep the voice Willy was born with"


async def _knows_people(house: House, settings: willy_example.Settings, speech_url: str, own_key: str) -> None:
    async with _step("L3 unknown voice", _HEARD_BOUND):
        entries = await speech_server.Catalog(
            settings.tts_url, source=willy_example.VOICE_SOURCE, languages=willy_example.LANGUAGES
        ).voices()
        others = [entry.ref.key for entry in entries if entry.ref.key != own_key]
        assert len(others) >= 2, "L3: the lab's speech engine lists fewer than two voices that are not Willy's"
        a, b = (
            openai_compat.SpeechEncoder(model=willy_example.SPEECH_MODEL, base_url=speech_url, voice=key)
            for key in others[:2]
        )
        house.report.write("strangers", a=others[0], b=others[1])
        _ = await house.microphone.streaming.wait()
        # Knowing who someone is takes thought (effort M): intuition's and reasoning's calls offer it, when reflex
        # hands the turn on. Whether it does is Willy's; A says it again, within the step's bound, until it does.
        heard = await house.heard_saying(
            await _spoken(a, _A_ARRIVES),
            "L3",
            lambda calls: any(_others(call) and memory.KnowEvent.name in _offered(call) for call in calls),
        )
        turn = next(call for call in heard if memory.KnowEvent.name in _offered(call))
    (stranger,) = _others(turn)
    # The typed unknown fact: a voice nobody is known by. What to do about it is Willy's.
    assert stranger.known is False, "L3: a voice Willy has never heard was shown as someone it knows"
    assert memory.KnowEvent.name in _offered(turn), "L3: knowing who a voice is was not offered"
    a_ref = stranger.ref

    async with _step("L4 knowing is a capacity", _HEARD_BOUND):
        known = await house.select(
            memory.KnowEvent,
            memory.KnowData(
                about=a_ref, fact=memory.NameFact(name="Sam"), note="the live check stands in for a selection"
            ),
        )
    await house.settled()
    async with _step("L4 knowing is a capacity", _HEARD_BOUND):
        (turn, *_) = await house.heard_saying(await _spoken(a, _A_AGAIN), "L4", bool)
    (again,) = _others(turn)
    assert isinstance(known, memory.KnownData), f"L4: remembering who A is failed: {known}"
    a_person = stranger.person
    if again.ref == a_ref:
        assert again.known is True, "L4: A's named voice was not shown as someone Willy knows"
        # Sam, unless Willy has since called them something else itself: what it remembers then is its own.
        names = {"Sam", *_names_chosen(house.model)}
        assert _person(turn, again.person).name in names, "L4: A's person was not called what was remembered"
    else:
        # A voiceprint of the same speaker can fall below the match threshold on another sentence: then the voice
        # is new, and Willy is told whom it resembles (R-13); whether it is Sam is Willy's to infer.
        assert again.known is False, "L4: A's voice was taken for someone else Willy knows"
        resembled = {resemblance.person for resemblance in again.resembles}
        assert a_person in resembled, "L4: A's new voice was not shown as resembling the person Willy named"

    together = await _spoken(a, _A_ASKS) + await _spoken(b, _B_ANSWERS)
    async with _step("L5 two voices, one utterance", _TALKED_OVER_BOUND):
        heard = await house.heard_saying(
            together, "L5", lambda calls: len({present.ref for call in calls for present in _others(call)}) >= 2
        )
    voices = {present.ref for call in heard for present in _others(call)}
    assert len(voices) >= 2, "L5: one utterance of two voices was not two voices"

    await house.settled()
    async with _step("L6 own voice", _OWN_VOICE):
        said_at = time.monotonic()
        reply = speaking.InputData(parts=(vocal.speech.Part(words=_OWN_WORDS),))
        _ = await hsm.dispatch(
            house.run.environment, house.mouth, speaking.InputEvent.with_data_and_id(reply, uuid.uuid4().hex)
        )
        # What the room's speaker played of the reply, once it has been quiet for a moment: Willy's voice.
        while not [frame for frame in house.speaker.played if frame.at >= said_at] or (
            time.monotonic() - house.speaker.played[-1].at < _QUIET_SECONDS
        ):
            await asyncio.sleep(_POLL_SECONDS)
        reply_played = [frame for frame in house.speaker.played if frame.at >= said_at]
        (rate,) = {frame.sample_rate_hz for frame in reply_played}
        echo = _resampled(b"".join(frame.pcm for frame in reply_played), rate)
        house.report.write("echo", seconds=len(echo) / 2 / room.MICROPHONE_RATE_HZ)
        after = house.mark()
        await house.microphone.say(echo)
        while True:
            # A turn shows Willy who it is; if none comes soon, A speaking in the room gives it one.
            with contextlib.suppress(TimeoutError):
                turn = await asyncio.wait_for(
                    house.turn(after, "L6", lambda call: bool(_itself(call))), _TURN_PATIENCE_SECONDS
                )
                break
            await house.microphone.say(await _spoken(a, _A_PROMPTS))
    you = _recalled(turn).you
    assert you.id == memory.SELF
    assert you.refs and all(ref.startswith("voice:") for ref in you.refs), "L6: you lists no voice of Willy's own"
    assert all(present.known for present in _itself(turn))
    assert a_ref not in you.refs, "L6: A's voice was taken for Willy's own"

    async with _step("L7 forget", _HEARD_BOUND):
        forgot = await house.select(
            memory.ForgetEvent, memory.ForgetData(person=a_person, note="the live check stands in for a selection")
        )
        assert isinstance(forgot, memory.KnownData), f"L7: forgetting A's person failed: {forgot}"
    leaves = await _spoken(a, _A_LEAVES)
    async with _step("L7 forget", _TALKED_OVER_BOUND):
        (turn, *_) = await house.heard_saying(leaves, "L7", bool)
    (forgotten,) = _others(turn)
    # The print went with the person: A's voice is a new voice to Willy now, and not the person it forgot.
    assert forgotten.ref != a_ref, "L7: A's print outlived the person it was linked to"
    assert forgotten.person != a_person, "L7: a forgotten person was shown again"
    # Nobody Willy knows, unless Willy itself has since come to know this voice again: that is its own.
    if forgotten.known:
        assert forgotten.ref in _known_by_willy(house.model), "L7: a forgotten voice was shown as someone Willy knows"
