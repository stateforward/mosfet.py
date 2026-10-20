"""Willy, end to end: voices in a room, Willy's ear and mouth, its memory of people, and its capacity to know them.

Each test writes its own ``bot.define`` of Willy's shape: the speech seed and associations on a scripted model
(``testing.Scripted``), the real ``listening.Listening`` on the speech server's own wire, the real
``speaking.Speaking`` and ``communication.Communication``, a voice ``identification.Registry`` over an
``InProcessStore`` given to the cognition, and an in-process ``memory.Memory`` that knows people. The fakes sit at the provider
seams only: a scripted speech server behind ``speech_server.Transcription``'s ``connect`` (it hears each second
of audio as one turn), and the voiceprint and diarization servers behind ``httpx.MockTransport``, where a voice
is the sample value its audio is made of. Sound reaches the bot the way a room's computer puts it in the house: as
``environment.sound`` from that room. A line typed in a room reaches it the way the bot file's rooms carry it: a
real ``computer.Computer`` attached to Willy, served by a real ``websocket.Server`` on loopback, with
``willy_example.room`` as its far side on ``support``'s fake devices.

These tests pin perception and capacity: who Willy is shown is speaking, who is present and who it is itself,
and what it is offered to do about them. They never pin a choice. Every fact is scripted, or dispatched by the
test standing in for a selection, only to prove that the capacity works; with a model that chooses nothing,
nothing is known, named or forgotten, and nobody is asked anything.
"""

import asyncio
import collections.abc
import contextlib
import dataclasses
import datetime
import json
import typing
import uuid

import httpx
import hsm
import mosfet
import willy_example
from mosfet import bot, environment
from mosfet.abilities import communication, identification, listening, memory, processing, speaking, vocal
from mosfet.abilities.communication import conversation
from mosfet.bot import testing
from mosfet.devices import audio, computer
from mosfet.providers import speech_server
from mosfet.providers import websocket
from mosfet.providers.speech_server import transcription

from tests import support

_RATE_HZ = 16_000
# Two voices in one utterance get a second each: what a voice nobody knows needs to be made a new voice from.
_UTTERANCE_SECONDS = 2.0
_WAIT = 10.0
# How many more turns a test waits through for something to have reached Willy's memory.
_ATTEMPTS = 5
_MODEL = "wespeaker-test"
# Each voice is the sample value its audio is made of; the voiceprint server knows each by that value.
_SAM, _ADA, _WILLY = 1, 2, 3
_PRINTS = {
    _SAM: (1.0, 0.0, 0.0),
    _ADA: (0.0, 1.0, 0.0),
    _WILLY: (0.0, 0.0, 1.0),
}
_HEARD = conversation.OutputEvent.name


def _pcm(voice: int, seconds: float) -> bytes:
    return voice.to_bytes(2, "little", signed=True) * round(seconds * _RATE_HZ)


class Socket:
    """The speech server's end of one streaming session: every second of audio it is sent is one turn."""

    def __init__(self) -> None:
        self._incoming: asyncio.Queue[str | transcription.Closed] = asyncio.Queue()
        self._heard: float = 0.0
        self._turns: int = 0
        self._incoming.put_nowait(json.dumps({"type": "ready"}))

    async def send(self, message: str | bytes) -> None:
        if not isinstance(message, bytes):
            return
        self._heard += len(message) / 2 / _RATE_HZ
        while self._heard >= (self._turns + 1) * _UTTERANCE_SECONDS - 1e-6:
            start = self._turns * _UTTERANCE_SECONDS
            self._turns += 1
            for frame in (
                {"type": "speech_start", "t": start},
                {"type": "final", "text": "hello there", "words": [{"t": start}, {"t": start + 0.9}]},
                {"type": "turn_end", "t": start + _UTTERANCE_SECONDS},
            ):
                self._incoming.put_nowait(json.dumps(frame))

    async def recv(self) -> str | bytes:
        item = await self._incoming.get()
        if isinstance(item, transcription.Closed):
            raise item
        return item

    async def close(self) -> None:
        self._incoming.put_nowait(transcription.Closed(1000))


async def _connect(url: str) -> transcription.Socket:
    del url
    return Socket()


def _samples(content: bytes) -> list[int]:
    return [int.from_bytes(content[index : index + 2], "little", signed=True) for index in range(0, len(content), 2)]


def _serve(request: httpx.Request) -> httpx.Response:
    """The voiceprint server (``/v1/embed``) and the diarization server (``/v1/diarize``)."""

    samples = _samples(request.content)
    if request.url.path == "/v1/embed":
        return httpx.Response(200, json={"embedding": list(_PRINTS[samples[0]]), "model": _MODEL})
    # Diarization: one segment per run of one voice, each a speaker of its own.
    segments: list[dict[str, float | int]] = []
    start = 0
    for index in range(1, len(samples) + 1):
        if index == len(samples) or samples[index] != samples[start]:
            segments.append({"start": start / _RATE_HZ, "end": index / _RATE_HZ, "speaker": len(segments)})
            start = index
    return httpx.Response(200, json={"segments": segments})


class Mouth(vocal.speech.SpeechEncoder):
    """Willy's voice: whatever it says comes out as two seconds of its own voice."""

    def __init__(self) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=_RATE_HZ, channels=1)
        self.said: list[str] = []

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        self.said.append(said)
        yield vocal.speech.Audio(pcm=_pcm(_WILLY, 2.0), text=said)


class Prints(identification.InProcessStore):
    """The registry's store of voiceprints, which also keeps the id of each of Willy's own voices it enrolls."""

    def __init__(self) -> None:
        super().__init__(max_ids=64)
        self.own: list[str] = []

    @typing.override
    def enroll(
        self,
        kind: identification.Kind,
        id: identification.Id | None,
        print: identification.Print,
        own: bool,
        policy: identification.MatchPolicy,
        *,
        at: datetime.datetime,
    ) -> identification.Recognition:
        enrolled = super().enroll(kind, id, print, own, policy, at=at)
        if own:
            self.own.append(enrolled.id)
        return enrolled


@dataclasses.dataclass
class House:
    """Willy, running as ``bot.run`` runs it, and what the test can do in its house."""

    run: testing.Run
    model: testing.Scripted
    intuition: testing.Decider
    mouth: Mouth
    prints: Prints

    async def say(self, *voices: int, room: str = "kitchen") -> None:
        """One utterance in ``room``: each voice in turn, sharing two seconds of audio."""

        share = _UTTERANCE_SECONDS / len(voices)
        sound = environment.SoundData(
            audio=b"".join(_pcm(voice, share) for voice in voices),
            media_type="audio/pcm",
            sample_rate_hz=_RATE_HZ,
            channels=1,
            amplitude_db=60.0,
            received_level_db=60.0,
        )
        event = dataclasses.replace(environment.SoundEvent.with_data(sound), source=room)
        _ = await hsm.dispatch(self.run.environment, self.run.bot, event)

    def heard(self) -> tuple[processing.InputData, ...]:
        """The model calls of turns a conversation woke: what Willy perceived of what was said."""

        return self.model.woken_by(_HEARD)

    async def turns(self, count: int, name: str = _HEARD) -> tuple[processing.InputData, ...]:
        """Wait until ``count`` turns ``name`` woke have reached the model; their first calls, in order."""

        async with asyncio.timeout(_WAIT):
            while len(_first_calls(self.model.woken_by(name))) < count:
                await asyncio.sleep(0.01)
        return _first_calls(self.model.woken_by(name))

    async def select(self, request: hsm.Event[typing.Any], data: object) -> None:
        """Stand in for Willy selecting a memory tool on a turn it was offered one; wait until Willy is told of it."""

        # The memory a turn offered the model is the actor cognition dispatches the selection to.
        offered = next(call.actors["memory"] for call in self.model.calls() if "memory" in call.actors)
        request_id = uuid.uuid4().hex
        _ = await hsm.dispatch(self.run.environment, offered, request.with_data_and_id(data, request_id))
        async with asyncio.timeout(_WAIT):
            while not any(_stimulus_id(call) == request_id for call in self.model.calls()):
                await asyncio.sleep(0.01)


def _first_calls(calls: tuple[processing.InputData, ...]) -> tuple[processing.InputData, ...]:
    """One call per turn: the first the model was given on it."""

    seen: set[str] = set()
    first: list[processing.InputData] = []
    for call in calls:
        turn = _stimulus_id(call)
        if turn not in seen:
            seen.add(turn)
            first.append(call)
    return tuple(first)


def _stimulus_id(call: processing.InputData) -> str:
    given = call.input
    if isinstance(given, mosfet.abilities.cognition.reasoning.ProcessorInput):
        given = given.host_input.input
    return str(typing.cast(hsm.Event[typing.Any], given).id) if isinstance(given, hsm.Event) else ""


def _recalled(call: processing.InputData) -> memory.AssociationsData:
    recalled = call.recalled
    assert isinstance(recalled, memory.AssociationsData)
    return recalled


def _sources(call: processing.InputData) -> list[memory.Present]:
    return [present for present in _recalled(call).people if present.part is mosfet.ParticipantRole.SOURCE]


def _person(call: processing.InputData, person: str) -> memory.Person:
    (found,) = [known for known in _recalled(call).persons if known.id == person]
    return found


def _offered(call: processing.InputData) -> frozenset[str]:
    return frozenset(schema.name for schema in call.schemas)


@contextlib.asynccontextmanager
async def willy(
    *choices: testing.Choice,
    heard_before: tuple[tuple[str, int], ...] = (),
    intuition: testing.Decider | None = None,
    rooms: tuple[computer.Computer, ...] = (),
    actors: tuple[hsm.Instance, ...] = (),
) -> collections.abc.AsyncGenerator[House]:
    """Run a Willy-shaped bot for the block, on wire fakes, with the model's ``choices`` scripted.

    ``heard_before`` gives voices Willy's registry already holds unnamed, as (id, voice) pairs, as a voice heard on an earlier
    day would be: so a test can write a choice about that voice's ref before the bot runs. ``intuition`` is the
    decision model, a ``testing.Decider`` that chooses nothing when not given. ``rooms`` are computers attached to
    Willy, as the bot file attaches one per room, and ``actors`` the shared actors they need (their server).
    """

    model = testing.Scripted(*choices)
    decider = intuition if intuition is not None else testing.Decider()
    prints = Prints()
    for id, voice in heard_before:
        _ = prints.enroll(
            "voice",
            id,
            identification.Print(vector=_PRINTS[voice], model=_MODEL),
            False,
            willy_example.VOICE_POLICY,
            at=datetime.datetime.now(datetime.UTC),
        )
    voices = identification.Registry(
        identifier=identification.Matcher(store=prints, policy=willy_example.VOICE_POLICY),
        kind="voice",
        max_unlinked_age=willy_example.VOICE_RETENTION,
        max_ids=64,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(_serve)) as client:
        store = memory.Memory()
        hearing = listening.Listening(
            transcription=lambda: speech_server.Transcription(url="ws://speech.test", connect=_connect),
            voice_classifier=speech_server.VoiceEmbedder(base_url="http://voiceprint.test", client=client),
            registry=voices,
            voice_diarizer=speech_server.Diarizer(base_url="http://diarize.test", client=client),
        )
        talk = conversation.Conversation()
        mouth = Mouth()
        voice = speaking.Speaking(encoder=mouth, speaker=audio.Speaker(), listening=(hearing,), conversation=(talk,))
        defined = bot.define(
            "willy",
            bot.traits(*willy_example.DEFAULT_TRAITS.traits),
            bot.cognition(
                model,
                intuition=decider,
                memory=store,
                registries=(voices,),
                associative=True,
                autonomy=True,
                seeds=(communication.speech_heard_seed(),),
            ),
            bot.input(hearing),
            bot.output(voice),
            bot.attach(communication.Communication(conversations=(talk,), speaking=voice)),
            *(bot.attach(room) for room in rooms),
        )
        async with testing.running(defined, actors=actors) as run:
            yield House(run=run, model=model, intuition=decider, mouth=mouth, prints=prints)


class Busy(testing.Decider):
    """Willy's intuition, scripted, holding the first turn it is given until the test lets it go."""

    def __init__(self, release: asyncio.Event) -> None:
        super().__init__()
        self.release: asyncio.Event = release
        self.holding: asyncio.Event = asyncio.Event()

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        if not self.holding.is_set():
            _ = self.holding.set()
            _ = await self.release.wait()
        return await super().process(input, drafts=drafts)


def _heard_count(calls: tuple[processing.InputData, ...]) -> int:
    """How many conversation turns Willy's intuition perceived on ``calls``, alone or with what else queued."""

    heard = 0
    for call in calls:
        given = call.input
        if not isinstance(given, hsm.Event):
            continue
        event = typing.cast(hsm.Event[typing.Any], given)
        batch = event.data
        stimuli = (
            [percept.stimulus for percept in batch.percepts] if isinstance(batch, mosfet.PerceivedData) else [event]
        )
        heard += sum(1 for stimulus in stimuli if isinstance(stimulus, hsm.Event) and stimulus.name == _HEARD)
    return heard


def test_speech_heard_while_willy_is_busy_still_reaches_the_conversation() -> None:
    """Two utterances that queue behind a busy turn are perceived together, and each is admitted as speech."""

    async def run() -> int:
        release = asyncio.Event()
        busy = Busy(release)
        async with willy(intuition=busy) as house:
            await house.say(_SAM)
            async with asyncio.timeout(_WAIT):
                _ = await busy.holding.wait()
            await house.say(_ADA)
            await house.say(_SAM)
            async with asyncio.timeout(_WAIT):
                # Both utterances wait, held, while the turn in progress lasts.
                while (testing.snapshot(house.run.bot).Attributes or {}).get("/willy/held_percepts") != 2:
                    await asyncio.sleep(0.01)
            _ = release.set()
            async with asyncio.timeout(_WAIT):
                while _heard_count(busy.calls()) < 3:
                    await asyncio.sleep(0.01)
            return _heard_count(busy.calls())

    assert asyncio.run(run()) == 3


def test_a_strangers_voice_reaches_willy_unknown_and_knowing_them_is_offered() -> None:
    async def run() -> processing.InputData:
        async with willy() as house:
            await house.say(_SAM)
            (turn,) = await house.turns(1)
            return turn

    turn = asyncio.run(run())

    (stranger,) = _sources(turn)
    assert stranger.ref.startswith("voice:")
    # The typed unknown fact: a voice nobody has named. What to do about it is Willy's.
    assert stranger.known is False
    assert _person(turn, stranger.person).name is None
    assert {memory.KnowEvent.name, memory.ForgetEvent.name} <= _offered(turn)
    assert _recalled(turn).you.id == memory.SELF


def test_a_voice_willy_names_is_known_by_that_name_next_turn() -> None:
    """Nobody lists the household: Willy learns who a voice is, and the next turn shows it what it learned."""

    know = testing.Choice(
        stimulus=_HEARD,
        event=memory.KnowEvent.name,
        data={"about": "voice:ada-voice", "fact": {"kind": "name", "name": "Ada"}},
    )

    async def run() -> tuple[processing.InputData, processing.InputData]:
        async with willy(know, heard_before=(("ada-voice", _ADA),)) as house:
            await house.say(_ADA)
            _ = await house.turns(1)
            _ = await house.turns(1, memory.KnownEvent.name)
            await house.say(_ADA)
            first, second = await house.turns(2)
            return first, second

    first, second = asyncio.run(run())

    (before,) = _sources(first)
    assert (before.ref, before.known) == ("voice:ada-voice", False)
    (after,) = _sources(second)
    assert (after.ref, after.known) == ("voice:ada-voice", True)
    ada = _person(second, after.person)
    assert ada.name == "Ada"
    assert "voice:ada-voice" in ada.refs


def test_two_voices_in_one_utterance_are_two_sources_of_that_turn() -> None:
    async def run() -> processing.InputData:
        async with willy() as house:
            await house.say(_SAM, _ADA)
            (turn,) = await house.turns(1)
            return turn

    turn = asyncio.run(run())

    sources = _sources(turn)
    assert len(sources) == 2
    assert len({source.ref for source in sources}) == 2
    assert len({source.person for source in sources}) == 2
    assert all(source.ref.startswith("voice:") and not source.known for source in sources)


def test_once_willy_has_spoken_in_its_voice_that_voice_is_willy_itself() -> None:
    def reply() -> testing.Choice:
        return testing.Choice(
            stimulus=_HEARD, event=communication.RespondEvent.name, data={"parts": [{"words": "Hello, I'm here."}]}
        )

    async def run() -> tuple[processing.InputData, list[str]]:
        async with willy(reply(), reply()) as house:
            await house.say(_SAM)
            _ = await house.turns(1)
            # Once it has spoken, it has learnt its own voice from what it said.
            async with asyncio.timeout(_WAIT):
                while not house.prints.own:
                    await asyncio.sleep(0.01)
            # What it says from then on is said in a voice it knows is its own.
            await house.say(_SAM)
            _ = await house.turns(2)
            async with asyncio.timeout(_WAIT):
                while len(house.mouth.said) < 2:
                    await asyncio.sleep(0.01)
            # Its next turns show it who it is, once its own words are in the conversation it remembers.
            for count in range(3, 3 + _ATTEMPTS):
                await house.say(_SAM)
                turn = (await house.turns(count))[-1]
                if _recalled(turn).you.refs:
                    return turn, house.prints.own
            raise AssertionError(f"Willy was not shown its own voice in {_ATTEMPTS} turns after it spoke")

    turn, own = asyncio.run(run())

    (own_id,) = own
    you = _recalled(turn).you
    assert you.id == memory.SELF
    assert you.refs == (identification.ref("voice", own_id),)
    (itself,) = [present for present in _recalled(turn).people if present.ref in you.refs]
    assert (itself.person, itself.known) == (memory.SELF, True)
    # Nobody else is ever taken for Willy.
    assert all(present.person != memory.SELF for present in _sources(turn) if present.ref not in you.refs)


def test_forgetting_a_person_erases_their_voiceprint() -> None:
    async def run() -> tuple[processing.InputData, processing.InputData]:
        async with willy(heard_before=(("sam-voice", _SAM),)) as house:
            await house.say(_SAM)
            (first,) = await house.turns(1)
            (sam,) = _sources(first)
            await house.select(memory.ForgetEvent, memory.ForgetData(person=sam.person, note="asked to forget"))
            await house.say(_SAM)
            _, second = await house.turns(2)
            return first, second

    first, second = asyncio.run(run())

    (before,) = _sources(first)
    assert before.ref == "voice:sam-voice"
    (after,) = _sources(second)
    # The print went with the person: the same voice is a new voice to Willy now, and nobody it knows.
    assert after.ref != before.ref
    assert after.ref.startswith("voice:")
    assert after.known is False
    assert after.person != before.person


def _typed(call: processing.InputData) -> computer.TextEnteredData:
    """What was typed, on a turn a line typed on a room's computer woke."""

    given = call.input
    if isinstance(given, mosfet.abilities.cognition.reasoning.ProcessorInput):
        given = given.host_input.input
    assert isinstance(given, mosfet.InputEventData) and given.observation is not None
    entered = given.observation.data
    assert isinstance(entered, computer.TextEnteredData)
    return entered


def test_a_line_typed_in_a_room_reaches_willy_as_what_woke_its_turn() -> None:
    """Typing is perceived: the room's far side types a line on its computer, and a turn of Willy's is woken by it."""

    async def run() -> processing.InputData:
        port = support.free_port()
        server = websocket.Server(host="127.0.0.1", port=port, verifier=support.TOKENS)
        kitchen = computer.Computer(
            id=support.PEER, transport=server, peer_id=support.PEER, placement=support.COUNTER.surface()
        )
        async with willy(rooms=(kitchen,), actors=(server,)) as house:
            keyboard = support.Keyboard()
            session = await support.enter(port, support.Microphone(), support.Speaker(), keyboard=keyboard)
            try:
                await support.wait_until(lambda: keyboard.typed is not None)
                keyboard.type("Hello, Willy")
                (turn,) = await house.turns(1, computer.TextEnteredEvent.name)
                return turn
            finally:
                _ = await session.leave()

    turn = asyncio.run(run())

    typed = _typed(turn)
    assert (typed.device_id, typed.field_id, typed.text) == (support.PEER, None, "Hello, Willy")
    # A room's computer is shared: it stamps nobody on what was typed on it.
    assert typed.participants == ()
    # Answering is offered: what, if anything, to say back is Willy's.
    assert communication.RespondEvent.name in _offered(turn)


def test_willy_can_answer_a_typed_line_aloud_and_the_room_plays_it() -> None:
    """The capacity, not the choice: the answer is scripted only to prove a reply reaches the room's speaker."""

    answer = testing.Choice(
        stimulus=computer.TextEnteredEvent.name,
        event=communication.RespondEvent.name,
        data={"parts": [{"words": "Hello, I'm here."}]},
    )

    async def run() -> tuple[list[str], list[support.Played]]:
        port = support.free_port()
        server = websocket.Server(host="127.0.0.1", port=port, verifier=support.TOKENS)
        kitchen = computer.Computer(
            id=support.PEER, transport=server, peer_id=support.PEER, placement=support.COUNTER.surface()
        )
        async with willy(answer, rooms=(kitchen,), actors=(server,)) as house:
            keyboard, speaker = support.Keyboard(), support.Speaker()
            session = await support.enter(port, support.Microphone(), speaker, keyboard=keyboard)
            try:
                await support.wait_until(lambda: keyboard.typed is not None)
                keyboard.type("Hello, Willy")
                await support.wait_until(lambda: bool(house.mouth.said) and bool(speaker.played), timeout=_WAIT)
                return house.mouth.said, speaker.played
            finally:
                _ = await session.leave()

    said, played = asyncio.run(run())

    assert said == ["Hello, I'm here."]
    assert all(sound.sample_rate_hz == _RATE_HZ and sound.channels == 1 for sound in played)
