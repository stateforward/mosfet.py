"""Transcription: Listening's streaming hearing path, driven through Listening with fake streaming sessions.

A fake session is a small attachment HSM that records what it is sent and lets the test speak for it
the way a real session does: by dispatching ``hearing.speech.stream`` outputs to its attached
controller with its own id as the source.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import pathlib
import typing

import hsm
import mosfet
from mosfet import lifecycle
import pytest

from mosfet.abilities import cognition
from mosfet.abilities import identification
from mosfet.abilities import listening
from mosfet.abilities import memory
from mosfet.abilities.hearing import voice
from mosfet.abilities.hearing.speech import stream
from mosfet.abilities.speaking import EfferenceData, EfferenceEvent
from mosfet.environment import SoundData, SoundEvent
from mosfet.protocols import attachment
from mosfet.telemetry.hsm import Traced
from tests import metric_reader
from tests.bot.abilities.support import EventRecorder, recent_turns, shared_hsm_context, start_abilities_for_test
from tests.hsm_instance_state import ability_terminal_owner, start_ability_tree

MOUTH = "the-mouth-that-is-producing"
CALLER = "the-callers-microphone"
_TENTH = b"\x01\x00" * 1_600  # 0.1 s of 16 kHz mono PCM
_ADA = mosfet.Participant(ref="person-ada", role=mosfet.ParticipantRole.SOURCE)
_POLICY = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65)


class FakeSession(Traced, attachment.Attachment):
    """A streaming session that records what it is sent; the test speaks for it."""

    _attachment_limit: typing.ClassVar[int | None] = 1
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str
    received: list[hsm.Event[typing.Any]]
    controllers: list[hsm.Instance]
    detached: bool

    def __init__(self) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self.received = []
        self.controllers = []
        self.detached = False

    @typing.override
    def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> collections.abc.Awaitable[bool]:
        return self.dispatch(ctx, event)

    @typing.override
    def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> collections.abc.Awaitable[bool]:
        return self.dispatch(ctx, event)

    @staticmethod
    def _record(ctx: hsm.Context, instance: "FakeSession", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.received.append(event)

    @staticmethod
    def _keep_controller(ctx: hsm.Context, instance: "FakeSession", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, attachment.AttachData)
        instance.controllers.append(data.actor)

    @staticmethod
    def _mark_detached(ctx: hsm.Context, instance: "FakeSession", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance.detached = True

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "FakeSession",
        hsm.initial(hsm.target("/FakeSession/serving")),
        hsm.state(
            "serving",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._can_attach),
                hsm.effect(
                    attachment.Attachment._attach,
                    _keep_controller,
                    attachment.Attachment._dispatch_attach_complete_created,
                ),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._is_attached),
                hsm.effect(
                    attachment.Attachment._detach,
                    _mark_detached,
                    attachment.Attachment._dispatch_detach_complete_removed,
                ),
            ),
            hsm.transition(hsm.on(stream.AudioEvent, stream.ProducingEvent), hsm.effect(_record)),
        ),
    )

    async def say(self, event: hsm.Event[typing.Any]) -> None:
        """Speak as the session does: an output to the attached controller, from this session."""

        (controller,) = self.controllers
        _ = await hsm.dispatch(
            self.context(), controller, dataclasses.replace(event, source=hsm.id(self), target=hsm.id(controller))
        )

    def audio(self) -> list[stream.AudioData]:
        return [event.data for event in self.received if isinstance(event.data, stream.AudioData)]

    def producing(self) -> list[bool]:
        return [event.data.producing for event in self.received if isinstance(event.data, stream.ProducingData)]


class SlowToCloseSession(FakeSession):
    """A session that takes its detach but never answers it: its source stays closing until its stage timeout."""

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "SlowToCloseSession",
        hsm.initial(hsm.target("/SlowToCloseSession/serving")),
        hsm.state(
            "serving",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._can_attach),
                hsm.effect(
                    attachment.Attachment._attach,
                    FakeSession._keep_controller,
                    attachment.Attachment._dispatch_attach_complete_created,
                ),
            ),
            hsm.transition(hsm.on(attachment.DetachEvent), hsm.effect(FakeSession._mark_detached)),
            hsm.transition(hsm.on(stream.AudioEvent, stream.ProducingEvent), hsm.effect(FakeSession._record)),
        ),
    )


class Sessions:
    """The session factory a composition passes: a fresh fake per call, all kept for the test."""

    made: list[FakeSession]
    _kind: type[FakeSession]

    def __init__(self, kind: type[FakeSession] = FakeSession) -> None:
        self.made = []
        self._kind = kind

    def __call__(self) -> hsm.Instance:
        session = self._kind()
        self.made.append(session)
        return session


class FixedVoiceClassifier(
    mosfet.abilities.Classifier[voice.identification.InputData, voice.identification.OutputData]
):
    """Embeds every segment the same, except segments shorter than ``shortest`` it cannot fingerprint."""

    calls: list[voice.identification.InputData]
    embedding: voice.identification.VoiceEmbedding
    shortest: float

    def __init__(self, embedding: voice.identification.VoiceEmbedding, *, shortest: float = 0.0) -> None:
        self.calls = []
        self.embedding = embedding
        self.shortest = shortest

    @typing.override
    async def classify(self, input: voice.identification.InputData) -> voice.identification.OutputData:
        self.calls.append(input)
        return voice.identification.OutputData(
            embeddings=tuple(
                None if segment.duration_seconds < self.shortest else self.embedding for segment in input.segments
            )
        )


class StreamingListening(listening.Listening):
    handoffs: list[cognition.InputData]
    failures: list[listening.FailedEventData]

    def __init__(
        self,
        sessions: Sessions,
        *,
        voice_classifier: FixedVoiceClassifier | None = None,
        registry: identification.Registry | None = None,
        voice_diarizer: voice.diarization.VoiceDiarizer | None = None,
        affect_classifier: voice.AffectClassifier | None = None,
        stm_memory: memory.StmMemory | None = None,
    ) -> None:
        super().__init__(
            stm_memory=stm_memory,
            transcription=sessions,
            voice_classifier=voice_classifier,
            registry=registry,
            voice_diarizer=voice_diarizer,
            affect_classifier=affect_classifier,
        )
        self.handoffs = []
        self.failures = []


def _registry(store: identification.InProcessStore | None = None) -> identification.Registry:
    return identification.Registry(
        identifier=identification.Matcher(
            store=identification.InProcessStore(max_ids=16) if store is None else store, policy=_POLICY
        ),
        kind="voice",
        max_unlinked_age=datetime.timedelta(days=1),
        max_ids=16,
    )


async def _wait(condition: collections.abc.Callable[[], bool], *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.002)
    raise AssertionError("condition not met")


def heard(
    tenths: int = 1, *, level: float | None = 60.0, participants: tuple[mosfet.Participant, ...] = ()
) -> SoundData:
    return SoundData(
        audio=_TENTH * tenths,
        media_type="audio/pcm",
        sample_rate_hz=16_000,
        channels=1,
        amplitude_db=None if level is None else 60.0,
        received_level_db=level,
        participants=participants,
    )


async def arrives(ear: listening.Listening, sound: SoundData, *, source: str = CALLER) -> None:
    _ = await hsm.dispatch(hsm.Context(), ear, dataclasses.replace(SoundEvent.with_data(sound), source=source))


async def produce(ear: listening.Listening, *, duration: float = 60.0) -> None:
    _ = await hsm.dispatch(
        hsm.Context(),
        ear,
        EfferenceEvent.with_data(
            EfferenceData(mouth=MOUTH, duration=duration, media_type="audio/pcm", sample_rate_hz=16_000)
        ),
    )


async def _ear(
    sessions: Sessions,
    *,
    voice_classifier: FixedVoiceClassifier | None = None,
    registry: identification.Registry | None = None,
    voice_diarizer: voice.diarization.VoiceDiarizer | None = None,
    affect_classifier: voice.AffectClassifier | None = None,
    ctx: hsm.Context | None = None,
) -> StreamingListening:
    ear = StreamingListening(
        sessions,
        voice_classifier=voice_classifier,
        registry=registry,
        voice_diarizer=voice_diarizer,
        affect_classifier=affect_classifier,
    )
    await start_ability_tree(ctx, ear)
    return ear


def _stimuli(ear: StreamingListening) -> list[object]:
    stimuli = [handoff.stimulus for handoff in ear.handoffs if isinstance(handoff.stimulus, hsm.Event)]
    return [typing.cast(object, stimulus.data) for stimulus in stimuli]


def _speech(ear: StreamingListening) -> list[listening.SpeechData]:
    return [data for data in _stimuli(ear) if isinstance(data, listening.SpeechData)]


def _facts(ear: StreamingListening) -> list[listening.OverlapData]:
    return [data for data in _stimuli(ear) if isinstance(data, listening.OverlapData)]


def test_listening_takes_exactly_one_hearing_path() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _ = listening.Listening()
    with pytest.raises(ValueError, match="exactly one"):
        _ = listening.Listening(
            transcription=Sessions(),
            voice_activity_classifier=typing.cast(voice.detection.VoiceActivityClassifier, object()),
        )


def test_the_session_contract_is_typed_events() -> None:
    assert stream.AudioEvent.name == "bot.ability.hearing.speech.stream.audio"
    assert stream.TurnEndedEvent.schema is stream.TurnEndedData
    assert stream.FailedData(kind="overloaded").kind == "overloaded"
    with pytest.raises(ValueError):
        _ = stream.FailedData.model_validate({"kind": "refused"})


def test_each_sound_source_gets_its_own_session_up_to_max_sessions(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> Sessions:
        sessions = Sessions()
        ear = await _ear(sessions)
        for source in ("mic-1", "mic-2", "mic-3", "mic-4", "mic-5", "mic-1"):
            await arrives(ear, heard(), source=source)
        await _wait(lambda: len(sessions.made) == 4 and len(sessions.made[0].audio()) == 2)
        return sessions

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = sum(point.value for point in reader.points("bot.listening.transcription.dropped.count"))
        sessions = asyncio.run(run())
        after = sum(point.value for point in reader.points("bot.listening.transcription.dropped.count"))

    assert len(sessions.made) == 4
    assert [len(session.audio()) for session in sessions.made] == [2, 1, 1, 1]
    assert sessions.made[0].audio()[0] == stream.AudioData(audio=_TENTH, sample_rate_hz=16_000, channels=1)
    assert after - before == 1


def test_the_bots_own_voice_arriving_as_predicted_is_not_streamed_and_everything_else_is(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> Sessions:
        sessions = Sessions()
        ear = await _ear(sessions)
        await arrives(ear, heard(), source=CALLER)  # outside the window: streamed
        await produce(ear)
        await arrives(ear, heard(), source=MOUTH)  # own mouth, calibrates: 0 dB left, gated
        await arrives(ear, heard(level=48.0), source=MOUTH)  # own mouth 12 dB off: streamed
        await arrives(ear, heard(), source=CALLER)  # somebody else in the window: streamed
        await _wait(lambda: len(sessions.made) == 2 and len(sessions.made[1].audio()) == 1)
        await _wait(lambda: len(sessions.made[0].audio()) == 2)
        return sessions

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = sum(point.value for point in reader.points("bot.listening.transcription.gated.count"))
        sessions = asyncio.run(run())
        gated = sum(point.value for point in reader.points("bot.listening.transcription.gated.count")) - before

    caller, mouth = sessions.made
    assert len(caller.audio()) == 2 and len(mouth.audio()) == 1
    assert gated == 1
    # The session hears whether the bot is producing, on change only.
    assert caller.producing() == [True]
    assert mouth.producing() == [True]


def test_a_turn_the_session_ends_is_one_text_speech_product() -> None:
    async def run() -> StreamingListening:
        sessions = Sessions()
        ear = await _ear(sessions)
        await arrives(ear, heard(participants=(_ADA,)))
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        session = sessions.made[0]
        await session.say(stream.VoiceStartedEvent.with_data(stream.VoiceStartedData(at_seconds=0.02)))
        await session.say(stream.PartialEvent.with_data(stream.PartialData(text="what is")))
        await session.say(
            stream.TranscriptEvent.with_data(
                stream.TranscriptData(text="what is the time", confidence=0.9, start_seconds=0.02, end_seconds=0.09)
            )
        )
        await session.say(stream.VoiceEndedEvent.with_data(stream.VoiceEndedData(at_seconds=0.09)))
        await session.say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=0.1, probability=0.8)))
        await _wait(lambda: bool(ear.handoffs))
        return ear

    ear = asyncio.run(run())
    (speech,) = _speech(ear)
    assert isinstance(speech, listening.SpeechData)
    assert (speech.content, speech.content_type) == ("what is the time", "text/plain")
    assert speech.confidence == 0.9
    assert (speech.sample_rate_hz, speech.channels, speech.media_type) == (16_000, 1, "audio/pcm")
    (span,) = speech.voice_detection.segments
    assert (span.start_seconds, span.end_seconds) == (pytest.approx(0.02), pytest.approx(0.09))
    assert speech.voices == () and speech.source_ids == frozenset()
    assert speech.participants == (_ADA,)
    (handoff,) = ear.handoffs
    assert handoff.participants == (_ADA,)
    assert isinstance(handoff.stimulus, hsm.Event) and handoff.stimulus.source == CALLER
    # Partial words never reach cognition on their own.
    assert len(ear.handoffs) == 1


def test_a_turns_voice_is_identified_and_stamped() -> None:
    async def run() -> tuple[StreamingListening, FixedVoiceClassifier]:
        sessions = Sessions()
        classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"))
        ear = await _ear(sessions, voice_classifier=classifier, registry=_registry())
        # A second of a voice nobody knows: enough to make a new voice from (recognition.MIN_NEW_VOICE_SECONDS).
        await arrives(ear, heard(10))
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        await sessions.made[0].say(
            stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=1.0, text="hello there"))
        )
        await _wait(lambda: bool(ear.handoffs))
        return ear, classifier

    ear, classifier = asyncio.run(run())
    (speech,) = _speech(ear)
    (voice_heard,) = speech.voices
    ref = identification.ref("voice", voice_heard.recognition.id)
    assert speech.content == "hello there"
    assert speech.source_ids == frozenset({ref})
    assert speech.participants == (identification.participant("voice", voice_heard.recognition.id),)
    # The turn's audio, all of it, was embedded once.
    assert classifier.calls[0].segments[0].audio == _TENTH * 10


def test_a_turn_heard_from_an_identified_voice_is_written_to_short_term_memory_spoken_by_it() -> None:
    async def run() -> tuple[StreamingListening, tuple[memory.stm.Turn, ...]]:
        ctx = shared_hsm_context()
        register = memory.StmMemory()
        await start_abilities_for_test(ctx, register)
        sessions = Sessions()
        classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"))
        ear = StreamingListening(sessions, voice_classifier=classifier, registry=_registry(), stm_memory=register)
        await start_ability_tree(ctx, ear)
        await arrives(ear, heard(10))
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        await sessions.made[0].say(
            stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=1.0, text="hello there"))
        )
        await _wait(lambda: bool(ear.handoffs))
        return ear, await recent_turns(ctx, register)

    ear, turns = asyncio.run(run())

    (handoff,) = ear.handoffs
    assert isinstance(handoff.stimulus, hsm.Event)
    (turn,) = turns
    assert (turn.direction, turn.text) == ("heard", "hello there")
    assert turn.speakers == _speech(ear)[0].participants and len(turn.speakers) == 1
    assert turn.speakers[0].ref.startswith("voice:")
    assert turn.id == handoff.stimulus.id


@pytest.mark.parametrize(("tenths", "known", "named"), [(9, False, False), (10, False, True), (5, True, True)])
def test_a_new_voice_is_made_only_from_at_least_a_second_of_it(tenths: int, known: bool, named: bool) -> None:
    """D-W8-21 D7: a print from under a second of a voice is too little to make a voice ref from. A voice nobody
    knows is made into a new voice only from at least 1.0 s of it; shorter, the turn is heard without naming it
    and nothing is made. A voice already known is named from any length."""

    now = datetime.datetime.now(datetime.UTC)
    store = identification.InProcessStore(max_ids=16)
    if known:
        _ = store.enroll("voice", "known-a", identification.Print(vector=(0.6, 0.8), model="m"), False, _POLICY, at=now)

    async def run() -> StreamingListening:
        sessions = Sessions()
        classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"))
        ear = await _ear(sessions, voice_classifier=classifier, registry=_registry(store))
        await arrives(ear, heard(tenths))
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        await sessions.made[0].say(
            stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=tenths / 10, text="hello there"))
        )
        await _wait(lambda: bool(ear.handoffs))
        return ear

    ear = asyncio.run(run())

    (speech,) = _speech(ear)
    assert speech.content == "hello there"
    assert len(speech.voices) == int(named)
    if known:
        assert speech.voices[0].recognition.id == "known-a"
    held = store.prune("voice", unseen_before=now + datetime.timedelta(days=1), max_ids=16)
    assert held == int(named)


@pytest.mark.parametrize(
    ("while_producing", "quiet", "after_tenths", "made"),
    [
        (True, False, 0, False),  # the turn itself was heard while the bot spoke
        (False, False, 0, False),  # it began right as the bot stopped
        (False, False, 5, False),  # half a second after: still the echo window
        (False, True, 5, False),  # the bot's voice was heard only as predicted (not streamed), then stopped
        (False, False, 11, True),  # past recognition.ECHO_SECONDS: anybody's voice again
    ],
)
def test_a_voice_nobody_knows_heard_while_the_bot_speaks_or_just_after_makes_no_new_voice(
    while_producing: bool, quiet: bool, after_tenths: int, made: bool
) -> None:
    """D-R-37 (companion run 7, D-4): the companion's own sentence, heard back in its own voice and not matched to its
    own print, made a phantom stranger that the companion then linked as a person. While the bot speaks, and for
    ``recognition.ECHO_SECONDS`` of heard audio after, a voice nobody knows is identified match-only: a voice already
    known is still named, and no new one is made. Time is the audio the ear heard, never a wall clock."""

    now = datetime.datetime.now(datetime.UTC)
    store = identification.InProcessStore(max_ids=16)

    async def run() -> StreamingListening:
        sessions = Sessions()
        classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"))
        ear = await _ear(sessions, voice_classifier=classifier, registry=_registry(store))
        await arrives(ear, heard())  # the caller's source is open before the bot speaks
        await _wait(lambda: len(sessions.made) == 1 and len(sessions.made[0].audio()) == 1)
        session = sessions.made[0]

        async def streamed(sound: SoundData) -> None:
            # A turn ends only after its sound reached the session: the session hears before it can say.
            count = len(session.audio())
            await arrives(ear, sound)
            await _wait(lambda: len(session.audio()) == count + 1)

        await produce(ear)
        if while_producing:
            await streamed(heard(12))
        else:
            # The bot's voice in the room: loud enough to stream, or quiet enough to be predicted and not.
            if quiet:
                await arrives(ear, heard(level=1.0))
            else:
                await streamed(heard())
            await session.say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=0.2, text="")))
            await produce(ear, duration=0.0)
            if after_tenths:
                await streamed(heard(after_tenths))
                await session.say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=1.0, text="")))
            await streamed(heard(12))
        await produce(ear, duration=0.0)
        await session.say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=2.0, text="hello there")))
        await _wait(lambda: bool(_speech(ear)))
        return ear

    ear = asyncio.run(run())

    (speech,) = _speech(ear)
    assert speech.content == "hello there"
    assert len(speech.voices) == int(made)
    held = store.prune("voice", unseen_before=now + datetime.timedelta(days=1), max_ids=16)
    assert held == int(made)
    assert listening.recognition.ECHO_SECONDS == 1.0


def test_a_voice_already_known_heard_in_the_echo_window_is_still_named() -> None:
    now = datetime.datetime.now(datetime.UTC)
    store = identification.InProcessStore(max_ids=16)
    _ = store.enroll("voice", "known-a", identification.Print(vector=(0.6, 0.8), model="m"), False, _POLICY, at=now)

    async def run() -> StreamingListening:
        sessions = Sessions()
        classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"))
        ear = await _ear(sessions, voice_classifier=classifier, registry=_registry(store))
        await produce(ear)
        await arrives(ear, heard(12))
        await _wait(lambda: len(sessions.made) == 1 and len(sessions.made[0].audio()) == 1)
        await produce(ear, duration=0.0)
        await sessions.made[0].say(
            stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=1.2, text="hello there"))
        )
        await _wait(lambda: bool(_speech(ear)))
        return ear

    ear = asyncio.run(run())

    (speech,) = _speech(ear)
    (named,) = speech.voices
    assert (named.recognition.id, named.recognition.recognized) == ("known-a", True)


def test_the_bots_own_voice_enrolled_on_the_stream_path_is_never_a_source_and_its_id_is_told() -> None:
    async def run() -> tuple[StreamingListening, list[hsm.Event[typing.Any]]]:
        sessions = Sessions()
        classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"))
        ctx = shared_hsm_context()
        enroller = EventRecorder()
        _ = await mosfet.started(ctx, enroller, enroller.model)
        ear = await _ear(sessions, voice_classifier=classifier, registry=_registry(), ctx=ctx)
        copy = listening.EnrollData(
            segment=voice.VoiceSegment(
                audio=b"\x02\x00" * 24_000,
                media_type="audio/pcm",
                sample_rate_hz=16_000,
                channels=1,
                start_seconds=0.0,
                end_seconds=1.5,
            ),
        )
        _ = await hsm.dispatch(
            ctx,
            ear,
            dataclasses.replace(
                listening.EnrollEvent.with_data_and_id(copy, "enroll"), source=hsm.id(enroller), target=hsm.id(ear)
            ),
        )
        await _wait(lambda: bool(enroller.named(listening.EnrolledEvent)))
        await arrives(ear, heard(5))
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        await sessions.made[0].say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=0.5, text="echo")))
        await _wait(lambda: bool(ear.handoffs))
        return ear, enroller.named(listening.EnrolledEvent)

    ear, enrolled = asyncio.run(run())
    (told,) = enrolled
    assert told.id == "enroll"
    assert isinstance(told.data, listening.EnrolledData)
    own_id = told.data.voice_id
    (speech,) = _speech(ear)
    (own,) = speech.voices
    assert own.recognition.id == own_id and own.recognition.own
    assert speech.source_ids == frozenset()
    assert speech.participants == (identification.participant("voice", own_id, own=True),)


# 1.5 s of the mouth's 24 kHz PCM, 0, 300, 600 repeating; the same second and a half at 16 kHz alternates 0 and 450.
_OWN_24K = b"".join(((index % 3) * 300).to_bytes(2, "little", signed=True) for index in range(36_000))
_OWN_16K = b"".join((0 if index % 2 == 0 else 450).to_bytes(2, "little", signed=True) for index in range(24_000))


@pytest.mark.parametrize(
    ("hears", "expected_rate_hz", "expected_audio"),
    [(True, 16_000, _OWN_16K), (False, 24_000, _OWN_24K)],
    ids=["after-hearing-16khz", "before-hearing-anything"],
)
def test_the_bots_own_print_is_enrolled_at_the_rate_the_stream_ear_last_heard(
    hears: bool, expected_rate_hz: int, expected_audio: bytes
) -> None:
    async def run() -> voice.VoiceSegment:
        sessions = Sessions()
        classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"))
        ctx = shared_hsm_context()
        ear = await _ear(sessions, voice_classifier=classifier, registry=_registry(), ctx=ctx)
        if hears:
            await arrives(ear, heard(5))
            await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].audio()))
        copy = listening.EnrollData(
            segment=voice.VoiceSegment(
                audio=_OWN_24K,
                media_type="audio/pcm",
                sample_rate_hz=24_000,
                channels=1,
                start_seconds=0.0,
                end_seconds=1.5,
            ),
        )
        _ = await hsm.dispatch(ctx, ear, listening.EnrollEvent.with_data_and_id(copy, "enroll"))
        await _wait(lambda: bool(classifier.calls))
        return classifier.calls[0].segments[0]

    enrolled = asyncio.run(run())

    assert (enrolled.sample_rate_hz, enrolled.channels) == (expected_rate_hz, 1)
    assert (enrolled.start_seconds, enrolled.end_seconds) == (0.0, 1.5)
    assert enrolled.audio == expected_audio


def test_an_overlap_while_producing_is_a_fact_carrying_the_words_so_far(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> StreamingListening:
        sessions = Sessions()
        ear = await _ear(sessions)
        await produce(ear)
        await arrives(ear, heard(participants=(_ADA,)))
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        session = sessions.made[0]
        await session.say(stream.PartialEvent.with_data(stream.PartialData(text="wait, hold on")))
        await session.say(stream.OverlapEvent.with_data(stream.OverlapData(at_seconds=1.2, voiced_seconds=0.8)))
        await _wait(lambda: bool(ear.handoffs))
        return ear

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = sum(
            point.value
            for point in reader.points("bot.listening.overlap.count")
            if point.attributes.get("bot.listening.overlap.path") == "stream"
        )
        ear = asyncio.run(run())
        stated = (
            sum(
                point.value
                for point in reader.points("bot.listening.overlap.count")
                if point.attributes.get("bot.listening.overlap.path") == "stream"
            )
            - before
        )

    (fact,) = _facts(ear)
    assert isinstance(fact, listening.OverlapData)
    assert (fact.voiced_seconds, fact.text, fact.voices) == (0.8, "wait, hold on", ())
    assert fact.participants == (_ADA,)
    assert stated == 1


@pytest.mark.parametrize("known", [True, False])
def test_an_overlap_names_only_a_voice_already_known_and_never_makes_a_new_one(known: bool) -> None:
    """Companion live check (D-W8-21 D5, run6): an overlap is heard while the bot speaks, so its audio is the bot's
    voice and another's at once, and identifying it minted a phantom voice that later took a real voice's place.
    Overlap identification is match-only: a voice already known is named on the fact, one nobody knows is not,
    and no voice is made from it."""

    now = datetime.datetime.now(datetime.UTC)
    store = identification.InProcessStore(max_ids=16)
    if known:
        _ = store.enroll("voice", "known-a", identification.Print(vector=(0.6, 0.8), model="m"), False, _POLICY, at=now)

    async def run() -> StreamingListening:
        sessions = Sessions()
        classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"))
        ear = await _ear(sessions, voice_classifier=classifier, registry=_registry(store))
        await produce(ear)
        await arrives(ear, heard(12, participants=(_ADA,)))
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        await sessions.made[0].say(
            stream.OverlapEvent.with_data(stream.OverlapData(at_seconds=1.2, voiced_seconds=1.2))
        )
        await _wait(lambda: bool(_facts(ear)))
        return ear

    ear = asyncio.run(run())

    (fact,) = _facts(ear)
    held = store.prune("voice", unseen_before=now + datetime.timedelta(days=1), max_ids=16)
    if known:
        (heard_voice,) = fact.voices
        assert (heard_voice.recognition.id, heard_voice.recognition.recognized) == ("known-a", True)
        assert fact.participants == (_ADA, identification.participant("voice", "known-a"))
        assert held == 1
    else:
        assert fact.voices == ()
        assert fact.participants == (_ADA,)
        assert held == 0


def test_an_overlap_after_the_bot_stopped_producing_is_no_fact() -> None:
    async def run() -> StreamingListening:
        sessions = Sessions()
        ear = await _ear(sessions)
        await arrives(ear, heard())
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        await sessions.made[0].say(
            stream.OverlapEvent.with_data(stream.OverlapData(at_seconds=1.2, voiced_seconds=0.8))
        )
        await asyncio.sleep(0.05)
        return ear

    ear = asyncio.run(run())
    assert _facts(ear) == []


class RecordingTranscription(listening.Transcription):
    handoffs: list[cognition.InputData]

    def __init__(self, sessions: Sessions, *, idle_close: datetime.timedelta) -> None:
        super().__init__(session=sessions, idle_close=idle_close)
        self.handoffs = []


async def scored(hearing: listening.Transcription, sound: SoundData, *, source: str = CALLER) -> None:
    """Hand Transcription a scored sound the way Listening does, the transducer on the envelope."""

    sensed = listening.sensitivity.OutputData(sound=sound, perceived_level_db=sound.received_level_db)
    _ = await hsm.dispatch(
        hsm.Context(),
        hearing,
        dataclasses.replace(listening.sensitivity.OutputEvent.with_data(sensed), source=source, target=hsm.id(hearing)),
    )


def test_a_source_idle_for_idle_close_has_its_session_closed_and_a_new_one_opened_after() -> None:
    async def run() -> Sessions:
        sessions = Sessions()
        hearing = RecordingTranscription(sessions, idle_close=datetime.timedelta(milliseconds=50))
        await start_ability_tree(None, hearing)
        await scored(hearing, heard())
        await _wait(lambda: len(sessions.made) == 1 and sessions.made[0].detached)
        await _wait(lambda: not lifecycle.is_started(sessions.made[0]))
        await scored(hearing, heard())
        await _wait(lambda: len(sessions.made) == 2 and len(sessions.made[1].audio()) == 1)
        return sessions

    sessions = asyncio.run(run())
    assert sessions.made[0].detached
    assert len(sessions.made[1].audio()) == 1


def test_a_sound_arriving_while_an_idle_source_is_still_closing_opens_a_new_session() -> None:
    async def run() -> Sessions:
        sessions = Sessions(SlowToCloseSession)
        hearing = RecordingTranscription(sessions, idle_close=datetime.timedelta(milliseconds=50))
        await start_ability_tree(None, hearing)
        await scored(hearing, heard())
        # The idle source is detaching from its session, which has not answered yet.
        await _wait(lambda: len(sessions.made) == 1 and sessions.made[0].detached)
        await scored(hearing, heard())
        await _wait(lambda: len(sessions.made) == 2 and len(sessions.made[1].audio()) == 1)
        return sessions

    sessions = asyncio.run(run())
    assert len(sessions.made[0].audio()) == 1
    assert len(sessions.made[1].audio()) == 1


def test_transcription_refuses_meaningless_bounds() -> None:
    classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(1.0,), model="m"))
    builds: list[collections.abc.Callable[[], listening.Transcription]] = [
        lambda: listening.Transcription(session=Sessions(), max_sessions=0),
        lambda: listening.Transcription(session=Sessions(), idle_close=datetime.timedelta(0)),
        lambda: listening.Transcription(session=Sessions(), product_threshold_db=-1.0),
        lambda: listening.Transcription(session=Sessions(), voice_classifier=classifier),
    ]
    for build in builds:
        with pytest.raises(ValueError):
            _ = build()


def test_a_session_failure_loses_the_turn_but_keeps_the_session() -> None:
    async def run() -> tuple[Sessions, StreamingListening]:
        sessions = Sessions()
        ear = await _ear(sessions)
        await arrives(ear, heard())
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        session = sessions.made[0]
        await session.say(
            stream.TranscriptEvent.with_data(stream.TranscriptData(text="lost", start_seconds=0.0, end_seconds=0.1))
        )
        await session.say(stream.FailedEvent.with_data(stream.FailedData(kind="unavailable")))
        await session.say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=0.2)))
        await arrives(ear, heard())
        await _wait(lambda: len(session.audio()) == 2)
        return sessions, ear

    sessions, ear = asyncio.run(run())
    assert len(sessions.made) == 1
    assert ear.handoffs == []


def test_detaching_the_ear_closes_every_session_and_stopping_it_stops_them() -> None:
    async def run() -> Sessions:
        sessions = Sessions()
        hearing = RecordingTranscription(sessions, idle_close=datetime.timedelta(seconds=60))
        await start_ability_tree(None, hearing)
        for source in ("mic-1", "mic-2"):
            await scored(hearing, heard(), source=source)
        await _wait(lambda: len(sessions.made) == 2 and all(session.controllers for session in sessions.made))
        owner = ability_terminal_owner(hearing)
        assert owner is not None
        _ = await hearing.detach(
            hsm.Context(), attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=owner), "detach")
        )
        await _wait(lambda: all(session.detached for session in sessions.made))
        await hsm.stop(hearing)
        await _wait(lambda: not any(lifecycle.is_started(session) for session in sessions.made))
        return sessions

    sessions = asyncio.run(run())
    assert all(session.detached for session in sessions.made)


class TwoSpeakers(voice.diarization.VoiceDiarizer):
    calls: list[voice.diarization.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def classify(self, input: voice.diarization.InputData) -> voice.diarization.OutputData:
        self.calls.append(input)
        half = len(input.audio) // 2
        return voice.diarization.OutputData(
            segments=tuple(
                voice.diarization.VoiceDiarizationSegment(
                    audio=audio,
                    media_type="audio/pcm",
                    sample_rate_hz=input.sample_rate_hz,
                    channels=input.channels,
                    start_seconds=start,
                    end_seconds=start + len(audio) / (2 * input.sample_rate_hz),
                )
                for audio, start in ((input.audio[:half], 0.0), (input.audio[half:], half / (2 * input.sample_rate_hz)))
            )
        )


def test_a_diarizing_ear_identifies_each_speaker_of_the_turn() -> None:
    async def run() -> tuple[StreamingListening, TwoSpeakers, FixedVoiceClassifier]:
        sessions = Sessions()
        diarizer = TwoSpeakers()
        classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"))
        ear = await _ear(sessions, voice_classifier=classifier, registry=_registry(), voice_diarizer=diarizer)
        # A second of each speaker: enough to make a new voice from.
        await arrives(ear, heard(20))
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        await sessions.made[0].say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=2.0, text="hi")))
        await _wait(lambda: bool(ear.handoffs))
        return ear, diarizer, classifier

    ear, diarizer, classifier = asyncio.run(run())
    assert diarizer.calls[0].audio == _TENTH * 20
    assert len(classifier.calls[0].segments) == 2
    (speech,) = _speech(ear)
    assert [(item.start_seconds, item.end_seconds) for item in speech.voices] == [(0.0, 1.0), (1.0, 2.0)]


class SplitAt(voice.diarization.VoiceDiarizer):
    """Two speakers: the turn's audio before ``seconds`` and after; a shorter turn is one."""

    def __init__(self, seconds: float) -> None:
        self.seconds: float = seconds

    @typing.override
    async def classify(self, input: voice.diarization.InputData) -> voice.diarization.OutputData:
        cut = int(self.seconds * input.sample_rate_hz) * 2 * input.channels
        if len(input.audio) <= cut:
            parts = ((input.audio, 0.0),)
        else:
            parts = ((input.audio[:cut], 0.0), (input.audio[cut:], self.seconds))
        return voice.diarization.OutputData(
            segments=tuple(
                voice.diarization.VoiceDiarizationSegment(
                    audio=audio,
                    media_type="audio/pcm",
                    sample_rate_hz=input.sample_rate_hz,
                    channels=input.channels,
                    start_seconds=start,
                    end_seconds=start + len(audio) / (2 * input.channels * input.sample_rate_hz),
                )
                for audio, start in parts
            )
        )


def test_a_voice_nobody_knows_after_the_echo_window_of_the_same_turn_is_a_new_voice() -> None:
    """Companion live check (D-R-44): the echo guard held a whole turn match-only when the bot had spoken anywhere in
    it, so a stranger who spoke well after the companion went quiet, in the same turn, was never heard as anyone. Only a
    diarized segment that overlaps the bot producing, or ``recognition.ECHO_SECONDS`` after, is match-only."""

    async def run() -> StreamingListening:
        sessions = Sessions()
        classifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"))
        ear = await _ear(sessions, voice_classifier=classifier, registry=_registry(), voice_diarizer=SplitAt(2.0))
        await arrives(ear, heard())  # the caller's source is open before the bot speaks
        await _wait(lambda: len(sessions.made) == 1 and len(sessions.made[0].audio()) == 1)
        session = sessions.made[0]

        async def streamed(sound: SoundData) -> None:
            count = len(session.audio())
            await arrives(ear, sound)
            await _wait(lambda: len(session.audio()) == count + 1)

        await session.say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=0.1, text="")))
        await produce(ear)
        await streamed(heard(10))  # the bot speaking
        await produce(ear, duration=0.0)
        await streamed(heard(10))  # its echo window, a second of heard audio
        await streamed(heard(20))  # a stranger, after it
        await session.say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=4.0, text="hello there")))
        await _wait(lambda: bool(_speech(ear)))
        return ear

    ear = asyncio.run(run())

    (speech,) = _speech(ear)
    (stranger,) = speech.voices
    assert (stranger.start_seconds, stranger.recognition.recognized) == (2.0, False)


_CHEERFUL = voice.Affect(label="happy", scores={"happy": 0.7, "neutral": 0.3}, valence=0.6, confidence=0.7)


class FixedAffectClassifier(voice.AffectClassifier):
    calls: list[voice.VoiceSegment]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def classify(self, input: voice.VoiceSegment) -> voice.Affect:
        self.calls.append(input)
        return _CHEERFUL


class HangingAffectClassifier(voice.AffectClassifier):
    @typing.override
    async def classify(self, input: voice.VoiceSegment) -> voice.Affect:
        del input
        _ = await asyncio.Event().wait()
        raise AssertionError("unreachable")


class UnevenSpeakers(voice.diarization.VoiceDiarizer):
    """Two speakers: the first a fifth of the turn, the second the rest."""

    @typing.override
    async def classify(self, input: voice.diarization.InputData) -> voice.diarization.OutputData:
        cut = len(input.audio) // 5
        rate = input.sample_rate_hz
        return voice.diarization.OutputData(
            segments=tuple(
                voice.diarization.VoiceDiarizationSegment(
                    audio=audio,
                    media_type="audio/pcm",
                    sample_rate_hz=rate,
                    channels=input.channels,
                    start_seconds=start,
                    end_seconds=start + len(audio) / (2 * rate),
                )
                for audio, start in ((input.audio[:cut], 0.0), (input.audio[cut:], cut / (2 * rate)))
            )
        )


def test_a_turns_backchannel_too_short_to_fingerprint_leaves_the_rest_identified_and_the_turn_heard() -> None:
    async def run() -> tuple[StreamingListening, FixedVoiceClassifier]:
        sessions = Sessions()
        classifier = FixedVoiceClassifier(
            voice.identification.VoiceEmbedding(embedding=(0.6, 0.8), model="m"), shortest=0.3
        )
        ear = await _ear(sessions, voice_classifier=classifier, registry=_registry(), voice_diarizer=UnevenSpeakers())
        await arrives(ear, heard(14))
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        await sessions.made[0].say(
            stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=1.4, text="mm, yes, go on"))
        )
        await _wait(lambda: bool(ear.handoffs))
        return ear, classifier

    ear, classifier = asyncio.run(run())
    # A 0.28 s backchannel, then 1.12 s of the other voice: both sent to the classifier in one request.
    assert [round(segment.duration_seconds, 3) for segment in classifier.calls[0].segments] == [0.28, 1.12]
    (speech,) = _speech(ear)
    assert speech.content == "mm, yes, go on"
    (identified,) = speech.voices
    assert (identified.start_seconds, identified.end_seconds) == (pytest.approx(0.28), pytest.approx(1.4))
    assert speech.source_ids == frozenset({identification.ref("voice", identified.recognition.id)})


async def _toned_turn(ear: StreamingListening, sessions: Sessions, tenths: int) -> None:
    await arrives(ear, heard(tenths))
    await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
    await sessions.made[0].say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=1.0, text="lovely")))
    await _wait(lambda: bool(ear.handoffs))


def test_a_turn_carries_the_tone_of_its_voice() -> None:
    async def run() -> tuple[StreamingListening, FixedAffectClassifier]:
        sessions = Sessions()
        classifier = FixedAffectClassifier()
        ear = await _ear(sessions, affect_classifier=classifier)
        await _toned_turn(ear, sessions, 5)
        return ear, classifier

    ear, classifier = asyncio.run(run())
    (speech,) = _speech(ear)
    assert speech.content == "lovely"
    assert speech.affect == _CHEERFUL
    (segment,) = classifier.calls
    assert (segment.audio, segment.sample_rate_hz) == (_TENTH * 5, 16_000)


def test_a_diarized_turn_is_heard_in_the_tone_of_its_longest_voice() -> None:
    async def run() -> tuple[StreamingListening, FixedAffectClassifier]:
        sessions = Sessions()
        classifier = FixedAffectClassifier()
        ear = await _ear(sessions, voice_diarizer=UnevenSpeakers(), affect_classifier=classifier)
        await _toned_turn(ear, sessions, 10)
        return ear, classifier

    ear, classifier = asyncio.run(run())
    (speech,) = _speech(ear)
    assert speech.affect == _CHEERFUL
    (segment,) = classifier.calls
    assert segment.audio == (_TENTH * 10)[len(_TENTH * 10) // 5 :]


class TonedTranscription(listening.Transcription):
    handoffs: list[cognition.InputData]

    def __init__(self, sessions: Sessions, *, affect_classifier: voice.AffectClassifier) -> None:
        super().__init__(
            session=sessions, affect_classifier=affect_classifier, embed_timeout=datetime.timedelta(milliseconds=50)
        )
        self.handoffs = []


def test_a_tone_not_classified_in_time_leaves_the_turn_without_one(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> TonedTranscription:
        sessions = Sessions()
        hearing = TonedTranscription(sessions, affect_classifier=HangingAffectClassifier())
        await start_ability_tree(None, hearing)
        await scored(hearing, heard(5))
        await _wait(lambda: len(sessions.made) == 1 and bool(sessions.made[0].controllers))
        await sessions.made[0].say(stream.TurnEndedEvent.with_data(stream.TurnEndedData(at_seconds=0.5, text="lovely")))
        await _wait(lambda: bool(hearing.handoffs))
        return hearing

    timeout = {"bot.outcome": "failed", "bot.failure.kind": "timeout"}

    def timed_out(reader: metric_reader.Reader) -> float:
        return sum(
            point.value
            for point in reader.points("bot.listening.affect.count")
            if all(point.attributes.get(key) == value for key, value in timeout.items())
        )

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = timed_out(reader)
        hearing = asyncio.run(run())
        after = timed_out(reader)

    (handoff,) = hearing.handoffs
    assert isinstance(handoff.stimulus, hsm.Event)
    speech = handoff.stimulus.data
    assert isinstance(speech, listening.SpeechData)
    assert speech.content == "lovely"
    assert speech.affect is None
    assert after - before == 1


def test_a_new_voice_stamps_whose_known_voices_it_resembles_once_per_voice() -> None:
    """R-13: the ear copies what the registry perceived onto the voice's one participant; it links nothing."""

    like = identification.Resemblance(id="sam", similarity=0.42)
    minted = identification.Recognition(id="new", recognized=False, similarity=0.42, own=False, resembles=(like,))
    again = identification.Recognition(id="new", recognized=True, similarity=0.9, own=False)
    heard = (
        voice.HeardVoice(recognition=minted, start_seconds=0.0, end_seconds=1.0),
        voice.HeardVoice(recognition=again, start_seconds=1.0, end_seconds=2.0),
    )

    sources, participants = listening.recognition.stamps(heard)

    assert sources == frozenset({"voice:new"})
    assert participants == (
        mosfet.Participant(
            ref="voice:new",
            role=mosfet.ParticipantRole.SOURCE,
            resembles=(mosfet.Resemblance(ref="voice:sam", similarity=0.42),),
        ),
    )
