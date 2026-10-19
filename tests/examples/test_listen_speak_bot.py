"""listen_speak_bot: the owner's ``Settings``, its bot file, its encoders, and one observed run of a bot it defines.

Every bot here is a small ``bot.define`` of listen_speak's shape, written in the test that runs it, on a scripted
model (``testing.Scripted``) and fakes at the provider seams only: a fixed voice-activity classifier and a recorded
PyAnnote inference on the local path, and on the lab path a scripted speech server behind
``speech_server.Transcription``'s ``connect`` and the voiceprint server behind ``httpx.MockTransport``. Sound
reaches the bot as ``environment.sound``, as ``hear`` plays it.

These tests pin perception and capacity: what the bot hears, who it is shown is speaking, what it is offered, and
that what it selects is said. They never pin a choice. A respond is scripted only to prove that the capacity works;
with a model that chooses nothing, nothing is said, and the run is ``incomplete``, which is the bot's to decide.
"""

import ast
import asyncio
import collections.abc
import dataclasses
import datetime
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import types
import typing
import wave

import hsm
import httpx
import mosfet
import openai
import pydantic
import pytest
from mosfet import bot
from mosfet.abilities import (
    cognition,
    communication,
    encoding,
    identification,
    listening,
    memory,
    processing,
    speaking,
    vocal,
)
from mosfet.abilities.communication import conversation
from mosfet.abilities.hearing import voice
from mosfet.bot import testing
from mosfet.environment import SoundData, SoundEvent
from mosfet.providers import openai_compat
from mosfet.providers import pyannote
from mosfet.providers import speech_server
from mosfet.providers.speech_server import transcription

from tests.bot.abilities.vocal.speech import contract

if typing.TYPE_CHECKING:
    import listen_speak_bot_example as example
else:
    example = pytest.importorskip(
        "listen_speak_bot_example",
        reason="listen_speak_bot example package not installed (run via the examples/listen_speak_bot project env)",
    )

_KEY = "gemini-key-5d1a"
_WAIT = 10.0
_RATE_HZ = 16_000
# Starting the child imports mosfet and the providers before the bot file runs; that is not under test.
_RUN_BOUND = 60.0


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _example_root() -> pathlib.Path:
    return _repo_root() / "examples" / "listen_speak_bot"


# --- The owner's settings ---------------------------------------------------------------------------------------


def test_the_gemini_key_alone_takes_every_default() -> None:
    settings = example.Settings.model_validate({"BOT_GEMINI_API_KEY": _KEY})

    assert settings.gemini_api_key.get_secret_value() == _KEY
    assert settings.gemini_model == example.DEFAULT_GEMINI_MODEL == "gemini-3.5-flash"
    assert settings.vad_model == example.DEFAULT_SILERO_VAD_MODEL == "mlx-community/silero-vad"
    assert settings.stt_model is None
    assert (
        settings.voice_identity_model
        == example.DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL
        == "pyannote/wespeaker-voxceleb-resnet34-LM"
    )
    # Off the lab: Silero hears, PyAnnote knows voices apart, and macOS say speaks.
    assert (settings.stream_url, settings.voiceprint_url, settings.tts_url) == (None, None, None)


@pytest.mark.parametrize(
    ("variable", "field", "value"),
    [
        ("BOT_GEMINI_API_KEY", "gemini_api_key", _KEY),
        ("GEMINI_API_KEY", "gemini_api_key", _KEY),
        ("GOOGLE_API_KEY", "gemini_api_key", _KEY),
        ("BOT_GEMINI_MODEL", "gemini_model", "gemini-test"),
        ("GEMINI_MODEL", "gemini_model", "gemini-test"),
        ("BOT_SILERO_VAD_MODEL", "vad_model", "local/silero-test"),
        ("BOT_VAD_MODEL", "vad_model", "local/silero-test"),
        ("SILERO_VAD_MODEL", "vad_model", "local/silero-test"),
        ("BOT_STT_MODEL", "stt_model", "local/whisper-test"),
        ("BOT_WHISPER_MODEL", "stt_model", "local/whisper-test"),
        ("STT_MODEL", "stt_model", "local/whisper-test"),
        ("WHISPER_MODEL", "stt_model", "local/whisper-test"),
        ("BOT_PYANNOTE_VOICE_IDENTITY_MODEL", "voice_identity_model", "local/test-speaker-model"),
        ("PYANNOTE_VOICE_IDENTITY_MODEL", "voice_identity_model", "local/test-speaker-model"),
    ],
)
def test_every_setting_reads_each_of_its_variables(variable: str, field: str, value: str) -> None:
    given = {"GEMINI_API_KEY": "another-key"} if field != "gemini_api_key" else {}
    settings = example.Settings.model_validate({**given, variable: value})

    read = getattr(settings, field)
    assert (read.get_secret_value() if isinstance(read, pydantic.SecretStr) else read) == value


def test_the_lab_wiring_is_read_from_the_environment() -> None:
    settings = example.Settings.model_validate(
        {
            "BOT_GEMINI_API_KEY": _KEY,
            "BOT_SPEECH_SERVER_STREAM_URL": "ws://lab.test:8001",
            "BOT_SPEECH_SERVER_URL": "http://lab.test:8003",
            "BOT_SPEECH_SERVER_TTS_URL": "http://lab.test:8002",
        }
    )

    assert (settings.stream_url, settings.voiceprint_url, settings.tts_url) == (
        "ws://lab.test:8001",
        "http://lab.test:8003",
        "http://lab.test:8002",
    )


def test_an_empty_variable_is_unset() -> None:
    settings = example.Settings.model_validate(
        {"BOT_GEMINI_API_KEY": _KEY, "BOT_GEMINI_MODEL": "", "BOT_SPEECH_SERVER_TTS_URL": ""}
    )

    assert (settings.gemini_model, settings.tts_url) == (example.DEFAULT_GEMINI_MODEL, None)


def test_a_missing_key_is_named_and_no_value_is_shown() -> None:
    with pytest.raises(pydantic.ValidationError) as missing:
        _ = example.Settings.model_validate({"BOT_GEMINI_MODEL": "gemini-secret-model"})

    assert "BOT_GEMINI_API_KEY" in str(missing.value)
    assert "gemini-secret-model" not in str(missing.value)


def test_the_env_file_reader_takes_simple_assignments(tmp_path: pathlib.Path) -> None:
    path = tmp_path / ".env"
    _ = path.write_text("# a comment\nBOT_GEMINI_API_KEY='quoted'\n\nBOT_GEMINI_MODEL = gemini-test\nnoise\n")

    assert example.load_env(path) == {"BOT_GEMINI_API_KEY": "quoted", "BOT_GEMINI_MODEL": "gemini-test"}
    assert example.load_env(tmp_path / "absent.env") == {}


# --- The bot file ---------------------------------------------------------------------------------------------


def _environ() -> dict[str, str]:
    """This process's environment for the child, without its telemetry, model and speech variables."""

    environ = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OTEL_", "BOT_", "GEMINI_", "GOOGLE_", "VA_", "MOSFET_"))
    }
    _ = environ.pop("PYTHONPATH", None)
    environ["BOT_OTEL_DISABLED"] = "1"
    return environ


def test_python_m_names_a_missing_key_and_exits_1(tmp_path: pathlib.Path) -> None:
    env_file = tmp_path / ".env"
    _ = env_file.write_text("BOT_GEMINI_MODEL=gemini-test\n", encoding="utf-8")
    finished = subprocess.run(
        [sys.executable, "-m", "listen_speak_bot_example", "--env", str(env_file), "--assets-dir", str(tmp_path)],
        cwd=tmp_path,
        env=_environ(),
        capture_output=True,
        timeout=_RUN_BOUND,
        check=False,
    )

    assert finished.returncode == 1, finished.stderr.decode(errors="replace")
    assert finished.stdout == b""
    assert b"BOT_GEMINI_API_KEY" in finished.stderr
    # Nothing was heard, so nothing was rendered.
    assert not (tmp_path / "hey_gabe.wav").exists()


def test_python_m_refuses_a_trait_that_is_not_one(tmp_path: pathlib.Path) -> None:
    finished = subprocess.run(
        [sys.executable, "-m", "listen_speak_bot_example", "--trait", ""],
        cwd=tmp_path,
        env={**_environ(), "BOT_GEMINI_API_KEY": _KEY},
        capture_output=True,
        timeout=_RUN_BOUND,
        check=False,
    )

    assert finished.returncode == 2, finished.stderr.decode(errors="replace")
    assert b"--trait" in finished.stderr
    assert _KEY.encode() not in finished.stderr


def test_the_bot_file_is_one_inline_bot_define() -> None:
    package = _example_root() / "src" / "listen_speak_bot_example"
    main = ast.parse((package / "__main__.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "define"
    ]

    assert len(calls) == 1
    assert isinstance(calls[0].func, ast.Attribute) and ast.unparse(calls[0].func) == "bot.define"
    # The bot file defines no function or class of its own: its one definition is the bot.
    assert not [node for node in ast.walk(main) if isinstance(node, ast.FunctionDef | ast.ClassDef)]
    library = ast.parse((package / "__init__.py").read_text(encoding="utf-8"))
    for node in ast.walk(library):
        if isinstance(node, ast.ClassDef):
            assert "Bot" not in {ast.unparse(base).rpartition(".")[2] for base in node.bases}, node.name
        assert not (isinstance(node, ast.Attribute) and ast.unparse(node) == "bot.define"), (
            "only the bot file defines a bot"
        )


# --- One run, observed -----------------------------------------------------------------------------------------


# A second of speech: what a voice nobody knows needs to be made into a new voice from.
_SPEECH = b"\x01\x00" * 16_000


class _FixedVoiceActivityClassifier(voice.detection.VoiceActivityClassifier):
    """Speech is the audio ``_SPEECH``, a second of it; anything else is silence."""

    @typing.override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        if input == _SPEECH:
            return voice.detection.ApplyData(
                segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.0, confidence=0.9),)
            )
        return voice.detection.ApplyData(segments=())


class _RecordingInference:
    """PyAnnote's embedding inference: one fixed print for every clip, each clip checked and kept."""

    def __init__(self) -> None:
        self.calls: list[pathlib.Path] = []

    def __call__(self, audio: pathlib.Path) -> object:
        with wave.open(str(audio), "rb") as stream:
            assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) == (16_000, 1, 2)
        self.calls.append(audio)
        return [0.12, -0.08, 0.31]


class _Hanging(processing.Processor):
    """A model that never answers: these runs are about what reaches the bot, not its turns."""

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input, drafts
        _ = await asyncio.Event().wait()
        raise AssertionError("a cancelled model resumed")


def _registry() -> identification.Registry:
    return identification.Registry(
        identifier=identification.Matcher(store=identification.InProcessStore(max_ids=8), policy=example.VOICE_POLICY),
        kind="voice",
        max_unlinked_age=datetime.timedelta(days=1),
        max_ids=8,
    )


def _pcm(sample: int, seconds: float) -> bytes:
    return sample.to_bytes(2, "little", signed=True) * round(seconds * _RATE_HZ)


def _sound(audio: bytes) -> SoundData:
    return SoundData(audio=audio, media_type="audio/pcm", sample_rate_hz=_RATE_HZ, channels=1)


def test_speech_reaches_the_bot_from_a_voice_it_does_not_know_on_the_local_path() -> None:
    inference = _RecordingInference()
    loaded: list[str] = []

    def load(model_id: str) -> pyannote.SpeakerEmbeddingInference:
        loaded.append(model_id)
        return inference

    heard: list[cognition.InputData] = []

    def record(ctx: hsm.Context, instance: hsm.Instance, observed: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        data = typing.cast(dict[str, hsm.Event[typing.Any]], observed.data)["event"].data
        if isinstance(data, cognition.InputData) and isinstance(data.stimulus, hsm.Event):
            heard.append(data)

    hearing = listening.Listening(
        voice_activity_classifier=_FixedVoiceActivityClassifier(),
        speech_decoder=None,
        voice_classifier=pyannote.Classifier(model_id="local/test-speaker-model", load_inference=load),
        registry=_registry(),
    )
    listener = bot.define(
        "listen_speak",
        bot.traits(*example.DEFAULT_TRAITS.traits),
        bot.cognition(_Hanging()),
        bot.input(hearing),
    )

    async def run() -> listening.SpeechData:
        async with testing.running(listener, observe=(bot.observe(cognition.InputEvent, record),)) as running:
            for audio in (_SPEECH, b"silence"):
                _ = await hsm.dispatch(running.environment, running.bot, SoundEvent.with_data(_sound(audio)))
            async with asyncio.timeout(_WAIT):
                while not heard:
                    await asyncio.sleep(0.01)
        stimulus = heard[0].stimulus
        assert isinstance(stimulus, hsm.Event)
        assert stimulus.name == listening.SpeechEvent.name
        assert isinstance(stimulus.data, listening.SpeechData)
        return stimulus.data

    speech = asyncio.run(run())

    assert len(inference.calls) == 1
    assert loaded == ["local/test-speaker-model"]
    (voiced,) = speech.voices
    # A voice the bot has never heard: an opaque ref, recognized as nobody, and never a vector.
    assert voiced.recognition.recognized is False
    assert speech.source_ids == frozenset({identification.ref("voice", voiced.recognition.id)})


class _Socket:
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
        while self._heard >= self._turns + 1 - 1e-6:
            start = float(self._turns)
            self._turns += 1
            for frame in (
                {"type": "speech_start", "t": start},
                {"type": "final", "text": "hey how are you", "words": [{"t": start}, {"t": start + 0.9}]},
                {"type": "turn_end", "t": start + 1.0},
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
    return _Socket()


def _voiceprints(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/embed"
    return httpx.Response(200, json={"embedding": [1.0, 0.0, 0.0], "model": "wespeaker-test"})


class _Mouth(vocal.speech.SpeechEncoder):
    """The bot's voice: whatever it says comes out as a tenth of a second of quiet, and is kept."""

    def __init__(self) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=_RATE_HZ, channels=1)
        self.said: list[str] = []

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        self.said.append(said)
        yield vocal.speech.Audio(pcm=_pcm(0, 0.1), text=said)


@dataclasses.dataclass(frozen=True)
class _Ran:
    summary: example.Summary
    model: testing.Scripted
    mouth: _Mouth


def _respond(words: str) -> testing.Choice:
    return testing.Choice(
        stimulus=conversation.OutputEvent.name,
        event=communication.RespondEvent.name,
        data={"parts": [{"words": words}]},
    )


def _hear_on_the_lab(*choices: testing.Choice) -> _Ran:
    """One run of a listen_speak bot on the lab's wire, with its model's ``choices`` scripted: what ``hear`` does."""

    model = testing.Scripted(*choices)
    mouth = _Mouth()
    run = example.Run(timeout=datetime.timedelta(seconds=_WAIT))

    async def played() -> example.SettledData:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_voiceprints)) as client:
            hearing = listening.Listening(
                transcription=lambda: speech_server.Transcription(url="ws://speech.test", connect=_connect),
                voice_classifier=speech_server.VoiceEmbedder(base_url="http://voiceprint.test", client=client),
                registry=_registry(),
            )
            talk = conversation.Conversation()
            voiced = speaking.Speaking(
                encoder=example.LabEncoder(mouth, run=run), listening=(hearing,), conversation=(talk,)
            )
            listener = bot.define(
                "listen_speak",
                bot.traits(*example.DEFAULT_TRAITS.traits),
                bot.cognition(model, autonomy=True, seeds=(communication.speech_heard_seed(),)),
                bot.input(hearing),
                bot.output(voiced),
                bot.attach(communication.Communication(conversations=(talk,), speaking=voiced)),
            )
            return await example.hear(listener, (_sound(_pcm(1, 1.0)),), run=run)

    settled = asyncio.run(played())
    summary = example.Summary.model_validate(
        settled.model_dump() | {"heard_audio_bytes": len(_pcm(1, 1.0)), "reply_audio_bytes": 0}
    )
    return _Ran(summary=summary, model=model, mouth=mouth)


def test_a_heard_turn_offers_the_bot_its_answer_and_what_it_selects_is_said() -> None:
    ran = _hear_on_the_lab(_respond("I am well."))

    (turn, *_) = ran.model.woken_by(conversation.OutputEvent.name)
    assert communication.RespondEvent.name in frozenset(schema.name for schema in turn.schemas)
    assert ran.mouth.said == ["I am well."]
    assert ran.summary.status == "ok"
    assert ran.summary.status_reason == "response_spoken"
    assert (ran.summary.listening_handoffs, ran.summary.response_selections) == (1, 1)
    assert ran.summary.processing_completed >= 1


def test_a_bot_that_selects_nothing_says_nothing_and_the_run_is_incomplete() -> None:
    ran = _hear_on_the_lab()

    # It was offered the answer and chose not to give one: that is the bot's to decide.
    assert ran.model.woken_by(conversation.OutputEvent.name)
    assert ran.mouth.said == []
    assert (ran.summary.status, ran.summary.status_reason) == ("incomplete", "no_response_selected")
    assert ran.summary.response_selections == 0


def test_the_summary_carries_counts_and_outcomes_only() -> None:
    ran = _hear_on_the_lab(_respond("I am well."))
    encoded = ran.summary.model_dump_json()

    assert set(json.loads(encoded)) == {
        "status",
        "status_reason",
        "heard_audio_bytes",
        "reply_audio_bytes",
        "listening_handoffs",
        "processing_completed",
        "response_selections",
    }
    for forbidden in ("embedding", "source_ids", "voice:", "I am well", "hey how are you", "media", "content"):
        assert forbidden not in encoded


class _Closing:
    """A worker the run owns, such as an MLX model's thread: it records that it was closed."""

    def __init__(self) -> None:
        self.closed: bool = False

    async def aclose(self) -> None:
        self.closed = True


def test_hear_closes_its_workers_when_the_run_never_settles() -> None:
    workers = (_Closing(), _Closing())
    listener = bot.define("listen_speak", bot.cognition(_Hanging()), bot.allow("no_devices"))

    async def run() -> None:
        with pytest.raises(TimeoutError):
            _ = await example.hear(
                listener, (), run=example.Run(timeout=datetime.timedelta(seconds=0.05)), workers=workers
            )

    asyncio.run(run())

    assert all(worker.closed for worker in workers)


# --- The run, as a machine -------------------------------------------------------------------------------------


def _handed(turn: str) -> hsm.Event[typing.Any]:
    """The body handing its cognition a conversation turn, as the run's observer sees it."""

    history = conversation.OutputEvent.with_data(conversation.Messages())
    return cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=history), turn)


def _decided(turn: str, *, responses: int) -> hsm.Event[typing.Any]:
    """The bot's decision on ``turn``, selecting ``responses`` responses."""

    respond = cognition.EventData(
        event=communication.RespondEvent.name,
        target="communication",
        data=communication.RespondData(parts=(vocal.speech.Part(words="Reply."),)),
    )
    return mosfet.ProcessingCompletedEvent.with_data_and_id(
        mosfet.ProcessingCompletedEventData(output=(respond,) * responses, focus_candidates=()), turn
    )


def _failed(turn: str) -> hsm.Event[typing.Any]:
    return mosfet.ProcessingFailedEvent.with_data_and_id(mosfet.ProcessingFailedEventData(message="failed"), turn)


class _Ended:
    """How a run ended, as a boundary observer of its terminals sees it, the way ``hear`` waits for it."""

    def __init__(self) -> None:
        self.seen: list[example.SettledData | example.TimedOutData] = []

    def record(self, ctx: hsm.Context, instance: hsm.Instance, observed: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        data = typing.cast(dict[str, hsm.Event[typing.Any]], observed.data)["event"].data
        assert isinstance(data, example.SettledData | example.TimedOutData)
        self.seen.append(data)

    async def started(self, run: example.Run) -> None:
        observed = hsm.redefine(
            example.Run.model, bot.observe(example.SettledEvent, example.TimedOutEvent, self.record)
        )
        _ = await mosfet.started(None, run, observed)

    async def ended(self) -> example.SettledData | example.TimedOutData:
        async with asyncio.timeout(_WAIT):
            while not self.seen:
                await asyncio.sleep(0.01)
        return self.seen[0]

    async def open(self) -> None:
        await asyncio.sleep(0.05)
        assert self.seen == []


def _settled(status: example.Status, reason: example.Reason, *, completed: int, selections: int) -> example.SettledData:
    return example.SettledData(
        status=status,
        status_reason=reason,
        listening_handoffs=1,
        processing_completed=completed,
        response_selections=selections,
    )


def test_the_run_settles_once_the_selected_response_is_said_and_ignores_a_reply_after_it_settled() -> None:
    async def run() -> None:
        ended = _Ended()
        progress = example.Run()
        await ended.started(progress)
        for event in (_handed("turn"), _decided("turn", responses=1)):
            _ = await hsm.dispatch(None, progress, event)
        await ended.open()
        progress.said()
        assert await ended.ended() == _settled("ok", "response_spoken", completed=1, selections=1)
        progress.said()
        progress.not_said()
        await asyncio.sleep(0.05)
        assert len(ended.seen) == 1
        await hsm.stop(progress)

    asyncio.run(run())


def test_a_reply_said_before_its_decision_reached_the_run_waits_for_that_decision() -> None:
    async def run() -> None:
        ended = _Ended()
        progress = example.Run()
        await ended.started(progress)
        _ = await hsm.dispatch(None, progress, _handed("turn"))
        progress.said()
        await ended.open()
        _ = await hsm.dispatch(None, progress, _decided("turn", responses=1))
        assert await ended.ended() == _settled("ok", "response_spoken", completed=1, selections=1)
        await hsm.stop(progress)

    asyncio.run(run())


def test_the_run_counts_only_decisions_on_the_turns_it_saw_handed() -> None:
    async def run() -> None:
        ended = _Ended()
        progress = example.Run()
        await ended.started(progress)
        for event in (_handed("turn"), _decided("other", responses=1), _failed("other")):
            _ = await hsm.dispatch(None, progress, event)
        await ended.open()
        _ = await hsm.dispatch(None, progress, _decided("turn:autonomy", responses=0))
        assert await ended.ended() == _settled("incomplete", "no_response_selected", completed=1, selections=0)
        await hsm.stop(progress)

    asyncio.run(run())


def test_the_run_waits_for_every_response_and_fails_when_one_could_not_be_said() -> None:
    async def run() -> None:
        ended = _Ended()
        progress = example.Run()
        await ended.started(progress)
        for event in (_handed("turn"), _decided("turn", responses=2)):
            _ = await hsm.dispatch(None, progress, event)
        progress.said()
        await ended.open()
        progress.not_said()
        assert await ended.ended() == _settled("failed", "response_execution_failed", completed=1, selections=2)
        await hsm.stop(progress)

    asyncio.run(run())


def test_a_turn_whose_processing_failed_fails_the_run() -> None:
    async def run() -> None:
        ended = _Ended()
        progress = example.Run()
        await ended.started(progress)
        for event in (_handed("turn"), _failed("turn")):
            _ = await hsm.dispatch(None, progress, event)
        assert await ended.ended() == _settled("failed", "processing_failed", completed=0, selections=0)
        await hsm.stop(progress)

    asyncio.run(run())


def test_the_run_times_out_on_its_own_deadline_while_a_response_is_outstanding() -> None:
    bound = datetime.timedelta(seconds=0.2)

    async def run() -> None:
        ended = _Ended()
        progress = example.Run(timeout=bound)
        await ended.started(progress)
        for event in (_handed("turn"), _decided("turn", responses=1)):
            _ = await hsm.dispatch(None, progress, event)
        assert await ended.ended() == example.TimedOutData(timeout=bound)
        await hsm.stop(progress)

    asyncio.run(run())


def test_the_run_is_a_state_machine_with_its_own_deadline_and_no_asyncio_wait_of_its_own() -> None:
    source = (_example_root() / "src" / "listen_speak_bot_example" / "__init__.py").read_text(encoding="utf-8")

    assert "class Run(hsm.Instance):" in source
    assert "hsm.after(" in source
    for forbidden in ("asyncio.Event", "asyncio.timeout", "asyncio.wait_for", "async def settled(", ".state()"):
        assert forbidden not in source, forbidden


# --- Its voice ---------------------------------------------------------------------------------------------------


_DESCRIBED = "A bright, quick young voice with a smile in it, never hurried."


class _Maker(processing.Processor):
    """The bot's model, scripted at its birth: it conceives one described voice whenever it is asked for one."""

    def __init__(self) -> None:
        self.asked: int = 0

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del drafts
        if vocal.voice.ConceiveEvent.name not in {schema.name for schema in input.schemas}:
            return ()
        self.asked += 1
        return (
            processing.SelectedEvent(
                event=vocal.voice.ConceiveEvent.name,
                data={"conception": {"kind": "described", "description": _DESCRIBED}},
            ),
        )


class _LabVoices:
    """The lab TTS engine's /v1/voices protocol: its built-in voice, and each voice it designs, kept for good."""

    def __init__(self) -> None:
        self.held: list[dict[str, object]] = [{"id": "warm", "description": "A warm voice."}]
        self.designed: list[object] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/voices"
        if request.method == "POST":
            self.designed.append(json.loads(request.content))
            made: dict[str, object] = {"id": f"v-{len(self.designed):012x}", "description": _DESCRIBED, "custom": True}
            self.held.append(made)
            return httpx.Response(201, json={"voices": [made]})
        return httpx.Response(200, json={"default": "warm", "voices": self.held, "emotive_tags": ["laugh"]})


def test_on_the_lab_the_bot_is_given_its_voice_at_birth_and_keeps_it_for_life(tmp_path: pathlib.Path) -> None:
    lab = _LabVoices()
    database = tmp_path / "memory.sqlite3"

    async def born(maker: _Maker) -> example.LabEncoder | None:
        given = example.LabVoice("http://lab.test:8002", destination=tmp_path / "reply.wav")
        # Nothing speaks before the voice is given: the lab voice is the one its birth gives it.
        assert given.encoder is None
        async with httpx.AsyncClient(transport=httpx.MockTransport(lab.handle)) as voices:
            listener = bot.define(
                "listen_speak",
                bot.traits(*example.DEFAULT_TRAITS.traits),
                bot.voice(
                    speech_server.Catalog(
                        "http://lab.test:8002", source=example.LAB_SOURCE, languages=example.LANGUAGES, client=voices
                    ),
                    designer=speech_server.Designer("http://lab.test:8002", source=example.LAB_SOURCE, client=voices),
                    languages=example.LANGUAGES,
                ),
                bot.cognition(maker, memory=memory.Memory(database=str(database))),
                bot.output(speaking.Speaking(encoder=given)),
                # Being born is all it does here: it has nothing to hear, and it is never woken.
                bot.allow("no_devices"),
            )
            async with testing.running(listener):
                async with asyncio.timeout(_WAIT):
                    while given.encoder is None:
                        await asyncio.sleep(0.01)
        return given.encoder

    first, again = _Maker(), _Maker()
    made = asyncio.run(born(first))
    kept = asyncio.run(born(again))

    # Made once, from its traits: one conception, one voice designed for it, which it speaks in.
    assert isinstance(made, example.LabEncoder)
    assert first.asked == 1
    assert lab.designed == [{"description": _DESCRIBED, "count": 1}]
    # Started again, it recalls that voice, finds the lab still holds it, and is not made again.
    assert isinstance(kept, example.LabEncoder)
    assert again.asked == 0
    assert len(lab.designed) == 1


class _Served(vocal.speech.SpeechEncoder):
    """A served encoder: two chunks per utterance, aligned to the whole of what it read."""

    def __init__(self, expressiveness: vocal.speech.Expressiveness = vocal.speech.Expressiveness()) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=24_000, channels=1)
        self.expressiveness: vocal.speech.Expressiveness = expressiveness

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        if not said.strip():
            raise encoding.Error("invalid_input", "blank text")
        yield vocal.speech.Audio(pcm=b"\x01\x00", text="")
        yield vocal.speech.Audio(pcm=b"\x02\x00", text=said)


def test_lab_encoder_streams_through_and_keeps_what_it_said(tmp_path: pathlib.Path) -> None:
    async def run() -> tuple[list[bytes], example.LabEncoder]:
        encoder = example.LabEncoder(_Served(), destination=tmp_path / "reply.wav")
        chunks = [chunk.pcm for chunk in await contract.assert_speech_encoder(encoder, contract.words("Hel", "lo."))]
        return chunks, encoder

    chunks, encoder = asyncio.run(run())
    assert chunks == [b"\x01\x00", b"\x02\x00"]
    assert encoder.calls == ["Hello."]
    assert encoder.audio == b"\x01\x00\x02\x00"
    assert (tmp_path / "reply.wav").read_bytes().startswith(b"RIFF")


def test_lab_encoder_can_do_what_the_served_encoder_can_with_an_expression() -> None:
    encoder = example.LabEncoder(_Served(example.LAB_EXPRESSIVENESS))

    # Speaking reports each part's cues against this, and the self block shows it.
    assert encoder.expressiveness == example.LAB_EXPRESSIVENESS


_LAB_PCM_HEADERS = {"x-sample-rate": "24000", "x-sample-format": "s16le", "content-type": "application/octet-stream"}


def _lab_speech_client(requests: list[httpx.Request]) -> openai_compat.AsyncOpenAIClient:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers=_LAB_PCM_HEADERS, content=b"\x01\x00\x02\x00")

    sdk = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="http://lab.test:8002/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    # The SDK's create takes its own keywords; the protocol is the narrow shape the provider calls.
    return typing.cast(openai_compat.AsyncOpenAIClient, typing.cast(object, sdk))


def test_lab_voice_speaks_in_the_voice_it_is_given_and_keeps_its_description(tmp_path: pathlib.Path) -> None:
    requests: list[httpx.Request] = []
    lab = example.LabVoice(
        "http://lab.test:8002", destination=tmp_path / "reply.wav", client=_lab_speech_client(requests)
    )

    encoder = lab("v-0123456789ab", "A bright, quick young voice.")
    parts = (
        vocal.speech.Piece(part=0, words="Oh, ", expression=vocal.speech.Expression(sound="laugh")),
        vocal.speech.Piece(part=1, words="you got me. "),
        vocal.speech.Piece(part=2, words="Shh.", expression=vocal.speech.Expression(emotion="secretive")),
    )
    chunks = asyncio.run(contract.assert_speech_encoder(encoder, parts))

    assert lab.encoder is encoder
    assert encoder.format == vocal.speech.Format(sample_rate_hz=24_000, channels=1)
    assert encoder.expressiveness == example.LAB_EXPRESSIVENESS
    sent = [typing.cast(dict[str, object], json.loads(request.content)) for request in requests]
    # One request per run of parts said the same way; the voice and its description come from inception.
    assert sent[:2] == [
        {"model": "tts", "input": "<laugh>Oh, you got me. ", "voice": "v-0123456789ab", "response_format": "pcm"},
        {
            "model": "tts",
            "input": "Shh.",
            "voice": "v-0123456789ab",
            "instructions": sent[1]["instructions"],
            "response_format": "pcm",
        },
    ]
    assert str(sent[1]["instructions"]).startswith("A bright, quick young voice. ")
    assert "".join(chunk.text for chunk in chunks) == "Oh, you got me. Shh."


def _listen_check() -> types.ModuleType:
    path = _example_root() / "scripts" / "listen_check.py"
    spec = importlib.util.spec_from_file_location("listen_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_listen_check_renders_each_declared_sound_before_and_after_its_words_and_each_style() -> None:
    lines = typing.cast(dict[str, tuple[vocal.speech.Part, ...]], _listen_check().replies())

    sounds = sorted(example.LAB_EXPRESSIVENESS.sounds)
    assert sorted(lines) == sorted(
        [f"{sound}_{at}" for sound in sounds for at in ("before", "after")] + list(example.LAB_EXPRESSIVENESS.styles)
    )
    for name, parts in lines.items():
        # One two-part reply each: the first part carries the sound or the style, the second is said naturally.
        assert len(parts) == 2, name
        opening, closing = parts
        assert opening.expression is not None and closing.expression is None, name
    assert lines["laugh_after"][0].expression == vocal.speech.Expression(sound="laugh", sound_at="after")
    assert lines["whisper"][0].expression == vocal.speech.Expression(style="whisper")


def test_listen_check_writes_one_wav_per_line(tmp_path: pathlib.Path) -> None:
    listen_check = _listen_check()

    def encoder(destination: pathlib.Path) -> vocal.speech.SpeechEncoder:
        return example.LabEncoder(_Served(), destination=destination)

    written = typing.cast(list[pathlib.Path], asyncio.run(listen_check.render(encoder, tmp_path)))

    assert sorted(path.name for path in written) == sorted(f"{name}.wav" for name in listen_check.replies())
    assert all(path.read_bytes().startswith(b"RIFF") for path in written)


# --- macOS say ---------------------------------------------------------------------------------------------------


def _rendered(*samples: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(22_050)
        stream.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
    return output.getvalue()


async def _encoded(encoder: vocal.speech.SpeechEncoder, text: str) -> list[bytes]:
    said = vocal.speech.Text()
    said.write(vocal.speech.Piece(part=0, words=text))
    said.close()
    return [chunk.pcm async for chunk in encoder.encode(said)]


def test_say_encoder_streams_the_rendered_pcm_and_renders_in_a_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[pathlib.Path] = []

    def render(text: str, destination: pathlib.Path, *, sample_rate_hz: int) -> bytes:
        assert (text, sample_rate_hz) == ("reply", 22_050)
        rendered.append(destination)
        _ = destination.write_bytes(_rendered(1, 2))
        return _rendered(1, 2)

    monkeypatch.setattr(example, "say_to_wav", render)
    encoder = example.SayEncoder()

    assert asyncio.run(_encoded(encoder, " reply ")) == [b"\x01\x00\x02\x00"]
    assert encoder.format == vocal.speech.Format(sample_rate_hz=22_050, channels=1)
    assert encoder.calls == [" reply "]
    assert [
        chunk.text for chunk in asyncio.run(contract.assert_speech_encoder(encoder, contract.words(" re", "ply ")))
    ] == [" reply "]
    assert not rendered[0].parent.exists()


def test_say_encoder_failures_are_encoding_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_render(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise RuntimeError("render failed")

    monkeypatch.setattr(example, "say_to_wav", fail_render)
    encoder = example.SayEncoder()

    with pytest.raises(encoding.Error) as failed:
        _ = asyncio.run(_encoded(encoder, "reply"))
    assert failed.value.kind == "encode_failed"
    with pytest.raises(encoding.Error) as blank:
        _ = asyncio.run(_encoded(encoder, "   "))
    assert blank.value.kind == "invalid_input"
    assert (encoder.calls, encoder.audio) == ([], None)


def test_say_encoder_cancellation_does_not_publish_the_cancelled_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    wait_timeout_seconds = 2.0

    def blocked_say_to_wav(text: str, destination: pathlib.Path, *, sample_rate_hz: int) -> bytes:
        del text, destination, sample_rate_hz
        started.set()
        try:
            assert release.wait(timeout=wait_timeout_seconds)
            return _rendered(1)
        finally:
            finished.set()

    monkeypatch.setattr(example, "say_to_wav", blocked_say_to_wav)
    encoder = example.SayEncoder()

    async def run() -> None:
        task = asyncio.create_task(_encoded(encoder, "reply"))
        _ = await asyncio.to_thread(started.wait, wait_timeout_seconds)
        _ = task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert encoder.calls == []
        assert encoder.audio is None
        assert await asyncio.to_thread(finished.wait, wait_timeout_seconds)

    asyncio.run(run())


def test_say_encoder_keeps_everything_said_in_an_explicit_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    samples = iter((1, 2))

    def render(text: str, destination: pathlib.Path, *, sample_rate_hz: int) -> bytes:
        del text, destination, sample_rate_hz
        return _rendered(next(samples))

    monkeypatch.setattr(example, "say_to_wav", render)
    destination = tmp_path / "reply.wav"
    encoder = example.SayEncoder(destination=destination)

    _ = asyncio.run(_encoded(encoder, "First."))
    _ = asyncio.run(_encoded(encoder, "Second."))

    with wave.open(str(destination), "rb") as stream:
        assert stream.readframes(stream.getnframes()) == b"\x01\x00\x02\x00"
    assert encoder.calls == ["First.", "Second."]


@pytest.mark.skipif(
    shutil.which("say") is None or shutil.which("afconvert") is None, reason="macOS say and afconvert render speech"
)
def test_silence_and_a_rendering_are_wavs_whose_frames_the_bot_can_hear(tmp_path: pathlib.Path) -> None:
    silence = example.silence_wav()
    rendered = example.say_to_wav("Hi.", tmp_path / "hi.wav")

    assert example.wav_pcm(silence) == b"\x00" * (round(0.5 * example.SAMPLE_RATE_HZ) * 2)
    assert (tmp_path / "hi.wav").read_bytes() == rendered
    assert len(example.wav_pcm(rendered)) > 0


# --- CI ----------------------------------------------------------------------------------------------------------


def test_ci_runs_fatal_root_and_listen_speak_python_gates() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "uv run --locked ruff check ." in workflow
    assert "EVENT_NAME: ${{ github.event_name }}" in workflow
    assert "PR_BASE_SHA: ${{ github.event.pull_request.base.sha }}" in workflow
    assert "PR_HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in workflow
    assert "PUSH_BASE_SHA: ${{ github.event.before }}" in workflow
    assert "PUSH_HEAD_SHA: ${{ github.sha }}" in workflow
    assert 'if [[ "$EVENT_NAME" == "pull_request" ]]' in workflow
    assert 'if [[ "$base_sha" =~ ^0+$ ]]' in workflow
    assert 'base_sha="$(git hash-object -t tree /dev/null)"' in workflow
    assert "git diff --name-only --diff-filter=ACMR -z" in workflow
    assert '"$base_sha" "$head_sha" -- \'*.py\'' in workflow
    assert 'changed_python_files+=("$path")' in workflow
    assert 'uv run --locked ruff format --check -- "${changed_python_files[@]}"' in workflow
    assert "uv run --locked basedpyright --level error" in workflow
    assert "uv run --locked python -m pytest -q -m 'not live'" in workflow
    assert "uv run --locked --project examples/listen_speak_bot ruff check" in workflow
    assert "uv run --locked --project examples/listen_speak_bot ruff format --check" in workflow
    assert "uv run --locked basedpyright --project examples/listen_speak_bot" in workflow
