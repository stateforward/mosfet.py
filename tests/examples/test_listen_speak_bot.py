from mosfet import abilities
from mosfet.abilities import cognition
from mosfet.abilities import listening
from mosfet.abilities.hearing import voice
from mosfet.devices import audio
from mosfet.environment import Environment, SoundData, SoundEvent
from mosfet.protocols import attachment
from mosfet.providers import pyannote

import asyncio
import collections.abc
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import typing
import wave
from typing import ClassVar, Protocol, TypeVar, override

import hsm
import pytest

if typing.TYPE_CHECKING:
    from listen_speak_bot_example import ListenSpeakBot
else:
    ListenSpeakBot = pytest.importorskip(
        "listen_speak_bot_example",
        reason="listen_speak_bot example package not installed (run via the examples/listen_speak_bot project env)",
    ).ListenSpeakBot


class _FixedVoiceActivityClassifier(voice.detection.VoiceActivityClassifier):
    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        if input == b"speech":
            return voice.detection.ApplyData(
                segments=(
                    voice.detection.VoiceDetectionSegment(
                        start_seconds=0.0,
                        end_seconds=0.1,
                        confidence=0.9,
                    ),
                )
            )
        return voice.detection.ApplyData(segments=())


class _CloseTrackingDecoder:
    instances: ClassVar[list["_CloseTrackingDecoder"]] = []

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.closed = False
        self.instances.append(self)

    async def decode(self, input: bytes) -> bytes:
        return input

    async def aclose(self) -> None:
        self.closed = True


class _CloseTrackingVoiceActivityClassifier(_FixedVoiceActivityClassifier):
    instances: ClassVar[list["_CloseTrackingVoiceActivityClassifier"]] = []

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.closed = False
        self.instances.append(self)

    async def aclose(self) -> None:
        self.closed = True


class _RecordingListening(listening.Listening):
    handoffs: list[cognition.InputData]

    def __init__(
        self,
        *,
        voice_activity_classifier: voice.detection.VoiceActivityClassifier,
        voice_classifier: abilities.Classifier[
            voice.identification.InputData,
            voice.identification.OutputData,
        ],
    ) -> None:
        super().__init__(
            voice_activity_classifier=voice_activity_classifier,
            voice_classifier=voice_classifier,
        )
        self.handoffs = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name and isinstance(event.data, cognition.InputData):
            self.handoffs.append(event.data)
        return super().dispatch(ctx, event)


class _RecordingInference:
    def __init__(self) -> None:
        self.calls: list[pathlib.Path] = []

    def __call__(self, audio: pathlib.Path) -> object:
        with wave.open(str(audio), "rb") as stream:
            assert stream.getframerate() == 16_000
            assert stream.getnchannels() == 1
            assert stream.getsampwidth() == 2
        self.calls.append(audio)
        return [0.12, -0.08, 0.31]


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


_EventData = TypeVar("_EventData")


class _WaitableBot(Protocol):
    async def wait_for_conversation_processing(self) -> None: ...


def _routed_event(
    event: hsm.Event[_EventData],
    *,
    source: str,
    target: str,
) -> hsm.Event[_EventData]:
    return dataclasses.replace(event, source=source, target=target)


async def _assert_not_ready(body: _WaitableBot) -> None:
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(body.wait_for_conversation_processing(), timeout=0.01)


async def _assert_ready(body: _WaitableBot) -> None:
    await asyncio.wait_for(body.wait_for_conversation_processing(), timeout=0.1)


def _empty_cognition() -> cognition.Cognition:
    from mosfet.abilities import memory, processing

    class _EmptyProcessor(processing.Processor):
        @override
        async def process(self, input: processing.InputData) -> processing.Events:
            del input
            await asyncio.Event().wait()
            raise AssertionError("cancelled test processor resumed")

    processor = _EmptyProcessor()
    return cognition.Cognition(
        intuition=cognition.Intuition(processor=processor),
        reasoning=cognition.Reasoning(processor=processor),
        reflection=cognition.Reflection(processor=processor, memory=memory.Memory()),
    )


def _test_bot(
    tmp_path: pathlib.Path,
    *,
    inference: pyannote.SpeakerEmbeddingInference | None = None,
    load_inference: pyannote.SpeakerEmbeddingInferenceLoader | None = None,
) -> ListenSpeakBot:
    return ListenSpeakBot(
        reply_wav=tmp_path / "reply.wav",
        cognition=_empty_cognition(),
        voice_activity_classifier=_FixedVoiceActivityClassifier(),
        voice_identification_inference=inference,
        voice_identification_loader=load_inference,
    )


class _HandoffProbeBot(ListenSpeakBot):
    _handoff_waiter: asyncio.Future[cognition.InputData] | None = None

    def prepare_handoff_waiter(self) -> asyncio.Future[cognition.InputData]:
        self._handoff_waiter = asyncio.get_running_loop().create_future()
        return self._handoff_waiter

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if (
            event.name == cognition.InputEvent.name
            and isinstance(event.data, cognition.InputData)
            and event.source == hsm.id(self.listening())
            and self._handoff_waiter is not None
            and not self._handoff_waiter.done()
        ):
            self._handoff_waiter.set_result(event.data)
        return super().dispatch(ctx, event)


class _FakeRunEncoder:
    calls: list[str]
    audio: bytes | None

    def __init__(self, *, calls: tuple[str, ...], audio: bytes | None) -> None:
        self.calls = list(calls)
        self.audio = audio


class _FakeRunBot:
    _encoder: _FakeRunEncoder

    configured_response_selections: ClassVar[int] = 1
    configured_response_execution_failed: ClassVar[bool] = False
    configured_encoder_calls: ClassVar[tuple[str, ...]] = ("reply",)
    configured_encoder_audio: ClassVar[bytes | None] = b"reply-audio"

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self._encoder = _FakeRunEncoder(
            calls=self.configured_encoder_calls,
            audio=self.configured_encoder_audio,
        )

    async def attach(self, environment: Environment) -> None:
        del environment

    async def dispatch(self, context: hsm.Context, event: hsm.Event) -> None:
        del context, event

    async def wait_for_conversation_processing(self) -> None:
        return

    def response_selection_count(self) -> int:
        return self.configured_response_selections

    def listening_handoffs(self) -> tuple[object, ...]:
        return ()

    def completed(self) -> tuple[object, ...]:
        return ()

    def conversation_failures(self) -> tuple[object, ...]:
        return ()

    def failures(self) -> tuple[object, ...]:
        return ()

    def response_execution_failed(self) -> bool:
        return self.configured_response_execution_failed

    async def detach(self, environment: Environment) -> None:
        del environment

    def encoder(self) -> _FakeRunEncoder:
        return self._encoder


def _run_with_public_seams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    *,
    response_selections: int,
    response_execution_failed: bool = False,
    encoder_calls: tuple[str, ...] = ("reply",),
    encoder_audio: bytes | None = b"reply-audio",
) -> dict[str, object]:
    import listen_speak_bot_example as example

    _FakeRunBot.configured_response_selections = response_selections
    _FakeRunBot.configured_response_execution_failed = response_execution_failed
    _FakeRunBot.configured_encoder_calls = encoder_calls
    _FakeRunBot.configured_encoder_audio = encoder_audio
    monkeypatch.setattr(example, "ListenSpeakBot", _FakeRunBot)

    def fake_which(name: str) -> str:
        return name

    monkeypatch.setattr(shutil, "which", fake_which)

    def fake_subprocess_run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        if args[0] == "afconvert":
            _ = pathlib.Path(args[-1]).write_bytes(b"heard")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    from listen_speak_bot_example import AppConfig, CognitionConfig, run

    config = AppConfig(cognition=CognitionConfig(api_key="gemini", intuition_api_key="mercury"))
    return asyncio.run(run(assets_dir=tmp_path, config=config))


def test_listen_speak_bot_owns_speaker(tmp_path: pathlib.Path) -> None:
    body = ListenSpeakBot(
        reply_wav=tmp_path / "reply.wav",
        cognition=_empty_cognition(),
        voice_activity_classifier=_FixedVoiceActivityClassifier(),
        voice_identity_model_id="local/test-speaker-model",
        voice_identification_inference=_RecordingInference(),
    )

    assert isinstance(body.speaker(), audio.Speaker)


def test_environment_sound_reaches_listening_after_attach(tmp_path: pathlib.Path) -> None:
    async def run() -> None:
        body = _HandoffProbeBot(
            reply_wav=tmp_path / "reply.wav",
            cognition=_empty_cognition(),
            voice_activity_classifier=_FixedVoiceActivityClassifier(),
            voice_identification_inference=_RecordingInference(),
        )
        environment = Environment()
        waiter = body.prepare_handoff_waiter()
        try:
            _ = await body.attach(environment)
            await body.dispatch(
                environment,
                SoundEvent.with_data(
                    SoundData(
                        audio=b"speech",
                        media_type="audio/pcm",
                        sample_rate_hz=16_000,
                        channels=1,
                    )
                ),
            )
            await body.dispatch(
                environment,
                SoundEvent.with_data(
                    SoundData(
                        audio=b"silence",
                        media_type="audio/pcm",
                        sample_rate_hz=16_000,
                        channels=1,
                    )
                ),
            )
            handoff = await asyncio.wait_for(waiter, timeout=1.0)
            assert isinstance(handoff.stimulus, hsm.Event)
        finally:
            _ = await body.detach(environment)

    asyncio.run(run())


def test_listen_speak_run_progress_waits_for_correlated_speaking_terminal(tmp_path: pathlib.Path) -> None:
    import mosfet
    from mosfet.abilities import speaking
    from mosfet.abilities.communication import communication, conversation

    async def run() -> None:
        body = _test_bot(tmp_path)
        environment = Environment()
        _ = await body.attach(environment)
        turn_id = "conversation-turn"
        history = conversation.OutputEvent.with_data(conversation.Messages())
        await body.dispatch(
            environment,
            _routed_event(
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=history), turn_id),
                source=hsm.id(body.listening()),
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                mosfet.ProcessingCompletedEvent.with_data_and_id(
                    mosfet.ProcessingCompletedEventData(
                        output=(
                            cognition.EventData(
                                event=communication.RespondEvent.name,
                                target="communication",
                                data=communication.RespondData(text="I am well."),
                            ),
                        ),
                        focus_candidates=(),
                    ),
                    turn_id,
                ),
                source="cognition",
                target=hsm.id(body),
            ),
        )

        await _assert_not_ready(body)

        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(speaking.OutputData(text="I am well."), "stale-operation"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_not_ready(body)

        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(speaking.OutputData(text="I am well."), f"{turn_id}:intuition"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_ready(body)
        _ = await body.detach(environment)

    asyncio.run(run())


def test_listen_speak_run_progress_requires_one_terminal_per_response_selection(tmp_path: pathlib.Path) -> None:
    import mosfet
    from mosfet.abilities import speaking
    from mosfet.abilities.communication import communication, conversation

    async def run() -> None:
        body = _test_bot(tmp_path)
        environment = Environment()
        _ = await body.attach(environment)
        turn_id = "conversation-turn"
        history = conversation.OutputEvent.with_data(conversation.Messages())
        await body.dispatch(
            environment,
            _routed_event(
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=history), turn_id),
                source=hsm.id(body.listening()),
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                mosfet.ProcessingCompletedEvent.with_data_and_id(
                    mosfet.ProcessingCompletedEventData(
                        output=(
                            cognition.EventData(
                                event=communication.RespondEvent.name,
                                target="communication",
                                data=communication.RespondData(text="Reply A."),
                            ),
                            cognition.EventData(
                                event=communication.RespondEvent.name,
                                target="communication",
                                data=communication.RespondData(text="Reply B."),
                            ),
                        ),
                        focus_candidates=(),
                    ),
                    turn_id,
                ),
                source="cognition",
                target=hsm.id(body),
            ),
        )

        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(
                    speaking.OutputData(text="Reply A."),
                    "different-response-operation-a",
                ),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_not_ready(body)

        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(
                    speaking.OutputData(text="Reply B."),
                    "different-response-operation-b",
                ),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_not_ready(body)

        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(speaking.OutputData(text="Reply A."), turn_id),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_ready(body)
        assert body.response_selection_count() == 1
        _ = await body.detach(environment)

    asyncio.run(run())


def test_listen_speak_run_progress_waits_for_earlier_response_before_later_no_response(tmp_path: pathlib.Path) -> None:
    import mosfet
    from mosfet.abilities import speaking
    from mosfet.abilities.communication import communication, conversation

    async def run() -> None:
        body = _test_bot(tmp_path)
        environment = Environment()
        _ = await body.attach(environment)
        history = conversation.OutputEvent.with_data(conversation.Messages())
        await body.dispatch(
            environment,
            _routed_event(
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=history), "turn-a"),
                source=hsm.id(body.listening()),
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                mosfet.ProcessingCompletedEvent.with_data_and_id(
                    mosfet.ProcessingCompletedEventData(
                        output=(
                            cognition.EventData(
                                event=communication.RespondEvent.name,
                                target="communication",
                                data=communication.RespondData(text="Reply A."),
                            ),
                        ),
                        focus_candidates=(),
                    ),
                    "turn-a",
                ),
                source="cognition",
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=history), "turn-b"),
                source=hsm.id(body.listening()),
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                mosfet.ProcessingCompletedEvent.with_data_and_id(
                    mosfet.ProcessingCompletedEventData(output=(), focus_candidates=()),
                    "turn-b",
                ),
                source="cognition",
                target=hsm.id(body),
            ),
        )

        await _assert_not_ready(body)

        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(speaking.OutputData(text="Reply A."), "response-a"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_not_ready(body)

        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(speaking.OutputData(text="Reply A."), "turn-a"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_ready(body)
        _ = await body.detach(environment)

    asyncio.run(run())


def test_listen_speak_run_progress_reports_processing_failure(tmp_path: pathlib.Path) -> None:
    import mosfet
    from mosfet.abilities.communication import conversation

    async def run() -> None:
        body = _test_bot(tmp_path)
        environment = Environment()
        _ = await body.attach(environment)
        turn_id = "conversation-turn"
        history = conversation.OutputEvent.with_data(conversation.Messages())
        await body.dispatch(
            environment,
            _routed_event(
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=history), turn_id),
                source=hsm.id(body.listening()),
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                mosfet.ProcessingFailedEvent.with_data_and_id(
                    mosfet.ProcessingFailedEventData(message="processing failed"),
                    turn_id,
                ),
                source="cognition",
                target=hsm.id(body),
            ),
        )
        await _assert_ready(body)
        _ = await body.detach(environment)

    asyncio.run(run())


def test_listen_speak_run_progress_accepts_no_response_completion(tmp_path: pathlib.Path) -> None:
    import mosfet
    from mosfet.abilities.communication import conversation

    async def run() -> None:
        body = _test_bot(tmp_path)
        environment = Environment()
        _ = await body.attach(environment)
        turn_id = "conversation-turn"
        history = conversation.OutputEvent.with_data(conversation.Messages())
        await body.dispatch(
            environment,
            _routed_event(
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=history), turn_id),
                source=hsm.id(body.listening()),
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                mosfet.ProcessingCompletedEvent.with_data_and_id(
                    mosfet.ProcessingCompletedEventData(output=(), focus_candidates=()),
                    turn_id,
                ),
                source="cognition",
                target=hsm.id(body),
            ),
        )
        await _assert_ready(body)
        _ = await body.detach(environment)

    asyncio.run(run())


def test_listen_speak_run_progress_reports_conversation_failure(tmp_path: pathlib.Path) -> None:
    from mosfet.abilities import ability
    from mosfet.abilities.communication import conversation

    async def run() -> None:
        body = _test_bot(tmp_path)
        environment = Environment()
        _ = await body.attach(environment)
        await body.dispatch(
            environment,
            _routed_event(
                conversation.FailedEvent.with_data_and_id(
                    ability.FailureData(message="MLX Audio speech decoding failed."),
                    "conversation-turn",
                ),
                source=hsm.id(body.conversation()),
                target=hsm.id(body),
            ),
        )
        await _assert_ready(body)
        assert body.conversation_failures()
        _ = await body.detach(environment)

    asyncio.run(run())


def test_listen_speak_communication_drives_body_conversation_and_speaking(tmp_path: pathlib.Path) -> None:
    """The composed Communication ability admits handoffs into the body conversation.

    Behavioral wiring proof for the example composition (replaces
    constructor-text asserts): a listening handoff plus a selected
    ``communication.respond`` completes the run through the body's own
    conversation and speaking ports, recorded on public seams.
    """

    import mosfet
    from mosfet.abilities import speaking
    from mosfet.abilities.communication import communication, conversation

    async def run() -> tuple[int, tuple[object, ...]]:
        body = _test_bot(tmp_path)
        assert isinstance(body.communication(), communication.Communication)
        assert isinstance(body.conversation(), conversation.Conversation)
        assert isinstance(body.speaking(), speaking.Speaking)
        environment = Environment()
        _ = await body.attach(environment)
        turn_id = "wiring-turn"
        history = conversation.OutputEvent.with_data(conversation.Messages())
        await body.dispatch(
            environment,
            _routed_event(
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=history), turn_id),
                source=hsm.id(body.listening()),
                target=hsm.id(body),
            ),
        )
        await asyncio.sleep(0.05)
        await body.dispatch(
            environment,
            _routed_event(
                mosfet.ProcessingCompletedEvent.with_data_and_id(
                    mosfet.ProcessingCompletedEventData(
                        output=(
                            cognition.EventData(
                                event=communication.RespondEvent.name,
                                target="communication",
                                data=communication.RespondData(text="Hello."),
                            ),
                        ),
                        focus_candidates=(),
                    ),
                    turn_id,
                ),
                source="cognition",
                target=hsm.id(body),
            ),
        )
        # Let run progress observe the selected response before its speaking
        # terminal arrives; production always separates the two by Speaking work.
        await asyncio.sleep(0.05)
        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(speaking.OutputData(text="Hello."), f"{turn_id}:intuition"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_ready(body)
        selections = body.response_selection_count()
        done = body.completed()
        _ = await body.detach(environment)
        return selections, done

    selections, completed = asyncio.run(run())
    assert selections == 1
    assert completed


def test_listen_speak_example_uses_configured_silero_vad_and_real_silence_boundary() -> None:
    root = _repo_root()
    source = (root / "examples" / "listen_speak_bot" / "src" / "listen_speak_bot_example" / "__init__.py").read_text(
        encoding="utf-8"
    )
    pyproject = (root / "examples" / "listen_speak_bot" / "pyproject.toml").read_text(encoding="utf-8")
    readme = (root / "examples" / "listen_speak_bot" / "README.md").read_text(encoding="utf-8")

    assert "mosfet-provider-mlx-audio" in pyproject
    assert "mosfet-provider-pyannote" in pyproject
    assert "from mosfet.providers.mlx_audio import VoiceActivityClassifier as SileroVoiceActivityClassifier" in source
    assert "from mosfet.providers.mlx_audio import SpeechDecoder" in source
    assert "from mosfet.providers.mlx_audio import VoiceDecoder as MlxVoiceDecoder" in source
    assert "from mosfet.providers.pyannote import Classifier as PyannoteVoiceClassifier" in source
    assert "from mosfet.abilities import communication" in source
    assert "from mosfet.abilities.communication import conversation" in source
    assert "seeded_behaviors=(communication.speech_heard_seed(),)" in source
    assert "memory=store" in source
    assert 'DEFAULT_SILERO_VAD_MODEL = "mlx-community/silero-vad"' in source
    assert 'DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"' in source
    assert '"BOT_SILERO_VAD_MODEL", "BOT_VAD_MODEL", "SILERO_VAD_MODEL"' in source
    assert '"BOT_STT_MODEL", "BOT_WHISPER_MODEL", "STT_MODEL", "WHISPER_MODEL"' in source
    assert "stt_model_id: str | None = None" in source
    assert "voice_activity_classifier: voice.detection.VoiceActivityClassifier | None = None" in source
    assert "voice_classifier=classifier" in source
    assert "voice_identity_model_id: str = DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL" in source
    assert '"BOT_PYANNOTE_VOICE_IDENTITY_MODEL",' in source
    assert "speech_decoder=None" in source
    assert "MlxVoiceDecoder(speech_decoder=speech_decoder)" in source
    assert "_SequencedVoiceActivityClassifier" not in source
    assert "model_id=vad_model_id" in source
    assert "sample_rate_hz=sample_rate_hz" in source
    assert "channels=_DEFAULT_CHANNELS" in source
    assert "def _silence_wav(" in source
    assert 'media_type="audio/wav"' in source
    assert 'kind="silence"' in source
    assert source.count("await body.dispatch(") >= 2
    assert "wait_for_first_listening_handoff" not in source
    assert "_first_listening_handoff" not in source
    assert "FixedTranscriptDecoder" not in source
    assert ".decoder().calls" not in source
    assert "await body.wait_for_conversation_processing()" in source
    assert "await self.wait_for_activation()" in source
    assert "ActivatingDoneEventData" in source
    assert "ActivatingFailedEventData" in source
    assert "_BotLifecycle" not in source
    assert "bot.lifecycle." not in source
    # Response-selection tracking is pinned behaviorally, not textually: one
    # terminal per selection, stale/duplicate rejection, and failure correlation
    # are covered by the run-progress tests (one-terminal-per-selection, rejects
    # stale/duplicates, correlates speaking failure).
    assert 'reason="response_execution_failed"' in source
    assert "observe_response_request" not in source
    assert "hsm.choice(" in source
    assert "_RunTerminalEvent" in source
    assert "bool(body.completed()) or bool(body.failures())" not in source
    assert "body.encoder().audio is not None and bool(body.completed())" not in source
    assert 'status="ok" if has_speaking_output else "incomplete"' in source
    assert "status_reason:" in source
    assert "_OperatorSummary" in source
    assert '"response_execution_failed"' in source
    assert "processing_outputs" not in source
    assert "source_ids" not in source
    assert "not select communication.respond" not in source
    assert '"reply_wav": str(reply_wav) if has_speaking_output else None' not in source
    assert 'return 0 if summary.get("status") == "ok" else 1' in source
    assert "status: incomplete" in readme
    assert "real PyAnnote speaker classifier" in readme
    assert "provider-neutral source embedding" in readme
    assert "`communication.respond` action" in readme


def test_listen_speak_reads_pyannote_model_override_without_credentials(tmp_path: pathlib.Path) -> None:
    from listen_speak_bot_example import AppConfig

    env_file = tmp_path / ".env"
    _ = env_file.write_text(
        "BOT_PYANNOTE_VOICE_IDENTITY_MODEL=local/test-speaker-model\n",
        encoding="utf-8",
    )

    config = AppConfig.from_env_file(env_file)

    assert config.voice_identity_model_id == "local/test-speaker-model"


def test_listen_speak_injects_pyannote_classifier_and_emits_source_embedding() -> None:
    async def run() -> tuple[object, _RecordingInference, list[str]]:
        inference = _RecordingInference()
        loaded_models: list[str] = []

        def load_inference(model_id: str) -> pyannote.SpeakerEmbeddingInference:
            loaded_models.append(model_id)
            return inference

        classifier = pyannote.Classifier(
            model_id="local/test-speaker-model",
            load_inference=load_inference,
        )
        listening_ability = _RecordingListening(
            voice_activity_classifier=_FixedVoiceActivityClassifier(),
            voice_classifier=classifier,
        )
        from tests.hsm_instance_state import start_ability_tree

        await start_ability_tree(None, listening_ability)
        from mosfet.environment import SoundData

        await listening_ability.apply(
            SoundData(
                audio=b"speech",
                media_type="audio/pcm",
                sample_rate_hz=16_000,
                channels=1,
            )
        )
        await listening_ability.apply(
            SoundData(
                audio=b"silence",
                media_type="audio/pcm",
                sample_rate_hz=16_000,
                channels=1,
            )
        )
        for _ in range(1000):
            if listening_ability.handoffs:
                break
            await asyncio.sleep(0)
        assert listening_ability.handoffs
        stimulus = listening_ability.handoffs[0].stimulus
        assert isinstance(stimulus, hsm.Event)
        assert stimulus.name == listening.SpeechEvent.name
        assert isinstance(stimulus.data, listening.SpeechData)
        return stimulus.data, inference, loaded_models

    stimulus, inference, loaded_models = asyncio.run(run())
    assert len(inference.calls) == 1
    assert loaded_models == ["local/test-speaker-model"]
    assert isinstance(stimulus, listening.SpeechData)
    assert stimulus.source_ids == frozenset({(0.12, -0.08, 0.31)})


def test_listen_speak_operator_summary_excludes_raw_provider_and_content_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    summary = _run_with_public_seams(
        monkeypatch,
        tmp_path,
        response_selections=1,
    )

    encoded = json.dumps(summary)
    assert set(summary) == {
        "status",
        "status_reason",
        "heard_audio_bytes",
        "reply_audio_bytes",
        "listening_handoffs",
        "processing_completed",
        "response_selections",
    }
    for forbidden in ("embedding", "source_ids", "media", "content", "provenance", "processing_outputs"):
        assert forbidden not in encoded


def test_listen_speak_operator_summary_distinguishes_no_response_and_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    no_response = _run_with_public_seams(
        monkeypatch,
        tmp_path / "no-response",
        response_selections=0,
        encoder_calls=(),
        encoder_audio=None,
    )
    failed_response = _run_with_public_seams(
        monkeypatch,
        tmp_path / "failed-response",
        response_selections=1,
        response_execution_failed=True,
    )

    assert no_response["status"] == "incomplete"
    assert no_response["status_reason"] == "no_response_selected"
    assert failed_response["status"] == "failed"
    assert failed_response["status_reason"] == "response_execution_failed"


def test_say_encoder_releases_default_destination_after_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    import listen_speak_bot_example as example
    from listen_speak_bot_example import SayEncoder

    descriptor, destination = tempfile.mkstemp(dir=tmp_path, suffix=".wav")
    monkeypatch.setattr(example.tempfile, "mkstemp", lambda **kwargs: (descriptor, destination))
    monkeypatch.setattr(example, "_say_to_wav", lambda *args, **kwargs: b"wav")

    assert asyncio.run(SayEncoder().encode(b"reply")) == b"wav"
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert not pathlib.Path(destination).exists()


def test_say_encoder_releases_default_destination_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    import listen_speak_bot_example as example
    from listen_speak_bot_example import SayEncoder

    descriptor, destination = tempfile.mkstemp(dir=tmp_path, suffix=".wav")
    monkeypatch.setattr(example.tempfile, "mkstemp", lambda **kwargs: (descriptor, destination))

    def fail_render(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise RuntimeError("render failed")

    monkeypatch.setattr(example, "_say_to_wav", fail_render)

    with pytest.raises(RuntimeError, match="render failed"):
        asyncio.run(SayEncoder().encode(b"reply"))
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert not pathlib.Path(destination).exists()


def test_say_encoder_cancellation_does_not_publish_cancelled_encoding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    import listen_speak_bot_example as example
    from listen_speak_bot_example import SayEncoder

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    wait_timeout_seconds = 2.0
    descriptor, destination = tempfile.mkstemp(dir=tmp_path, suffix=".wav")
    monkeypatch.setattr(example.tempfile, "mkstemp", lambda **kwargs: (descriptor, destination))

    def blocked_say_to_wav(text: str, destination: pathlib.Path, *, sample_rate_hz: int) -> bytes:
        del text, destination, sample_rate_hz
        started.set()
        try:
            assert release.wait(timeout=wait_timeout_seconds)
            return b"wav"
        finally:
            finished.set()

    monkeypatch.setattr(example, "_say_to_wav", blocked_say_to_wav)
    encoder = SayEncoder()

    async def run() -> None:
        task = asyncio.create_task(encoder.encode(b"reply"))
        _ = await asyncio.to_thread(started.wait, wait_timeout_seconds)
        _ = task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert encoder.calls == []
        assert encoder.audio is None
        assert await asyncio.to_thread(finished.wait, wait_timeout_seconds)

    asyncio.run(run())
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert not pathlib.Path(destination).exists()


def test_say_encoder_preserves_explicit_destination(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    import listen_speak_bot_example as example
    from listen_speak_bot_example import SayEncoder

    destination = tmp_path / "reply.wav"

    def render(text: str, destination: pathlib.Path, *, sample_rate_hz: int) -> bytes:
        del text, sample_rate_hz
        destination.write_bytes(b"wav")
        return b"wav"

    monkeypatch.setattr(example, "_say_to_wav", render)

    assert asyncio.run(SayEncoder(destination=destination).encode(b"reply")) == b"wav"
    assert destination.read_bytes() == b"wav"


def test_listen_speak_progress_rejects_stale_and_duplicate_speaking_terminals(tmp_path: pathlib.Path) -> None:
    import mosfet
    from mosfet.abilities import speaking
    from mosfet.abilities.communication import communication, conversation

    async def run() -> None:
        body = _test_bot(tmp_path)
        environment = Environment()
        _ = await body.attach(environment)
        history = conversation.OutputEvent.with_data(conversation.Messages())
        await body.dispatch(
            environment,
            _routed_event(
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=history), "turn-id"),
                source=hsm.id(body.listening()),
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                mosfet.ProcessingCompletedEvent.with_data_and_id(
                    mosfet.ProcessingCompletedEventData(
                        output=(
                            cognition.EventData(
                                event=communication.RespondEvent.name,
                                target="communication",
                                data=communication.RespondData(text="Reply."),
                            ),
                        ),
                        focus_candidates=(),
                    ),
                    "turn-id",
                ),
                source="cognition",
                target=hsm.id(body),
            ),
        )

        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(speaking.OutputData(text="stale"), "stale-id"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_not_ready(body)

        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(speaking.OutputData(text="Reply."), "turn-id"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(speaking.OutputData(text="duplicate"), "turn-id"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_ready(body)
        _ = await body.detach(environment)

    asyncio.run(run())


def test_listen_speak_progress_correlates_speaking_failure_to_selected_operation(tmp_path: pathlib.Path) -> None:
    import mosfet
    from mosfet.abilities import ability, speaking
    from mosfet.abilities.communication import communication, conversation

    async def run() -> None:
        body = _test_bot(tmp_path)
        environment = Environment()
        _ = await body.attach(environment)
        history = conversation.OutputEvent.with_data(conversation.Messages())
        await body.dispatch(
            environment,
            _routed_event(
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=history), "turn-id"),
                source=hsm.id(body.listening()),
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                mosfet.ProcessingCompletedEvent.with_data_and_id(
                    mosfet.ProcessingCompletedEventData(
                        output=(
                            cognition.EventData(
                                event=communication.RespondEvent.name,
                                target="communication",
                                data=communication.RespondData(text="Reply."),
                            ),
                        ),
                        focus_candidates=(),
                    ),
                    "turn-id",
                ),
                source="cognition",
                target=hsm.id(body),
            ),
        )

        failure = ability.FailureData(message="encoding failed")
        await body.dispatch(
            environment,
            _routed_event(
                ability.FailedEvent.with_data_and_id(failure, "stale-id"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_not_ready(body)
        assert not body.response_execution_failed()

        await body.dispatch(
            environment,
            _routed_event(
                ability.FailedEvent.with_data_and_id(failure, "turn-id"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await body.dispatch(
            environment,
            _routed_event(
                speaking.OutputEvent.with_data_and_id(speaking.OutputData(text="duplicate"), "turn-id"),
                source=hsm.id(body.speaking()),
                target=hsm.id(body),
            ),
        )
        await _assert_ready(body)
        assert body.response_execution_failed()
        _ = await body.detach(environment)

    asyncio.run(run())


def test_listen_speak_progress_has_no_snapshot_or_asyncio_event_protocol() -> None:
    root = _repo_root()
    source = (root / "examples" / "listen_speak_bot" / "src" / "listen_speak_bot_example" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert "asyncio.Event" not in source
    assert "body.state()" not in source
    assert "_RunProgress(hsm.Instance)" in source


def test_listen_speak_models_progress_without_dispatch_interception_or_actor_waiters() -> None:
    source = (
        _repo_root() / "examples" / "listen_speak_bot" / "src" / "listen_speak_bot_example" / "__init__.py"
    ).read_text(encoding="utf-8")
    bot_source = source.split("class ListenSpeakBot", 1)[1]
    progress_source = source.split("class _RunProgress", 1)[1].split("class ListenSpeakBot", 1)[0]

    assert "machine.dispatch =" not in source
    assert "def dispatch(" not in bot_source
    assert "asyncio.Future" not in bot_source
    assert "event.name in" not in bot_source
    assert "def is_ready(" not in progress_source
    assert "async def wait(" not in progress_source
    assert "def conversation_failures(" not in progress_source
    assert "def response_selection_count(" not in progress_source
    assert "def response_execution_failed(" not in progress_source


def test_listen_speak_closes_owned_decoder_and_vad_on_detach(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    import listen_speak_bot_example as example

    _CloseTrackingDecoder.instances.clear()
    _CloseTrackingVoiceActivityClassifier.instances.clear()
    monkeypatch.setattr(example, "SpeechDecoder", _CloseTrackingDecoder)
    monkeypatch.setattr(example, "SileroVoiceActivityClassifier", _CloseTrackingVoiceActivityClassifier)

    async def run() -> None:
        body = ListenSpeakBot(reply_wav=tmp_path / "reply.wav", cognition=_empty_cognition())
        environment = Environment()
        _ = await body.attach(environment)
        _ = await body.detach(environment)

    asyncio.run(run())

    assert _CloseTrackingDecoder.instances and _CloseTrackingDecoder.instances[0].closed
    assert _CloseTrackingVoiceActivityClassifier.instances and _CloseTrackingVoiceActivityClassifier.instances[0].closed


def test_listen_speak_closes_owned_decoder_and_vad_when_attach_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    import listen_speak_bot_example as example

    _CloseTrackingDecoder.instances.clear()
    _CloseTrackingVoiceActivityClassifier.instances.clear()
    monkeypatch.setattr(example, "SpeechDecoder", _CloseTrackingDecoder)
    monkeypatch.setattr(example, "SileroVoiceActivityClassifier", _CloseTrackingVoiceActivityClassifier)

    async def fail_attach(
        instance: object,
        environment: Environment,
        *,
        placement: object | None = None,
    ) -> object:
        del instance, environment, placement
        raise RuntimeError("attach failed")

    monkeypatch.setattr(example.Bot, "attach", fail_attach)

    async def run() -> None:
        body = ListenSpeakBot(reply_wav=tmp_path / "reply.wav", cognition=_empty_cognition())
        with pytest.raises(RuntimeError, match="attach failed"):
            _ = await body.attach(Environment())

    asyncio.run(run())

    assert _CloseTrackingDecoder.instances and _CloseTrackingDecoder.instances[0].closed
    assert _CloseTrackingVoiceActivityClassifier.instances and _CloseTrackingVoiceActivityClassifier.instances[0].closed


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


def test_listen_speak_detach_waits_for_lifecycle_terminal_and_leaves_no_lifecycle_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    original_detach = attachment.Group.detach
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_detach(
        group: attachment.Group,
        ctx: hsm.Context,
        event: hsm.Event[attachment.DetachData],
    ) -> None:
        _ = entered.set()
        await release.wait()
        await original_detach(group, ctx, event)

    monkeypatch.setattr(attachment.Group, "detach", gated_detach)

    async def run() -> tuple[str, ...]:
        body = _test_bot(tmp_path)
        environment = Environment()
        _ = await body.attach(environment)
        detach_task = asyncio.create_task(body.detach(environment))
        await entered.wait()
        assert not detach_task.done()
        _ = release.set()
        _ = await detach_task
        current = asyncio.current_task()
        return tuple(repr(task.get_coro()) for task in asyncio.all_tasks() if task is not current and not task.done())

    pending = asyncio.run(run())

    assert not any(
        lifecycle_name in coroutine
        for coroutine in pending
        for lifecycle_name in ("AttachmentGroup", "BotLifecycleReply", "HSM._process")
    )
