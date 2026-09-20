"""Device-free Listening/cognition bot example (macOS ``say`` + mixed cognition).

Flow:

1. macOS ``say`` renders *Hey I'm Gabe how are you* into a WAV asset.
2. ``environment.sound`` carries that audio into bot **input** (Listening).
3. MLX Audio Silero VAD + PyAnnote voice embeddings hand a provider-neutral source identity
   with the speech product to cognition.
4. Communication is acquired and its seeded behavior can admit the identity-bearing speech into
   Conversation; Conversation retains the existing MLX Audio Whisper STT path.
5. Mercury 2 intuition (OpenAI-compatible) / Gemini reasoning produce cognition output.
6. Cognition may select Communication's semantic response, which routes through the internal
   Speaking port and writes the reply WAV.
"""

from __future__ import annotations

import argparse
import asyncio
import collections.abc
import dataclasses
import datetime
import functools
import io
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import typing
import wave
from typing import override

import hsm
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

import mosfet
from mosfet.abilities import cognition
from mosfet.abilities import ability
from mosfet.abilities import communication
from mosfet.abilities import decoding
from mosfet.abilities import encoding
from mosfet.abilities import listening
from mosfet.abilities import memory
from mosfet.abilities import speaking
from mosfet.abilities.communication import conversation
from mosfet.abilities.hearing import voice
from mosfet.bot import Bot
from mosfet.providers.gemini import ChatClient as GeminiChatClient
from mosfet.providers.gemini import Processor as GeminiProcessor
from mosfet.providers.mlx_audio import SpeechDecoder
from mosfet.providers.mlx_audio import VoiceDecoder as MlxVoiceDecoder
from mosfet.providers.mlx_audio import VoiceActivityClassifier as SileroVoiceActivityClassifier
from mosfet.providers.openai_compat import ChatClient as OpenAIChatClient
from mosfet.providers.openai_compat import Processor as OpenAIProcessor
from mosfet.providers.pyannote import Classifier as PyannoteVoiceClassifier
from mosfet.providers.pyannote import SpeakerEmbeddingInference
from mosfet.providers.pyannote import SpeakerEmbeddingInferenceLoader
from mosfet.devices import audio
from mosfet.environment import Environment, SoundData, SoundEvent, space
from mosfet.abilities import classifying
from mosfet.abilities.communication.conversation import turn_detector

# Capture before any local named ``cognition`` shadows the package (constructor param).
_Cognition = cognition.Cognition

_LOG = logging.getLogger("listen_speak_bot_example")

_EXAMPLE_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REPO_ROOT = _EXAMPLE_ROOT.parent.parent
_ASSETS_DIR = _EXAMPLE_ROOT / "assets"
_DEFAULT_ENV_PATH = _EXAMPLE_ROOT / ".env"
_REPO_ENV_PATH = _REPO_ROOT / ".env"
_HEARD_PHRASE = "Hey I'm Gabe how are you"
_DEFAULT_SAMPLE_RATE_HZ = 22_050
_DEFAULT_CHANNELS = 1
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_MERCURY_MODEL = "mercury-2"
DEFAULT_MERCURY_BASE_URL = "https://api.inceptionlabs.ai/v1"
DEFAULT_SILERO_VAD_MODEL = "mlx-community/silero-vad"
DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


def _configure_logging(*, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    _LOG.setLevel(level)


def _summarize_selections(output: object) -> str:
    if output is None:
        return "unhandled"
    if isinstance(output, tuple):
        if not output:
            return "handled empty"
        names: list[str] = []
        for item in output:
            event = getattr(item, "event", None)
            names.append(str(event) if event is not None else type(item).__name__)
        return f"handled events={names}"
    return f"handled type={type(output).__name__}"


_SummaryStatus = typing.Literal["ok", "incomplete", "failed"]
_SummaryReason = typing.Literal[
    "response_spoken",
    "no_response_selected",
    "conversation_failed",
    "processing_failed",
    "response_execution_failed",
]


class _OperatorSummary(BaseModel):
    """Bounded, provider-neutral output for operators and JSON consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: _SummaryStatus
    status_reason: _SummaryReason
    heard_audio_bytes: int = Field(ge=0)
    reply_audio_bytes: int = Field(ge=0)
    listening_handoffs: int = Field(ge=0)
    processing_completed: int = Field(ge=0)
    response_selections: int = Field(ge=0)


def _build_operator_summary(
    *,
    status: _SummaryStatus,
    status_reason: _SummaryReason,
    heard_audio_bytes: int,
    reply_audio_bytes: int,
    listening_handoffs: int,
    processing_completed: int,
    response_selections: int,
) -> dict[str, object]:
    """Project one run into a stable JSON-safe summary without provider payloads."""

    summary = _OperatorSummary(
        status=status,
        status_reason=status_reason,
        heard_audio_bytes=heard_audio_bytes,
        reply_audio_bytes=reply_audio_bytes,
        listening_handoffs=listening_handoffs,
        processing_completed=processing_completed,
        response_selections=response_selections,
    )
    return typing.cast(dict[str, object], summary.model_dump(mode="json"))


def _strip_env_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def load_env(path: pathlib.Path) -> dict[str, str]:
    """Load a simple KEY=VALUE provider file."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key:
            values[key] = _strip_env_value(value)
    return values


def _env_first(env: collections.abc.Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return None


def _merged_env(path: pathlib.Path | None) -> dict[str, str]:
    """Process env as base; repo/.env then example/.env win (local credentials)."""

    values = {key: value for key, value in os.environ.items() if value}
    if _REPO_ENV_PATH.exists():
        values.update(load_env(_REPO_ENV_PATH))
    if path is not None and path.exists():
        values.update(load_env(path))
    return values


@dataclasses.dataclass(frozen=True)
class CognitionConfig:
    """Reasoning stays on Gemini; intuition defaults to Mercury 2 (OpenAI-compatible)."""

    model: str = DEFAULT_GEMINI_MODEL
    api_key: str | None = None
    intuition_model: str = DEFAULT_MERCURY_MODEL
    intuition_api_key: str | None = None
    intuition_base_url: str = DEFAULT_MERCURY_BASE_URL

    @classmethod
    def from_env(cls, env: collections.abc.Mapping[str, str]) -> typing.Self:
        return cls(
            model=_env_first(env, "BOT_GEMINI_MODEL", "GEMINI_MODEL", "VA_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
            api_key=_env_first(
                env,
                "BOT_GEMINI_API_KEY",
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
                "VA_GEMINI_API_KEY",
            ),
            intuition_model=_env_first(
                env,
                "BOT_MERCURY_MODEL",
                "BOT_INTUITION_MODEL",
                "MERCURY_MODEL",
            )
            or DEFAULT_MERCURY_MODEL,
            intuition_api_key=_env_first(
                env,
                "BOT_MERCURY_API_KEY",
                "MERCURY_API_KEY",
                "INCEPTION_API_KEY",
            ),
            intuition_base_url=_env_first(
                env,
                "BOT_MERCURY_BASE_URL",
                "MERCURY_BASE_URL",
                "INCEPTION_BASE_URL",
            )
            or DEFAULT_MERCURY_BASE_URL,
        )

    def can_process(self) -> bool:
        return self.api_key is not None and self.intuition_api_key is not None


@dataclasses.dataclass(frozen=True)
class AppConfig:
    env_path: pathlib.Path | None = None
    cognition: CognitionConfig = dataclasses.field(default_factory=CognitionConfig)
    vad_model_id: str = DEFAULT_SILERO_VAD_MODEL
    stt_model_id: str | None = None
    voice_identity_model_id: str = DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL

    @classmethod
    def from_env_file(cls, path: pathlib.Path | None = None) -> typing.Self:
        env = _merged_env(path)
        return cls(
            env_path=path,
            cognition=CognitionConfig.from_env(env),
            vad_model_id=(
                _env_first(env, "BOT_SILERO_VAD_MODEL", "BOT_VAD_MODEL", "SILERO_VAD_MODEL") or DEFAULT_SILERO_VAD_MODEL
            ),
            stt_model_id=_env_first(env, "BOT_STT_MODEL", "BOT_WHISPER_MODEL", "STT_MODEL", "WHISPER_MODEL"),
            voice_identity_model_id=(
                _env_first(
                    env,
                    "BOT_PYANNOTE_VOICE_IDENTITY_MODEL",
                    "PYANNOTE_VOICE_IDENTITY_MODEL",
                )
                or DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL
            ),
        )

    def with_cognition_overrides(self, *, model: str | None = None, api_key: str | None = None) -> typing.Self:
        return dataclasses.replace(
            self,
            cognition=dataclasses.replace(
                self.cognition,
                model=model or self.cognition.model,
                api_key=api_key if api_key is not None else self.cognition.api_key,
            ),
        )


def _require_macos_speech_tools() -> None:
    missing = [name for name in ("say", "afconvert") if shutil.which(name) is None]
    if missing:
        raise SystemExit(
            "This example needs macOS speech tools "
            f"({', '.join(missing)} not found). Run on macOS with `say` and `afconvert`."
        )


def _say_to_wav(text: str, destination: pathlib.Path, *, sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ) -> bytes:
    """Render ``text`` with macOS ``say`` and convert to 16-bit mono WAV.

    The subprocess boundary is synchronous inside its worker thread. Cancelling an
    awaiter does not terminate an already-running ``say`` or ``afconvert`` process;
    the worker either leaves the existing destination untouched on failure or
    publishes the complete converted file after both commands succeed.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="listen-speak-") as tmp:
        aiff = pathlib.Path(tmp) / "utterance.aiff"
        wav = pathlib.Path(tmp) / "utterance.wav"
        _ = subprocess.run(
            ["say", "-o", str(aiff), text],
            check=True,
            capture_output=True,
        )
        _ = subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{sample_rate_hz}",
                str(aiff),
                str(wav),
            ],
            check=True,
            capture_output=True,
        )
        data = wav.read_bytes()
    destination.write_bytes(data)
    return data


def _silence_wav(
    *,
    duration_seconds: float = 0.5,
    sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ,
    channels: int = _DEFAULT_CHANNELS,
) -> bytes:
    """Build a valid signed 16-bit PCM WAV chunk for the end of an utterance."""

    if duration_seconds <= 0:
        raise ValueError("silence duration must be positive.")
    frame_count = max(1, round(duration_seconds * sample_rate_hz))
    pcm = b"\x00" * (frame_count * channels * 2)
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate_hz)
        stream.writeframes(pcm)
    return output.getvalue()


class SayEncoder(encoding.Encoder[bytes, bytes]):
    """Offline TTS: encode UTF-8 text with macOS ``say`` into WAV bytes.

    A caller-supplied destination remains caller-owned. Without one, the encoder
    owns a temporary destination and closes and removes it after rendering,
    including after failure or cancellation. Cancellation waits for the native
    render worker to settle before releasing that temporary path, and a cancelled
    invocation does not publish ``audio`` or ``calls``.
    """

    def __init__(
        self,
        *,
        destination: pathlib.Path | None = None,
        sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ,
    ) -> None:
        self.destination = destination
        self.sample_rate_hz = sample_rate_hz
        self.calls: list[str] = []
        self.audio: bytes | None = None

    @override
    async def encode(self, input: bytes) -> bytes:
        text = input.decode("utf-8").strip()
        if not text:
            raise ValueError("SayEncoder requires non-blank text.")
        owns_path = self.destination is None
        if owns_path:
            descriptor, raw_path = tempfile.mkstemp(suffix=".wav", prefix="say-reply-")
            os.close(descriptor)
            path = pathlib.Path(raw_path)
        else:
            assert self.destination is not None
            path = self.destination
        render_audio = functools.partial(
            _say_to_wav,
            text=text,
            destination=path,
            sample_rate_hz=self.sample_rate_hz,
        )
        render = asyncio.create_task(asyncio.to_thread(render_audio))
        try:
            audio = await asyncio.shield(render)
        except asyncio.CancelledError:
            try:
                _ = await render
            except Exception as error:
                _LOG.debug("Say render failed while a cancelled encoding settled.", exc_info=error)
            raise
        finally:
            if owns_path:
                path.unlink(missing_ok=True)
        self.audio = audio
        self.calls.append(text)
        return audio


def _gemini_client(config: CognitionConfig, *, model: str | None = None) -> GeminiChatClient:
    return GeminiChatClient(model=model or config.model, api_key=config.api_key)


def _mercury_intuition_client(config: CognitionConfig) -> OpenAIChatClient:
    if not config.intuition_api_key:
        raise ValueError("intuition_api_key is required for Mercury 2 (set BOT_MERCURY_API_KEY).")
    return OpenAIChatClient(
        model=config.intuition_model,
        api_key=config.intuition_api_key,
        base_url=config.intuition_base_url,
    )


def _pyannote_voice_classifier(
    model_id: str = DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL,
    *,
    inference: SpeakerEmbeddingInference | None = None,
    load_inference: SpeakerEmbeddingInferenceLoader | None = None,
) -> PyannoteVoiceClassifier:
    """Create the local PyAnnote embedding adapter, with an explicit test seam."""

    if load_inference is not None:
        return PyannoteVoiceClassifier(
            model_id=model_id,
            inference=inference,
            load_inference=load_inference,
        )
    return PyannoteVoiceClassifier(model_id=model_id, inference=inference)


def _conversation(*, speech_decoder: SpeechDecoder) -> conversation.Conversation:
    """Keep MLX Whisper on the identity-correlated Conversation turn path."""

    return conversation.Conversation(
        turn_detector=turn_detector.TurnDetector(
            # TurnDetector invokes its decoder only for audio ParticipationStimulus values;
            # the provider's narrower VoiceDecoder contract is the intended boundary here.
            decoder=typing.cast(
                decoding.Decoder[turn_detector.ParticipationStimulus, str],
                MlxVoiceDecoder(speech_decoder=speech_decoder),
            ),
            end_of_turn_silence_seconds=0.5,
        ),
    )


def _listen_speak_cognition(config: CognitionConfig) -> cognition.Cognition:
    """Run seeded Communication behaviors before Mercury intuition and Gemini reasoning."""

    store = memory.Memory()

    intuition_ability = cognition.Intuition(
        processor=OpenAIProcessor(
            client=_mercury_intuition_client(config),
            provider="mercury2_intuition",
        ),
    )
    deliberate_processor = GeminiProcessor(
        client=_gemini_client(config),
        provider="gemini_slow_reasoning",
    )
    reasoning_ability = cognition.Reasoning(processor=deliberate_processor)
    _LOG.info(
        "cognition wired intuition_model=%s intuition_base_url=%s reasoning_model=%s",
        config.intuition_model,
        config.intuition_base_url,
        config.model,
    )
    return cognition.Cognition(
        autonomy=cognition.Autonomy(
            memory=store,
            seeded_behaviors=(communication.speech_heard_seed(),),
        ),
        intuition=intuition_ability,
        reasoning=reasoning_ability,
        reflection=cognition.Reflection(
            processor=deliberate_processor,
            memory=store,
        ),
    )


class _ProgressDecisionData(BaseModel):
    """One processing outcome selected by the progress topology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_id: str = Field(min_length=1)


_ResponseSelectedEvent = hsm.Event[_ProgressDecisionData](
    name="listen_speak.progress.response_selected",
    schema=_ProgressDecisionData,
)
_NoResponseEvent = hsm.Event[_ProgressDecisionData](
    name="listen_speak.progress.no_response",
    schema=_ProgressDecisionData,
)
_ContinueProcessingEvent = hsm.Event[_ProgressDecisionData](
    name="listen_speak.progress.continue_processing",
    schema=_ProgressDecisionData,
)


class _RunTerminalData(BaseModel):
    """Typed terminal emitted by the demo progress actor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: typing.Literal[
        "completed",
        "conversation_failed",
        "processing_failed",
        "response_execution_failed",
        "timed_out",
    ]
    response_selections: int = Field(ge=0)


_RunTerminalEvent = hsm.Event[_RunTerminalData](
    name="listen_speak.progress.terminal",
    schema=_RunTerminalData,
)


_TerminalData = typing.TypeVar("_TerminalData")


class _TerminalSubscription(typing.Generic[_TerminalData]):
    """Non-actor subscription boundary for one typed terminal."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_TerminalData] = asyncio.Queue(maxsize=1)
        self._terminal: _TerminalData | None = None

    def publish(self, terminal: _TerminalData) -> None:
        if self._terminal is None:
            self._terminal = terminal
            self._queue.put_nowait(terminal)

    async def receive(self, *, timeout: float) -> _TerminalData:
        if self._terminal is not None:
            return self._terminal
        return await asyncio.wait_for(self._queue.get(), timeout=timeout)

    def latest(self) -> _TerminalData | None:
        return self._terminal


class _RunRecord:
    """External observation record; it never decides actor progression."""

    def __init__(self) -> None:
        self.terminals = _TerminalSubscription[hsm.Event[_RunTerminalData]]()
        self.completed: list[mosfet.ProcessingCompletedEventData] = []
        self.failures: list[mosfet.ProcessingFailedEventData] = []
        self.listening_handoffs: list[cognition.InputData] = []
        self.conversation_failures: list[ability.FailureData] = []


class _RunProgress(hsm.Instance):
    """HSM-owned demo progress with explicit classification and terminal states."""

    _processing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(seconds=120)
    _record: _RunRecord
    _conversation_handoff_ids: set[str]
    _processing_completed_ids: set[str]
    _selected_response_operation_ids: set[str]
    _speaking_terminal_ids: set[str]
    _no_response_completed: bool

    def __init__(self, record: _RunRecord) -> None:
        super().__init__()
        self._record = record
        self._conversation_handoff_ids = set()
        self._processing_completed_ids = set()
        self._selected_response_operation_ids = set()
        self._speaking_terminal_ids = set()
        self._no_response_completed = False

    @staticmethod
    def _is_conversation_handoff(
        ctx: hsm.Context,
        instance: "_RunProgress",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        data = event.data
        return (
            isinstance(data, cognition.InputData)
            and isinstance(data.stimulus, hsm.Event)
            and isinstance(data.stimulus.data, conversation.Messages)
        )

    @staticmethod
    def _record_handoff(ctx: hsm.Context, instance: "_RunProgress", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._conversation_handoff_ids.add(event.id)

    @staticmethod
    def _matches_handoff(ctx: hsm.Context, instance: "_RunProgress", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return any(
            event.id == operation_id or event.id.startswith(f"{operation_id}:")
            for operation_id in instance._conversation_handoff_ids
        )

    @staticmethod
    def _record_processing_completed(
        ctx: hsm.Context,
        instance: "_RunProgress",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        if not isinstance(data, mosfet.ProcessingCompletedEventData):
            return
        instance._processing_completed_ids.add(event.id)
        selected = isinstance(data.output, tuple) and any(
            isinstance(selection, cognition.EventData) and selection.event == communication.RespondEvent.name
            for selection in data.output
        )
        if selected:
            outcome = _ResponseSelectedEvent
        elif not data.output:
            outcome = _NoResponseEvent
        else:
            outcome = _ContinueProcessingEvent
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                outcome.with_data(_ProgressDecisionData(operation_id=event.id)),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _record_response_selected(
        ctx: hsm.Context,
        instance: "_RunProgress",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        data = event.data
        if isinstance(data, _ProgressDecisionData):
            instance._selected_response_operation_ids.add(data.operation_id)

    @staticmethod
    def _record_no_response(
        ctx: hsm.Context,
        instance: "_RunProgress",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, event
        instance._no_response_completed = True

    @staticmethod
    def _matches_selected_response(
        ctx: hsm.Context,
        instance: "_RunProgress",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return any(
            event.id == operation_id or event.id.startswith(f"{operation_id}:")
            for operation_id in instance._selected_response_operation_ids
        )

    @staticmethod
    def _record_speaking_terminal(
        ctx: hsm.Context,
        instance: "_RunProgress",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        matching = (
            operation_id
            for operation_id in instance._selected_response_operation_ids
            if event.id == operation_id or event.id.startswith(f"{operation_id}:")
        )
        operation_id = max(matching, key=len)
        instance._speaking_terminal_ids.add(operation_id)

    @staticmethod
    def _ready(ctx: hsm.Context, instance: "_RunProgress", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        selections = instance._selected_response_operation_ids
        return (bool(selections) or instance._no_response_completed) and selections <= instance._speaking_terminal_ids

    @staticmethod
    def _publish_terminal(
        ctx: hsm.Context,
        instance: "_RunProgress",
        event: hsm.Event[typing.Any],
        *,
        reason: typing.Literal[
            "completed",
            "conversation_failed",
            "processing_failed",
            "response_execution_failed",
            "timed_out",
        ],
    ) -> None:
        terminal = dataclasses.replace(
            _RunTerminalEvent.with_data(
                _RunTerminalData(
                    reason=reason,
                    response_selections=len(instance._selected_response_operation_ids),
                )
            ),
            id=event.id,
            source=hsm.id(instance),
            metadata=dict(event.metadata),
        )
        instance._record.terminals.publish(terminal)
        del ctx

    @staticmethod
    def _publish_completed(ctx: hsm.Context, instance: "_RunProgress", event: hsm.Event[typing.Any]) -> None:
        _RunProgress._publish_terminal(ctx, instance, event, reason="completed")

    @staticmethod
    def _publish_conversation_failed(
        ctx: hsm.Context,
        instance: "_RunProgress",
        event: hsm.Event[typing.Any],
    ) -> None:
        _RunProgress._publish_terminal(ctx, instance, event, reason="conversation_failed")

    @staticmethod
    def _publish_processing_failed(
        ctx: hsm.Context,
        instance: "_RunProgress",
        event: hsm.Event[typing.Any],
    ) -> None:
        _RunProgress._publish_terminal(ctx, instance, event, reason="processing_failed")

    @staticmethod
    def _publish_response_failed(
        ctx: hsm.Context,
        instance: "_RunProgress",
        event: hsm.Event[typing.Any],
    ) -> None:
        _RunProgress._record_speaking_terminal(ctx, instance, event)
        _RunProgress._publish_terminal(ctx, instance, event, reason="response_execution_failed")

    @staticmethod
    def _publish_timed_out(ctx: hsm.Context, instance: "_RunProgress", event: hsm.Event[typing.Any]) -> None:
        _RunProgress._publish_terminal(ctx, instance, event, reason="timed_out")

    @staticmethod
    def _timeout(ctx: hsm.Context, instance: "_RunProgress", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._processing_timeout

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "ListenSpeakRunProgress",
        hsm.initial(hsm.target("tracking")),
        hsm.state(
            "tracking",
            hsm.transition(
                hsm.on(cognition.InputEvent),
                hsm.guard(_is_conversation_handoff),
                hsm.effect(_record_handoff),
            ),
            hsm.transition(
                hsm.on(mosfet.ProcessingCompletedEvent),
                hsm.guard(_matches_handoff),
                hsm.effect(_record_processing_completed),
                hsm.target("../classifying_processing"),
            ),
            hsm.transition(
                hsm.on(mosfet.ProcessingFailedEvent),
                hsm.guard(_matches_handoff),
                hsm.effect(_publish_processing_failed),
                hsm.target("../done"),
            ),
            hsm.transition(
                hsm.on(conversation.FailedEvent),
                hsm.effect(_publish_conversation_failed),
                hsm.target("../done"),
            ),
            hsm.transition(
                hsm.on(speaking.OutputEvent),
                hsm.guard(_matches_selected_response),
                hsm.effect(_record_speaking_terminal),
                hsm.target("../routing_readiness"),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.guard(_matches_selected_response),
                hsm.effect(_publish_response_failed),
                hsm.target("../done"),
            ),
            hsm.transition(
                hsm.after(_timeout),
                hsm.effect(_publish_timed_out),
                hsm.target("../done"),
            ),
        ),
        hsm.state(
            "classifying_processing",
            hsm.transition(
                hsm.on(_ResponseSelectedEvent),
                hsm.effect(_record_response_selected),
                hsm.target("../routing_readiness"),
            ),
            hsm.transition(
                hsm.on(_NoResponseEvent),
                hsm.effect(_record_no_response),
                hsm.target("../routing_readiness"),
            ),
            hsm.transition(
                hsm.on(_ContinueProcessingEvent),
                hsm.target("../tracking"),
            ),
        ),
        hsm.choice(
            "routing_readiness",
            hsm.transition(
                hsm.guard(_ready),
                hsm.effect(_publish_completed),
                hsm.target("/ListenSpeakRunProgress/done"),
            ),
            hsm.transition(hsm.target("/ListenSpeakRunProgress/tracking")),
        ),
        hsm.final("done"),
    )


class _TrackedCommunication(communication.Communication):
    """Demo communication route that forwards response terminals to run progress."""

    _run_progress: _RunProgress

    def __init__(
        self,
        *,
        active_conversation: conversation.Conversation,
        speaking: speaking.Speaking,
        run_progress: _RunProgress,
    ) -> None:
        self._run_progress = run_progress
        super().__init__(active_conversation=active_conversation, speaking=speaking)

    @staticmethod
    def _observed_event(observation: hsm.Event[typing.Any]) -> hsm.Event[typing.Any] | None:
        data = observation.data
        if not isinstance(data, dict):
            return None
        event = data.get("event")
        return event if isinstance(event, hsm.Event) else None

    @staticmethod
    def _forward_response_terminal(
        ctx: hsm.Context,
        instance: "_TrackedCommunication",
        observation: hsm.Event[typing.Any],
    ) -> None:
        event = _TrackedCommunication._observed_event(observation)
        if event is not None:
            _ = hsm.dispatch(ctx, instance._run_progress, event)

    submodel: typing.ClassVar[hsm.Model | None] = hsm.redefine(
        typing.cast(hsm.Model, communication.Communication.submodel),
        "Communication",
        hsm.observe(speaking.OutputEvent, _forward_response_terminal),
        hsm.observe(ability.FailedEvent, _forward_response_terminal),
    )


class ListenSpeakBot(Bot):
    """Bot with no devices: real Listening input plus cognition and an internal Speaking port."""

    _processing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(seconds=120)
    _listening: listening.Listening
    _speaking: speaking.Speaking
    _speaker: audio.Speaker
    _decoder: SpeechDecoder
    _owned_voice_activity_classifier: SileroVoiceActivityClassifier | None
    _encoder: SayEncoder
    _cognition_config: CognitionConfig
    _conversation: conversation.Conversation
    _communication: _TrackedCommunication
    _run_record: _RunRecord
    _run_progress: _RunProgress
    _activation_terminals: _TerminalSubscription[hsm.Event[typing.Any]]
    _deactivation_terminals: _TerminalSubscription[hsm.Event[typing.Any]]

    def __init__(
        self,
        *,
        reply_wav: pathlib.Path,
        cognition: CognitionConfig | cognition.Cognition | None = None,
        sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ,
        vad_model_id: str = DEFAULT_SILERO_VAD_MODEL,
        stt_model_id: str | None = None,
        voice_identity_model_id: str = DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL,
        voice_activity_classifier: voice.detection.VoiceActivityClassifier | None = None,
        voice_classifier: classifying.Classifier[
            voice.identification.InputData,
            voice.identification.OutputData,
        ]
        | None = None,
        voice_identification_inference: SpeakerEmbeddingInference | None = None,
        voice_identification_loader: SpeakerEmbeddingInferenceLoader | None = None,
    ) -> None:
        self._run_record = _RunRecord()
        self._run_progress = _RunProgress(self._run_record)
        self._decoder = SpeechDecoder() if stt_model_id is None else SpeechDecoder(model_id=stt_model_id)
        self._encoder = SayEncoder(destination=reply_wav, sample_rate_hz=sample_rate_hz)
        classifier = (
            voice_classifier
            if voice_classifier is not None
            else _pyannote_voice_classifier(
                model_id=voice_identity_model_id,
                inference=voice_identification_inference,
                load_inference=voice_identification_loader,
            )
        )
        if voice_activity_classifier is None:
            owned_voice_activity_classifier = SileroVoiceActivityClassifier(
                model_id=vad_model_id,
                sample_rate_hz=sample_rate_hz,
                channels=_DEFAULT_CHANNELS,
            )
            selected_voice_activity_classifier = owned_voice_activity_classifier
        else:
            owned_voice_activity_classifier = None
            selected_voice_activity_classifier = voice_activity_classifier
        self._owned_voice_activity_classifier = owned_voice_activity_classifier
        self._listening = listening.Listening(
            voice_activity_classifier=selected_voice_activity_classifier,
            speech_decoder=None,
            voice_classifier=classifier,
        )
        self._conversation = _conversation(speech_decoder=self._decoder)
        # The internal Speaker produces WAV and keeps composition-time efference wiring active.
        self._speaker = audio.Speaker()
        self._speaking = speaking.Speaking(
            encoder=self._encoder,
            speaker=self._speaker,
            listening=self._listening,
            conversation=self._conversation,
            sample_rate_hz=sample_rate_hz,
            channels=_DEFAULT_CHANNELS,
            media_type="audio/wav",
        )
        self._communication = _TrackedCommunication(
            active_conversation=self._conversation,
            speaking=self._speaking,
            run_progress=self._run_progress,
        )
        if isinstance(cognition, _Cognition):
            cognition_instance = cognition
            self._cognition_config = CognitionConfig()
        else:
            self._cognition_config = cognition if cognition is not None else CognitionConfig()
            cognition_instance = _listen_speak_cognition(self._cognition_config)
        super().__init__(
            devices={},
            cognition=cognition_instance,
            input=(self._listening,),
            output=(self._speaking,),
            acquired_abilities=(self._communication,),
        )
        self._activation_terminals = _TerminalSubscription()
        self._deactivation_terminals = _TerminalSubscription()

    @staticmethod
    def _observation_event(event: hsm.Event[typing.Any]) -> hsm.Event[typing.Any] | None:
        data = event.data
        if not isinstance(data, dict):
            return None
        observed = data.get("event")
        return observed if isinstance(observed, hsm.Event) else None

    @staticmethod
    def _forward_progress(ctx: hsm.Context, instance: "ListenSpeakBot", event: hsm.Event[typing.Any]) -> None:
        _ = hsm.dispatch(ctx, instance._run_progress, event)

    @staticmethod
    def _observe_cognition_handoff(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        observation: hsm.Event[typing.Any],
    ) -> None:
        event = ListenSpeakBot._observation_event(observation)
        if event is None:
            return
        ListenSpeakBot._record_cognition_handoff_event(ctx, instance, event)

    @staticmethod
    def _record_cognition_handoff_event(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        event: hsm.Event[typing.Any],
    ) -> None:
        if not isinstance(event.data, cognition.InputData):
            return
        stimulus = event.data.stimulus
        if (
            event.source == hsm.id(instance._listening)
            and isinstance(stimulus, hsm.Event)
            and isinstance(stimulus.data, conversation.Messages)
        ):
            instance._run_record.listening_handoffs.append(event.data)
        ListenSpeakBot._forward_progress(ctx, instance, event)

    @staticmethod
    def _observe_processing_completed(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        observation: hsm.Event[typing.Any],
    ) -> None:
        event = ListenSpeakBot._observation_event(observation)
        if event is None:
            return
        ListenSpeakBot._record_processing_completed_event(ctx, instance, event)

    @staticmethod
    def _record_processing_completed_event(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        event: hsm.Event[typing.Any],
    ) -> None:
        if not isinstance(event.data, mosfet.ProcessingCompletedEventData):
            return
        instance._run_record.completed.append(event.data)
        _LOG.info("bot processing completed id=%s selections=%s", event.id, _summarize_selections(event.data.output))
        ListenSpeakBot._forward_progress(ctx, instance, event)

    @staticmethod
    def _observe_processing_failed(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        observation: hsm.Event[typing.Any],
    ) -> None:
        event = ListenSpeakBot._observation_event(observation)
        if event is None:
            return
        ListenSpeakBot._record_processing_failed_event(ctx, instance, event)

    @staticmethod
    def _record_processing_failed_event(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        event: hsm.Event[typing.Any],
    ) -> None:
        if not isinstance(event.data, mosfet.ProcessingFailedEventData):
            return
        instance._run_record.failures.append(event.data)
        _LOG.warning("bot processing failed id=%s", event.id)
        ListenSpeakBot._forward_progress(ctx, instance, event)

    @staticmethod
    def _from_conversation(ctx: hsm.Context, instance: "ListenSpeakBot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, ability.FailureData) and event.source == hsm.id(instance._conversation)

    @staticmethod
    def _consume_conversation_failure(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Consume the conversation ability's own failure without failing the run.

        ``_from_conversation`` narrows this to a failure raised by this bot's conversation. The
        record of it is kept by ``_observe_conversation_failure_transition``, wired through
        ``hsm.observe(conversation.FailedEvent, ...)``; this transition is what admits the event
        into ``active`` so that observation runs. The bot itself does nothing further: a
        conversation turn that failed is a turn it can decline to continue, not a reason for the
        run to end, and the terminals that do end it are the deactivation transitions above.
        """

        del ctx, instance, event

    @staticmethod
    def _observe_conversation_failure(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        if not isinstance(data, ability.FailureData):
            return
        instance._run_record.conversation_failures.append(data)
        _LOG.warning("conversation failed id=%s message=%s", event.id, data.message)
        ListenSpeakBot._forward_progress(ctx, instance, event)

    @staticmethod
    def _observe_conversation_failure_transition(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        observation: hsm.Event[typing.Any],
    ) -> None:
        event = ListenSpeakBot._observation_event(observation)
        if event is not None and event.source == hsm.id(instance._conversation):
            ListenSpeakBot._observe_conversation_failure(ctx, instance, event)

    @staticmethod
    def _from_speaking(ctx: hsm.Context, instance: "ListenSpeakBot", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return event.source == hsm.id(instance._speaking)

    @staticmethod
    def _forward_speaking_terminal(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        event: hsm.Event[typing.Any],
    ) -> None:
        ListenSpeakBot._forward_progress(ctx, instance, event)

    @staticmethod
    def _is_activation_terminal(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, (mosfet.ActivatingDoneEventData, mosfet.ActivatingFailedEventData))
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _publish_activation_terminal(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        instance._activation_terminals.publish(event)

    @staticmethod
    def _is_deactivation_terminal(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, mosfet.DeactivatingDoneEventData)
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _publish_deactivation_terminal(
        ctx: hsm.Context,
        instance: "ListenSpeakBot",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        instance._deactivation_terminals.publish(event)

    model: typing.ClassVar[hsm.Model] = hsm.redefine(
        Bot.model,
        "ListenSpeakBot",
        hsm.transition(
            hsm.source("active"),
            hsm.on(mosfet.ActivatingDoneEvent),
            hsm.guard(_is_activation_terminal),
            hsm.effect(_publish_activation_terminal),
        ),
        hsm.transition(
            hsm.source("activation_cleanup"),
            hsm.on(mosfet.ActivatingFailedEvent),
            hsm.guard(_is_activation_terminal),
            hsm.effect(_publish_activation_terminal),
        ),
        hsm.transition(
            hsm.source("inactive"),
            hsm.on(mosfet.DeactivatingDoneEvent),
            hsm.guard(_is_deactivation_terminal),
            hsm.effect(_publish_deactivation_terminal),
        ),
        hsm.transition(
            hsm.source("active"),
            hsm.on(cognition.InputEvent),
            hsm.effect(_record_cognition_handoff_event),
        ),
        hsm.transition(
            hsm.source("active"),
            hsm.on(mosfet.ProcessingCompletedEvent),
            hsm.effect(_record_processing_completed_event),
        ),
        hsm.transition(
            hsm.source("active"),
            hsm.on(mosfet.ProcessingFailedEvent),
            hsm.effect(_record_processing_failed_event),
        ),
        hsm.transition(
            hsm.source("active"),
            hsm.on(conversation.FailedEvent),
            hsm.guard(_from_conversation),
            hsm.effect(_consume_conversation_failure),
        ),
        hsm.transition(
            hsm.source("active"),
            hsm.on(speaking.OutputEvent),
            hsm.guard(_from_speaking),
            hsm.effect(_forward_speaking_terminal),
        ),
        hsm.transition(
            hsm.source("active"),
            hsm.on(ability.FailedEvent),
            hsm.guard(_from_speaking),
            hsm.effect(_forward_speaking_terminal),
        ),
        hsm.observe(cognition.InputEvent, _observe_cognition_handoff),
        hsm.observe(mosfet.ProcessingCompletedEvent, _observe_processing_completed),
        hsm.observe(mosfet.ProcessingFailedEvent, _observe_processing_failed),
        hsm.observe(conversation.FailedEvent, _observe_conversation_failure_transition),
    )

    @override
    async def attach(self, environment: Environment, *, placement: space.Placement | None = None) -> typing.Self:
        _ = await mosfet.started(environment, self._run_progress, self._run_progress.model)
        try:
            attached = await super().attach(environment, placement=placement)
            await self.wait_for_activation()
            return attached
        except BaseException:
            await hsm.stop(self._run_progress, hsm.Context())
            await self._close_owned_audio_workers()
            raise

    @override
    async def detach(self, environment: Environment) -> typing.Self:
        try:
            detached = await super().detach(environment)
            try:
                _ = await self._deactivation_terminals.receive(timeout=self._deactivation_timeout.total_seconds())
            except TimeoutError as error:
                raise RuntimeError("Timed out waiting for ListenSpeakBot deactivation.") from error
            return detached
        finally:
            await hsm.stop(self._run_progress, hsm.Context())
            await self._close_owned_audio_workers()

    async def _close_owned_audio_workers(self) -> None:
        close_operations: list[collections.abc.Awaitable[None]] = [self._decoder.aclose()]
        if self._owned_voice_activity_classifier is not None:
            close_operations.append(self._owned_voice_activity_classifier.aclose())
        results = await asyncio.gather(*close_operations, return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("ListenSpeakBot audio worker cleanup failed.", failures)

    def listening(self) -> listening.Listening:
        return self._listening

    def speaking(self) -> speaking.Speaking:
        return self._speaking

    def speaker(self) -> audio.Speaker:
        return self._speaker

    def conversation(self) -> conversation.Conversation:
        return self._conversation

    def communication(self) -> communication.Communication:
        return self._communication

    def decoder(self) -> SpeechDecoder:
        return self._decoder

    def encoder(self) -> SayEncoder:
        return self._encoder

    def cognition_config(self) -> CognitionConfig:
        return self._cognition_config

    def completed(self) -> tuple[mosfet.ProcessingCompletedEventData, ...]:
        return tuple(self._run_record.completed)

    def failures(self) -> tuple[mosfet.ProcessingFailedEventData, ...]:
        return tuple(self._run_record.failures)

    def listening_handoffs(self) -> tuple[cognition.InputData, ...]:
        return tuple(self._run_record.listening_handoffs)

    async def wait_for_conversation_processing(self) -> None:
        terminal = await self._run_record.terminals.receive(timeout=self._processing_timeout.total_seconds())
        if isinstance(terminal.data, _RunTerminalData) and terminal.data.reason == "timed_out":
            raise RuntimeError("Timed out waiting for ListenSpeakBot conversation processing.")

    async def wait_for_activation(self) -> None:
        """Wait for the public activation terminal and surface typed activation failure."""

        try:
            terminal = await self._activation_terminals.receive(timeout=self._processing_timeout.total_seconds())
        except TimeoutError as error:
            raise RuntimeError("Timed out waiting for ListenSpeakBot activation.") from error
        if isinstance(terminal.data, mosfet.ActivatingFailedEventData):
            raise RuntimeError("ListenSpeakBot activation failed.")

    def conversation_failures(self) -> tuple[ability.FailureData, ...]:
        return tuple(self._run_record.conversation_failures)

    def response_selection_count(self) -> int:
        terminal = self._run_record.terminals.latest()
        if terminal is None or not isinstance(terminal.data, _RunTerminalData):
            return 0
        return terminal.data.response_selections

    def response_execution_failed(self) -> bool:
        terminal = self._run_record.terminals.latest()
        return (
            terminal is not None
            and isinstance(terminal.data, _RunTerminalData)
            and terminal.data.reason == "response_execution_failed"
        )


async def run(
    *,
    play: bool = False,
    assets_dir: pathlib.Path | None = None,
    config: AppConfig | None = None,
) -> dict[str, object]:
    """Run the device-free Listening→cognition pipeline once and return a summary dict."""

    _require_macos_speech_tools()
    app_config = config or AppConfig.from_env_file(_DEFAULT_ENV_PATH if _DEFAULT_ENV_PATH.exists() else None)
    if not app_config.cognition.can_process():
        raise RuntimeError(
            "Missing cognition credentials. Need BOT_GEMINI_API_KEY (reasoning) and "
            "BOT_MERCURY_API_KEY (intuition), e.g. in repo .env or examples/listen_speak_bot/.env."
        )

    assets = assets_dir if assets_dir is not None else _ASSETS_DIR
    assets.mkdir(parents=True, exist_ok=True)
    heard_wav = assets / "hey_gabe.wav"
    reply_wav = assets / "reply.wav"

    _LOG.info("generating heard audio with say path=%s phrase=%r", heard_wav, _HEARD_PHRASE)
    heard_audio = await asyncio.to_thread(_say_to_wav, _HEARD_PHRASE, heard_wav)

    body = ListenSpeakBot(
        reply_wav=reply_wav,
        cognition=app_config.cognition,
        vad_model_id=app_config.vad_model_id,
        stt_model_id=app_config.stt_model_id,
        voice_identity_model_id=app_config.voice_identity_model_id,
    )
    environment = Environment()
    _ = await body.attach(environment)
    _LOG.info("bot attached")
    try:
        sound = SoundEvent.with_data(
            SoundData(
                audio=heard_audio,
                media_type="audio/wav",
                sample_rate_hz=_DEFAULT_SAMPLE_RATE_HZ,
                channels=_DEFAULT_CHANNELS,
                kind="speech",
            )
        )
        _LOG.info(
            "dispatching environment.sound bytes=%s intuition_model=%s reasoning_model=%s",
            len(heard_audio),
            app_config.cognition.intuition_model,
            app_config.cognition.model,
        )
        await body.dispatch(environment, sound)

        silence_audio = _silence_wav(sample_rate_hz=_DEFAULT_SAMPLE_RATE_HZ, channels=_DEFAULT_CHANNELS)
        _LOG.info("dispatching environment.sound silence bytes=%s", len(silence_audio))
        await body.dispatch(
            environment,
            SoundEvent.with_data(
                SoundData(
                    audio=silence_audio,
                    media_type="audio/wav",
                    sample_rate_hz=_DEFAULT_SAMPLE_RATE_HZ,
                    channels=_DEFAULT_CHANNELS,
                    kind="silence",
                )
            ),
        )

        await body.wait_for_conversation_processing()
        response_selections = body.response_selection_count()
        summary_counts = {
            "heard_audio_bytes": len(heard_audio),
            "reply_audio_bytes": 0,
            "listening_handoffs": len(body.listening_handoffs()),
            "processing_completed": len(body.completed()),
            "response_selections": response_selections,
        }
        if body.conversation_failures():
            return _build_operator_summary(status="failed", status_reason="conversation_failed", **summary_counts)
        if body.failures():
            return _build_operator_summary(status="failed", status_reason="processing_failed", **summary_counts)
        if body.response_execution_failed():
            return _build_operator_summary(status="failed", status_reason="response_execution_failed", **summary_counts)
    finally:
        await body.detach(environment)

    reply_audio = body.encoder().audio or b""
    has_speaking_output = bool(body.encoder().calls)
    summary_counts["reply_audio_bytes"] = len(reply_audio)
    summary = _build_operator_summary(
        status="ok" if has_speaking_output else "incomplete",
        status_reason="response_spoken" if has_speaking_output else "no_response_selected",
        **summary_counts,
    )

    if play:
        for label, path in (("heard", heard_wav), ("reply", reply_wav)):
            if path.exists() and (label == "heard" or body.encoder().calls):
                print(f"Playing {label}: {path}", file=sys.stderr)
                _ = subprocess.run(["afplay", str(path)], check=False)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Device-free Listening→cognition demo with Mercury 2 intuition + Gemini reasoning (macOS say)."
    )
    _ = parser.add_argument(
        "--play",
        action="store_true",
        help="Play the heard WAV, and any reply WAV produced by a Speaking route, with afplay.",
    )
    _ = parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of a labeled block.",
    )
    _ = parser.add_argument(
        "--env",
        type=pathlib.Path,
        default=None,
        help=f"Provider env file (default: {_DEFAULT_ENV_PATH} if present, else process env).",
    )
    _ = parser.add_argument(
        "--gemini-model",
        default=None,
        help=f"Override Gemini cognition model (default: {DEFAULT_GEMINI_MODEL}).",
    )
    _ = parser.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=None,
        help=f"Directory for hey_gabe.wav and reply.wav (default: {_ASSETS_DIR}).",
    )
    _ = parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logs (default is INFO ability/bot pipeline logs on stderr).",
    )
    args = parser.parse_args(argv)
    _configure_logging(verbose=bool(args.verbose))
    env_path = args.env
    if env_path is None and _DEFAULT_ENV_PATH.exists():
        env_path = _DEFAULT_ENV_PATH
    config = AppConfig.from_env_file(env_path).with_cognition_overrides(model=args.gemini_model)
    try:
        summary = asyncio.run(run(play=bool(args.play), assets_dir=args.assets_dir, config=config))
    except Exception as error:
        _LOG.exception("listen_speak_bot failed")
        print(f"listen_speak_bot failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("stateforward.mosfet listen→cognition (device-free, Gemini cognition)")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    return 0 if summary.get("status") == "ok" else 1


__all__ = [
    "AppConfig",
    "CognitionConfig",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL",
    "DEFAULT_SILERO_VAD_MODEL",
    "ListenSpeakBot",
    "SayEncoder",
    "load_env",
    "main",
    "run",
]
