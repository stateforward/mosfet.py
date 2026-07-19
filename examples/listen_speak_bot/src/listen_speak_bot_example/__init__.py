"""Device-free listen → speak bot example (macOS ``say`` + Gemini cognition).

Flow:

1. macOS ``say`` renders *Hey I'm Gabe how are you* into a WAV asset.
2. ``world.sound`` carries that audio into bot **input** (Listening).
3. Offline VAD + fixed STT hand a transcript stimulus to cognition.
4. Gemini intuition (``gemini-3.1-flash-lite``) / reasoning (``gemini-3.5-flash``)
   select ``bot.ability.speaking.input``.
5. Speaking encodes the reply with ``say`` again and writes a reply WAV.
"""

from __future__ import annotations

import argparse
import asyncio
import collections.abc
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import typing
from typing import override

import hsm

import bot
from bot.abilities import cognition
from bot.abilities import encoding
from bot.abilities import listening
from bot.abilities import memory
from bot.abilities import speaking
from bot.abilities.hearing import speech
from bot.abilities.hearing import voice
from bot.bot import Bot
from bot.providers.gemini import ChatClient, Processor
from bot.world import SoundData, SoundEvent, World

# Capture before any local named ``cognition`` shadows the package (constructor param).
_Cognition = cognition.Cognition

_LOG = logging.getLogger("listen_speak_bot_example")

_EXAMPLE_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ASSETS_DIR = _EXAMPLE_ROOT / "assets"
_DEFAULT_ENV_PATH = _EXAMPLE_ROOT / ".env"
_HEARD_PHRASE = "Hey I'm Gabe how are you"
_DEFAULT_SAMPLE_RATE_HZ = 22_050
_DEFAULT_CHANNELS = 1
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_INTUITION_MODEL = "gemini-3.1-flash-lite"


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


def _log_ability_terminals(machine: typing.Any, *, name: str, model: str) -> None:
    """Log intuition/reasoning terminals so a run shows which ability closed the turn."""

    from bot.abilities import ability

    original = machine.dispatch

    def dispatch(ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        # Ability terminals arrive as TerminalOutput/Error wrapping the public event.
        if event.name == ability.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
            public = typing.cast(hsm.Event[typing.Any], event.data)
            if public.name == machine.output_event.name:
                # Typed terminal data only — never event.metadata for behavior (HSM-COMPLETION-001).
                _LOG.info(
                    "%s terminal model=%s outcome=%s",
                    name,
                    model,
                    _summarize_selections(public.data),
                )
        elif event.name == ability.TerminalErrorEvent.name and isinstance(event.data, hsm.Event):
            public = typing.cast(hsm.Event[typing.Any], event.data)
            if public.name == machine.failed_event.name:
                message = getattr(public.data, "message", public.data)
                _LOG.warning("%s failed model=%s message=%s", name, model, message)
        return original(ctx, event)

    machine.dispatch = dispatch  # type: ignore[method-assign]


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
    """Process environment as base; keys defined in an env file win (local credentials)."""

    values = {key: value for key, value in os.environ.items() if value}
    if path is not None and path.exists():
        values.update(load_env(path))
    return values


@dataclasses.dataclass(frozen=True)
class CognitionConfig:
    model: str = DEFAULT_GEMINI_MODEL
    api_key: str | None = None

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
        )

    def can_process(self) -> bool:
        return self.api_key is not None


@dataclasses.dataclass(frozen=True)
class AppConfig:
    env_path: pathlib.Path | None = None
    cognition: CognitionConfig = dataclasses.field(default_factory=CognitionConfig)

    @classmethod
    def from_env_file(cls, path: pathlib.Path | None = None) -> typing.Self:
        env = _merged_env(path)
        return cls(env_path=path, cognition=CognitionConfig.from_env(env))

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
    """Render ``text`` with macOS ``say`` and convert to 16-bit mono WAV."""

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


class AlwaysVoiceDetector(voice.detection.VoiceDetector):
    """Offline VAD: every acoustic chunk is voice (no ML dependency)."""

    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        return voice.detection.OutputData(is_voice=True, confidence=1.0)


class FixedTranscriptDecoder(speech.SpeechDecoder):
    """Offline STT: return a fixed UTF-8 transcript for the demo asset."""

    def __init__(self, transcript: str = _HEARD_PHRASE) -> None:
        self.transcript = transcript
        self.calls: list[int] = []

    @override
    async def decode(self, input: bytes) -> bytes:
        self.calls.append(len(input))
        return self.transcript.encode("utf-8")


class SayEncoder(encoding.Encoder[bytes, bytes]):
    """Offline TTS: encode UTF-8 text with macOS ``say`` into WAV bytes."""

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
        path = self.destination or pathlib.Path(tempfile.mkstemp(suffix=".wav", prefix="say-reply-")[1])
        audio = await asyncio.to_thread(_say_to_wav, text, path, sample_rate_hz=self.sample_rate_hz)
        self.audio = audio
        self.calls.append(text)
        return audio


def _chat_client(config: CognitionConfig, *, model: str | None = None) -> ChatClient:
    return ChatClient(model=model or config.model, api_key=config.api_key)


def _listen_speak_cognition(config: CognitionConfig) -> cognition.Cognition:
    """Fast intuition on Flash-Lite; deliberate reasoning on the configured cognition model."""

    intuition_ability = cognition.Intuition(
        processor=Processor(
            client=_chat_client(config, model=DEFAULT_GEMINI_INTUITION_MODEL),
            provider="gemini_fast_intuition",
        ),
    )
    deliberate_processor = Processor(
        client=_chat_client(config),
        provider="gemini_slow_reasoning",
    )
    reasoning_ability = cognition.Reasoning(processor=deliberate_processor)
    _log_ability_terminals(
        intuition_ability,
        name="intuition",
        model=DEFAULT_GEMINI_INTUITION_MODEL,
    )
    _log_ability_terminals(
        reasoning_ability,
        name="reasoning",
        model=config.model,
    )
    _LOG.info(
        "cognition wired intuition_model=%s reasoning_model=%s",
        DEFAULT_GEMINI_INTUITION_MODEL,
        config.model,
    )
    return cognition.Cognition(
        intuition=intuition_ability,
        reasoning=reasoning_ability,
        reflection=cognition.Reflection(
            processor=deliberate_processor,
            memory=memory.Memory(),
        ),
    )


class ListenSpeakBot(Bot):
    """Bot with no devices: Listening input + Speaking output + Gemini cognition."""

    _processing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(seconds=120)
    _listening: listening.Listening
    _speaking: speaking.Speaking
    _decoder: FixedTranscriptDecoder
    _encoder: SayEncoder
    _cognition_config: CognitionConfig
    _completed: list[bot.ProcessingCompletedEventData]
    _failures: list[bot.ProcessingFailedEventData]
    _listening_handoffs: list[cognition.InputData]

    def __init__(
        self,
        *,
        reply_wav: pathlib.Path,
        cognition: CognitionConfig | cognition.Cognition | None = None,
        sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ,
    ) -> None:
        self._decoder = FixedTranscriptDecoder()
        self._encoder = SayEncoder(destination=reply_wav, sample_rate_hz=sample_rate_hz)
        self._listening = listening.Listening(
            voice_detector=AlwaysVoiceDetector(),
            speech_decoder=self._decoder,
        )
        # No Speaker device: encoder still produces WAV; avoids self-hearing loop.
        self._speaking = speaking.Speaking(
            encoder=self._encoder,
            speaker=None,
            sample_rate_hz=sample_rate_hz,
            channels=_DEFAULT_CHANNELS,
            media_type="audio/wav",
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
        )
        self._completed = []
        self._failures = []
        self._listening_handoffs = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == bot.ProcessingCompletedEvent.name:
            completed = event.data
            assert isinstance(completed, bot.ProcessingCompletedEventData)
            self._completed.append(completed)
            _LOG.info("bot processing completed selections=%s", _summarize_selections(completed.output))
        if event.name == bot.ProcessingFailedEvent.name:
            failure = event.data
            assert isinstance(failure, bot.ProcessingFailedEventData)
            self._failures.append(failure)
            _LOG.warning("bot processing failed message=%s", failure.message)
        if event.name == cognition.InputEvent.name and event.source == hsm.id(self._listening):
            handoff = event.data
            assert isinstance(handoff, cognition.InputData)
            self._listening_handoffs.append(handoff)
            stimulus = handoff.stimulus
            stimulus_name = getattr(stimulus, "name", type(stimulus).__name__)
            _LOG.info("listening handoff to cognition stimulus=%s", stimulus_name)
        if event.name == speaking.InputEvent.name:
            data = event.data
            text = getattr(data, "text", None)
            _LOG.info("speaking.input text=%r", text)
        return super().dispatch(ctx, event)

    def listening(self) -> listening.Listening:
        return self._listening

    def speaking(self) -> speaking.Speaking:
        return self._speaking

    def decoder(self) -> FixedTranscriptDecoder:
        return self._decoder

    def encoder(self) -> SayEncoder:
        return self._encoder

    def cognition_config(self) -> CognitionConfig:
        return self._cognition_config

    def completed(self) -> tuple[bot.ProcessingCompletedEventData, ...]:
        return tuple(self._completed)

    def failures(self) -> tuple[bot.ProcessingFailedEventData, ...]:
        return tuple(self._failures)

    def listening_handoffs(self) -> tuple[cognition.InputData, ...]:
        return tuple(self._listening_handoffs)


async def _wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 90.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out waiting for listen→speak pipeline.")


async def run(
    *,
    play: bool = False,
    assets_dir: pathlib.Path | None = None,
    config: AppConfig | None = None,
) -> dict[str, object]:
    """Run the device-free listen→speak pipeline once and return a summary dict."""

    _require_macos_speech_tools()
    app_config = config or AppConfig.from_env_file(_DEFAULT_ENV_PATH if _DEFAULT_ENV_PATH.exists() else None)
    if not app_config.cognition.can_process():
        raise RuntimeError(
            "Missing Gemini API key. Set BOT_GEMINI_API_KEY / GEMINI_API_KEY or pass --env with that key."
        )

    assets = assets_dir if assets_dir is not None else _ASSETS_DIR
    assets.mkdir(parents=True, exist_ok=True)
    heard_wav = assets / "hey_gabe.wav"
    reply_wav = assets / "reply.wav"

    _LOG.info("generating heard audio with say path=%s phrase=%r", heard_wav, _HEARD_PHRASE)
    heard_audio = await asyncio.to_thread(_say_to_wav, _HEARD_PHRASE, heard_wav)

    body = ListenSpeakBot(reply_wav=reply_wav, cognition=app_config.cognition)
    world = World()
    _ = await body.attach(world)
    await _wait_until(lambda: (body.state() or "").endswith("/unfocused"))
    _LOG.info("bot active state=%s", body.state())

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
        "dispatching world.sound bytes=%s intuition_model=%s reasoning_model=%s",
        len(heard_audio),
        DEFAULT_GEMINI_INTUITION_MODEL,
        app_config.cognition.model,
    )
    await body.dispatch(world.context, sound)

    await _wait_until(
        lambda: (
            (bool(body.encoder().calls) and body.encoder().audio is not None and bool(body.completed()))
            or bool(body.failures())
        )
        and bool(body.listening_handoffs())
        and bool(body.decoder().calls)
    )
    if body.failures() and not body.encoder().calls:
        failure_messages = [failure.message for failure in body.failures()]
        await body.detach(world)
        return {
            "status": "failed",
            "heard_phrase": _HEARD_PHRASE,
            "cognition_model": app_config.cognition.model,
            "processing_failures": failure_messages,
            "bot_state": body.state(),
        }

    await body.detach(world)
    await _wait_until(lambda: (body.state() or "").endswith("/inactive") or body.state() is None, timeout=5.0)

    spoken = body.encoder().calls[0] if body.encoder().calls else ""
    reply_audio = body.encoder().audio or b""
    if not reply_audio and reply_wav.exists():
        reply_audio = reply_wav.read_bytes()
    completed_outputs: list[object] = []
    for completed in body.completed():
        output = completed.output
        if isinstance(output, tuple):
            completed_outputs.append(
                [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in output]
            )
        else:
            completed_outputs.append(output)
    summary: dict[str, object] = {
        "status": "ok",
        "heard_phrase": _HEARD_PHRASE,
        "heard_wav": str(heard_wav),
        "heard_bytes": len(heard_audio),
        "cognition_model": app_config.cognition.model,
        "intuition_model": DEFAULT_GEMINI_INTUITION_MODEL,
        "cognition_client": f"ChatClient({app_config.cognition.model})",
        "stt_calls": len(body.decoder().calls),
        "listening_handoffs": len(body.listening_handoffs()),
        "spoken_text": spoken,
        "reply_wav": str(reply_wav),
        "reply_bytes": len(reply_audio),
        "processing_completed": len(body.completed()),
        "processing_outputs": completed_outputs,
        "processing_failures": [failure.message for failure in body.failures()],
        "bot_state": body.state(),
    }

    if play:
        for label, path in (("heard", heard_wav), ("reply", reply_wav)):
            if path.exists():
                print(f"Playing {label}: {path}", file=sys.stderr)
                _ = subprocess.run(["afplay", str(path)], check=False)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Device-free listen→speak bot with Gemini intuition/reasoning (macOS say)."
    )
    _ = parser.add_argument(
        "--play",
        action="store_true",
        help="Play the heard and reply WAVs with afplay after the run.",
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
        print("stateforward.bot listen→speak (device-free, Gemini cognition)")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    return 0 if summary.get("status") == "ok" else 1


__all__ = [
    "AppConfig",
    "AlwaysVoiceDetector",
    "CognitionConfig",
    "DEFAULT_GEMINI_MODEL",
    "FixedTranscriptDecoder",
    "ListenSpeakBot",
    "SayEncoder",
    "load_env",
    "main",
    "run",
]
