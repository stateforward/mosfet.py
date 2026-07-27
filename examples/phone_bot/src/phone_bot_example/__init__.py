from __future__ import annotations

from bot import abilities
import bot
from bot.abilities import ability
from bot.abilities import cognition
from bot.abilities import decoding
from bot.abilities import listening
from bot.abilities import memory
from bot.abilities import participating
from bot.abilities import speaking

# ``start_bot`` binds ``cognition`` as a parameter, so directives are imported by module name.
from bot.abilities.cognition import directives
from bot.abilities.hearing import sound as sound_hearing
from bot.abilities.hearing import speech
from bot.abilities.hearing import voice

import argparse
import asyncio
import collections.abc
import dataclasses
import datetime
import json
import logging
import pathlib
import typing

import hsm

from bot.bot import Bot

from bot.devices import audio
from bot.devices import phone as phone_device
from bot.providers.gemini import ChatClient as GeminiChatClient
from bot.providers.gemini import SpeechDecoder as GeminiSpeechDecoder
from bot.providers.gemini import SpeechEncoder as GeminiSpeechEncoder
from bot.providers.mlx_audio import VoiceDetector as SileroVoiceDetector
from bot.providers.openai_compat import ChatClient as OpenAIChatClient
from bot.providers.openai_compat import Processor as OpenAIProcessor
from bot.providers.livekit import PhoneService
from bot.providers.livekit.audio import PcmWavDecoder
from bot.telemetry import observed_event, observed_occurrence
from bot.environment import Environment, space

_LOG = logging.getLogger("phone_bot_example.hsm")

_EXAMPLE_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REPO_ROOT = _EXAMPLE_ROOT.parent.parent
_REPO_ENV_PATH = _REPO_ROOT / ".env"

DEFAULT_MERCURY_MODEL = "mercury-2"
DEFAULT_MERCURY_BASE_URL = "https://api.inceptionlabs.ai/v1"
DEFAULT_OPENAI_REASONING_MODEL = "gpt-5.6-terra"
DEFAULT_OPENAI_REFLECTION_MODEL = "gpt-5.6-terra"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
# Local Silero VAD (cheap) + off-device Gemini STT/TTS (no local whisper/Qwen).
DEFAULT_GEMINI_STT_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_GEMINI_TTS_VOICE = "Kore"
DEFAULT_SILERO_VAD_MODEL = "mlx-community/silero-vad"
DEFAULT_LIVEKIT_TRACK_NAME = "bot-phone-bot"
DEFAULT_LIVEKIT_INPUT_SAMPLE_RATE_HZ = 48_000
# Gemini TTS default playout rate (speaker / LiveKit metadata).
DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE_HZ = 24_000
DEFAULT_AUDIO_CHANNELS = 1


def _strip_env_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def load_env(path: pathlib.Path) -> dict[str, str]:
    """Load the simple KEY=VALUE provider file used by this local example."""

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


def _env_int(env: collections.abc.Mapping[str, str], default: int, *names: str) -> int:
    value = _env_first(env, *names)
    if value is None:
        return default
    return int(value)


def mint_livekit_access_token(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room: str,
    name: str | None = None,
    ttl_seconds: int = 6 * 60 * 60,
) -> str:
    """Mint a LiveKit join JWT (HS256) with publish + subscribe grants."""

    import base64
    import hashlib
    import hmac
    import time

    def b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload: dict[str, object] = {
        "iss": api_key,
        "sub": identity,
        "nbf": now - 10,
        "exp": now + ttl_seconds,
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
    }
    if name is not None:
        payload["name"] = name
    segments = (
        f"{b64url(json.dumps(header, separators=(',', ':')).encode())}."
        f"{b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    )
    signature = hmac.new(api_secret.encode("utf-8"), segments.encode("ascii"), hashlib.sha256).digest()
    return f"{segments}.{b64url(signature)}"


@dataclasses.dataclass(frozen=True)
class LiveKitConfig:
    url: str | None = None
    token: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    room: str = "bot-phone-bot"
    identity: str = "bot-phone-bot"
    track_name: str = DEFAULT_LIVEKIT_TRACK_NAME

    @classmethod
    def from_env(cls, env: collections.abc.Mapping[str, str]) -> typing.Self:
        url = _env_first(env, "BOT_LIVEKIT_URL", "LIVEKIT_URL", "VA_LIVEKIT_URL")
        token = _env_first(env, "BOT_LIVEKIT_TOKEN", "LIVEKIT_TOKEN", "VA_LIVEKIT_TOKEN")
        api_key = _env_first(env, "BOT_LIVEKIT_API_KEY", "LIVEKIT_API_KEY", "VA_LIVEKIT_API_KEY")
        api_secret = _env_first(env, "BOT_LIVEKIT_API_SECRET", "LIVEKIT_API_SECRET", "VA_LIVEKIT_API_SECRET")
        room = _env_first(env, "BOT_LIVEKIT_ROOM", "LIVEKIT_ROOM", "VA_LIVEKIT_ROOM") or "bot-phone-bot"
        identity = _env_first(env, "BOT_LIVEKIT_IDENTITY", "LIVEKIT_IDENTITY", "VA_LIVEKIT_IDENTITY") or "bot-phone-bot"
        track_name = (
            _env_first(env, "BOT_LIVEKIT_TRACK_NAME", "LIVEKIT_TRACK_NAME", "VA_LIVEKIT_TRACK_NAME")
            or DEFAULT_LIVEKIT_TRACK_NAME
        )
        if token is None and api_key is not None and api_secret is not None:
            token = mint_livekit_access_token(
                api_key=api_key,
                api_secret=api_secret,
                identity=identity,
                room=room,
            )
        return cls(
            url=url,
            token=token,
            api_key=api_key,
            api_secret=api_secret,
            room=room,
            identity=identity,
            track_name=track_name,
        )

    def can_connect_room(self) -> bool:
        return self.url is not None and self.token is not None


@dataclasses.dataclass(frozen=True)
class CognitionConfig:
    """Intuition on Mercury 2 (OpenAI-compatible); reasoning and reflection on OpenAI Terra."""

    model: str = DEFAULT_OPENAI_REASONING_MODEL
    api_key: str | None = None
    base_url: str = DEFAULT_OPENAI_BASE_URL
    intuition_model: str = DEFAULT_MERCURY_MODEL
    intuition_api_key: str | None = None
    intuition_base_url: str = DEFAULT_MERCURY_BASE_URL
    reflection_model: str = DEFAULT_OPENAI_REFLECTION_MODEL
    reflection_api_key: str | None = None
    reflection_base_url: str = DEFAULT_OPENAI_BASE_URL

    def can_process(self) -> bool:
        reflection_key = self.reflection_api_key if self.reflection_api_key is not None else self.api_key
        return self.api_key is not None and self.intuition_api_key is not None and reflection_key is not None


@dataclasses.dataclass(frozen=True)
class SpeechConfig:
    """Silero VAD (local) + off-device Gemini STT/TTS."""

    api_key: str | None = None
    voice_name: str = DEFAULT_GEMINI_TTS_VOICE
    tts_model: str = DEFAULT_GEMINI_TTS_MODEL
    stt_model: str = DEFAULT_GEMINI_STT_MODEL
    vad_model_id: str = DEFAULT_SILERO_VAD_MODEL
    input_sample_rate_hz: int = DEFAULT_LIVEKIT_INPUT_SAMPLE_RATE_HZ
    input_channels: int = DEFAULT_AUDIO_CHANNELS
    output_sample_rate_hz: int = DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE_HZ
    output_channels: int = DEFAULT_AUDIO_CHANNELS

    @classmethod
    def from_env(cls, env: collections.abc.Mapping[str, str]) -> typing.Self:
        return cls(
            api_key=_env_first(
                env,
                "BOT_GEMINI_API_KEY",
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
            ),
            voice_name=_env_first(
                env,
                "BOT_GEMINI_TTS_VOICE",
                "GEMINI_TTS_VOICE",
                "BOT_TTS_VOICE",
            )
            or DEFAULT_GEMINI_TTS_VOICE,
            tts_model=_env_first(
                env,
                "BOT_GEMINI_TTS_MODEL",
                "GEMINI_TTS_MODEL",
                "BOT_TTS_MODEL",
            )
            or DEFAULT_GEMINI_TTS_MODEL,
            stt_model=_env_first(
                env,
                "BOT_GEMINI_STT_MODEL",
                "GEMINI_STT_MODEL",
                "BOT_STT_MODEL",
            )
            or DEFAULT_GEMINI_STT_MODEL,
            vad_model_id=_env_first(
                env,
                "BOT_SILERO_VAD_MODEL",
                "BOT_VAD_MODEL",
                "SILERO_VAD_MODEL",
            )
            or DEFAULT_SILERO_VAD_MODEL,
            input_sample_rate_hz=_env_int(
                env,
                DEFAULT_LIVEKIT_INPUT_SAMPLE_RATE_HZ,
                "BOT_LIVEKIT_INPUT_SAMPLE_RATE_HZ",
                "LIVEKIT_INPUT_SAMPLE_RATE_HZ",
            ),
            input_channels=_env_int(
                env,
                DEFAULT_AUDIO_CHANNELS,
                "BOT_LIVEKIT_INPUT_CHANNELS",
                "LIVEKIT_INPUT_CHANNELS",
            ),
            output_sample_rate_hz=_env_int(
                env,
                DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE_HZ,
                "BOT_GEMINI_OUTPUT_SAMPLE_RATE_HZ",
                "GEMINI_OUTPUT_SAMPLE_RATE_HZ",
            ),
            output_channels=_env_int(
                env,
                DEFAULT_AUDIO_CHANNELS,
                "BOT_GEMINI_OUTPUT_CHANNELS",
                "GEMINI_OUTPUT_CHANNELS",
            ),
        )

    def can_encode_speech(self) -> bool:
        return self.api_key is not None

    def can_decode_speech(self) -> bool:
        return self.api_key is not None

    def can_publish_livekit_audio(self) -> bool:
        return self.can_encode_speech() and self.can_decode_speech()


@dataclasses.dataclass(frozen=True)
class AppConfig:
    env_path: pathlib.Path | None = None
    cognition: CognitionConfig = dataclasses.field(default_factory=CognitionConfig)
    livekit: LiveKitConfig = dataclasses.field(default_factory=LiveKitConfig)
    speech: SpeechConfig = dataclasses.field(default_factory=SpeechConfig)
    # Things this bot was told, in the words they were told in. They become standing directives
    # in its memory before the first turn; the example never reads them.
    told: tuple[str, ...] = ()

    @classmethod
    def from_env_file(cls, path: pathlib.Path | None = None) -> typing.Self:
        # Repo root .env then example .env (example wins).
        env: dict[str, str] = {}
        if _REPO_ENV_PATH.exists():
            env.update(load_env(_REPO_ENV_PATH))
        if path is not None and path.exists():
            env.update(load_env(path))
        # One thing the env file can say to the bot. `--tell` is the repeatable form; a KEY=VALUE
        # line holds one instruction and cannot hold two without splitting somebody's prose.
        told = _env_first(env, "BOT_TELL")
        openai_api_key = _env_first(env, "BOT_OPENAI_API_KEY", "OPENAI_API_KEY")
        openai_base_url = _env_first(env, "BOT_OPENAI_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        return cls(
            env_path=path,
            cognition=CognitionConfig(
                model=_env_first(
                    env,
                    "BOT_REASONING_MODEL",
                    "BOT_OPENAI_REASONING_MODEL",
                )
                or DEFAULT_OPENAI_REASONING_MODEL,
                api_key=openai_api_key,
                base_url=openai_base_url,
                intuition_model=_env_first(
                    env,
                    "BOT_INTUITION_MODEL",
                    "BOT_MERCURY_MODEL",
                    "MERCURY_MODEL",
                )
                or DEFAULT_MERCURY_MODEL,
                intuition_api_key=_env_first(
                    env,
                    "BOT_MERCURY_API_KEY",
                    "MERCURY_API_KEY",
                    "INCEPTION_API_KEY",
                    "BOT_INTUITION_API_KEY",
                ),
                intuition_base_url=_env_first(
                    env,
                    "BOT_MERCURY_BASE_URL",
                    "MERCURY_BASE_URL",
                    "INCEPTION_BASE_URL",
                )
                or DEFAULT_MERCURY_BASE_URL,
                reflection_model=_env_first(env, "BOT_REFLECTION_MODEL", "BOT_OPENAI_REFLECTION_MODEL")
                or DEFAULT_OPENAI_REFLECTION_MODEL,
                reflection_api_key=openai_api_key,
                reflection_base_url=openai_base_url,
            ),
            livekit=LiveKitConfig.from_env(env),
            speech=SpeechConfig.from_env(env),
            told=() if told is None else (told,),
        )

    def with_cognition_overrides(self, *, model: str | None) -> typing.Self:
        return dataclasses.replace(
            self,
            cognition=dataclasses.replace(
                self.cognition,
                model=model or self.cognition.model,
            ),
        )


def _mercury_intuition_client(config: CognitionConfig) -> OpenAIChatClient:
    # Missing keys stay constructible so smoke runs can report blocked; live calls fail closed.
    return OpenAIChatClient(
        model=config.intuition_model,
        api_key=config.intuition_api_key or "",
        base_url=config.intuition_base_url,
    )


def _openai_reasoning_client(config: CognitionConfig) -> OpenAIChatClient:
    return OpenAIChatClient(
        model=config.model,
        api_key=config.api_key or "",
        base_url=config.base_url,
    )


def _openai_reflection_client(config: CognitionConfig) -> OpenAIChatClient:
    reflection_key = config.reflection_api_key if config.reflection_api_key is not None else config.api_key
    return OpenAIChatClient(
        model=config.reflection_model,
        api_key=reflection_key or "",
        base_url=config.reflection_base_url or config.base_url,
    )


def _phone_cognition(
    config: CognitionConfig | None = None,
    *,
    memory: memory.Memory | None = None,
) -> cognition.Cognition:
    config = config or CognitionConfig()
    store = memory if memory is not None else _memory()
    # Mercury 2 intuition (OpenAI-compat); OpenAI Terra reasoning + reflection.
    # Reflection owns the shared Memory lifecycle. Autonomy and Reasoning use its public
    # execute capability as injected collaborators without attaching it again.
    intuition = OpenAIProcessor(
        client=_mercury_intuition_client(config),
        provider="mercury2_intuition",
    )
    deliberate = OpenAIProcessor(
        client=_openai_reasoning_client(config),
        provider="openai_terra_reasoning",
    )
    reflection_processor = OpenAIProcessor(
        client=_openai_reflection_client(config),
        provider="openai_terra_reflection",
    )
    _LOG.info(
        "cognition wired intuition_model=%s intuition_base_url=%s reasoning_model=%s "
        "reasoning_base_url=%s reflection_model=%s",
        config.intuition_model,
        config.intuition_base_url,
        config.model,
        config.base_url,
        config.reflection_model,
    )
    return cognition.Cognition(
        autonomy=cognition.Autonomy(memory=store),
        intuition=cognition.Intuition(processor=intuition),
        reasoning=cognition.Reasoning(processor=deliberate, memory=store),
        reflection=cognition.Reflection(processor=reflection_processor, memory=store),
    )


class TranscriptTextDecoder(decoding.Decoder[participating.ParticipationStimulus, str]):
    """Decode TextStimulus (Listening STT transcript) for conversation participation."""

    @typing.override
    async def decode(self, input: participating.ParticipationStimulus) -> str:
        if isinstance(input, participating.TextStimulus):
            return input.content
        if isinstance(input, participating.EventStimulus):
            text = input.payload.get("text")
            if isinstance(text, str):
                return text
        raise AssertionError(f"unexpected conversation stimulus {input!r}")


class ExampleTextConversation(abilities.TextConversation):
    """Thin text conversation for product path: Listening STT → Message → contribution → Bot → Speaking."""

    _outputs: list[abilities.Response]
    _failures: list[abilities.FailureData]

    def __init__(
        self,
        *,
        participating: abilities.Participating,
        decoder: decoding.Decoder[participating.ParticipationStimulus, str] | None = None,
    ) -> None:
        super().__init__(
            participating=participating,
            decoding=decoding.Decoding(decoder=decoder if decoder is not None else TranscriptTextDecoder()),
        )
        self._outputs = []
        self._failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        maybe_terminal = (
            event.data if event.name in {ability.TerminalOutputEvent.name, ability.TerminalErrorEvent.name} else event
        )
        terminal_event = (
            typing.cast(hsm.Event[typing.Any], maybe_terminal) if isinstance(maybe_terminal, hsm.Event) else event
        )
        if event.name == self.output_event.name or terminal_event.name == self.output_event.name:
            output = typing.cast(abilities.Response, terminal_event.data)
            assert isinstance(output, abilities.Response)
            self._outputs.append(output)
        if event.name == self.failed_event.name or terminal_event.name == self.failed_event.name:
            failure = typing.cast(abilities.FailureData, terminal_event.data)
            assert isinstance(failure, abilities.FailureData)
            self._failures.append(failure)
        return super().dispatch(ctx, event)

    def outputs(self) -> tuple[abilities.Response, ...]:
        return tuple(self._outputs)

    def failures(self) -> tuple[abilities.FailureData, ...]:
        return tuple(self._failures)


def _is_wav_container(audio: bytes) -> bool:
    """True when bytes already look like a RIFF/WAVE container (e.g. ring clip)."""

    return len(audio) >= 12 and audio.startswith(b"RIFF") and audio[8:12] == b"WAVE"


@dataclasses.dataclass(frozen=True, kw_only=True)
class PcmAwareVoiceDetector(voice.detection.VoiceDetector):
    """Silero VAD with PCM→WAV wrap for LiveKit chunks; pass through real WAV (ring)."""

    pcm_decoder: PcmWavDecoder
    voice_detector: voice.detection.VoiceDetector

    @typing.override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        wav = input if _is_wav_container(input) else await self.pcm_decoder.decode(input)
        return await self.voice_detector.classify(wav)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PcmListeningSpeechDecoder(speech.SpeechDecoder):
    """Cloud Gemini STT: wrap LiveKit PCM as WAV; leave WAV (ring clip) alone."""

    pcm_decoder: PcmWavDecoder
    speech_decoder: speech.SpeechDecoder

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        wav = input if _is_wav_container(input) else await self.pcm_decoder.decode(input)
        return await self.speech_decoder.decode(wav)


def _gemini_speech_client(config: SpeechConfig) -> GeminiChatClient:
    return GeminiChatClient(api_key=config.api_key, model=config.stt_model)


def _gemini_speech_decoder(config: SpeechConfig) -> GeminiSpeechDecoder:
    return GeminiSpeechDecoder(
        client=_gemini_speech_client(config),
        model=config.stt_model,
        mime_type="audio/wav",
    )


def _gemini_speech_encoder(config: SpeechConfig) -> GeminiSpeechEncoder:
    return GeminiSpeechEncoder(
        client=_gemini_speech_client(config),
        model=config.tts_model,
        voice_name=config.voice_name,
        sample_rate_hz=config.output_sample_rate_hz,
        # Raw PCM, not a WAV container: this audio ends up on the LiveKit uplink, and
        # AudioFrameEncoder frames the bytes as samples. A RIFF header would be framed as audio.
        output_format="pcm",
    )


def _silero_voice_detector(config: SpeechConfig) -> SileroVoiceDetector:
    """Cheap local Silero VAD (MLX Audio). STT/TTS remain off-device Gemini."""

    return SileroVoiceDetector(model_id=config.vad_model_id)


def _conversation(speech_config: SpeechConfig | None = None) -> ExampleTextConversation:
    """Acquired Conversation for product path: Listening STT transcript → text Message → Speaking.

    TTS remains on Bot Speaking (not host_turn / Conversation encoder).
    """

    del speech_config
    return ExampleTextConversation(participating=abilities.Participating())


def _listening(speech_config: SpeechConfig | None = None) -> listening.Listening:
    config = speech_config or SpeechConfig()
    pcm_decoder = PcmWavDecoder(sample_rate_hz=config.input_sample_rate_hz, channels=config.input_channels)
    return listening.Listening(
        voice_detector=PcmAwareVoiceDetector(
            pcm_decoder=pcm_decoder,
            voice_detector=_silero_voice_detector(config),
        ),
        # Non-voice environment.sound with SoundData.kind (e.g. ring) becomes cognition.InputEvent.
        sound_classifier=sound_hearing.classification.KindSoundClassifier(),
        speech_decoder=PcmListeningSpeechDecoder(
            pcm_decoder=pcm_decoder,
            speech_decoder=_gemini_speech_decoder(config),
        ),
    )


# Where this robot is and how loud its parts are. dB SPL at one metre.
#
# The handset earpiece sits at the robot's head, so it is loud in the robot's own ears (45 dB of
# margin) and far too quiet by the time it reaches the mouthpiece 15 cm away (28.5 dB below the
# mouthpiece threshold) — which is what stops the far end hearing itself. The bot's voice comes
# out at the mouth, right at the mouthpiece, 30 dB over the threshold, so it goes up the wire.
_BOT_ORIGIN = space.Position(x=0.0, y=0.0)
_MOUTH = space.Position(x=phone_device.MOUTH_OFFSET_M, y=0.0)

_EARPIECE_DB = 25.0
"""Handset receiver at one metre: about 65 dB at the ear, where the ear actually is."""

_VOICE_DB = 60.0
"""Normal conversational speech at one metre."""

_EARS_THRESHOLD_DB = 20.0
"""Quiet-room hearing floor for the robot's own ears."""

_MOUTHPIECE_THRESHOLD_DB = 70.0
"""Close-talk handset mouthpiece: it hears the mouth it is held next to, and little else."""


def _earpiece() -> audio.Speaker:
    """The handset receiver: at the robot's ear, quiet, for the robot to hear the far end."""

    return audio.Speaker(
        placement=space.Placement(position=_BOT_ORIGIN),
        amplitude_db=_EARPIECE_DB,
    )


def _voice() -> audio.Speaker:
    """The robot's mouth: at the mouthpiece, conversational, for the far end to hear it."""

    return audio.Speaker(placement=space.Placement(position=_MOUTH), amplitude_db=_VOICE_DB)


def _mouthpiece() -> audio.Microphone:
    """The handset transmitter: close-talk, so it picks up the mouth and not the earpiece."""

    return audio.Microphone(placement=space.Placement(position=_MOUTH, threshold_db=_MOUTHPIECE_THRESHOLD_DB))


def _handset(*, service: phone_device.PhoneService | None = None) -> phone_device.Phone:
    """A handset held to the robot's ear, with its own transducers placed on it."""

    return phone_device.Phone(
        service=service,
        speaker=_earpiece(),
        microphone=_mouthpiece(),
        placement=space.Placement(position=_BOT_ORIGIN),
    )


def _speaking(
    *,
    speaker: audio.Speaker,
    speech_config: SpeechConfig | None = None,
) -> speaking.Speaking:
    """Bot output ability: Gemini TTS + phone speaker playout (environment.sound elevation)."""

    config = speech_config or SpeechConfig()
    return speaking.Speaking(
        encoder=_gemini_speech_encoder(config),
        speaker=speaker,
        sample_rate_hz=config.output_sample_rate_hz,
        channels=config.output_channels,
        media_type="audio/pcm",
    )


def _memory() -> memory.ShortTermMemory:
    return memory.ShortTermMemory()


def _observed_event(event: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
    data = event.data
    if not isinstance(data, dict):
        return hsm.Event(name="phone_bot_example.unknown_observation")
    values = typing.cast(dict[str, object], data)
    observed = values.get("event")
    if isinstance(observed, hsm.Event):
        return typing.cast(hsm.Event[typing.Any], observed)
    return hsm.Event(name="phone_bot_example.unknown_observation")


def _record_phone_bot_output(
    ctx: hsm.Context,
    instance: "PhoneBot",
    event: hsm.Event[typing.Any],
) -> None:
    del ctx
    observed = _observed_event(event)
    data = observed.data
    if isinstance(data, bot.ProcessingCompletedEventData):
        instance.record_output(data.output)


def _record_phone_bot_failure(
    ctx: hsm.Context,
    instance: "PhoneBot",
    event: hsm.Event[typing.Any],
) -> None:
    del ctx
    observed = _observed_event(event)
    data = observed.data
    if isinstance(data, bot.ProcessingFailedEventData):
        instance.record_failure(data)


def _summarize_event_data(data: object) -> str:
    """Low-cardinality data summary for logs (never raw audio/text payloads)."""

    if data is None:
        return "none"
    if isinstance(data, (bytes, bytearray)):
        return f"bytes[{len(data)}]"
    if isinstance(data, str):
        return f"str[{len(data)}]"
    if isinstance(data, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], data)
        keys = sorted(str(key) for key in mapping)
        audio_len: int | None = None
        raw_audio = mapping.get("audio")
        if isinstance(raw_audio, (bytes, bytearray)):
            audio_len = len(raw_audio)
        sample_rate = mapping.get("sample_rate_hz")
        parts = [f"keys={keys}"]
        if audio_len is not None:
            parts.append(f"audio_bytes={audio_len}")
        if isinstance(sample_rate, int):
            parts.append(f"sample_rate_hz={sample_rate}")
        return " ".join(parts)
    type_name = type(data).__name__
    audio = getattr(data, "audio", None)
    if isinstance(audio, (bytes, bytearray)):
        sample_rate = getattr(data, "sample_rate_hz", None)
        rate = f" sample_rate_hz={sample_rate}" if isinstance(sample_rate, int) else ""
        return f"{type_name}(audio_bytes={len(audio)}{rate})"
    return type_name


def _log_phone_bot_observation(
    ctx: hsm.Context,
    instance: "PhoneBot",
    observation: hsm.Event[typing.Any],
) -> None:
    """hsm.observe callback: print HSM activity for the phone bot body."""

    del ctx
    event = observed_event(observation)
    occurrence = observed_occurrence(observation)
    _LOG.info(
        "phone_bot observe state=%s occurrence=%s event=%s kind=%s data=%s",
        instance.state(),
        occurrence,
        event.name,
        event.kind,
        _summarize_event_data(event.data),
    )


class PhoneBot(Bot):
    """Bot with LiveKit phone, input Listening, output Speaking, and optional conversation."""

    _processing_timeout: typing.ClassVar[datetime.timedelta] = datetime.timedelta(seconds=600)
    model: typing.ClassVar[hsm.Model] = hsm.define(
        "PhoneBot",
        Bot.model,
        hsm.observe(_log_phone_bot_observation),
        hsm.observe(bot.ProcessingCompletedEvent, _record_phone_bot_output),
        hsm.observe(bot.ProcessingFailedEvent, _record_phone_bot_failure),
    )
    _label: str
    _phone: phone_device.Phone
    _listening: listening.Listening
    _speaking: speaking.Speaking
    _conversation: ExampleTextConversation
    _memory: memory.Memory
    _outputs: list[cognition.types.OutputData]
    _failures: list[bot.ProcessingFailedEventData]
    _conversation_outputs: list[abilities.Response]
    _conversation_failures: list[abilities.FailureData]
    _listening_handoffs: list[cognition.InputData]
    _listening_failures: list[listening.FailedEventData]

    def __init__(
        self,
        label: str,
        *,
        phone: phone_device.Phone | None = None,
        voice: audio.Speaker | None = None,
        cognition_config: CognitionConfig | None = None,
        speech_config: SpeechConfig | None = None,
        cognition: cognition.Cognition | None = None,
        listening: listening.Listening | None = None,
        speaking: speaking.Speaking | None = None,
        conversation: ExampleTextConversation | None = None,
        memory: memory.Memory | None = None,
    ) -> None:
        self._label = label
        # Two transducers, because one object cannot be both at the ear and at the mouth. The
        # earpiece belongs to the handset; the voice belongs to the robot. Nothing to enforce
        # between them any more — Environment.join rejects a speaker placed in two places.
        self._phone = phone if phone is not None else _handset()
        # The voice transducer is the robot's own device, not a peripheral of its telephone: the
        # robot speaks, and the handset is only what it holds to its ear. Registering it here is
        # what powers it — Speaking attaches to it, but device lifetime belongs to the body.
        self._voice = voice if voice is not None else _voice()
        speaking_instance = (
            speaking if speaking is not None else _speaking(speaker=self._voice, speech_config=speech_config)
        )
        self._memory = memory if memory is not None else _memory()
        cognition_instance = (
            cognition if cognition is not None else _phone_cognition(cognition_config, memory=self._memory)
        )
        self._listening = listening if listening is not None else _listening(speech_config)
        self._speaking = speaking_instance
        self._conversation = conversation if conversation is not None else _conversation(speech_config)
        super().__init__(
            # Phone first: an unfocused turn falls back to the first configured device.
            devices={"phone": self._phone, "voice": self._voice},
            cognition=cognition_instance,
            input=(self._listening,),
            output=(self._speaking,),
            # Conversation is bot-acquired. Memory is attached under Reflection only
            # (Ability attachment is exclusive; do not double-attach the same instance).
            acquired_abilities=(self._conversation,),
        )
        self._outputs = []
        self._failures = []
        self._conversation_outputs = []
        self._conversation_failures = []
        self._listening_handoffs = []
        self._listening_failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self._conversation.output_event.name:
            output = event.data
            assert isinstance(output, abilities.Response)
            self._conversation_outputs.append(output)
        if event.name == self._conversation.failed_event.name:
            failure = event.data
            assert isinstance(failure, abilities.FailureData)
            self._conversation_failures.append(failure)
        if event.name == cognition.InputEvent.name:
            cognitive = event.data
            assert isinstance(cognitive, cognition.InputData)
            # Sensory Listening terminal is cognition.InputEvent (source is the listening ability).
            if event.source == hsm.id(self._listening):
                self._listening_handoffs.append(cognitive)
        if event.name == self._listening.failed_event.name:
            listening_failure = event.data
            assert isinstance(listening_failure, listening.FailedEventData)
            self._listening_failures.append(listening_failure)
        return super().dispatch(ctx, event)

    def record_output(self, output: cognition.types.OutputData) -> None:
        self._outputs.append(output)

    def record_failure(self, failure: bot.ProcessingFailedEventData) -> None:
        self._failures.append(failure)

    def label(self) -> str:
        return self._label

    def phone(self) -> phone_device.Phone:
        return self._phone

    def conversation(self) -> ExampleTextConversation:
        return self._conversation

    def speaking(self) -> speaking.Speaking:
        return self._speaking

    def listening(self) -> listening.Listening:
        return self._listening

    def memory(self) -> memory.Memory:
        return self._memory

    def cognition(self) -> cognition.Cognition:
        return typing.cast(cognition.Cognition, self._cognition)

    def outputs(self) -> tuple[cognition.types.OutputData, ...]:
        return tuple(self._outputs)

    def failures(self) -> tuple[bot.ProcessingFailedEventData, ...]:
        return tuple(self._failures)

    def conversation_outputs(self) -> tuple[abilities.Response, ...]:
        return tuple(self._conversation_outputs)

    def conversation_failures(self) -> tuple[abilities.FailureData, ...]:
        return tuple(self._conversation_failures)

    def listening_handoffs(self) -> tuple[cognition.InputData, ...]:
        return tuple(self._listening_handoffs)

    def listening_failures(self) -> tuple[listening.FailedEventData, ...]:
        return tuple(self._listening_failures)


async def _wait_until(condition: collections.abc.Callable[[], bool], *, timeout_seconds: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise RuntimeError("Timed out waiting for the phone-bot example.")


async def _wait_for_active_bot(body: PhoneBot) -> None:
    await _wait_until(
        lambda: (body.state() or "").endswith("/active/unfocused") or (body.state() or "").endswith("/inactive"),
        timeout_seconds=30.0,
    )
    if not (body.state() or "").endswith("/active/unfocused"):
        raise RuntimeError(f"Phone bot activation failed in state {body.state()}.")
    await _wait_until(
        lambda: (body.conversation().state() or "").endswith("/behavior/silent"),
        timeout_seconds=30.0,
    )


async def start_bot(
    label: str,
    *,
    config: AppConfig | None = None,
    connect_livekit: bool = False,
    phone: phone_device.Phone | None = None,
    voice: audio.Speaker | None = None,
    phone_service: PhoneService | None = None,
    cognition: cognition.Cognition | None = None,
    listening: listening.Listening | None = None,
    speaking: speaking.Speaking | None = None,
    conversation: ExampleTextConversation | None = None,
    memory: memory.Memory | None = None,
) -> PhoneBot:
    app_config = config or AppConfig.from_env_file()
    body = PhoneBot(
        label,
        phone=phone,
        voice=voice,
        cognition_config=app_config.cognition,
        speech_config=app_config.speech,
        cognition=cognition,
        listening=listening,
        speaking=speaking,
        conversation=conversation,
        memory=memory,
    )
    # What this bot was told, written where a bot keeps what it was told, before it is awake to
    # have a turn about any of it. The text is opaque here: nothing reads it, nothing routes on
    # it, and a bot that recalls "Call Bob" is as free to ignore that as a person would be.
    for instruction in app_config.told:
        _ = body.memory().execute(
            directives.directive_insert_input(directives.Directive(text=instruction), context_ref=None)
        )

    environment = Environment()
    _ = await body.attach(environment, placement=space.Placement(position=_BOT_ORIGIN, threshold_db=_EARS_THRESHOLD_DB))
    await _wait_for_active_bot(body)
    if connect_livekit and app_config.livekit.can_connect_room():
        if phone_service is None:
            raise RuntimeError("connect_livekit requires an explicit LiveKit PhoneService reference.")
        assert app_config.livekit.url is not None
        assert app_config.livekit.token is not None
        await phone_service.connect_room(
            url=app_config.livekit.url,
            token=app_config.livekit.token,
            track_name=app_config.livekit.track_name,
        )
        await _wait_for_livekit_room_audio_result(phone_service)
    return body


async def _wait_for_livekit_room_audio_result(service: PhoneService) -> None:
    await _wait_until(
        lambda: service.media_snapshot().local_track_sid is not None
        or (_service_track_path_state(service) or "").endswith("/failed"),
        timeout_seconds=30.0,
    )


def _service_track_path_state(service: PhoneService) -> str | None:
    track_path = object.__getattribute__(service, "_track_path")
    if track_path is None:
        return None
    return typing.cast(str | None, track_path.state())


def _warnings(config: AppConfig, *, connect_livekit: bool) -> list[str]:
    warnings: list[str] = []
    if config.env_path is None:
        warnings.append("No provider env file was provided; using example defaults.")
    elif not config.env_path.exists():
        warnings.append("Provider env file was not found; using example defaults.")
    if not config.cognition.api_key:
        warnings.append("BOT_OPENAI_API_KEY / OPENAI_API_KEY is missing; OpenAI Terra reasoning will fail.")
    if not config.cognition.intuition_api_key:
        warnings.append("BOT_MERCURY_API_KEY / MERCURY_API_KEY is missing; Mercury intuition will fail.")
    reflection_key = (
        config.cognition.reflection_api_key
        if config.cognition.reflection_api_key is not None
        else config.cognition.api_key
    )
    if not reflection_key:
        warnings.append("BOT_OPENAI_API_KEY / OPENAI_API_KEY is missing; OpenAI Terra reflection will fail.")
    if not config.speech.api_key:
        warnings.append(
            "BOT_GEMINI_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY is missing; "
            "off-device Gemini STT/TTS will fail (VAD stays local Silero)."
        )
    if not config.livekit.can_connect_room():
        warnings.append(
            "BOT_LIVEKIT_URL/VA_LIVEKIT_URL and BOT_LIVEKIT_TOKEN/VA_LIVEKIT_TOKEN are missing; "
            + "LiveKit room audio is not connected."
        )
    if config.livekit.can_connect_room() and not connect_livekit:
        warnings.append("LiveKit room config is loaded but connection is opt-in; pass --connect-livekit to attempt it.")
    return warnings


def _summary_for(
    *,
    app_config: AppConfig,
    body: PhoneBot,
    phone_service: PhoneService,
    connect_livekit: bool,
) -> dict[str, object]:
    snapshot = phone_service.media_snapshot()
    room_connected = snapshot.local_track_sid is not None
    room_attempted = connect_livekit and app_config.livekit.can_connect_room()
    can_talk = room_connected and app_config.cognition.can_process() and app_config.speech.can_publish_livekit_audio()
    return {
        "status": "ready" if can_talk else "blocked",
        "can_talk": can_talk,
        "bot": body.label(),
        "bot_state": body.state(),
        "phone_component": "bot.devices.phone.Phone",
        "phone_service_component": "bot.providers.livekit.PhoneService",
        "cognition_client": (
            f"mercury={app_config.cognition.intuition_model}@"
            f"{app_config.cognition.intuition_base_url} "
            f"openai_terra_reasoning={app_config.cognition.model}@"
            f"{app_config.cognition.base_url} "
            f"openai_terra_reflection={app_config.cognition.reflection_model}"
        ),
        "mercury_intuition": {
            "model": app_config.cognition.intuition_model,
            "base_url": app_config.cognition.intuition_base_url,
            "api_key_loaded": bool(app_config.cognition.intuition_api_key),
        },
        # How many things this bot was told, not what they were: the operator gets confirmation
        # that the words were heard without the readiness summary quoting them back.
        "told": len(app_config.told),
        "livekit_room_audio_attempted": room_attempted,
        "livekit_room_audio_configured": app_config.livekit.can_connect_room(),
        "livekit_room_audio_connected": room_connected,
        "livekit_remote_audio_chunks": snapshot.remote_audio_chunks,
        "livekit_remote_audio_bytes": snapshot.remote_audio_bytes,
        "warnings": _warnings(app_config, connect_livekit=connect_livekit),
        "openai_reasoning": {
            "model": app_config.cognition.model,
            "api_key_loaded": bool(app_config.cognition.api_key),
            "base_url": app_config.cognition.base_url,
        },
        "openai_reflection": {
            "model": app_config.cognition.reflection_model,
            "api_key_loaded": bool(
                app_config.cognition.reflection_api_key
                if app_config.cognition.reflection_api_key is not None
                else app_config.cognition.api_key
            ),
            "base_url": app_config.cognition.reflection_base_url,
        },
        "livekit": {
            "url_loaded": bool(app_config.livekit.url),
            "token_loaded": bool(app_config.livekit.token),
            "api_key_loaded": bool(app_config.livekit.api_key),
            "room": app_config.livekit.room,
            "identity": app_config.livekit.identity,
            "track_name": app_config.livekit.track_name,
        },
        "speech": {
            "voice_name": app_config.speech.voice_name,
            "tts_model": app_config.speech.tts_model,
            "stt_model": app_config.speech.stt_model,
            "vad_model_id": app_config.speech.vad_model_id,
            "api_key_loaded": bool(app_config.speech.api_key),
            "stt_provider": "gemini",
            "vad_provider": "silero",
            "tts_provider": "gemini",
            "input_sample_rate_hz": app_config.speech.input_sample_rate_hz,
            "input_channels": app_config.speech.input_channels,
            "output_sample_rate_hz": app_config.speech.output_sample_rate_hz,
            "output_channels": app_config.speech.output_channels,
        },
    }


async def run(
    config: AppConfig | None = None,
    *,
    connect_livekit: bool = False,
    hold: bool = False,
    on_ready: collections.abc.Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    app_config = config or AppConfig.from_env_file()
    store = _memory()
    cognition_ability = _phone_cognition(app_config.cognition, memory=store)
    conversation = _conversation(app_config.speech)
    livekit_url = app_config.livekit.url if connect_livekit and app_config.livekit.can_connect_room() else None
    livekit_token = app_config.livekit.token if connect_livekit and app_config.livekit.can_connect_room() else None
    phone_service = PhoneService(
        url=livekit_url,
        token=livekit_token,
        track_name=app_config.livekit.track_name,
        # One value decides the robot's voice rate: the TTS encoder, Speaking's label, and the
        # LiveKit source all take it from here. They used to be three defaults that happened to
        # agree in two places and not in the third, which is a silent mute rather than an error.
        uplink_sample_rate_hz=app_config.speech.output_sample_rate_hz,
    )
    phone = _handset(service=phone_service)
    # Explicit speech wiring: Silero VAD (local, cheap) + Gemini STT/TTS (off-device).
    listening_ability = _listening(app_config.speech)
    voice = _voice()
    speaking_ability = _speaking(speaker=voice, speech_config=app_config.speech)
    _LOG.info(
        "speech path stt_provider=gemini stt_model=%s tts_provider=gemini tts_model=%s "
        "tts_voice=%s vad_provider=silero vad_model_id=%s api_key_loaded=%s",
        app_config.speech.stt_model,
        app_config.speech.tts_model,
        app_config.speech.voice_name,
        app_config.speech.vad_model_id,
        bool(app_config.speech.api_key),
    )
    body = await start_bot(
        "alice",
        config=app_config,
        connect_livekit=False,
        phone=phone,
        voice=voice,
        phone_service=phone_service,
        cognition=cognition_ability,
        listening=listening_ability,
        speaking=speaking_ability,
        conversation=conversation,
        memory=store,
    )
    if connect_livekit and app_config.livekit.can_connect_room():
        await _wait_for_livekit_room_audio_result(phone_service)
    summary = _summary_for(
        app_config=app_config,
        body=body,
        phone_service=phone_service,
        connect_livekit=connect_livekit,
    )
    if on_ready is not None:
        on_ready(summary)
    if hold:
        # Stay on the line until cancelled so a human can join the room.
        # Call setup addressed to this identity rings the phone; the bot decides whether to answer.
        # Periodically surface LiveKit media counters so deafness is obvious.
        while True:
            await asyncio.sleep(5.0)
            snap = phone_service.media_snapshot()
            _LOG.info(
                "livekit media local_track_sid=%s delivered_chunks=%s dropped_chunks=%s bot_state=%s phone_state=%s",
                snap.local_track_sid,
                snap.remote_audio_chunks,
                snap.remote_audio_dropped_chunks,
                body.state(),
                phone.state(),
            )
    return summary


def _render_text(summary: dict[str, object]) -> str:
    warnings = "\n".join(f"- {item}" for item in typing.cast(list[str], summary["warnings"]))
    return (
        "stateforward.bot phone voice-bot example\n"
        f"Status: {summary['status']}\n"
        f"Bot state: {summary['bot_state']}\n"
        f"Cognition: {summary['cognition_client']}\n"
        f"LiveKit room audio connected: {summary['livekit_room_audio_connected']}\n"
        f"Warnings:\n{warnings or '- none'}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--json", action="store_true", help="Print the compact machine-readable summary.")
    _ = parser.add_argument("--env", default=None, help="Provider env file.")
    _ = parser.add_argument(
        "--reasoning-model",
        default=None,
        help="Override the OpenAI Terra reasoning model (default gpt-5.6-terra).",
    )
    _ = parser.add_argument("--connect-livekit", action="store_true", help="Attempt the configured LiveKit room join.")
    _ = parser.add_argument(
        "--tell",
        action="append",
        default=[],
        metavar="TEXT",
        help="Tell the bot something before it starts, in plain words. Repeatable.",
    )
    args = parser.parse_args()
    env_arg = typing.cast(str | None, getattr(args, "env", None))
    env_path = pathlib.Path(env_arg).expanduser() if env_arg is not None else None
    config = AppConfig.from_env_file(env_path)
    config = config.with_cognition_overrides(
        model=typing.cast(str | None, getattr(args, "reasoning_model", None)),
    )
    config = dataclasses.replace(
        config,
        told=(*config.told, *typing.cast(list[str], getattr(args, "tell", []))),
    )
    summary = asyncio.run(
        run(
            config,
            connect_livekit=bool(getattr(args, "connect_livekit", False)),
        )
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(summary, indent=2))
    else:
        print(_render_text(summary))


__all__ = [
    "AppConfig",
    "CognitionConfig",
    "ExampleTextConversation",
    "TranscriptTextDecoder",
    "LiveKitConfig",
    "SpeechConfig",
    "PhoneBot",
    "load_env",
    "main",
    "mint_livekit_access_token",
    "run",
    "start_bot",
]
