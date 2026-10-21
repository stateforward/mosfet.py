"""phone_bot: a bot holding a telephone on a LiveKit line, in a room where somebody can talk to it.

    python -m phone_bot_example --ready-file /tmp/ready

``__main__`` is the bot file. It reads the owner's :class:`Settings` from the environment, then defines one bot,
``alice``, with one ``bot.define`` that constructs every part inline, and runs it with ``bot.run`` until it is told
to stop. Its phone is the core ``phone.Phone`` held to its ear, on a LiveKit ``PhoneService`` line whose
participant identity is the line's number. It hears through ``listening`` (MLX Audio's Silero VAD and PyAnnote
voice embeddings, local), its conversations are decoded by Gemini (:class:`GeminiVoiceDecoder`), and it speaks
through ``speaking`` with Gemini's voice, out of a mouth at the phone's mouthpiece. It thinks with one model,
OpenAI's reasoning model.

Somebody stands a metre in front of it: a :class:`person.Person`, a shared actor of the run, who says out loud
each line typed at the terminal. Nothing here reads those words, and nothing tells the bot what to do with a
ring, a voice or a number: answering, declining, dialing and hanging up are its own choices. :func:`log` is the
run's observer: it logs what the bot does, and never acts on it.
"""

import base64
import collections.abc
import datetime
import hashlib
import hmac
import json
import logging
import time
import typing

import hsm
import pydantic
from mosfet.abilities import decoding, identification
from mosfet.abilities.communication.conversation import turn_detector
from mosfet.abilities.hearing import speech
from mosfet.devices import phone
from mosfet.environment import space
from mosfet.providers.livekit.audio import PcmWavDecoder
from mosfet.telemetry import observed_event, observed_occurrence

from . import person

_LOG = logging.getLogger("phone_bot_example.hsm")

NAME: typing.Final = "alice"
"""The bot's name: its topology and its telemetry are named for it."""
DEFAULT_OPENAI_BASE_URL: typing.Final = "https://api.openai.com/v1"
DEFAULT_OPENAI_REASONING_MODEL: typing.Final = "gpt-5.6-terra"
DEFAULT_GEMINI_STT_MODEL: typing.Final = "gemini-3.5-flash"
DEFAULT_GEMINI_TTS_MODEL: typing.Final = "gemini-3.1-flash-tts-preview"
DEFAULT_GEMINI_TTS_VOICE: typing.Final = "Kore"
DEFAULT_SILERO_VAD_MODEL: typing.Final = "mlx-community/silero-vad"
DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL: typing.Final = "pyannote/wespeaker-voxceleb-resnet34-LM"
DEFAULT_LIVEKIT_ROOM: typing.Final = "bot-phone-bot"
DEFAULT_LIVEKIT_IDENTITY: typing.Final = "5550141"
DEFAULT_LIVEKIT_TRACK_NAME: typing.Final = "5550141"
DEFAULT_LIVEKIT_INPUT_SAMPLE_RATE_HZ: typing.Final = 48_000
DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE_HZ: typing.Final = 24_000
"""Gemini TTS's playout rate: the speaker, Speaking's label and the LiveKit uplink all take it."""
DEFAULT_AUDIO_CHANNELS: typing.Final = 1
END_OF_TURN_SILENCE_SECONDS: typing.Final = 0.5
"""How much silence ends a conversation turn."""

VOICE_POLICY: typing.Final = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65)
"""When two voiceprints are one voice: WeSpeaker's operating point."""
MAX_VOICE_IDS: typing.Final = 1000
"""Most voice ids this example remembers."""
VOICE_RETENTION: typing.Final = datetime.timedelta(days=30)
"""How long a voice never linked to a person is remembered after it was last heard."""

# Where this robot is and how loud its parts are. dB SPL at one metre.
#
# The handset earpiece sits at the robot's head, so it is loud in the robot's own ears and far too quiet by the
# time it reaches the mouthpiece 15 cm away (28.5 dB below the mouthpiece threshold) — which is what stops the far
# end hearing itself. The bot's voice comes out at the mouth, right at the mouthpiece, so it goes up the wire.
BOT_ORIGIN: typing.Final = space.Position(x=0.0, y=0.0)
"""Where the robot holds its handset: the earpiece is here."""
MOUTH: typing.Final = space.Position(x=phone.MOUTH_OFFSET_M, y=0.0)
"""The robot's mouth, at the handset's mouthpiece."""
EARPIECE_DB: typing.Final = 25.0
"""Handset receiver at one metre: about 65 dB at the ear, where the ear actually is."""
VOICE_DB: typing.Final = 60.0
"""Normal conversational speech at one metre."""
MOUTHPIECE_THRESHOLD_DB: typing.Final = 70.0
"""Close-talk handset mouthpiece: it hears the mouth it is held next to, and little else."""
ROOM_POSITION: typing.Final = space.Position(x=0.0, y=1.0)
"""Where somebody stands to talk to the robot: a metre away, facing it.

Close enough that a conversational 60 dB reaches the robot, and far enough that it arrives 10 dB *under* the
close-talk mouthpiece's threshold — so talking to the robot in the room does not put your voice down its line.
"""


def mint_livekit_access_token(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room: str,
    name: str | None = None,
    ttl_seconds: int = 6 * 60 * 60,
) -> str:
    """Mint a LiveKit join JWT (HS256) for ``identity`` in ``room``, with publish and subscribe grants (reads the clock)."""

    def b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload: dict[str, object] = {
        "iss": api_key,
        "sub": identity,
        "nbf": now - 10,
        "exp": now + ttl_seconds,
        "video": {"room": room, "roomJoin": True, "canPublish": True, "canSubscribe": True, "canPublishData": True},
    }
    if name is not None:
        payload["name"] = name
    segments = (
        f"{b64url(json.dumps(header, separators=(',', ':')).encode())}."
        f"{b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    )
    signature = hmac.new(api_secret.encode("utf-8"), segments.encode("ascii"), hashlib.sha256).digest()
    return f"{segments}.{b64url(signature)}"


class Settings(pydantic.BaseModel):
    """The owner's settings for the phone bot, read from the process environment by ``__main__``.

    Each field is read from the first of its variables that is set; a variable set to the empty string is unset.
    The OpenAI and Gemini keys are required, and so is the decision server the bot's reflex decides on, whose
    key is optional (a local decide takes none). The LiveKit line is optional: with no URL the phone has no line, and
    the bot can still be talked to in its room. With a URL, the token is ``BOT_LIVEKIT_TOKEN``, or one minted
    from the API key and secret for the line's identity and room. A value is never shown in an error: the error
    names the variable.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="ignore", hide_input_in_errors=True
    )

    openai_api_key: pydantic.SecretStr = pydantic.Field(
        validation_alias=pydantic.AliasChoices("BOT_OPENAI_API_KEY", "OPENAI_API_KEY"),
        description="The OpenAI API key the bot thinks with.",
    )
    openai_base_url: str = pydantic.Field(
        default=DEFAULT_OPENAI_BASE_URL,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
        description="The OpenAI-compatible endpoint the bot thinks with.",
        examples=[DEFAULT_OPENAI_BASE_URL],
    )
    reasoning_model: str = pydantic.Field(
        default=DEFAULT_OPENAI_REASONING_MODEL,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_REASONING_MODEL", "BOT_OPENAI_REASONING_MODEL"),
        description="The model the bot thinks with.",
        examples=[DEFAULT_OPENAI_REASONING_MODEL],
    )
    decision_base_url: str = pydantic.Field(
        min_length=1,
        validation_alias=pydantic.AliasChoices("DECISION_BASE_URL"),
        description=(
            "The decision server the bot's reflex decides on, which speaks the decisions wire format: anygame's"
            " decide, or a hosted server with the same format."
        ),
        examples=["http://127.0.0.1:8102"],
    )
    decision_api_key: pydantic.SecretStr | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("DECISION_API_KEY"),
        description="The decision server's key, sent as a bearer only when given; a local decide takes none.",
    )
    decision_max_context_tokens: int | None = pydantic.Field(
        default=None,
        gt=0,
        validation_alias=pydantic.AliasChoices("DECISION_MAX_CONTEXT_TOKENS"),
        description=(
            "The decision model's context in tokens, which each decision request is fitted to; unset, requests are"
            " sent whole."
        ),
        examples=[2048],
    )
    gemini_api_key: pydantic.SecretStr = pydantic.Field(
        validation_alias=pydantic.AliasChoices("BOT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
        description="The Gemini API key its conversations are decoded and its replies spoken with.",
    )
    stt_model: str = pydantic.Field(
        default=DEFAULT_GEMINI_STT_MODEL,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_GEMINI_STT_MODEL", "GEMINI_STT_MODEL", "BOT_STT_MODEL"),
        description="The Gemini model that decodes conversation turns.",
        examples=[DEFAULT_GEMINI_STT_MODEL],
    )
    tts_model: str = pydantic.Field(
        default=DEFAULT_GEMINI_TTS_MODEL,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_GEMINI_TTS_MODEL", "GEMINI_TTS_MODEL", "BOT_TTS_MODEL"),
        description="The Gemini model the bot speaks with.",
        examples=[DEFAULT_GEMINI_TTS_MODEL],
    )
    tts_voice: str = pydantic.Field(
        default=DEFAULT_GEMINI_TTS_VOICE,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_GEMINI_TTS_VOICE", "GEMINI_TTS_VOICE", "BOT_TTS_VOICE"),
        description="The Gemini voice the bot speaks in.",
        examples=[DEFAULT_GEMINI_TTS_VOICE],
    )
    vad_model: str = pydantic.Field(
        default=DEFAULT_SILERO_VAD_MODEL,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_SILERO_VAD_MODEL", "BOT_VAD_MODEL", "SILERO_VAD_MODEL"),
        description="The MLX Audio Silero model that hears where speech is.",
        examples=[DEFAULT_SILERO_VAD_MODEL],
    )
    voice_identity_model: str = pydantic.Field(
        default=DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_PYANNOTE_VOICE_IDENTITY_MODEL", "PYANNOTE_VOICE_IDENTITY_MODEL"),
        description="The PyAnnote speaker-embedding model that knows voices apart.",
        examples=[DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL],
    )
    input_sample_rate_hz: int = pydantic.Field(
        default=DEFAULT_LIVEKIT_INPUT_SAMPLE_RATE_HZ,
        gt=0,
        validation_alias=pydantic.AliasChoices("BOT_LIVEKIT_INPUT_SAMPLE_RATE_HZ", "LIVEKIT_INPUT_SAMPLE_RATE_HZ"),
        description="The rate of the raw PCM the line delivers.",
        examples=[DEFAULT_LIVEKIT_INPUT_SAMPLE_RATE_HZ],
    )
    input_channels: int = pydantic.Field(
        default=DEFAULT_AUDIO_CHANNELS,
        gt=0,
        validation_alias=pydantic.AliasChoices("BOT_LIVEKIT_INPUT_CHANNELS", "LIVEKIT_INPUT_CHANNELS"),
        description="The channels of the raw PCM the line delivers.",
        examples=[DEFAULT_AUDIO_CHANNELS],
    )
    output_sample_rate_hz: int = pydantic.Field(
        default=DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE_HZ,
        gt=0,
        validation_alias=pydantic.AliasChoices("BOT_GEMINI_OUTPUT_SAMPLE_RATE_HZ", "GEMINI_OUTPUT_SAMPLE_RATE_HZ"),
        description="The rate the bot's voice is rendered and sent up the line at.",
        examples=[DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE_HZ],
    )
    livekit_url: str | None = pydantic.Field(
        default=None,
        pattern=r"^wss?://",
        validation_alias=pydantic.AliasChoices("BOT_LIVEKIT_URL", "LIVEKIT_URL", "VA_LIVEKIT_URL"),
        description="The LiveKit server's WebSocket URL; the phone has no line without it.",
        examples=["ws://127.0.0.1:7880"],
    )
    livekit_token: pydantic.SecretStr | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("BOT_LIVEKIT_TOKEN", "LIVEKIT_TOKEN", "VA_LIVEKIT_TOKEN"),
        description="A LiveKit join token for the line; minted from the API key and secret when unset.",
    )
    livekit_api_key: str | None = pydantic.Field(
        default=None,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_LIVEKIT_API_KEY", "LIVEKIT_API_KEY", "VA_LIVEKIT_API_KEY"),
        description="The LiveKit API key a join token is minted with.",
        examples=["devkey"],
    )
    livekit_api_secret: pydantic.SecretStr | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("BOT_LIVEKIT_API_SECRET", "LIVEKIT_API_SECRET", "VA_LIVEKIT_API_SECRET"),
        description="The LiveKit API secret a join token is minted with.",
    )
    room: str = pydantic.Field(
        default=DEFAULT_LIVEKIT_ROOM,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_LIVEKIT_ROOM", "LIVEKIT_ROOM", "VA_LIVEKIT_ROOM"),
        description="The LiveKit room the line is in.",
        examples=[DEFAULT_LIVEKIT_ROOM],
    )
    identity: str = pydantic.Field(
        default=DEFAULT_LIVEKIT_IDENTITY,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_LIVEKIT_IDENTITY", "LIVEKIT_IDENTITY", "VA_LIVEKIT_IDENTITY"),
        description="The line's number, in the digits dialing uses (555-0141 is 5550141): its participant identity.",
        examples=[DEFAULT_LIVEKIT_IDENTITY],
    )
    track_name: str = pydantic.Field(
        default=DEFAULT_LIVEKIT_TRACK_NAME,
        min_length=1,
        validation_alias=pydantic.AliasChoices("BOT_LIVEKIT_TRACK_NAME", "LIVEKIT_TRACK_NAME", "VA_LIVEKIT_TRACK_NAME"),
        description="The name of the audio track the line publishes.",
        examples=[DEFAULT_LIVEKIT_TRACK_NAME],
    )
    verbose: bool = pydantic.Field(
        default=False,
        validation_alias=pydantic.AliasChoices("BOT_VERBOSE"),
        description="Whether the run logs at DEBUG (1) rather than INFO (0, the default), on stderr.",
        examples=["0"],
    )
    dial_plan: dict[str, str] = pydantic.Field(
        default_factory=dict[str, str],
        validation_alias=pydantic.AliasChoices("BOT_LIVEKIT_DIAL_PLAN"),
        description=(
            "Optional number=identity aliases, comma separated. Empty is normal: a dialled number is the identity "
            "it rings."
        ),
        examples=["5550142=alias-bob"],
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _unset_empty(cls, data: object) -> object:
        if isinstance(data, collections.abc.Mapping):
            given = typing.cast(collections.abc.Mapping[object, object], data)
            return {name: value for name, value in given.items() if value != ""}
        return data

    @pydantic.field_validator("dial_plan", mode="before")
    @classmethod
    def _read_dial_plan(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        plan: dict[str, str] = {}
        for entry in value.split(","):
            if not entry.strip():
                continue
            number, separator, endpoint = (part.strip() for part in entry.partition("="))
            if not separator or not number or not endpoint:
                raise ValueError("must be comma-separated number=identity entries, such as 5550142=alias-bob")
            plan[number] = endpoint
        return plan

    @pydantic.model_validator(mode="after")
    def _a_line_needs_a_token(self) -> typing.Self:
        if self.livekit_url is None:
            if self.livekit_token is not None:
                raise ValueError("BOT_LIVEKIT_TOKEN needs BOT_LIVEKIT_URL, the line it joins")
            return self
        if self.livekit_token is None and (self.livekit_api_key is None or self.livekit_api_secret is None):
            raise ValueError(
                "BOT_LIVEKIT_URL needs BOT_LIVEKIT_TOKEN, or BOT_LIVEKIT_API_KEY and BOT_LIVEKIT_API_SECRET"
            )
        return self

    def decision_key(self) -> str | None:
        """The key reflex's decision server is asked with: ``DECISION_API_KEY`` when given, else none at all."""

        return None if self.decision_api_key is None else self.decision_api_key.get_secret_value()

    def line_token(self) -> str | None:
        """The token the line joins with: the one given, or one minted now for its identity and room; none off a line."""

        if self.livekit_url is None:
            return None
        if self.livekit_token is not None:
            return self.livekit_token.get_secret_value()
        assert self.livekit_api_key is not None and self.livekit_api_secret is not None
        return mint_livekit_access_token(
            api_key=self.livekit_api_key,
            api_secret=self.livekit_api_secret.get_secret_value(),
            identity=self.identity,
            room=self.room,
        )


class GeminiVoiceDecoder(decoding.Decoder[turn_detector.ParticipationStimulus, str]):
    """A conversation turn's audio to its words: raw PCM wrapped as WAV at the turn's own rate, decoded by Gemini.

    A WAV container (the ring) describes itself and is decoded as it is. Raw PCM is wrapped with the stimulus' own
    rate and channels, because wrapping 16 kHz room speech at the line's 48 kHz makes a wrong WAV and an empty
    transcript; ``pcm_decoder``'s rate and channels are used when the stimulus has none.
    """

    def __init__(self, *, pcm_decoder: PcmWavDecoder, speech_decoder: speech.SpeechDecoder) -> None:
        self.pcm_decoder: PcmWavDecoder = pcm_decoder
        self.speech_decoder: speech.SpeechDecoder = speech_decoder

    @typing.override
    async def decode(self, input: turn_detector.ParticipationStimulus) -> str:
        """The words said in ``input``'s audio.

        Raises:
            TypeError: If ``input`` is not audio.
            ValueError: If Gemini returns no transcript, or one that is not UTF-8.
        """

        if not isinstance(input, turn_detector.AudioStimulus):
            raise TypeError(f"GeminiVoiceDecoder requires AudioStimulus, got {type(input).__name__}.")
        audio = input.content
        if len(audio) >= 12 and audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
            wav = audio
        else:
            rate = input.sample_rate_hz if input.sample_rate_hz is not None else self.pcm_decoder.sample_rate_hz
            channels = input.channels if input.channels is not None else self.pcm_decoder.channels
            wav = await PcmWavDecoder(sample_rate_hz=rate, channels=channels).decode(audio)
        transcript = await self.speech_decoder.decode(wav)
        try:
            text = transcript.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Gemini returned a transcript that is not UTF-8.") from error
        if not text:
            raise ValueError("Gemini returned an empty transcript.")
        return text


def _summary(data: object) -> str:
    """A low-cardinality summary of an event's data for logs: its type and sizes, never audio or text."""

    if data is None:
        return "none"
    if isinstance(data, (bytes, bytearray)):
        return f"bytes[{len(data)}]"
    if isinstance(data, str):
        return f"str[{len(data)}]"
    if isinstance(data, collections.abc.Mapping):
        keys = sorted(str(key) for key in typing.cast(collections.abc.Mapping[object, object], data))
        return f"keys={keys}"
    audio = typing.cast(object, getattr(data, "audio", None))
    if isinstance(audio, (bytes, bytearray)):
        return f"{type(data).__name__}(audio_bytes={len(audio)})"
    return type(data).__name__


def log(ctx: hsm.Context, instance: hsm.Instance, observation: hsm.Event[typing.Any]) -> None:
    """The run's observer (``bot.observe(log)``): log each thing the bot does, by name and kind, never its content."""

    del ctx
    event = observed_event(observation)
    _LOG.info(
        "phone_bot observe state=%s occurrence=%s event=%s kind=%s data=%s",
        instance.state(),
        observed_occurrence(observation),
        event.name,
        event.kind,
        _summary(event.data),
    )


__all__ = [
    "BOT_ORIGIN",
    "DEFAULT_AUDIO_CHANNELS",
    "DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE_HZ",
    "DEFAULT_GEMINI_STT_MODEL",
    "DEFAULT_GEMINI_TTS_MODEL",
    "DEFAULT_GEMINI_TTS_VOICE",
    "DEFAULT_LIVEKIT_IDENTITY",
    "DEFAULT_LIVEKIT_INPUT_SAMPLE_RATE_HZ",
    "DEFAULT_LIVEKIT_ROOM",
    "DEFAULT_LIVEKIT_TRACK_NAME",
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OPENAI_REASONING_MODEL",
    "DEFAULT_PYANNOTE_VOICE_IDENTITY_MODEL",
    "DEFAULT_SILERO_VAD_MODEL",
    "EARPIECE_DB",
    "END_OF_TURN_SILENCE_SECONDS",
    "MAX_VOICE_IDS",
    "MOUTH",
    "MOUTHPIECE_THRESHOLD_DB",
    "NAME",
    "ROOM_POSITION",
    "VOICE_DB",
    "VOICE_POLICY",
    "VOICE_RETENTION",
    "GeminiVoiceDecoder",
    "Settings",
    "log",
    "mint_livekit_access_token",
    "person",
]
