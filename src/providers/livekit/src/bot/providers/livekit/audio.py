from __future__ import annotations

import bot.abilities
from bot.abilities.communication import conversation
from bot.abilities.hearing import speech
from bot.devices import audio as audio_device

import asyncio
import collections.abc
import dataclasses
import io
import typing
import wave

from livekit import rtc

_PCM_MEDIA_TYPE = "audio/pcm"
_SAMPLE_WIDTH_BYTES = 2

TFrame = typing.TypeVar("TFrame")
TFrame_co = typing.TypeVar("TFrame_co", covariant=True)
TFrame_contra = typing.TypeVar("TFrame_contra", contravariant=True)


class AudioFrameError(RuntimeError):
    """Raised when stateforward.bot audio cannot be adapted to or from a LiveKit audio frame."""


class AudioFrame(typing.Protocol):
    """LiveKit audio frame shape consumed by stateforward.bot adapters."""

    @property
    def data(self) -> bytes | bytearray | memoryview[int]:
        """PCM audio bytes or a bytes-compatible view owned by the frame."""
        ...

    @property
    def sample_rate(self) -> int:
        """FrameData sample rate in hertz."""
        ...

    @property
    def num_channels(self) -> int:
        """Number of interleaved PCM channels in the frame."""
        ...

    @property
    def samples_per_channel(self) -> int:
        """Number of PCM samples per channel in the frame."""
        ...


class AudioFrameFactory(typing.Protocol[TFrame_co]):
    """Factory compatible with `livekit.rtc.AudioFrame` construction."""

    def __call__(
        self,
        *,
        data: bytes,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
    ) -> TFrame_co:
        """Build a LiveKit-compatible audio frame from signed 16-bit PCM bytes."""
        ...


class AudioSource(typing.Protocol[TFrame_contra]):
    """LiveKit audio source shape used to publish frames."""

    def capture_frame(self, frame: TFrame_contra) -> collections.abc.Awaitable[None]:
        """Queue a LiveKit-compatible audio frame for playout."""
        ...


def _livekit_audio_frame(
    *,
    data: bytes,
    sample_rate: int,
    num_channels: int,
    samples_per_channel: int,
) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


def _positive_int(value: int | None, fallback: int, *, label: str) -> int:
    resolved = fallback if value is None else value
    if isinstance(resolved, bool) or resolved < 1:
        message = f"LiveKit audio {label} must be a positive integer."
        raise AudioFrameError(message)
    return resolved


def _is_pcm_media_type(media_type: str | None) -> bool:
    if media_type is None:
        return True
    return media_type.split(";", 1)[0].strip().casefold() in {_PCM_MEDIA_TYPE, "audio/l16", "audio/raw"}


def _samples_per_channel(audio: bytes, *, channels: int) -> int:
    frame_width = channels * _SAMPLE_WIDTH_BYTES
    if len(audio) % frame_width != 0:
        message = "LiveKit audio frames require signed 16-bit PCM bytes aligned to the channel count."
        raise AudioFrameError(message)
    return len(audio) // frame_width


def _frame_audio_bytes(frame: AudioFrame) -> bytes:
    data = frame.data
    if isinstance(data, bytes):
        audio = data
    elif isinstance(data, bytearray):
        audio = bytes(data)
    else:
        audio = data.tobytes()
    if not audio:
        message = "LiveKit audio frame data must not be empty."
        raise AudioFrameError(message)
    return audio


def pcm_to_wav_bytes(
    audio: bytes,
    *,
    sample_rate_hz: int,
    channels: int,
    sample_width_bytes: int = _SAMPLE_WIDTH_BYTES,
) -> bytes:
    """Wrap signed PCM audio bytes in a WAV container."""

    if not audio:
        message = "LiveKit PCM WAV decoding requires non-empty audio bytes."
        raise AudioFrameError(message)
    if sample_width_bytes < 1:
        message = "LiveKit PCM sample width must be positive."
        raise AudioFrameError(message)

    _ = _positive_int(sample_rate_hz, sample_rate_hz, label="sample rate")
    _ = _positive_int(channels, channels, label="channel count")
    if len(audio) % (channels * sample_width_bytes) != 0:
        message = "LiveKit PCM WAV decoding requires audio bytes aligned to the sample width and channel count."
        raise AudioFrameError(message)

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width_bytes)
            wav.setframerate(sample_rate_hz)
            wav.writeframes(audio)
        return buffer.getvalue()


@dataclasses.dataclass(frozen=True, kw_only=True)
class AudioFrameEncoder(bot.abilities.Encoder[audio_device.AudioOutputData, TFrame], typing.Generic[TFrame]):
    """Encoder that converts stateforward.bot audio output chunks into LiveKit PCM audio frames."""

    default_sample_rate_hz: int = 48_000
    default_channels: int = 1
    audio_frame_factory: AudioFrameFactory[TFrame] = typing.cast(
        AudioFrameFactory[TFrame],
        _livekit_audio_frame,
    )

    @typing.override
    async def encode(self, input: audio_device.AudioOutputData) -> TFrame:
        if not _is_pcm_media_type(input.media_type):
            message = "LiveKit audio frames only support raw PCM input from stateforward.bot audio output data."
            raise AudioFrameError(message)

        sample_rate_hz = _positive_int(input.sample_rate_hz, self.default_sample_rate_hz, label="sample rate")
        channels = _positive_int(input.channels, self.default_channels, label="channel count")
        samples_per_channel = _samples_per_channel(input.audio, channels=channels)

        return self.audio_frame_factory(
            data=input.audio,
            sample_rate=sample_rate_hz,
            num_channels=channels,
            samples_per_channel=samples_per_channel,
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class AudioFrameDecoder(bot.abilities.Decoder[AudioFrame, audio_device.AudioInputData]):
    """Decoder that converts LiveKit PCM audio frames into stateforward.bot audio input chunks."""

    media_type: str = _PCM_MEDIA_TYPE

    @typing.override
    async def decode(self, input: AudioFrame) -> audio_device.AudioInputData:
        channels = _positive_int(input.num_channels, input.num_channels, label="channel count")
        sample_rate_hz = _positive_int(input.sample_rate, input.sample_rate, label="sample rate")
        audio = _frame_audio_bytes(input)
        expected_bytes = input.samples_per_channel * channels * _SAMPLE_WIDTH_BYTES
        if expected_bytes != len(audio):
            message = "LiveKit audio frame data length does not match its sample metadata."
            raise AudioFrameError(message)
        return audio_device.AudioInputData(
            audio=audio,
            media_type=self.media_type,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class AudioSourceWriter(typing.Generic[TFrame]):
    """Writer that publishes stateforward.bot audio output chunks to an injected LiveKit audio source."""

    source: AudioSource[TFrame]
    encoder: bot.abilities.Encoder[audio_device.AudioOutputData, TFrame]

    async def write(self, output: audio_device.AudioOutputData) -> None:
        """EncodeData and capture one stateforward.bot audio output chunk."""

        frame = await self.encoder.encode(output)
        await self.source.capture_frame(frame)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AudioBridge(typing.Generic[TFrame]):
    """Bidirectional LiveKit audio bridge for stateforward.bot audio device data.

    The bridge owns audio frame adaptation only. LiveKit room lifecycle,
    participant selection, SIP signaling, VAD, STT, and TTS remain injected
    layers around this provider-owned media boundary.
    """

    source_writer: AudioSourceWriter[TFrame]
    frame_decoder: bot.abilities.Decoder[AudioFrame, audio_device.AudioInputData] = dataclasses.field(
        default_factory=AudioFrameDecoder,
    )

    @classmethod
    def from_audio_source(
        cls: type[AudioBridge[TFrame]],
        *,
        source: AudioSource[TFrame],
        encoder: bot.abilities.Encoder[audio_device.AudioOutputData, TFrame],
        frame_decoder: bot.abilities.Decoder[AudioFrame, audio_device.AudioInputData] | None = None,
    ) -> AudioBridge[TFrame]:
        """Build a bridge from an injected LiveKit audio source and stateforward.bot encoder."""

        return cls(
            source_writer=AudioSourceWriter(source=source, encoder=encoder),
            frame_decoder=frame_decoder or AudioFrameDecoder(),
        )

    async def receive_frame(self, frame: AudioFrame) -> audio_device.AudioInputData:
        """Convert one remote LiveKit audio frame into stateforward.bot audio input data."""

        return await self.frame_decoder.decode(frame)

    async def publish_audio(self, output: audio_device.AudioOutputData) -> None:
        """Publish one stateforward.bot audio output chunk into the injected LiveKit source."""

        await self.source_writer.write(output)


def create_audio_bridge(
    *,
    sample_rate_hz: int = 48_000,
    channels: int = 1,
    queue_size_ms: int = 1_000,
    loop: asyncio.AbstractEventLoop | None = None,
) -> AudioBridge[rtc.AudioFrame]:
    """Create a stateforward.bot audio bridge backed by a LiveKit SDK audio source."""

    sample_rate_hz = _positive_int(sample_rate_hz, sample_rate_hz, label="sample rate")
    channels = _positive_int(channels, channels, label="channel count")
    queue_size_ms = _positive_int(queue_size_ms, queue_size_ms, label="queue size")
    try:
        loop = loop or asyncio.get_running_loop()
    except RuntimeError as error:
        message = "LiveKit audio bridge creation requires a running asyncio event loop or an explicit loop."
        raise AudioFrameError(message) from error
    source = typing.cast(
        AudioSource[rtc.AudioFrame],
        rtc.AudioSource(
            sample_rate=sample_rate_hz,
            num_channels=channels,
            queue_size_ms=queue_size_ms,
            loop=loop,
        ),
    )
    encoder: bot.abilities.Encoder[audio_device.AudioOutputData, rtc.AudioFrame] = AudioFrameEncoder(
        default_sample_rate_hz=sample_rate_hz,
        default_channels=channels,
    )
    source_writer: AudioSourceWriter[rtc.AudioFrame] = AudioSourceWriter(
        source=source,
        encoder=encoder,
    )
    return AudioBridge[rtc.AudioFrame](source_writer=source_writer)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PcmWavDecoder(speech.SpeechDecoder):
    """Speech decoder that wraps LiveKit-compatible signed PCM bytes as WAV bytes."""

    sample_rate_hz: int = 48_000
    channels: int = 1
    sample_width_bytes: int = _SAMPLE_WIDTH_BYTES

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        return pcm_to_wav_bytes(
            input,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            sample_width_bytes=self.sample_width_bytes,
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceDecoder(conversation.voice.VoiceDecoder):
    """Voice conversation decoder for LiveKit PCM audio with an injected speech decoder."""

    speech_decoder: speech.SpeechDecoder
    pcm_decoder: PcmWavDecoder = dataclasses.field(default_factory=PcmWavDecoder)

    @typing.override
    async def decode(self, input: conversation.turn_detector.AudioStimulus) -> str:
        rate = input.sample_rate_hz if input.sample_rate_hz is not None else self.pcm_decoder.sample_rate_hz
        channels = input.channels if input.channels is not None else self.pcm_decoder.channels
        pcm_decoder = (
            self.pcm_decoder
            if rate == self.pcm_decoder.sample_rate_hz and channels == self.pcm_decoder.channels
            else PcmWavDecoder(sample_rate_hz=rate, channels=channels)
        )
        wav = await pcm_decoder.decode(input.content)
        transcript = await self.speech_decoder.decode(wav)
        return transcript.decode("utf-8")


__all__ = [
    "AudioBridge",
    "AudioFrame",
    "AudioFrameDecoder",
    "AudioFrameEncoder",
    "AudioFrameError",
    "AudioSource",
    "AudioSourceWriter",
    "PcmWavDecoder",
    "VoiceDecoder",
    "create_audio_bridge",
    "pcm_to_wav_bytes",
]
