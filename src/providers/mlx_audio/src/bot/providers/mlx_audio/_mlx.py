from __future__ import annotations

import collections.abc
import io
import importlib
import typing


class SpeechGenerationResult(typing.Protocol):
    audio: object
    sample_rate: int


class SpeechEncodingModel(typing.Protocol):
    def generate(self, text: str, **kwargs: object) -> collections.abc.Iterable[SpeechGenerationResult]:
        """Return generated speech audio records for the supplied text."""
        ...


class SpeechDecodingModel(typing.Protocol):
    def generate(self, audio: str, **kwargs: object) -> object:
        """Return generated speech transcription data for the supplied audio file."""
        ...


VAD_TARGET_SAMPLE_RATE = 16_000
"""Sample rate MLX Audio's streaming VAD consumes (``realtime_vad.VAD_SAMPLE_RATE``)."""


class VoiceActivityEvent(typing.NamedTuple):
    """One streaming VAD boundary, timed from the start of the session."""

    started: bool
    audio_ms: int


class StreamingVoiceDetectionSession(typing.Protocol):
    """Stateful streaming VAD session that spans many audio chunks.

    The session is what makes chunk boundaries irrelevant: speech that begins in one chunk and
    continues into the next stays open across the seam, so a caller never has to guess where an
    utterance starts in order to be able to hear it.
    """

    def process_pcm(
        self,
        pcm: bytes,
        *,
        sample_rate_hz: int,
        channels: int,
    ) -> tuple[VoiceActivityEvent, ...]:
        """Feed one chunk of signed 16-bit PCM and return the boundaries it crossed."""
        ...

    def in_speech(self) -> bool:
        """True when the session is inside a talkspurt at the end of the last fed chunk."""
        ...


class VoiceDiarizationModel(typing.Protocol):
    def generate(self, audio: str, *, threshold: float, verbose: bool) -> object:
        """Return speaker diarization output for the supplied audio file."""
        ...


SpeechEncodingModelLoader = collections.abc.Callable[[str], SpeechEncodingModel]
SpeechDecodingModelLoader = collections.abc.Callable[[str], SpeechDecodingModel]
SpeechAudioWriter = collections.abc.Callable[[object, int, str], bytes]
StreamingVoiceDetectionSessionLoader = collections.abc.Callable[[str], StreamingVoiceDetectionSession]
VoiceDiarizationModelLoader = collections.abc.Callable[[str], VoiceDiarizationModel]


def load_speech_encoding_model(model_id: str) -> SpeechEncodingModel:
    module = importlib.import_module("mlx_audio.tts.utils")
    load = typing.cast(collections.abc.Callable[[str], SpeechEncodingModel], getattr(module, "load_model"))
    return load(model_id)


def write_speech_audio(audio: object, sample_rate: int, audio_format: str) -> bytes:
    module = importlib.import_module("mlx_audio.audio_io")
    write = typing.cast(collections.abc.Callable[..., None], getattr(module, "write"))
    output = io.BytesIO()
    write(output, audio, sample_rate, format=audio_format)
    return output.getvalue()


def load_speech_decoding_model(model_id: str) -> SpeechDecodingModel:
    module = importlib.import_module("mlx_audio.stt.utils")
    load = typing.cast(collections.abc.Callable[[str], SpeechDecodingModel], getattr(module, "load_model"))
    return load(model_id)


def _resample_to_mono_float32(
    numpy_module: typing.Any,
    pcm: bytes,
    *,
    sample_rate_hz: int,
    channels: int,
) -> typing.Any:
    """Adapt signed 16-bit PCM to the mono 16 kHz float32 the streaming VAD consumes.

    This is the SDK's frame format, not a stateforward.bot audio contract: the model reads
    fixed 512-sample windows at 16 kHz and nothing else. The offline file API used to do this
    conversion invisibly inside its audio loader; driving the model directly makes it ours.

    Resampling is linear interpolation, which is adequate for a voice/no-voice decision and
    keeps the provider free of a signal-processing dependency. Chunks are resampled
    independently, so a chunk seam can land up to one output sample away from where a
    continuous resampler would put it; that is far below the model's 32 ms frame.
    """

    samples = numpy_module.frombuffer(pcm, dtype="<i2").astype(numpy_module.float32) / 32768.0
    if channels > 1:
        usable = samples.size - (samples.size % channels)
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    if sample_rate_hz == VAD_TARGET_SAMPLE_RATE or samples.size == 0:
        return samples
    target_size = int(round(samples.size * VAD_TARGET_SAMPLE_RATE / sample_rate_hz))
    if target_size <= 0:
        return samples[:0]
    source_positions = numpy_module.linspace(0.0, samples.size - 1, target_size)
    resampled = numpy_module.interp(source_positions, numpy_module.arange(samples.size), samples)
    return resampled.astype(numpy_module.float32)


class _StreamingVoiceDetectionSession:
    """Adapter from ``mlx_audio.realtime_vad.StreamingVad`` to PCM chunks."""

    def __init__(self, model_id: str) -> None:
        vad_module = importlib.import_module("mlx_audio.vad")
        realtime_module = importlib.import_module("mlx_audio.realtime_vad")
        self._numpy = importlib.import_module("numpy")
        load = typing.cast(collections.abc.Callable[[str], object], getattr(vad_module, "load"))
        streaming_vad = typing.cast(collections.abc.Callable[..., object], getattr(realtime_module, "StreamingVad"))
        config = typing.cast(collections.abc.Callable[[], object], getattr(realtime_module, "ServerVadConfig"))
        self._streaming = streaming_vad(load(model_id), config())

    def process_pcm(
        self,
        pcm: bytes,
        *,
        sample_rate_hz: int,
        channels: int,
    ) -> tuple[VoiceActivityEvent, ...]:
        samples = _resample_to_mono_float32(self._numpy, pcm, sample_rate_hz=sample_rate_hz, channels=channels)
        process = typing.cast(collections.abc.Callable[[object], object], getattr(self._streaming, "process"))
        raw = coerce_iterable(process(samples), description="voice activity events")
        events: list[VoiceActivityEvent] = []
        for item in raw:
            kind = get_member(item, "kind")
            audio_ms = get_member(item, "audio_ms")
            value = typing.cast(object, getattr(kind, "value", kind))
            if not isinstance(audio_ms, int) or isinstance(audio_ms, bool):
                message = "MLX Audio streaming VAD event is missing an integer audio_ms."
                raise TypeError(message)
            events.append(VoiceActivityEvent(started=value == "speech_started", audio_ms=audio_ms))
        return tuple(events)

    def in_speech(self) -> bool:
        return bool(typing.cast(object, getattr(self._streaming, "in_speech")))


def load_streaming_voice_detection_session(model_id: str) -> StreamingVoiceDetectionSession:
    return _StreamingVoiceDetectionSession(model_id)


def load_voice_diarization_model(model_id: str) -> VoiceDiarizationModel:
    module = importlib.import_module("mlx_audio.vad")
    load = typing.cast(collections.abc.Callable[[str], VoiceDiarizationModel], getattr(module, "load"))
    return load(model_id)


def get_member(value: object, *names: str) -> object:
    if isinstance(value, collections.abc.Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None

    for name in names:
        member = typing.cast(object, getattr(value, name, None))
        if member is not None:
            return member
    return None


def coerce_iterable(value: object, *, description: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes | bytearray):
        message = f"MLX Audio returned {description} as scalar data."
        raise TypeError(message)
    if not isinstance(value, collections.abc.Iterable):
        message = f"MLX Audio returned {description} that is not iterable."
        raise TypeError(message)
    return tuple(value)


def optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def required_float(value: object, *, field_name: str) -> float:
    parsed = optional_float(value)
    if parsed is None:
        message = f"MLX Audio segment is missing numeric {field_name}."
        raise TypeError(message)
    return parsed
