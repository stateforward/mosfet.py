from __future__ import annotations

from bot.abilities.hearing import voice

import asyncio
import collections.abc
import dataclasses
import io
import importlib
import math
import pathlib
import typing
import wave

from bot.telemetry import span

from ._audio import configure_headless_matplotlib, temporary_audio_file

_SCOPE = "bot.providers.pyannote"
_COMPONENT = "pyannote.voice_diarization"


class VoiceDiarizationError(RuntimeError):
    """Raised when the pyannote diarization adapter cannot classify audio."""


class VoiceDiarizationPipeline(typing.Protocol):
    """The callable surface required from an injected pyannote pipeline."""

    def __call__(self, audio: pathlib.Path) -> object:
        """Return a pyannote-like diarization result for an audio path."""
        ...


VoiceDiarizationPipelineLoader = collections.abc.Callable[[str], VoiceDiarizationPipeline]
VoiceDiarizationAudioClipper = collections.abc.Callable[[voice.diarization.InputData, float, float], bytes]


def load_voice_diarization_pipeline(model_id: str) -> VoiceDiarizationPipeline:
    """Load a pyannote pipeline lazily from its model identifier.

    Authentication belongs to the pyannote/Hugging Face runtime setup. Callers
    that need explicit token handling can inject a loader instead.
    """

    configure_headless_matplotlib()
    module = importlib.import_module("pyannote.audio")
    pipeline_type = typing.cast(object, getattr(module, "Pipeline"))
    from_pretrained = typing.cast(collections.abc.Callable[[str], object], getattr(pipeline_type, "from_pretrained"))
    return typing.cast(VoiceDiarizationPipeline, from_pretrained(model_id))


def _wav_container(input: voice.diarization.InputData) -> bytes:
    """Wrap signed 16-bit PCM in the WAV container expected by pyannote."""

    frame_bytes = 2 * input.channels
    if len(input.audio) % frame_bytes:
        raise ValueError("pyannote diarization input is not aligned to complete PCM frames.")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(input.channels)
        wav.setsampwidth(2)
        wav.setframerate(input.sample_rate_hz)
        wav.writeframes(input.audio)
    return output.getvalue()


def _clip_audio(input: voice.diarization.InputData, start_seconds: float, end_seconds: float) -> bytes:
    """Clip raw signed 16-bit PCM to a validated half-open diarization span."""

    frame_bytes = 2 * input.channels
    if len(input.audio) % frame_bytes:
        raise ValueError("pyannote diarization input is not aligned to complete PCM frames.")
    duration_seconds = len(input.audio) / float(frame_bytes * input.sample_rate_hz)
    if (
        not math.isfinite(start_seconds)
        or not math.isfinite(end_seconds)
        or start_seconds < 0.0
        or end_seconds <= start_seconds
        or end_seconds > duration_seconds
    ):
        raise ValueError("pyannote diarization span is outside the source audio duration.")
    start_frame = math.floor(start_seconds * input.sample_rate_hz)
    end_frame = math.ceil(end_seconds * input.sample_rate_hz)
    frames = input.audio[start_frame * frame_bytes : end_frame * frame_bytes]
    if not frames:
        raise ValueError("pyannote diarization segment has no audio frames.")
    return frames


@dataclasses.dataclass(frozen=True, slots=True)
class _Segment:
    start_seconds: float
    end_seconds: float
    confidence: float | None


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceDiarizer(voice.VoiceDiarizer):
    """Diarizer backed by an injected pyannote-compatible pipeline.

    Raw provider speaker labels are validated inside each classification call
    and omitted from the provider-neutral core output.
    """

    model_id: str = "pyannote/speaker-diarization-community-1"
    audio_file_suffix: str = ".wav"
    pipeline: VoiceDiarizationPipeline | None = None
    load_pipeline: VoiceDiarizationPipelineLoader = load_voice_diarization_pipeline
    clip_audio: VoiceDiarizationAudioClipper = _clip_audio

    @typing.override
    async def classify(self, input: voice.diarization.InputData) -> voice.diarization.OutputData:
        """Diarize raw PCM without blocking the event loop.

        Raises:
            VoiceDiarizationError: If pipeline loading, audio staging, result
                conversion, or core-schema validation fails.
        """

        return await asyncio.to_thread(self._classify_blocking, input)

    def _classify_blocking(self, input: voice.diarization.InputData) -> voice.diarization.OutputData:
        # How many turns the pipeline found, never who took them: raw pyannote speaker labels do
        # not leave this call, and the clipped audio never becomes an attribute.
        with span.operation(
            "bot.provider.pyannote.voice_diarization.classify",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="diarize",
        ) as active:
            try:
                pipeline = self.pipeline if self.pipeline is not None else self.load_pipeline(self.model_id)
                with temporary_audio_file(_wav_container(input), suffix=self.audio_file_suffix) as audio_path:
                    result = pipeline(audio_path)
                raw_segments = tuple(_segment_from_pyannote(turn, label) for turn, label in _turns_from_result(result))
                ordered_segments = tuple(sorted(raw_segments, key=lambda segment: segment.start_seconds))
                segments = tuple(
                    voice.VoiceDiarizationSegment(
                        audio=self.clip_audio(input, segment.start_seconds, segment.end_seconds),
                        media_type=input.media_type,
                        sample_rate_hz=input.sample_rate_hz,
                        channels=input.channels,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        confidence=segment.confidence,
                    )
                    for segment in ordered_segments
                )
                output = voice.diarization.OutputData(segments=segments)
            except Exception as error:
                raise VoiceDiarizationError("pyannote voice diarization failed.") from error
            active.set_attribute("bot.voice.segments.count", len(output.segments))
            return output


def _turns_from_result(result: object) -> collections.abc.Iterable[tuple[object, object]]:
    diarization = _member(result, "speaker_diarization", "diarization")
    if diarization is None:
        diarization = result

    itertracks = _member(diarization, "itertracks")
    if callable(itertracks):
        iterator = itertracks(yield_label=True)
        return _turns_from_entries(iterator)
    return _turns_from_entries(diarization)


def _turns_from_entries(entries: object) -> collections.abc.Iterable[tuple[object, object]]:
    if isinstance(entries, str | bytes | bytearray) or not isinstance(entries, collections.abc.Iterable):
        raise TypeError("pyannote diarization result is not iterable.")

    def turns() -> collections.abc.Iterator[tuple[object, object]]:
        for entry in entries:
            if not isinstance(entry, collections.abc.Sequence) or isinstance(entry, str | bytes | bytearray):
                raise TypeError("pyannote diarization entry is not a sequence.")
            values = tuple(entry)
            if len(values) == 2:
                yield values[0], values[1]
            elif len(values) == 3:
                yield values[0], values[2]
            else:
                raise TypeError("pyannote diarization entry must contain a turn and label.")

    return turns()


def _segment_from_pyannote(turn: object, label: object) -> _Segment:
    _validate_provider_label(label)
    start_seconds = _required_float(_member(turn, "start", "start_seconds"), field_name="start")
    end_seconds = _required_float(_member(turn, "end", "end_seconds", "stop"), field_name="end")
    if end_seconds <= start_seconds:
        raise ValueError("pyannote segment end must be greater than start.")
    confidence = _optional_float(_member(turn, "confidence", "score", "probability"))
    return _Segment(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        confidence=confidence,
    )


def _member(value: object, *names: str) -> object:
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


def _validate_provider_label(value: object) -> None:
    if value is None or isinstance(value, bool):
        raise TypeError("pyannote segment is missing a speaker label.")
    label = str(value)
    if not label:
        raise ValueError("pyannote segment has an empty speaker label.")


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and 0.0 <= parsed <= 1.0 else None


def _required_float(value: object, *, field_name: str) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"pyannote segment is missing numeric {field_name}.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"pyannote segment has invalid {field_name}.")
    return parsed


__all__ = [
    "VoiceDiarizationError",
    "VoiceDiarizationAudioClipper",
    "VoiceDiarizationPipeline",
    "VoiceDiarizationPipelineLoader",
    "VoiceDiarizer",
    "load_voice_diarization_pipeline",
]
