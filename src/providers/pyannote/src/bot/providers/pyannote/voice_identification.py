from __future__ import annotations

from bot.abilities.hearing import voice
from bot.abilities import classifying

import asyncio
import collections.abc
import dataclasses
import importlib
import math
import pathlib
import typing

from bot.telemetry import span

from ._audio import temporary_audio_file

_SCOPE = "bot.providers.pyannote"
_COMPONENT = "pyannote.voice_identification"


class VoiceIdentificationError(RuntimeError):
    """Raised when the pyannote embedding adapter cannot classify audio."""


class SpeakerEmbeddingInference(typing.Protocol):
    """The callable surface of ``pyannote.audio.Inference(model)``.

    ``Inference`` accepts a WAV path and returns one speaker embedding for the
    input audio. The adapter deliberately receives this inference object,
    rather than a raw pyannote ``Model``, so the injected contract matches the
    public pyannote usage API exactly.
    """

    def __call__(self, audio: pathlib.Path) -> object:
        """Return the pyannote speaker embedding for an audio path."""
        ...


SpeakerEmbeddingInferenceLoader = collections.abc.Callable[[str], SpeakerEmbeddingInference]


def load_speaker_embedding_inference(model_id: str) -> SpeakerEmbeddingInference:
    """Load ``Inference(Model.from_pretrained(model_id))`` lazily.

    Authentication belongs to the pyannote/Hugging Face runtime setup. Callers
    that need explicit token handling can inject a loader instead.
    """

    module = importlib.import_module("pyannote.audio")
    model_type = typing.cast(object, getattr(module, "Model"))
    load_model = typing.cast(collections.abc.Callable[[str], object], getattr(model_type, "from_pretrained"))
    loaded_model = load_model(model_id)
    inference_type = typing.cast(object, getattr(module, "Inference"))
    return typing.cast(
        SpeakerEmbeddingInference,
        typing.cast(collections.abc.Callable[..., object], inference_type)(loaded_model, window="whole"),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Classifier(classifying.Classifier[voice.identification.InputData, voice.identification.OutputData]):
    """Classify voice segments with pyannote speaker embeddings.

    This is an embedding classifier adapter, not an identity resolver: it
    returns one provider-neutral embedding for each input segment in input
    order. Conversation-owned participant resolution happens downstream.
    Raw pyannote model objects and labels never enter the returned core schema.
    """

    model_id: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    audio_file_suffix: str = ".wav"
    inference: SpeakerEmbeddingInference | None = None
    load_inference: SpeakerEmbeddingInferenceLoader = load_speaker_embedding_inference

    @typing.override
    async def classify(self, input: voice.identification.InputData) -> voice.identification.OutputData:
        """Classify each input segment without blocking the event loop.

        Raises:
            VoiceIdentificationError: If inference loading, audio staging,
                embedding conversion, or core-schema validation fails.
        """

        return await asyncio.to_thread(self._classify_blocking, input)

    def _classify_blocking(self, input: voice.identification.InputData) -> voice.identification.OutputData:
        # Counts only: how many segments went in and how many embeddings came back. An embedding
        # is the voiceprint itself and never becomes an attribute, and neither does any identity
        # resolved from one downstream. `asyncio.to_thread` copies the caller's context, so this
        # span keeps the trace of the audio that produced the segments.
        with span.operation(
            "bot.provider.pyannote.voice_identification.classify",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="identify",
        ) as active:
            active.set_attribute("bot.voice.segments.count", len(input.segments))
            try:
                inference = self.inference if self.inference is not None else self.load_inference(self.model_id)
                embeddings: list[voice.VoiceEmbedding] = []
                for segment in input.segments:
                    with temporary_audio_file(_wav_container(segment), suffix=self.audio_file_suffix) as audio_path:
                        embedding = inference(audio_path)
                    embeddings.append(
                        voice.VoiceEmbedding(
                            embedding=_embedding_values(embedding),
                            model=self.model_id,
                            confidence=None,
                        )
                    )
                output = voice.identification.OutputData(embeddings=tuple(embeddings))
            except Exception as error:
                raise VoiceIdentificationError("pyannote voice identification failed.") from error
            active.set_attribute("bot.voice.embeddings.count", len(output.embeddings))
            return output


def _embedding_values(value: object) -> tuple[float, ...]:
    tolist = typing.cast(object, getattr(value, "tolist", None))
    if callable(tolist):
        return _embedding_values(typing.cast(collections.abc.Callable[[], object], tolist)())
    data = typing.cast(object, getattr(value, "data", None))
    if data is not None and data is not value:
        return _embedding_values(data)
    if isinstance(value, bool) or not isinstance(value, collections.abc.Sequence) or isinstance(value, str | bytes):
        raise TypeError("pyannote speaker embedding must be a numeric vector.")
    if not value:
        raise ValueError("pyannote speaker embedding must not be empty.")
    values: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            break
        component = float(item)
        if not math.isfinite(component):
            raise ValueError("pyannote speaker embedding values must be finite.")
        values.append(component)
    else:
        return tuple(values)
    if len(value) == 1:
        return _embedding_values(value[0])
    raise TypeError("pyannote speaker embedding must be a one-dimensional numeric vector.")


def _wav_container(segment: voice.VoiceSegment) -> bytes:
    """Wrap a provider-neutral raw PCM segment in pyannote's WAV container."""

    import io
    import wave

    frame_bytes = 2 * segment.channels
    if len(segment.audio) % frame_bytes:
        raise ValueError("pyannote identification input is not aligned to complete PCM frames.")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(segment.channels)
        wav.setsampwidth(2)
        wav.setframerate(segment.sample_rate_hz)
        wav.writeframes(segment.audio)
    return output.getvalue()


__all__ = [
    "SpeakerEmbeddingInference",
    "SpeakerEmbeddingInferenceLoader",
    "VoiceIdentificationError",
    "Classifier",
    "load_speaker_embedding_inference",
]
