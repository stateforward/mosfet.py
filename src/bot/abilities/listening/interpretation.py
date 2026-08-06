"""Interpretation: working out what a heard sound turns out to be.

Everything slow about hearing is here — voice detection, non-speech labelling, diarization,
direct voice identification, speech decoding — and it is a machine of its own for one reason: it is slow, and the thing in
front of it must not be. Scoring an arrival against what the body predicted is about a *moment*;
working out what the arrival was takes seconds. While these shared one queue, a sound that
reached the ear inside its window could be scored after the window closed, because the score was
taken when interpretation got round to the sound rather than when the sound arrived. That made
the score a fact about the backlog. A bot answering somebody is speaking while its ears are
still busy with what it is answering, so this was the ordinary case, not an edge one.

The split is what keeps the window honest. Arrivals are scored as they arrive and queue here
already scored, so how long this takes cannot change what anything was worth.

Whether a scored sound is worth a turn is still decided here, and deliberately at the very end:
voice detection and speech decoding run on the bot's own voice and the transcript is produced.
Nothing is filtered — the only thing that changes is whether any of it becomes something to
answer.
"""

from __future__ import annotations

from . import sensitivity
from .. import ability
from .. import classifying
from ..hearing import sound
from ..hearing import speech
from ..hearing import voice
from ..identity import value

import asyncio
import dataclasses
import io
import math
import typing
import uuid
import wave

import hsm

from bot.protocols import attachment
import pydantic

from bot.abilities import cognition
from bot.environment import SoundData, SoundEvent
from bot import telemetry
from bot.telemetry import observer
from bot.telemetry import span

_SCOPE = "bot.abilities.listening"
_COMPONENT = "listening.interpretation"

DEFAULT_PRODUCT_THRESHOLD_DB = 3.0
"""How loud what is left of an arrival must be, in dB, to be worth handing on.

Three decibels is a doubling of acoustic intensity — the smallest level difference conventionally
treated as a real one rather than as measurement slop. This is the single place a level becomes a
yes or a no, and every contributor that reduces a perceived level is compared against it here.
"""

ListeningStage: typing.TypeAlias = typing.Literal[
    "voice_detection",
    "sound_classification",
    "voice_diarization",
    "voice_identification",
    "speech_decoding",
]


class FailedEventData(pydantic.BaseModel):
    """FailureData signal produced when a listening stage cannot complete."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "FailureData signal produced when a listening stage cannot complete.",
            "examples": [{"stage": "speech_decoding", "message": "decoder timed out"}],
        },
    )

    stage: ListeningStage = pydantic.Field(
        description="Listening pipeline stage that failed.",
        examples=["speech_decoding"],
    )
    message: str = pydantic.Field(
        description="Human-readable failure message for the listening stage.",
        examples=["decoder timed out"],
    )

    @classmethod
    def from_ability_failure(cls, *, stage: ListeningStage, failure: ability.FailureData) -> typing.Self:
        """Adapt an ability-protocol failure into a listening stage failure."""

        return cls(stage=stage, message=failure.message)


class _VoiceDetectionCompletedEventData(pydantic.BaseModel):
    """Private completion payload for voice detection inside the listening ability."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    sensed: sensitivity.OutputData
    voice_detection: voice.detection.ApplyData


class _VoiceDiarizationCompletedEventData(pydantic.BaseModel):
    """Private completion payload for voice diarization inside the listening ability."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    sensed: sensitivity.OutputData
    voice_detection: voice.detection.ApplyData
    segments: tuple[voice.diarization.VoiceDiarizationSegment, ...]


class _VoiceIdentificationCompletedEventData(pydantic.BaseModel):
    """Private completion payload for direct voice identification inside Listening."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    sensed: sensitivity.OutputData
    voice_detection: voice.detection.ApplyData
    segments: tuple[voice.VoiceSegment, ...]
    embeddings: tuple[voice.identification.VoiceEmbedding, ...]


class _SpeechDecodingCompletedEventData(pydantic.BaseModel):
    """Private completion payload for speech decoding inside the listening ability."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    sensed: sensitivity.OutputData
    voice_detection: voice.detection.ApplyData
    speech: bytes


class _SoundClassificationCompletedEventData(pydantic.BaseModel):
    """Private completion payload for non-speech sound classification inside listening."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    sensed: sensitivity.OutputData
    voice_detection: voice.detection.ApplyData
    classification: "sound.classification.OutputData"


class SpeechData(pydantic.BaseModel):
    """One speech observation, before or after decode.

    Listening VAD classifies one admission of sound. Sticky conversational turns assemble these
    speech observations under Conversation turn Start/End.

    ``content`` / ``content_type`` are the whole payload: one field carrying what the
    observation *is* to whoever perceives it — ``audio/pcm`` bytes while the observation is
    still acoustic, rewritten in place to ``text/plain`` words once speech decoding resolves it. Decode rewrites this same
    observation — it never mints a separate product envelope — so the stimulus a bot
    perceives keeps one identity from admission through transcription.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "One Listening speech observation after voice detection: content plus relative voice "
                "spans. content_type is audio/pcm while the observation is acoustic and text/plain "
                "once speech decoding has rewritten it into words. Empty segments means no voice in "
                "this observation (including silence used to close a turn)."
            ),
            "examples": [
                {
                    "content": "AAAA",
                    "content_type": "audio/pcm",
                    "voice_detection": {
                        "segments": [{"start_seconds": 0.0, "end_seconds": 0.4, "confidence": 0.9}],
                    },
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 48000,
                    "channels": 1,
                },
                {
                    "content": "what is the weather like?",
                    "content_type": "text/plain",
                    "voice_detection": {
                        "segments": [{"start_seconds": 0.0, "end_seconds": 0.4, "confidence": 0.9}],
                    },
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 48000,
                    "channels": 1,
                },
                {
                    "content": "AAAA",
                    "content_type": "audio/pcm",
                    "voice_detection": {"segments": []},
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 48000,
                    "channels": 1,
                },
            ],
        },
    )

    content: str | bytes = pydantic.Field(
        description=(
            "The one payload this observation carries: signed 16-bit PCM bytes while content_type is "
            "audio/pcm, the decoded words once content_type is text/plain. PCM observations on the same "
            "conversation stream are homogeneous so turn assembly can concatenate their bytes."
        ),
        examples=["what is the weather like?"],
    )
    content_type: str = pydantic.Field(
        default="audio/pcm",
        min_length=1,
        description=(
            "Media type of content. audio/pcm before speech decoding, text/plain after it rewrites "
            "this observation into words."
        ),
        examples=["audio/pcm", "text/plain"],
    )
    start_seconds: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        description="Optional start time of this clipped voice segment in the source audio.",
        examples=[0.0],
    )
    end_seconds: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        description="Optional exclusive end time of this clipped voice segment in the source audio.",
        examples=[1.25],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional diarizer confidence for this clipped speech segment.",
        examples=[0.87],
    )
    voice_embedding: voice.identification.VoiceEmbedding | None = pydantic.Field(
        default=None,
        description="Optional voice embedding produced by the configured classifier.",
    )
    source_ids: value.IdentitySet = pydantic.Field(
        default_factory=frozenset,
        description=(
            "Opaque source identities for this speech observation. Empty source_ids are permitted when "
            "Listening has speech evidence but no identity classifier result; Conversation rejects them."
        ),
        examples=[[], [[0.12, -0.08, 0.31]]],
    )
    voice_detection: voice.detection.ApplyData = pydantic.Field(
        description="Voice spans from VoiceDetection relative to the start of this observation.",
    )
    sample_rate_hz: int = pydantic.Field(
        ge=1,
        description="Sample rate of audio in hertz (required to place the observation on a conversation timeline).",
        examples=[48000],
    )
    media_type: typing.Literal["audio/pcm"] = pydantic.Field(
        description="Validated media type for the signed 16-bit PCM audio bytes.",
        examples=["audio/pcm"],
    )
    channels: int = pydantic.Field(
        default=1,
        ge=1,
        description="Channel count of audio.",
        examples=[1],
    )

    @property
    def duration_seconds(self) -> float:
        """Duration of PCM content using the stream's signed 16-bit sample convention, in seconds.

        Zero once decode has rewritten this observation into words: text has no frames.
        """

        frame_bytes = 2 * self.channels
        if frame_bytes <= 0 or not isinstance(self.content, bytes):
            return 0.0
        return (len(self.content) // frame_bytes) / float(self.sample_rate_hz)

    @pydantic.model_validator(mode="after")
    def validate_segment_timing(self) -> typing.Self:
        """Require optional segment timing to be supplied as a positive span."""

        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("start_seconds and end_seconds must be supplied together.")
        if self.start_seconds is not None and self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds.")
        return self

    @pydantic.field_validator("source_ids", mode="before")
    @classmethod
    def validate_source_ids(cls, raw_value: object) -> value.IdentitySet:
        """Canonicalize named or embedding source identities."""

        return value.normalize_identity_set(raw_value, field_name="source_ids", allow_empty=True)

    @pydantic.field_serializer("source_ids", when_used="json")
    def serialize_source_ids(self, identities: value.IdentitySet) -> list[str | list[float]]:
        return value.identities_json(identities)


ListeningFailedEvent = hsm.Event[FailedEventData](
    name="bot.ability.listening.failed",
    schema=FailedEventData,
)
SpeechEvent = hsm.Event[SpeechData](
    name="bot.ability.listening.speech.output",
    schema=SpeechData,
)
_VoiceDetectionCompletedEvent = hsm.Event[_VoiceDetectionCompletedEventData](
    name="bot.ability.listening.voice_detection.completed",
    kind=hsm.CompletionEventKind,
    schema=_VoiceDetectionCompletedEventData,
)
_SoundClassificationCompletedEvent = hsm.Event[_SoundClassificationCompletedEventData](
    name="bot.ability.listening.sound_classification.completed",
    kind=hsm.CompletionEventKind,
    schema=_SoundClassificationCompletedEventData,
)
_VoiceDiarizationCompletedEvent = hsm.Event[_VoiceDiarizationCompletedEventData](
    name="bot.ability.listening.voice_diarization.completed",
    kind=hsm.CompletionEventKind,
    schema=_VoiceDiarizationCompletedEventData,
)
_VoiceIdentificationCompletedEvent = hsm.Event[_VoiceIdentificationCompletedEventData](
    name="bot.ability.listening.voice_identification.completed",
    kind=hsm.CompletionEventKind,
    schema=_VoiceIdentificationCompletedEventData,
)
_SpeechDecodingCompletedEvent = hsm.Event[_SpeechDecodingCompletedEventData](
    name="bot.ability.listening.speech_decoding.completed",
    kind=hsm.CompletionEventKind,
    schema=_SpeechDecodingCompletedEventData,
)
_ListeningStageFailedEvent = hsm.Event[FailedEventData](
    name="bot.ability.listening.stage.failed",
    kind=hsm.ErrorEventKind,
    schema=FailedEventData,
)


def _listening_sensed(event: hsm.Event[typing.Any]) -> sensitivity.OutputData | None:
    """The scored sound carried anywhere along the private completion chain."""

    data = event.data
    if isinstance(data, sensitivity.OutputData):
        return data
    if isinstance(
        data,
        _VoiceDetectionCompletedEventData
        | _SoundClassificationCompletedEventData
        | _VoiceDiarizationCompletedEventData
        | _VoiceIdentificationCompletedEventData
        | _SpeechDecodingCompletedEventData,
    ):
        return data.sensed
    return None


def _listening_event_with_context(
    event: hsm.Event[typing.Any],
    source: hsm.Event[typing.Any],
    *,
    operation_id: str | None = None,
) -> hsm.Event[typing.Any]:
    """Copy operation id, acoustic source, and metadata along the private completion chain."""

    resolved_operation_id = operation_id if operation_id is not None else source.id
    if resolved_operation_id is None:
        resolved_operation_id = uuid.uuid4().hex
    event = event.with_data_and_id(event.data, resolved_operation_id)
    # The completion chain crosses one HSM activity per stage, and each activity runs in its own
    # task: the ambient trace context does not survive the hop. Stamping the emitting span here is
    # what keeps one arriving sound one trace instead of one trace per stage.
    return telemetry.inject_context(
        dataclasses.replace(
            event,
            source=source.source or event.source,
            metadata=dict(source.metadata),
        )
    )


def _dispatch_listening_cognition_input_with_operation(
    ctx: hsm.Context,
    instance: "Interpretation",
    event: hsm.Event[typing.Any],
    stimulus: hsm.Event[typing.Any],
    *,
    operation_id: str | None,
) -> None:
    """Terminal handoff: ``cognition.InputEvent`` for the body owner when listening finishes.

    Stimulus is the acoustic product cognition should judge (``environment.sound`` or decoded
    speech). Provenance comes only from the event chain: the transducer the sound came off rides
    in on the scored product's envelope and rides out on the stimulus, which is how the body
    resolves which of its devices a sensory product belongs to.
    """

    with span.operation(
        "bot.listening.handoff",
        scope=_SCOPE,
        component=_COMPONENT,
        stage="cognition_handoff",
        context=telemetry.event_context(event),
    ) as active:
        resolved_operation_id = operation_id or event.id or uuid.uuid4().hex
        stimulus = _listening_event_with_context(stimulus, event, operation_id=resolved_operation_id)
        stimulus = dataclasses.replace(
            stimulus,
            source=event.source or stimulus.source,
            metadata=dict(event.metadata),
        )
        # Whether the product carries any speaker identity at all is the difference between a
        # product downstream can admit and one it silently rejects; the ids themselves never
        # leave the payload.
        source_ids = getattr(stimulus.data, "source_ids", None)
        active.set_attribute("bot.stimulus.name", stimulus.name)
        active.set_attribute("bot.identity.source.count", len(source_ids) if source_ids is not None else 0)
        handoff = cognition.InputEvent.with_data(cognition.InputData(stimulus=stimulus))
        handoff = _listening_event_with_context(handoff, event, operation_id=resolved_operation_id)
        handoff = dataclasses.replace(handoff, source=hsm.id(instance), metadata=dict(stimulus.metadata))
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(handoff))


def _dispatch_listening_terminal_failure_with_operation(
    ctx: hsm.Context,
    instance: "Interpretation",
    source: hsm.Event[typing.Any],
    failure: FailedEventData,
    *,
    operation_id: str | None,
) -> None:
    resolved_operation_id = operation_id or source.id or uuid.uuid4().hex
    terminal = _listening_event_with_context(
        instance.failed_event.with_data(failure), source, operation_id=resolved_operation_id
    )
    terminal = dataclasses.replace(terminal, source=hsm.id(instance))
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


def _dispatch_stage_failure_with_operation(
    ctx: hsm.Context,
    instance: "Interpretation",
    source: hsm.Event[typing.Any],
    *,
    stage: ListeningStage,
    message: str,
    operation_id: str | None,
) -> None:
    failure = FailedEventData(stage=stage, message=message)
    # The stage reports by dispatching, not raising, so its span has to be told it failed or the
    # trace would show a stage that completed. `stage` is a closed Literal, so the kind is too.
    span.record_current_failure(f"{stage}_failed")
    _ = hsm.dispatch(
        ctx,
        instance,
        _listening_event_with_context(
            _ListeningStageFailedEvent.with_data(failure),
            source,
            operation_id=operation_id or source.id or uuid.uuid4().hex,
        ),
    )



def _is_wav_container_bytes(audio: bytes) -> bool:
    """True when bytes look like a RIFF/WAVE container."""

    return len(audio) >= 12 and audio.startswith(b"RIFF") and audio[8:12] == b"WAVE"


def _is_wav_sound(sound: SoundData) -> bool:
    """True when sound is labeled or containerized as WAV."""

    media = (sound.media_type or "").lower()
    if media in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return True
    return _is_wav_container_bytes(sound.audio)

def _pcm_sound_data(sound: SoundData) -> SoundData:
    """Normalize environment sound to signed 16-bit PCM for identity and speech products.

    Ambient emitters (for example phone_bot Person/SayEncoder) may deliver RIFF/WAVE containers
    as ``audio/wav``. Voice identification and SpeechData require raw ``audio/pcm`` with sample
    rate and channels. Decode with the stdlib ``wave`` module; leave non-WAV media unchanged so
    kind-labeled non-speech paths keep their original bytes when the payload is not a real WAV.
    """

    media_type = (sound.media_type or "").strip().lower()
    if media_type in {"", "audio/pcm", "audio/l16", "audio/raw"}:
        if media_type in {"", "audio/l16", "audio/raw"} and sound.sample_rate_hz is not None and sound.channels is not None:
            return sound.model_copy(update={"media_type": "audio/pcm"})
        return sound
    if media_type not in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return sound
    try:
        with wave.open(io.BytesIO(sound.audio), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate_hz = stream.getframerate()
            frames = stream.readframes(stream.getnframes())
    except wave.Error as error:
        raise ValueError(f"invalid audio/wav container: {error}") from error
    if channels < 1:
        raise ValueError("audio/wav must include at least one channel.")
    if sample_rate_hz < 1:
        raise ValueError("audio/wav must include a positive sample rate.")
    if sample_width != 2:
        raise ValueError("audio/wav must be signed 16-bit PCM (sample width 2).")
    if not frames:
        raise ValueError("audio/wav contains no PCM frames.")
    return sound.model_copy(
        update={
            "audio": frames,
            "media_type": "audio/pcm",
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
        }
    )


def _sensed_with_pcm_sound(sensed: sensitivity.OutputData) -> sensitivity.OutputData:
    """Carry PCM bytes on the scored sound when the environment delivered a WAV container."""

    try:
        pcm_sound = _pcm_sound_data(sensed.sound)
    except ValueError:
        # Non-speech kind labels may use media_type audio/wav without a real container; leave
        # those bytes alone and let PCM-only stages fail closed if they later require PCM.
        return sensed
    if pcm_sound is sensed.sound:
        return sensed
    return sensed.model_copy(update={"sound": pcm_sound})


def _speech_from_sensed(
    *,
    sensed: sensitivity.OutputData,
    voice_detection: voice.detection.ApplyData,
    source_ids: value.IdentitySet = frozenset(),
    voice_embedding: voice.identification.VoiceEmbedding | None = None,
) -> SpeechData | None:
    """Build a public PCM speech product; unsupported or incomplete formats fail closed."""

    sound = sensed.sound
    sample_rate_hz = sound.sample_rate_hz
    channels = sound.channels if sound.channels is not None else 1
    if sound.media_type != "audio/pcm" or sample_rate_hz is None or not sound.audio:
        return None
    speech = SpeechData(
        content=sound.audio,
        voice_detection=voice_detection,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        media_type="audio/pcm",
        source_ids=source_ids,
        voice_embedding=voice_embedding,
    )
    return speech


def _speech_from_voice_segment(
    *,
    sensed: sensitivity.OutputData,
    segment: voice.VoiceSegment,
    source_ids: value.IdentitySet = frozenset(),
    voice_embedding: voice.identification.VoiceEmbedding | None = None,
    voice_detection: voice.detection.ApplyData | None = None,
) -> SpeechData | None:
    """Build one public speech product from one already-clipped voice segment.

    Conversion errors are raised so the owning Listening machine can publish its typed
    listening-stage failure instead of silently dropping a product.
    """

    sound = sensed.sound
    if sound.media_type != "audio/pcm" or sound.sample_rate_hz is None:
        raise ValueError("voice segment speech requires source audio/pcm with a sample rate.")
    source_channels = sound.channels if sound.channels is not None else 1
    if (
        segment.media_type != sound.media_type
        or segment.sample_rate_hz != sound.sample_rate_hz
        or segment.channels != source_channels
    ):
        raise ValueError("voice segment audio format does not match its source audio.")
    if not segment.audio:
        raise ValueError("voice segment audio is empty.")
    diarization_confidence = (
        segment.confidence if isinstance(segment, voice.diarization.VoiceDiarizationSegment) else None
    )
    segment_voice_detection = (
        voice.detection.ApplyData(
            segments=(
                voice.detection.VoiceDetectionSegment(
                    start_seconds=0.0,
                    end_seconds=segment.duration_seconds,
                    confidence=diarization_confidence,
                ),
            )
        )
        if isinstance(segment, voice.diarization.VoiceDiarizationSegment)
        else voice_detection or voice.detection.ApplyData(segments=())
    )
    return SpeechData(
        content=segment.audio,
        voice_detection=segment_voice_detection,
        sample_rate_hz=segment.sample_rate_hz,
        media_type="audio/pcm",
        channels=segment.channels,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        confidence=diarization_confidence,
        source_ids=source_ids,
        voice_embedding=voice_embedding,
    )


def _voice_segment_from_detection(
    *,
    sensed: sensitivity.OutputData,
    segment: voice.detection.VoiceDetectionSegment,
) -> voice.VoiceSegment:
    """Clip the first VAD span into the provider-neutral classifier input contract."""

    sound = sensed.sound
    if sound.media_type != "audio/pcm" or sound.sample_rate_hz is None or sound.channels is None:
        raise ValueError("voice identification requires audio/pcm with sample rate and channels.")
    frame_bytes = 2 * sound.channels
    if len(sound.audio) % frame_bytes:
        raise ValueError("voice identification requires complete 16-bit PCM frames.")
    duration_seconds = len(sound.audio) / float(frame_bytes * sound.sample_rate_hz)
    end_seconds = min(segment.end_seconds, duration_seconds)
    if end_seconds <= segment.start_seconds:
        raise ValueError("voice detection span contains no source audio.")
    start_frame = math.floor(segment.start_seconds * sound.sample_rate_hz)
    end_frame = math.ceil(end_seconds * sound.sample_rate_hz)
    audio = sound.audio[start_frame * frame_bytes : end_frame * frame_bytes]
    if not audio:
        raise ValueError("voice detection span contains no PCM frames.")
    return voice.VoiceSegment(
        audio=audio,
        media_type="audio/pcm",
        sample_rate_hz=sound.sample_rate_hz,
        channels=sound.channels,
        start_seconds=segment.start_seconds,
        end_seconds=end_seconds,
    )


def _dispatch_product_cognition_input_with_operation(
    ctx: hsm.Context,
    instance: "Interpretation",
    event: hsm.Event[typing.Any],
    *,
    operation_id: str | None,
) -> None:
    """Hand off whatever product this pipeline arrived at, whichever route it came by.

    - STT configured and completed: the same ``SpeechEvent``, rewritten to ``text/plain`` words.
    - No STT (acoustic path): ``SpeechEvent`` with VAD spans + audio for Conversation /
      conversation turn End (including empty-segment silence observations that close a sticky turn).
    - Labeled non-speech (ring, busy, …): original ``environment.sound``.
    """

    completion = event.data
    if isinstance(completion, _SpeechDecodingCompletedEventData):
        sound = completion.sensed.sound
        if sound.sample_rate_hz is None:
            _dispatch_listening_terminal_failure_with_operation(
                ctx,
                instance,
                event,
                FailedEventData(
                    stage="speech_decoding",
                    message="Speech decoding completed on sound with no sample rate to place it on a timeline.",
                ),
                operation_id=operation_id,
            )
            return
        # Decode rewrites this observation: it becomes words. The audio it was is dropped.
        _dispatch_listening_cognition_input_with_operation(
            ctx,
            instance,
            event,
            SpeechEvent.with_data(
                SpeechData(
                    content=completion.speech.decode("utf-8", errors="replace"),
                    content_type="text/plain",
                    voice_detection=completion.voice_detection,
                    sample_rate_hz=sound.sample_rate_hz,
                    channels=sound.channels if sound.channels is not None else 1,
                    media_type="audio/pcm",
                )
            ),
            operation_id=operation_id,
        )
        return
    if isinstance(completion, _VoiceDiarizationCompletedEventData):
        try:
            speech_products = tuple(
                _speech_from_voice_segment(sensed=completion.sensed, segment=segment) for segment in completion.segments
            )
        except Exception as error:
            _dispatch_listening_terminal_failure_with_operation(
                ctx,
                instance,
                event,
                FailedEventData(
                    stage="voice_diarization",
                    message=f"Voice diarization product conversion failed: {error}",
                ),
                operation_id=operation_id,
            )
            return
        for speech_data in speech_products:
            assert speech_data is not None
            _dispatch_listening_cognition_input_with_operation(
                ctx, instance, event, SpeechEvent.with_data(speech_data), operation_id=operation_id
            )
        return
    if isinstance(completion, _VoiceIdentificationCompletedEventData):
        if len(completion.segments) != len(completion.embeddings):
            _dispatch_listening_terminal_failure_with_operation(
                ctx,
                instance,
                event,
                FailedEventData(
                    stage="voice_identification",
                    message="Voice identification returned one embedding per voice segment; requirement unmet.",
                ),
                operation_id=operation_id,
            )
            return
        try:
            speech_products = tuple(
                _speech_from_voice_segment(
                    sensed=completion.sensed,
                    segment=segment,
                    voice_embedding=embedding,
                    source_ids=frozenset({embedding.embedding}),
                    voice_detection=completion.voice_detection,
                )
                for segment, embedding in zip(completion.segments, completion.embeddings, strict=True)
            )
        except Exception as error:
            _dispatch_listening_terminal_failure_with_operation(
                ctx,
                instance,
                event,
                FailedEventData(
                    stage="voice_identification",
                    message=f"Voice identification product conversion failed: {error}",
                ),
                operation_id=operation_id,
            )
            return
        for speech_data in speech_products:
            assert speech_data is not None
            _dispatch_listening_cognition_input_with_operation(
                ctx, instance, event, SpeechEvent.with_data(speech_data), operation_id=operation_id
            )
        return
    # Acoustic path reaches HandingOff with VAD/diarization/unlabeled-classification completions
    # only when STT is not configured (STT path always completes as SpeechDecoding). Labeled
    # non-speech falls through to environment.sound.
    if isinstance(
        completion,
        _VoiceDetectionCompletedEventData,
    ) or (isinstance(completion, _SoundClassificationCompletedEventData) and not completion.classification.is_labeled):
        speech_data = _speech_from_sensed(
            sensed=completion.sensed,
            voice_detection=completion.voice_detection,
        )
        if speech_data is not None:
            _dispatch_listening_cognition_input_with_operation(
                ctx,
                instance,
                event,
                SpeechEvent.with_data(speech_data),
                operation_id=operation_id,
            )
            return
    sensed = _listening_sensed(event)
    if sensed is None:
        raise AssertionError("listening cognition handoff requires sound on the event chain.")
    _dispatch_listening_cognition_input_with_operation(
        ctx, instance, event, SoundEvent.with_data(sensed.sound), operation_id=operation_id
    )


def _has_sensed_sound(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, sensitivity.OutputData)


def _has_current_operation(active_operation_id: str | None, event: hsm.Event[typing.Any]) -> bool:
    """Compare an event id with the operation id supplied by the owning machine callback."""

    return active_operation_id is not None and event.id == active_operation_id


class Interpretation(ability.Ability[sensitivity.OutputData, cognition.InputData]):
    """Turns a scored sound into something the bot can judge, or into nothing.

    Input is a sound that has already been weighed against what the body predicted about it
    (:class:`~bot.abilities.listening.sensitivity.OutputData`), which is why this can take as
    long as it likes: the weighing already happened, at the moment the sound arrived.

    Stages (VAD, optional non-speech classification, optional diarization, direct voice identification,
    and optional STT) are
    internal. After VAD Start (first voice observation), Interpretation stays in **HearingSpeech** and
    continues feeding scored frames to VoiceDetection until VAD End. Speech observations stream while
    speech is open; whole-utterance STT/diarization (when configured) run only after End.

    Success terminal is cognitive input whose stimulus is:

    - decoded speech (when STT is configured, after utterance End),
    - a :class:`SpeechData` product (acoustic path: per-observation stream + empty-segment End), or
    - the original labeled non-speech sound (ring/busy/…).

    With STT configured, unlabeled no-voice audio before speech is still skipped. Stage progress is
    carried only on typed private completion/failure events (HSM-COMPLETION-001).
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = sensitivity.OutputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = cognition.InputData
    # The scored sound is the front door: what Sensitivity produces is exactly what this consumes,
    # so the two compose through a typed event rather than through a shared queue.
    input_event: typing.ClassVar[hsm.Event[sensitivity.OutputData]] = sensitivity.OutputEvent
    output_event: typing.ClassVar[hsm.Event[cognition.InputData]] = cognition.InputEvent
    failed_event: typing.ClassVar[hsm.Event[FailedEventData]] = ListeningFailedEvent
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _product_threshold_db: float
    _voice_detection: voice.detection.VoiceDetection
    _sound_classification: sound.classification.SoundClassification | None
    _speech_decoding: speech.SpeechDecoding | None
    _voice_diarization: voice.diarization.VoiceDiarization | None
    _voice_identification: voice.identification.VoiceIdentification | None
    _attachment_group: attachment.Group
    # Open utterance PCM while HearingSpeech (machine-owned sticky stream, not peer coordination).
    _open_speech_audio: bytearray
    _open_speech_sample_rate_hz: int | None
    _open_speech_channels: int
    _open_speech_confidence: float | None
    _open_speech_sensed: sensitivity.OutputData | None
    _open_speech_format_error: str | None
    _active_source_ids: value.IdentitySet
    _active_voice_embedding: voice.identification.VoiceEmbedding | None
    _active_operation_id: str | None

    def __init__(
        self,
        *,
        voice_detector: voice.detection.VoiceDetector,
        sound_classifier: sound.classification.SoundClassifier | None = None,
        speech_decoder: speech.SpeechDecoder | None = None,
        voice_diarizer: voice.diarization.VoiceDiarizer | None = None,
        voice_classifier: classifying.Classifier[
            voice.identification.InputData,
            voice.identification.OutputData,
        ]
        | None = None,
        product_threshold_db: float = DEFAULT_PRODUCT_THRESHOLD_DB,
    ) -> None:
        super().__init__()
        if product_threshold_db < 0.0:
            raise ValueError("product_threshold_db must not be negative.")
        self._product_threshold_db = product_threshold_db
        self._voice_detection = voice.detection.VoiceDetection(classifier=voice_detector)
        self._sound_classification = (
            sound.classification.SoundClassification(classifier=sound_classifier)
            if sound_classifier is not None
            else None
        )
        self._speech_decoding = speech.SpeechDecoding(decoder=speech_decoder) if speech_decoder is not None else None
        self._voice_diarization = (
            voice.diarization.VoiceDiarization(classifier=voice_diarizer) if voice_diarizer is not None else None
        )
        if voice_classifier is not None and speech_decoder is not None:
            raise ValueError(
                "voice_classifier cannot be combined with speech_decoder because embedding correlation would be lost."
            )
        self._voice_identification = (
            voice.identification.VoiceIdentification(classifier=voice_classifier)
            if voice_classifier is not None
            else None
        )
        children: list[hsm.Instance] = [self._voice_detection]
        if self._sound_classification is not None:
            children.append(self._sound_classification)
        if self._voice_diarization is not None:
            children.append(self._voice_diarization)
        if self._voice_identification is not None:
            children.append(self._voice_identification)
        if self._speech_decoding is not None:
            children.append(self._speech_decoding)
        self._attachment_group = attachment.Group(*children)
        self._open_speech_audio = bytearray()
        self._open_speech_sample_rate_hz = None
        self._open_speech_channels = 1
        self._open_speech_confidence = None
        self._open_speech_sensed = None
        self._open_speech_format_error = None
        self._active_source_ids = frozenset()
        self._active_voice_embedding = None
        self._active_operation_id = None

    @staticmethod
    def _dispatch_listening_cognition_input(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
        stimulus: hsm.Event[typing.Any],
    ) -> None:
        _dispatch_listening_cognition_input_with_operation(
            ctx,
            instance,
            event,
            stimulus,
            operation_id=instance._active_operation_id,
        )

    @staticmethod
    def _dispatch_listening_failure(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = event.data
        assert isinstance(failure, FailedEventData)
        _dispatch_listening_terminal_failure_with_operation(
            ctx,
            instance,
            event,
            failure,
            operation_id=instance._active_operation_id,
        )

    @staticmethod
    def _dispatch_stage_failure(
        ctx: hsm.Context,
        instance: "Interpretation",
        source: hsm.Event[typing.Any],
        *,
        stage: ListeningStage,
        message: str,
    ) -> None:
        _dispatch_stage_failure_with_operation(
            ctx,
            instance,
            source,
            stage=stage,
            message=message,
            operation_id=instance._active_operation_id,
        )

    @staticmethod
    def _dispatch_product_cognition_input(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        _dispatch_product_cognition_input_with_operation(
            ctx,
            instance,
            event,
            operation_id=instance._active_operation_id,
        )

    @staticmethod
    def _begin_operation(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> None:
        """Stamp one operation id at the interpretation ingress, including id-less ingress."""

        del ctx
        instance._active_operation_id = event.id or uuid.uuid4().hex

    @staticmethod
    def _clear_operation(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> None:
        """Clear the owned operation correlation after terminal success, drop, or failure."""

        del ctx, event
        instance._active_operation_id = None

    @staticmethod
    def _has_detected_voice(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            _has_current_operation(instance._active_operation_id, event)
            and isinstance(data, _VoiceDetectionCompletedEventData)
            and bool(data.voice_detection.segments)
        )

    @staticmethod
    def _has_detected_voice_and_direct_identification(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Interpretation._has_detected_voice(ctx, instance, event)
            and instance._voice_identification is not None
            and instance._voice_diarization is None
        )

    @staticmethod
    def _has_detected_no_voice(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            _has_current_operation(instance._active_operation_id, event)
            and isinstance(data, _VoiceDetectionCompletedEventData)
            and not data.voice_detection.segments
        )

    @staticmethod
    def _has_labeled_sound(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            _has_current_operation(instance._active_operation_id, event)
            and isinstance(data, _SoundClassificationCompletedEventData)
            and data.classification.is_labeled
        )

    @staticmethod
    def _has_unlabeled_sound(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            _has_current_operation(instance._active_operation_id, event)
            and isinstance(data, _SoundClassificationCompletedEventData)
            and not data.classification.is_labeled
        )

    @staticmethod
    def _has_voice_diarization_completion(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return _has_current_operation(instance._active_operation_id, event) and isinstance(
            event.data, _VoiceDiarizationCompletedEventData
        )

    @staticmethod
    def _has_voice_identification_completion(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return _has_current_operation(instance._active_operation_id, event) and isinstance(
            event.data, _VoiceIdentificationCompletedEventData
        )

    @staticmethod
    def _has_speech_decoding_completion(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return _has_current_operation(instance._active_operation_id, event) and isinstance(
            event.data, _SpeechDecodingCompletedEventData
        )

    @staticmethod
    def _has_listening_stage_failure(
        ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]
    ) -> bool:
        del ctx
        return _has_current_operation(instance._active_operation_id, event) and isinstance(event.data, FailedEventData)

    @staticmethod
    async def _await_child_output(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
        *,
        child: ability.Ability[typing.Any, typing.Any],
        child_input: object,
        stage: ListeningStage,
    ) -> object | None:
        """Await one nested ability apply via typed input/terminal events (never reach into collab).

        Returns the child's successful terminal payload, or ``None`` after a stage failure.
        This is not the Listening pipeline product (cognition handoff); that is decided later at
        HandingOff.
        """

        operation_id = instance._active_operation_id or event.id or uuid.uuid4().hex
        try:
            terminal = await ability.Ability.await_child_terminal(
                ctx,
                owner=instance,
                child=child,
                operation_id=operation_id,
                input=child_input,
                metadata=event.metadata,
            )
        except asyncio.CancelledError:
            raise
        if terminal.name == child.failed_event.name:
            message_text = getattr(terminal.data, "message", f"Listening {stage} child failed.")
            Interpretation._dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage=stage,
                message=str(message_text),
            )
            return None
        return terminal.data

    @staticmethod
    def _clear_open_speech(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Drop sticky utterance PCM when leaving an open-speech window."""

        del ctx, event
        instance._open_speech_audio = bytearray()
        instance._open_speech_sample_rate_hz = None
        instance._open_speech_channels = 1
        instance._open_speech_confidence = None
        instance._open_speech_sensed = None
        instance._open_speech_format_error = None
        instance._active_source_ids = frozenset()
        instance._active_voice_embedding = None

    @staticmethod
    def _append_open_speech(
        instance: "Interpretation",
        sensed: sensitivity.OutputData,
        voice_detection: voice.detection.ApplyData,
    ) -> None:
        sound = sensed.sound
        if sound.media_type != "audio/pcm" and instance._open_speech_format_error is None:
            instance._open_speech_format_error = "source audio must use media_type audio/pcm."
        if sound.sample_rate_hz is None and instance._open_speech_format_error is None:
            instance._open_speech_format_error = "source audio must include a sample rate."
        if sound.channels is None and instance._open_speech_format_error is None:
            instance._open_speech_format_error = "source audio must include a channel count."
        if (
            sound.sample_rate_hz is not None
            and instance._open_speech_sample_rate_hz is not None
            and sound.sample_rate_hz != instance._open_speech_sample_rate_hz
            and instance._open_speech_format_error is None
        ):
            instance._open_speech_format_error = "source audio sample rate changed during the utterance."
        if (
            sound.channels is not None
            and instance._open_speech_audio
            and sound.channels != instance._open_speech_channels
            and instance._open_speech_format_error is None
        ):
            instance._open_speech_format_error = "source audio channel count changed during the utterance."
        if sound.sample_rate_hz is not None:
            if instance._open_speech_sample_rate_hz is None:
                instance._open_speech_sample_rate_hz = sound.sample_rate_hz
        if sound.channels is not None:
            if not instance._open_speech_audio:
                instance._open_speech_channels = sound.channels
        if sound.audio:
            instance._open_speech_audio.extend(sound.audio)
        confidences = tuple(
            segment.confidence for segment in voice_detection.segments if segment.confidence is not None
        )
        if confidences:
            confidence = min(confidences)
            instance._open_speech_confidence = (
                confidence
                if instance._open_speech_confidence is None
                else min(instance._open_speech_confidence, confidence)
            )

    @staticmethod
    def _remember_open_speech_sensed(
        instance: "Interpretation",
        sensed: sensitivity.OutputData,
    ) -> None:
        open_speech_sensed = instance._open_speech_sensed
        if open_speech_sensed is None:
            instance._open_speech_sensed = sensed
        elif open_speech_sensed.perceived_level_db is None and sensed.perceived_level_db is not None:
            instance._open_speech_sensed = sensed
        elif (
            open_speech_sensed.perceived_level_db is not None
            and sensed.perceived_level_db is not None
            and sensed.perceived_level_db > open_speech_sensed.perceived_level_db
        ):
            instance._open_speech_sensed = sensed

    @staticmethod
    def _open_speech_sensed_or(
        instance: "Interpretation",
        fallback: sensitivity.OutputData,
    ) -> sensitivity.OutputData:
        return instance._open_speech_sensed or fallback

    @staticmethod
    def _open_speech_bytes(instance: "Interpretation") -> bytes:
        return bytes(instance._open_speech_audio)

    @staticmethod
    def _emit_speech_product(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
        completion: _VoiceDetectionCompletedEventData,
    ) -> None:
        """Stream one acoustic Speech cognition product when Listening owns no STT decoder."""

        with span.operation(
            "bot.listening.speech_product",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="acoustic_product",
            context=telemetry.event_context(event),
        ) as active:
            # An acoustic product that is never emitted is the ordinary case here, not a failure —
            # but which of the three reasons stopped it is exactly what a silent bot's trace has to
            # say. Closed vocabulary; nothing from the payload.
            if (
                instance._speech_decoding is not None
                or instance._voice_diarization is not None
            ):
                active.set_attribute("bot.product.withheld.reason", "decoder_owns_product")
                return

            sensed = completion.sensed
            if sensed.perceived_level_db is not None and sensed.perceived_level_db < instance._product_threshold_db:
                active.set_attribute("bot.product.withheld.reason", "below_threshold")
                return
            speech = _speech_from_sensed(
                sensed=sensed,
                voice_detection=completion.voice_detection,
                source_ids=instance._active_source_ids,
                voice_embedding=instance._active_voice_embedding,
            )
            if speech is None:
                active.set_attribute("bot.product.withheld.reason", "no_speech_content")
                return
            active.set_attribute("bot.product.withheld.reason", "")
            active.set_attribute("bot.identity.source.count", len(speech.source_ids))
            Interpretation._dispatch_listening_cognition_input(
                ctx,
                instance,
                event,
                SpeechEvent.with_data(speech),
            )

    @staticmethod
    def _enter_hearing_speech(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        """VAD Start / first voice: open utterance window and stream the opening observation."""

        completion = event.data
        assert isinstance(completion, _VoiceDetectionCompletedEventData)
        Interpretation._clear_open_speech(ctx, instance, event)
        Interpretation._remember_open_speech_sensed(instance, completion.sensed)
        Interpretation._append_open_speech(instance, completion.sensed, completion.voice_detection)
        Interpretation._emit_speech_product(ctx, instance, event, completion)

    @staticmethod
    def _enter_hearing_speech_from_identification(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Commit the first-segment identity before emitting the opening product."""

        completion = event.data
        if not isinstance(completion, _VoiceIdentificationCompletedEventData) or not completion.embeddings:
            Interpretation._dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="voice_identification",
                message="Voice identification produced no opening embedding.",
            )
            return
        embedding = completion.embeddings[0]
        Interpretation._clear_open_speech(ctx, instance, event)
        instance._active_voice_embedding = embedding
        instance._active_source_ids = frozenset({embedding.embedding})
        detection = _VoiceDetectionCompletedEventData(
            sensed=completion.sensed,
            voice_detection=completion.voice_detection,
        )
        Interpretation._remember_open_speech_sensed(instance, completion.sensed)
        Interpretation._append_open_speech(instance, completion.sensed, completion.voice_detection)
        Interpretation._emit_speech_product(ctx, instance, event, detection)

    @staticmethod
    def _continue_hearing_speech(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Still in voice: append PCM and stream an Update observation."""

        completion = event.data
        assert isinstance(completion, _VoiceDetectionCompletedEventData)
        Interpretation._remember_open_speech_sensed(instance, completion.sensed)
        Interpretation._append_open_speech(instance, completion.sensed, completion.voice_detection)
        Interpretation._emit_speech_product(ctx, instance, event, completion)

    @staticmethod
    def _close_hearing_speech(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        """VAD End: stream silence/end observation; keep open PCM for optional STT/diarization."""

        completion = event.data
        assert isinstance(completion, _VoiceDetectionCompletedEventData)
        Interpretation._append_open_speech(instance, completion.sensed, completion.voice_detection)
        Interpretation._emit_speech_product(ctx, instance, event, completion)

    @staticmethod
    async def _run_voice_detection(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.listening.detect_voice",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="voice_detection",
            context=telemetry.event_context(event),
        ) as active:
            sensed = _listening_sensed(event)
            if sensed is None:
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_detection",
                    message="listening input is missing sound data.",
                )
                return
            # Normalize to PCM for clipping/identity products. For VAD, prefer the original WAV
            # container when present so file-oriented detectors (Silero) keep the true sample rate.
            # Raw PCM is only passed through when ingress is already PCM; wrappers that re-encode PCM
            # as WAV must not invent a different rate than SoundData.sample_rate_hz.
            pcm_sensed = _sensed_with_pcm_sound(sensed)
            vad_input = sensed.sound.audio if _is_wav_sound(sensed.sound) else pcm_sensed.sound.audio
            active.set_attribute("bot.audio.byte.count", len(vad_input))
            output = await Interpretation._await_child_output(
                ctx,
                instance,
                event,
                child=instance._voice_detection,
                child_input=vad_input,
                stage="voice_detection",
            )
            if output is None:
                return
            if not isinstance(output, voice.detection.ApplyData):
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_detection",
                    message="Voice detection child produced a non-ApplyData terminal.",
                )
                return
            # Whether this chunk was voiced at all, and how much of it — the one thing that
            # decides which way the whole rest of interpretation goes.
            active.set_attribute("bot.voice.segment.count", len(output.segments))
            completion = _VoiceDetectionCompletedEventData(sensed=pcm_sensed, voice_detection=output)
            _ = hsm.dispatch(
                ctx,
                instance,
                _listening_event_with_context(
                    _VoiceDetectionCompletedEvent.with_data(completion),
                    event,
                    operation_id=instance._active_operation_id,
                ),
            )

    @staticmethod
    async def _run_sound_classification(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.listening.classify_sound",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="sound_classification",
            context=telemetry.event_context(event),
        ) as active:
            detection = event.data
            if not isinstance(detection, _VoiceDetectionCompletedEventData):
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="sound_classification",
                    message="sound classification requires a voice-detection completion.",
                )
                return
            sound_classification = instance._sound_classification
            if sound_classification is None:
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="sound_classification",
                    message="sound classification is not configured.",
                )
                return
            # Full SoundData (kind/media provenance) is the child's typed input contract.
            output = await Interpretation._await_child_output(
                ctx,
                instance,
                event,
                child=sound_classification,
                child_input=detection.sensed.sound,
                stage="sound_classification",
            )
            if output is None:
                return
            if not isinstance(output, sound.classification.OutputData):
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="sound_classification",
                    message="Sound classification child produced invalid output.",
                )
                return
            active.set_attribute("bot.sound.label.count", len(output.labels))
            completion = _SoundClassificationCompletedEventData(
                sensed=detection.sensed,
                voice_detection=detection.voice_detection,
                classification=output,
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                _listening_event_with_context(
                    _SoundClassificationCompletedEvent.with_data(completion),
                    event,
                    operation_id=instance._active_operation_id,
                ),
            )

    @staticmethod
    async def _run_voice_diarization(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.listening.diarize_voice",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="voice_diarization",
            context=telemetry.event_context(event),
        ) as active:
            detection = event.data
            if not isinstance(detection, _VoiceDetectionCompletedEventData):
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_diarization",
                    message="voice diarization requires a voice-detection completion.",
                )
                return
            voice_diarization = instance._voice_diarization
            if voice_diarization is None:
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_diarization",
                    message="voice diarization ability is not configured.",
                )
                return
            utterance = Interpretation._open_speech_bytes(instance)
            diarize_audio = utterance if utterance else detection.sensed.sound.audio
            sound = detection.sensed.sound
            sample_rate_hz = instance._open_speech_sample_rate_hz or sound.sample_rate_hz
            channels = (
                instance._open_speech_channels if instance._open_speech_sample_rate_hz is not None else sound.channels
            )
            if sound.media_type != "audio/pcm" or sample_rate_hz is None or channels is None:
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_diarization",
                    message="voice diarization requires raw audio/pcm with sample rate and channels.",
                )
                return
            try:
                diarize_input = voice.diarization.InputData(
                    audio=diarize_audio,
                    media_type="audio/pcm",
                    sample_rate_hz=sample_rate_hz,
                    channels=channels,
                )
            except Exception as error:
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_diarization",
                    message=f"Voice diarization input conversion failed: {error}",
                )
                return
            output = await Interpretation._await_child_output(
                ctx,
                instance,
                event,
                child=voice_diarization,
                child_input=diarize_input,
                stage="voice_diarization",
            )
            if output is None:
                return
            if not isinstance(output, voice.diarization.OutputData):
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_diarization",
                    message="Voice diarization child produced invalid output.",
                )
                return
            active.set_attribute("bot.voice.segment.count", len(output.segments))
            completion = _VoiceDiarizationCompletedEventData(
                sensed=Interpretation._open_speech_sensed_or(instance, detection.sensed),
                voice_detection=detection.voice_detection,
                segments=output.segments,
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                _listening_event_with_context(
                    _VoiceDiarizationCompletedEvent.with_data(completion),
                    event,
                    operation_id=instance._active_operation_id,
                ),
            )

    @staticmethod
    async def _run_voice_identification(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.listening.identify_voice",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="voice_identification",
            context=telemetry.event_context(event),
        ) as active:
            completion = event.data
            voice_detection = (
                completion.voice_detection
                if isinstance(
                    completion,
                    _VoiceDiarizationCompletedEventData | _VoiceDetectionCompletedEventData,
                )
                else None
            )
            if isinstance(completion, _VoiceDiarizationCompletedEventData):
                segments = completion.segments
            elif isinstance(completion, _VoiceDetectionCompletedEventData) and instance._voice_diarization is None:
                # Direct identification owns the opening VAD span only. The resulting source id is
                # then carried by every product until this VAD window closes.
                if not completion.voice_detection.segments:
                    Interpretation._dispatch_stage_failure(
                        ctx,
                        instance,
                        event,
                        stage="voice_identification",
                        message="Voice identification requires a first voiced VAD segment.",
                    )
                    return
                try:
                    segments = (
                        _voice_segment_from_detection(
                            sensed=completion.sensed,
                            segment=completion.voice_detection.segments[0],
                        ),
                    )
                except Exception as error:
                    Interpretation._dispatch_stage_failure(
                        ctx,
                        instance,
                        event,
                        stage="voice_identification",
                        message=f"Voice identification opening segment conversion failed: {error}",
                    )
                    return
                voice_detection = completion.voice_detection
            else:
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_identification",
                    message="voice identification requires a voice-diarization or voice-detection completion.",
                )
                return
            voice_identification = instance._voice_identification
            if voice_identification is None:
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_identification",
                    message="voice identification ability is not configured.",
                )
                return
            try:
                identification_input = voice.identification.InputData(segments=segments)
            except Exception as error:
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_identification",
                    message=f"Voice identification input conversion failed: {error}",
                )
                return
            output = await Interpretation._await_child_output(
                ctx,
                instance,
                event,
                child=voice_identification,
                child_input=identification_input,
                stage="voice_identification",
            )
            if output is None:
                return
            if not isinstance(output, voice.identification.OutputData):
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_identification",
                    message="Voice identification child produced invalid output.",
                )
                return
            if len(output.embeddings) != len(segments):
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="voice_identification",
                    message=(
                        "Voice identification must return exactly one embedding per voice segment; "
                        f"received {len(output.embeddings)} for {len(segments)} segments."
                    ),
                )
                return
            active.set_attribute("bot.voice.segment.count", len(segments))
            active.set_attribute("bot.voice.embedding.count", len(output.embeddings))
            identified = _VoiceIdentificationCompletedEventData(
                sensed=Interpretation._open_speech_sensed_or(instance, completion.sensed),
                voice_detection=voice_detection or voice.detection.ApplyData(),
                segments=segments,
                embeddings=output.embeddings,
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                _listening_event_with_context(
                    _VoiceIdentificationCompletedEvent.with_data(identified),
                    event,
                    operation_id=instance._active_operation_id,
                ),
            )

    @staticmethod
    async def _run_detected_speech_decoding(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        detection = event.data
        if not isinstance(detection, _VoiceDetectionCompletedEventData):
            Interpretation._dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="speech_decoding",
                message="speech decoding requires a voice-detection completion.",
            )
            return
        await Interpretation._decode_speech(
            ctx,
            instance,
            event,
            sensed=Interpretation._open_speech_sensed_or(instance, detection.sensed),
            voice_detection=detection.voice_detection,
        )

    @staticmethod
    async def _run_diarized_speech_decoding(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        completion = event.data
        if not isinstance(completion, _VoiceDiarizationCompletedEventData):
            Interpretation._dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="speech_decoding",
                message="speech decoding requires a voice-diarization completion.",
            )
            return
        await Interpretation._decode_speech(
            ctx,
            instance,
            event,
            sensed=completion.sensed,
            voice_detection=completion.voice_detection,
        )

    @staticmethod
    async def _decode_speech(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
        *,
        sensed: sensitivity.OutputData,
        voice_detection: voice.detection.ApplyData,
    ) -> None:
        with span.operation(
            "bot.listening.decode_speech",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="speech_decoding",
            context=telemetry.event_context(event),
        ) as active:
            speech_decoding = instance._speech_decoding
            if speech_decoding is None:
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="speech_decoding",
                    message="speech decoding is not configured.",
                )
                return
            utterance = Interpretation._open_speech_bytes(instance)
            decode_audio = utterance if utterance else sensed.sound.audio
            active.set_attribute("bot.audio.byte.count", len(decode_audio))
            output = await Interpretation._await_child_output(
                ctx,
                instance,
                event,
                child=speech_decoding,
                child_input=decode_audio,
                stage="speech_decoding",
            )
            if output is None:
                return
            if not isinstance(output, bytes):
                Interpretation._dispatch_stage_failure(
                    ctx,
                    instance,
                    event,
                    stage="speech_decoding",
                    message="Speech decoding child produced a non-bytes terminal.",
                )
                return
            active.set_attribute("bot.speech.decoded.byte.count", len(output))
            completion = _SpeechDecodingCompletedEventData(
                sensed=sensed,
                voice_detection=voice_detection,
                speech=output,
            )
            Interpretation._clear_open_speech(ctx, instance, event)
            _ = hsm.dispatch(
                ctx,
                instance,
                _listening_event_with_context(
                    _SpeechDecodingCompletedEvent.with_data(completion),
                    event,
                    operation_id=instance._active_operation_id,
                ),
            )

    @staticmethod
    def _has_detected_no_voice_without_sound_classification(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """No voice, no sound classifier, STT path: skip cognitive handoff (legacy)."""

        return (
            Interpretation._has_detected_no_voice(ctx, instance, event)
            and instance._sound_classification is None
            and instance._speech_decoding is not None
        )

    @staticmethod
    def _has_detected_no_voice_acoustic_speech(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """No voice and no STT: hand off silence/ambient speech observation so sticky turns can close."""

        return (
            Interpretation._has_detected_no_voice(ctx, instance, event)
            and instance._speech_decoding is None
            and instance._sound_classification is None
        )

    @staticmethod
    def _has_detected_no_voice_with_sound_classification(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """No voice with a sound classifier: try non-speech acoustic labeling first."""

        return (
            Interpretation._has_detected_no_voice(ctx, instance, event) and instance._sound_classification is not None
        )

    @staticmethod
    def _has_unlabeled_sound_drop(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """Unlabeled non-speech with STT path: drop (no conversation silence product)."""

        return Interpretation._has_unlabeled_sound(ctx, instance, event) and instance._speech_decoding is not None

    @staticmethod
    def _has_unlabeled_sound_acoustic_speech(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """Unlabeled non-speech without STT: silence observation for turn detection."""

        return Interpretation._has_unlabeled_sound(ctx, instance, event) and instance._speech_decoding is None

    @staticmethod
    def _is_audible_product(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
        """The one comparison: is enough of this left, after everything predicting it, for a turn?

        Made once, here, at the end of the pipeline and nowhere earlier — voice detection,
        diarization, and speech decoding all run on the bot's own voice and the transcript is
        produced. The only thing that changes is whether any of it becomes something for the bot
        to judge, which is what it means for a sound to be quiet rather than absent.

        The level being read was worked out when the sound arrived, not when this pipeline
        reached it. That is load-bearing: this stage can take seconds, and a level taken here
        would be a level taken after the body had stopped doing whatever produced the sound.

        A perceived level of ``None`` means nothing could be measured, and an arrival that cannot
        be shown to be quiet is heard.
        """

        del ctx
        sensed = _listening_sensed(event)
        if sensed is None or sensed.perceived_level_db is None:
            return True
        return sensed.perceived_level_db >= instance._product_threshold_db

    @staticmethod
    def _has_voice_diarization_ability(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._voice_diarization is not None

    @staticmethod
    def _has_voice_identification_without_diarization(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._voice_identification is not None and instance._voice_diarization is None

    @staticmethod
    def _has_voice_identification_ability(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._voice_identification is not None

    @staticmethod
    def _has_speech_decoding_ability(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._speech_decoding is not None

    @staticmethod
    def _has_voice_diarization_completion_and_speech_decoding(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Interpretation._has_voice_diarization_completion(ctx, instance, event)
            and instance._speech_decoding is not None
        )

    @staticmethod
    def _has_voice_diarization_completion_and_voice_identification(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Interpretation._has_voice_diarization_completion(ctx, instance, event)
            and instance._voice_identification is not None
        )

    @staticmethod
    def _has_voice_diarization_completion_without_speech_decoding(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Interpretation._has_voice_diarization_completion(ctx, instance, event) and instance._speech_decoding is None
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Interpretation",
        hsm.initial(hsm.target("/Interpretation/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Interpretation/Idle"),
            ),
        ),
        # Scored sounds queue here, one at a time, because interpreting them is serial work. The
        # queue is behind the scoring rather than in front of it, so its depth cannot change what
        # any arrival was worth.
        hsm.state(
            "Idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_sensed_sound),
                hsm.effect(_begin_operation),
                hsm.target("/Interpretation/Working/DetectingVoice"),
            ),
        ),
        # One busy region: serial queue + stage failure policy live once (HI-01), not on every stage.
        hsm.state(
            "Working",
            hsm.initial(hsm.target("/Interpretation/Working/DetectingVoice")),
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(_ListeningStageFailedEvent),
                hsm.guard(_has_listening_stage_failure),
                hsm.effect(_clear_open_speech, _dispatch_listening_failure, _clear_operation),
                hsm.target("/Interpretation/Idle"),
            ),
            hsm.state(
                "DetectingVoice",
                hsm.activity(_run_voice_detection),
                hsm.transition(
                    hsm.on(_VoiceDetectionCompletedEvent),
                    hsm.guard(_has_detected_no_voice_without_sound_classification),
                    hsm.effect(_clear_open_speech, _clear_operation),
                    hsm.target("/Interpretation/Idle"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDetectionCompletedEvent),
                    hsm.guard(_has_detected_no_voice_acoustic_speech),
                    hsm.target("/Interpretation/Working/HandingOff"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDetectionCompletedEvent),
                    hsm.guard(_has_detected_no_voice_with_sound_classification),
                    hsm.target("/Interpretation/Working/ClassifyingSound"),
                ),
                # First voice (VAD Start): open HearingSpeech and stream frames until VAD End.
                hsm.transition(
                    hsm.on(_VoiceDetectionCompletedEvent),
                    hsm.guard(_has_detected_voice_and_direct_identification),
                    hsm.target("/Interpretation/Working/IdentifyingVoiceDirect"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDetectionCompletedEvent),
                    hsm.guard(_has_detected_voice),
                    hsm.effect(_enter_hearing_speech),
                    hsm.target("/Interpretation/Working/HearingSpeech"),
                ),
            ),
            # Open speech window: scored frames keep going to VAD until sticky End.
            hsm.state(
                "HearingSpeech",
                hsm.transition(
                    hsm.on(input_event),
                    hsm.guard(_has_sensed_sound),
                    hsm.effect(_begin_operation),
                    hsm.target("/Interpretation/Working/FeedingSpeech"),
                ),
            ),
            hsm.state(
                "FeedingSpeech",
                hsm.activity(_run_voice_detection),
                hsm.transition(
                    hsm.on(_VoiceDetectionCompletedEvent),
                    hsm.guard(_has_detected_voice),
                    hsm.effect(_continue_hearing_speech),
                    hsm.target("/Interpretation/Working/HearingSpeech"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDetectionCompletedEvent),
                    hsm.guard(_has_detected_no_voice),
                    hsm.effect(_close_hearing_speech),
                    hsm.target("/Interpretation/Working/SpeechEnded"),
                ),
            ),
            # After VAD End: direct identification, optional diarize/STT on the full utterance; else products already streamed.
            hsm.choice(
                "SpeechEnded",
                hsm.transition(
                    hsm.guard(_has_voice_diarization_ability),
                    hsm.target("/Interpretation/Working/DiarizingVoice"),
                ),
                hsm.transition(
                    hsm.guard(_has_speech_decoding_ability),
                    hsm.target("/Interpretation/Working/DecodingSpeech/Detected"),
                ),
                hsm.transition(
                    hsm.effect(_clear_open_speech, _clear_operation),
                    hsm.target("/Interpretation/Idle"),
                ),
            ),
            hsm.state(
                "ClassifyingSound",
                hsm.activity(_run_sound_classification),
                hsm.transition(
                    hsm.on(_SoundClassificationCompletedEvent),
                    hsm.guard(_has_labeled_sound),
                    hsm.target("/Interpretation/Working/HandingOff"),
                ),
                hsm.transition(
                    hsm.on(_SoundClassificationCompletedEvent),
                    hsm.guard(_has_unlabeled_sound_drop),
                    hsm.effect(_clear_operation),
                    hsm.target("/Interpretation/Idle"),
                ),
                hsm.transition(
                    hsm.on(_SoundClassificationCompletedEvent),
                    hsm.guard(_has_unlabeled_sound_acoustic_speech),
                    hsm.target("/Interpretation/Working/HandingOff"),
                ),
            ),
            # Every route through interpretation ends here, and this is the only place a level
            # becomes a yes or a no. One comparison, whatever produced the product and whatever
            # reduced its level on the way — which is what keeps a second contributor from
            # needing a second threshold, or a second opinion about what quiet means.
            hsm.choice(
                "HandingOff",
                hsm.transition(
                    hsm.guard(_is_audible_product),
                    hsm.effect(_dispatch_product_cognition_input, _clear_open_speech, _clear_operation),
                    hsm.target("/Interpretation/Idle"),
                ),
                hsm.transition(
                    hsm.effect(_clear_open_speech, _clear_operation),
                    hsm.target("/Interpretation/Idle"),
                ),
            ),
            hsm.state(
                "DiarizingVoice",
                hsm.activity(_run_voice_diarization),
                hsm.transition(
                    hsm.on(_VoiceDiarizationCompletedEvent),
                    hsm.guard(_has_voice_diarization_completion_and_voice_identification),
                    hsm.target("/Interpretation/Working/IdentifyingVoice"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDiarizationCompletedEvent),
                    hsm.guard(_has_voice_diarization_completion_and_speech_decoding),
                    hsm.target("/Interpretation/Working/DecodingSpeech/Diarized"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDiarizationCompletedEvent),
                    hsm.guard(_has_voice_diarization_completion_without_speech_decoding),
                    hsm.target("/Interpretation/Working/HandingOff"),
                ),
            ),
            hsm.state(
                "IdentifyingVoiceDirect",
                # Identify the assembled VAD utterance directly, without a diarization pass.
                hsm.activity(_run_voice_identification),
                hsm.transition(
                    hsm.on(_VoiceIdentificationCompletedEvent),
                    hsm.guard(_has_voice_identification_completion),
                    hsm.effect(_enter_hearing_speech_from_identification),
                    hsm.target("/Interpretation/Working/HearingSpeech"),
                ),
            ),
            hsm.state(
                "IdentifyingVoice",
                hsm.activity(_run_voice_identification),
                hsm.transition(
                    hsm.on(_VoiceIdentificationCompletedEvent),
                    hsm.guard(_has_voice_identification_completion),
                    hsm.target("/Interpretation/Working/HandingOff"),
                ),
            ),
            hsm.state(
                "DecodingSpeech",
                hsm.initial(hsm.target("/Interpretation/Working/DecodingSpeech/Detected")),
                hsm.transition(
                    hsm.on(_SpeechDecodingCompletedEvent),
                    hsm.guard(_has_speech_decoding_completion),
                    hsm.target("/Interpretation/Working/HandingOff"),
                ),
                hsm.state(
                    "Detected",
                    hsm.activity(_run_detected_speech_decoding),
                ),
                hsm.state(
                    "Diarized",
                    hsm.activity(_run_diarized_speech_decoding),
                ),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Interpretation/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Interpretation/Idle"),
            ),
        ),
        hsm.state("degraded"),
        hsm.observe(observer),
    )


__all__ = [
    "DEFAULT_PRODUCT_THRESHOLD_DB",
    "FailedEventData",
    "Interpretation",
    "ListeningFailedEvent",
    "ListeningStage",
    "SpeechData",
    "SpeechEvent",
]
