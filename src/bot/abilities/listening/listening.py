from .. import ability
from ..hearing import sound
from ..hearing import speech
from ..hearing import voice

import dataclasses
import typing
import uuid

import hsm

from bot.protocols import attachment
import pydantic

from bot.abilities import cognition
from bot.telemetry import observer
from bot.environment import SoundData, SoundEvent

ListeningStage: typing.TypeAlias = typing.Literal[
    "voice_detection",
    "sound_classification",
    "voice_diarization",
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

    sound: SoundData
    voice_detection: voice.detection.OutputData


class _VoiceDiarizationCompletedEventData(pydantic.BaseModel):
    """Private completion payload for voice diarization inside the listening ability."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    sound: SoundData
    voice_detection: voice.detection.OutputData
    diarization: voice.diarization.OutputData


class _SpeechDecodingCompletedEventData(pydantic.BaseModel):
    """Private completion payload for speech decoding inside the listening ability."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    sound: SoundData
    voice_detection: voice.detection.OutputData
    speech: bytes
    diarization: voice.diarization.OutputData | None = None


class _SoundClassificationCompletedEventData(pydantic.BaseModel):
    """Private completion payload for non-speech sound classification inside listening."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    sound: SoundData
    voice_detection: voice.detection.OutputData
    classification: "sound.classification.OutputData"


ListeningFailedEvent = hsm.Event[FailedEventData](
    name="bot.ability.listening.failed",
    schema=FailedEventData,
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


def _listening_sound(event: hsm.Event[typing.Any]) -> SoundData | None:
    data = event.data
    if isinstance(data, SoundData):
        return data
    if isinstance(
        data,
        _VoiceDetectionCompletedEventData
        | _SoundClassificationCompletedEventData
        | _VoiceDiarizationCompletedEventData
        | _SpeechDecodingCompletedEventData,
    ):
        return data.sound
    return None


def _listening_event_with_context(
    event: hsm.Event[typing.Any],
    source: hsm.Event[typing.Any],
) -> hsm.Event[typing.Any]:
    """Copy operation id, acoustic source, and metadata along the private completion chain."""

    operation_id = source.id if source.id else None
    if operation_id is None:
        operation_id = uuid.uuid4().hex
    event = event.with_data_and_id(event.data, operation_id)
    return dataclasses.replace(
        event,
        source=source.source or event.source,
        metadata=dict(source.metadata),
    )


def _dispatch_listening_cognition_input(
    ctx: hsm.Context,
    instance: "Listening",
    event: hsm.Event[typing.Any],
    stimulus: hsm.Event[typing.Any],
) -> None:
    """Terminal handoff: ``cognition.InputEvent`` for the body owner when listening finishes.

    Stimulus is the acoustic product cognition should judge (``environment.sound`` or decoded
    speech). Provenance comes only from the event chain.
    """

    stimulus = _listening_event_with_context(stimulus, event)
    stimulus = dataclasses.replace(
        stimulus,
        source=event.source or stimulus.source,
        metadata=dict(event.metadata),
    )
    handoff = cognition.InputEvent.with_data(cognition.InputData(stimulus=stimulus))
    handoff = _listening_event_with_context(handoff, event)
    handoff = dataclasses.replace(handoff, source=hsm.id(instance), metadata=dict(stimulus.metadata))
    _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(handoff))


def _dispatch_listening_terminal_failure(
    ctx: hsm.Context,
    instance: "Listening",
    source: hsm.Event[typing.Any],
    failure: FailedEventData,
) -> None:
    terminal = _listening_event_with_context(instance.failed_event.with_data(failure), source)
    terminal = dataclasses.replace(terminal, source=hsm.id(instance))
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


def _dispatch_stage_failure(
    ctx: hsm.Context,
    instance: "Listening",
    source: hsm.Event[typing.Any],
    *,
    stage: ListeningStage,
    message: str,
) -> None:
    failure = FailedEventData(stage=stage, message=message)
    _ = hsm.dispatch(
        ctx,
        instance,
        _listening_event_with_context(_ListeningStageFailedEvent.with_data(failure), source),
    )


def _dispatch_sound_cognition_input(
    ctx: hsm.Context,
    instance: "Listening",
    event: hsm.Event[typing.Any],
) -> None:
    """Hand off the original sound when speech decoding did not produce a transcript."""

    sound = _listening_sound(event)
    if sound is None:
        raise AssertionError("listening sound cognition handoff requires sound on the event chain.")
    _dispatch_listening_cognition_input(ctx, instance, event, SoundEvent.with_data(sound))


def _dispatch_speech_cognition_input(
    ctx: hsm.Context,
    instance: "Listening",
    event: hsm.Event[typing.Any],
) -> None:
    """Hand off speech-decoding product bytes (typically UTF-8 transcript after STT).

    Speech decoding must succeed before this terminal fires. When Conversation is
    Bot-acquired, Bot bridges this product into a Conversation Message (``text_turn`` for
    TextConversation, ``voice_turn`` for VoiceConversation with acoustic payload).
    """

    completion = event.data
    assert isinstance(completion, _SpeechDecodingCompletedEventData)
    _dispatch_listening_cognition_input(
        ctx,
        instance,
        event,
        speech.SpeechDecoding.output_event.with_data(completion.speech),
    )


def _dispatch_listening_failure(
    ctx: hsm.Context,
    instance: "Listening",
    event: hsm.Event[typing.Any],
) -> None:
    failure = event.data
    assert isinstance(failure, FailedEventData)
    _dispatch_listening_terminal_failure(ctx, instance, event, failure)


def _has_listening_input(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return _listening_sound(event) is not None


def _has_detected_voice(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, _VoiceDetectionCompletedEventData) and data.voice_detection.is_voice


def _has_detected_no_voice(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, _VoiceDetectionCompletedEventData) and not data.voice_detection.is_voice


def _has_labeled_sound(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, _SoundClassificationCompletedEventData) and data.classification.is_labeled


def _has_unlabeled_sound(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, _SoundClassificationCompletedEventData) and not data.classification.is_labeled


def _has_voice_diarization_completion(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, _VoiceDiarizationCompletedEventData)


def _has_speech_decoding_completion(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, _SpeechDecodingCompletedEventData)


def _has_listening_stage_failure(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, FailedEventData)


class Listening(ability.Ability[SoundData, cognition.InputData]):
    """Sensory ability that may turn environment sound into ``cognition.InputEvent``.

    Public input is ``environment.sound``. Stages (VAD, optional non-speech classification,
    optional diarization, optional STT) are internal. Success terminal is cognitive input
    whose stimulus is the original sound or decoded speech. No-voice audio is skipped
    unless an optional sound classifier assigns labels. Stage progress is carried only on
    typed private completion/failure events (HSM-COMPLETION-001).
    """

    input_event: typing.ClassVar[hsm.Event[SoundData]] = SoundEvent
    output_event: typing.ClassVar[hsm.Event[cognition.InputData]] = cognition.InputEvent
    failed_event: typing.ClassVar[hsm.Event[FailedEventData]] = ListeningFailedEvent
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _voice_detection: voice.detection.VoiceDetection
    _sound_classification: sound.classification.SoundClassification | None
    _speech_decoding: speech.SpeechDecoding | None
    _voice_diarization: voice.diarization.VoiceDiarization | None
    _attachment_group: attachment.Group

    def __init__(
        self,
        *,
        voice_detector: voice.detection.VoiceDetector,
        sound_classifier: sound.classification.SoundClassifier | None = None,
        speech_decoder: speech.SpeechDecoder | None = None,
        voice_diarizer: voice.diarization.VoiceDiarizer | None = None,
    ) -> None:
        super().__init__()
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
        children: list[hsm.Instance] = [self._voice_detection]
        if self._sound_classification is not None:
            children.append(self._sound_classification)
        if self._voice_diarization is not None:
            children.append(self._voice_diarization)
        if self._speech_decoding is not None:
            children.append(self._speech_decoding)
        self._attachment_group = attachment.Group(*children)

    @staticmethod
    async def _run_voice_detection(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        sound = _listening_sound(event)
        if sound is None:
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="voice_detection",
                message="listening input is missing sound data.",
            )
            return
        try:
            voice_detection = await instance._voice_detection.classifier.classify(sound.audio)
        except Exception as error:
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="voice_detection",
                message=str(error),
            )
            return
        completion = _VoiceDetectionCompletedEventData(sound=sound, voice_detection=voice_detection)
        _ = hsm.dispatch(
            ctx,
            instance,
            _listening_event_with_context(_VoiceDetectionCompletedEvent.with_data(completion), event),
        )

    @staticmethod
    async def _run_sound_classification(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        detection = event.data
        if not isinstance(detection, _VoiceDetectionCompletedEventData):
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="sound_classification",
                message="sound classification requires a voice-detection completion.",
            )
            return
        sound_classification = instance._sound_classification
        if sound_classification is None:
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="sound_classification",
                message="sound classification is not configured.",
            )
            return
        try:
            # Pass full SoundData so kind/media provenance is available to classifiers
            # (e.g. KindSoundClassifier labels from sound.kind without probing codecs).
            classification = await sound_classification.classifier.classify(detection.sound)
        except Exception as error:
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="sound_classification",
                message=str(error),
            )
            return
        completion = _SoundClassificationCompletedEventData(
            sound=detection.sound,
            voice_detection=detection.voice_detection,
            classification=classification,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            _listening_event_with_context(_SoundClassificationCompletedEvent.with_data(completion), event),
        )

    @staticmethod
    async def _run_voice_diarization(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        detection = event.data
        if not isinstance(detection, _VoiceDetectionCompletedEventData):
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="voice_diarization",
                message="voice diarization requires a voice-detection completion.",
            )
            return
        voice_diarization = instance._voice_diarization
        if voice_diarization is None:
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="voice_diarization",
                message="voice diarization ability is not configured.",
            )
            return
        try:
            diarization = await voice_diarization.classifier.classify(detection.sound.audio)
        except Exception as error:
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="voice_diarization",
                message=str(error),
            )
            return
        completion = _VoiceDiarizationCompletedEventData(
            sound=detection.sound,
            voice_detection=detection.voice_detection,
            diarization=diarization,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            _listening_event_with_context(_VoiceDiarizationCompletedEvent.with_data(completion), event),
        )

    @staticmethod
    async def _run_detected_speech_decoding(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        detection = event.data
        if not isinstance(detection, _VoiceDetectionCompletedEventData):
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="speech_decoding",
                message="speech decoding requires a voice-detection completion.",
            )
            return
        await Listening._decode_speech(
            ctx,
            instance,
            event,
            sound=detection.sound,
            voice_detection=detection.voice_detection,
            diarization=None,
        )

    @staticmethod
    async def _run_diarized_speech_decoding(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        diarization = event.data
        if not isinstance(diarization, _VoiceDiarizationCompletedEventData):
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="speech_decoding",
                message="speech decoding requires a voice-diarization completion.",
            )
            return
        await Listening._decode_speech(
            ctx,
            instance,
            event,
            sound=diarization.sound,
            voice_detection=diarization.voice_detection,
            diarization=diarization.diarization,
        )

    @staticmethod
    async def _decode_speech(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
        *,
        sound: SoundData,
        voice_detection: voice.detection.OutputData,
        diarization: voice.diarization.OutputData | None,
    ) -> None:
        speech_decoding = instance._speech_decoding
        if speech_decoding is None:
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="speech_decoding",
                message="speech decoding is not configured.",
            )
            return
        try:
            speech = await speech_decoding.decoder.decode(sound.audio)
        except Exception as error:
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="speech_decoding",
                message=str(error),
            )
            return
        completion = _SpeechDecodingCompletedEventData(
            sound=sound,
            voice_detection=voice_detection,
            speech=speech,
            diarization=diarization,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            _listening_event_with_context(_SpeechDecodingCompletedEvent.with_data(completion), event),
        )

    @staticmethod
    def _has_detected_no_voice_without_sound_classification(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """No voice and no sound classifier: skip cognitive handoff."""

        return _has_detected_no_voice(ctx, instance, event) and instance._sound_classification is None

    @staticmethod
    def _has_detected_no_voice_with_sound_classification(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """No voice with a sound classifier: try non-speech acoustic labeling."""

        return _has_detected_no_voice(ctx, instance, event) and instance._sound_classification is not None

    @staticmethod
    def _has_voice_diarization_ability(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._voice_diarization is not None

    @staticmethod
    def _has_speech_decoding_ability(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._speech_decoding is not None

    @staticmethod
    def _has_voice_diarization_completion_and_speech_decoding(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_voice_diarization_completion(ctx, instance, event) and instance._speech_decoding is not None

    @staticmethod
    def _has_voice_diarization_completion_without_speech_decoding(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_voice_diarization_completion(ctx, instance, event) and instance._speech_decoding is None

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Listening",
        hsm.initial(hsm.target("/Listening/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Listening/Listening"),
            ),
        ),
        hsm.state(
            "Listening",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_listening_input),
                hsm.target("/Listening/DetectingVoice"),
            ),
        ),
        hsm.state(
            "DetectingVoice",
            hsm.defer(input_event),
            hsm.activity(_run_voice_detection),
            hsm.transition(
                hsm.on(_VoiceDetectionCompletedEvent),
                hsm.guard(_has_detected_no_voice_without_sound_classification),
                hsm.target("/Listening/Listening"),
            ),
            hsm.transition(
                hsm.on(_VoiceDetectionCompletedEvent),
                hsm.guard(_has_detected_no_voice_with_sound_classification),
                hsm.target("/Listening/ClassifyingSound"),
            ),
            hsm.transition(
                hsm.on(_VoiceDetectionCompletedEvent),
                hsm.guard(_has_detected_voice),
                hsm.target("/Listening/RoutingDetectedVoice"),
            ),
            hsm.transition(
                hsm.on(_ListeningStageFailedEvent),
                hsm.guard(_has_listening_stage_failure),
                hsm.effect(_dispatch_listening_failure),
                hsm.target("/Listening/Listening"),
            ),
        ),
        hsm.state(
            "ClassifyingSound",
            hsm.defer(input_event),
            hsm.activity(_run_sound_classification),
            hsm.transition(
                hsm.on(_SoundClassificationCompletedEvent),
                hsm.guard(_has_labeled_sound),
                hsm.effect(_dispatch_sound_cognition_input),
                hsm.target("/Listening/Listening"),
            ),
            hsm.transition(
                hsm.on(_SoundClassificationCompletedEvent),
                hsm.guard(_has_unlabeled_sound),
                hsm.target("/Listening/Listening"),
            ),
            hsm.transition(
                hsm.on(_ListeningStageFailedEvent),
                hsm.guard(_has_listening_stage_failure),
                hsm.effect(_dispatch_listening_failure),
                hsm.target("/Listening/Listening"),
            ),
        ),
        hsm.choice(
            "RoutingDetectedVoice",
            hsm.transition(
                hsm.guard(_has_voice_diarization_ability),
                hsm.target("/Listening/DiarizingVoice"),
            ),
            hsm.transition(
                hsm.guard(_has_speech_decoding_ability),
                hsm.target("/Listening/DecodingSpeech/Detected"),
            ),
            hsm.transition(
                hsm.effect(_dispatch_sound_cognition_input),
                hsm.target("/Listening/Listening"),
            ),
        ),
        hsm.state(
            "DiarizingVoice",
            hsm.defer(input_event),
            hsm.activity(_run_voice_diarization),
            hsm.transition(
                hsm.on(_VoiceDiarizationCompletedEvent),
                hsm.guard(_has_voice_diarization_completion_and_speech_decoding),
                hsm.target("/Listening/DecodingSpeech/Diarized"),
            ),
            hsm.transition(
                hsm.on(_VoiceDiarizationCompletedEvent),
                hsm.guard(_has_voice_diarization_completion_without_speech_decoding),
                hsm.effect(_dispatch_sound_cognition_input),
                hsm.target("/Listening/Listening"),
            ),
            hsm.transition(
                hsm.on(_ListeningStageFailedEvent),
                hsm.guard(_has_listening_stage_failure),
                hsm.effect(_dispatch_listening_failure),
                hsm.target("/Listening/Listening"),
            ),
        ),
        hsm.state(
            "DecodingSpeech",
            hsm.initial(hsm.target("/Listening/DecodingSpeech/Detected")),
            hsm.transition(
                hsm.on(_SpeechDecodingCompletedEvent),
                hsm.guard(_has_speech_decoding_completion),
                hsm.effect(_dispatch_speech_cognition_input),
                hsm.target("/Listening/Listening"),
            ),
            hsm.transition(
                hsm.on(_ListeningStageFailedEvent),
                hsm.guard(_has_listening_stage_failure),
                hsm.effect(_dispatch_listening_failure),
                hsm.target("/Listening/Listening"),
            ),
            hsm.state(
                "Detected",
                hsm.defer(input_event),
                hsm.activity(_run_detected_speech_decoding),
            ),
            hsm.state(
                "Diarized",
                hsm.defer(input_event),
                hsm.activity(_run_diarized_speech_decoding),
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
                hsm.target("/Listening/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Listening/Listening"),
            ),
        ),
        hsm.state("degraded"),
        hsm.observe(observer),
    )


__all__ = [
    "ListeningFailedEvent",
    "FailedEventData",
    "Listening",
    "ListeningStage",
]
