"""Interpretation: working out what a heard sound turns out to be.

Everything slow about hearing is here — voice detection, non-speech labelling, diarization,
speech decoding — and it is a machine of its own for one reason: it is slow, and the thing in
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
from bot.environment import SoundEvent
from bot.telemetry import observer

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
    voice_detection: voice.detection.OutputData


class _VoiceDiarizationCompletedEventData(pydantic.BaseModel):
    """Private completion payload for voice diarization inside the listening ability."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    sensed: sensitivity.OutputData
    voice_detection: voice.detection.OutputData
    diarization: voice.diarization.OutputData


class _SpeechDecodingCompletedEventData(pydantic.BaseModel):
    """Private completion payload for speech decoding inside the listening ability."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    sensed: sensitivity.OutputData
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

    sensed: sensitivity.OutputData
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
        | _SpeechDecodingCompletedEventData,
    ):
        return data.sensed
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
    instance: "Interpretation",
    event: hsm.Event[typing.Any],
    stimulus: hsm.Event[typing.Any],
) -> None:
    """Terminal handoff: ``cognition.InputEvent`` for the body owner when listening finishes.

    Stimulus is the acoustic product cognition should judge (``environment.sound`` or decoded
    speech). Provenance comes only from the event chain: the provenance the sound arrived with —
    the stamped holder's id when the sound declares one, the transducer's id otherwise — rides
    in on the scored product's envelope and rides out on the stimulus, which is how the body
    resolves which of its devices a sensory product belongs to.
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
    instance: "Interpretation",
    source: hsm.Event[typing.Any],
    failure: FailedEventData,
) -> None:
    terminal = _listening_event_with_context(instance.failed_event.with_data(failure), source)
    terminal = dataclasses.replace(terminal, source=hsm.id(instance))
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


def _dispatch_stage_failure(
    ctx: hsm.Context,
    instance: "Interpretation",
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


def _dispatch_product_cognition_input(
    ctx: hsm.Context,
    instance: "Interpretation",
    event: hsm.Event[typing.Any],
) -> None:
    """Hand off whatever product this pipeline arrived at, whichever route it came by.

    Speech-decoding product bytes (typically a UTF-8 transcript after STT) when there is a
    transcript, the original sound otherwise. When Conversation is Bot-acquired, Bot bridges a
    speech product into a Conversation Message (``text_turn`` for TextConversation,
    ``voice_turn`` for VoiceConversation with acoustic payload).
    """

    completion = event.data
    if isinstance(completion, _SpeechDecodingCompletedEventData):
        _dispatch_listening_cognition_input(
            ctx,
            instance,
            event,
            speech.SpeechDecoding.output_event.with_data(completion.speech),
        )
        return
    sensed = _listening_sensed(event)
    if sensed is None:
        raise AssertionError("listening cognition handoff requires sound on the event chain.")
    _dispatch_listening_cognition_input(ctx, instance, event, SoundEvent.with_data(sensed.sound))


def _dispatch_listening_failure(
    ctx: hsm.Context,
    instance: "Interpretation",
    event: hsm.Event[typing.Any],
) -> None:
    failure = event.data
    assert isinstance(failure, FailedEventData)
    _dispatch_listening_terminal_failure(ctx, instance, event, failure)


def _has_sensed_sound(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, sensitivity.OutputData)


def _has_detected_voice(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, _VoiceDetectionCompletedEventData) and data.voice_detection.is_voice


def _has_detected_no_voice(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, _VoiceDetectionCompletedEventData) and not data.voice_detection.is_voice


def _has_labeled_sound(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, _SoundClassificationCompletedEventData) and data.classification.is_labeled


def _has_unlabeled_sound(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, _SoundClassificationCompletedEventData) and not data.classification.is_labeled


def _has_voice_diarization_completion(
    ctx: hsm.Context,
    instance: "Interpretation",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, _VoiceDiarizationCompletedEventData)


def _has_speech_decoding_completion(
    ctx: hsm.Context,
    instance: "Interpretation",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, _SpeechDecodingCompletedEventData)


def _has_listening_stage_failure(ctx: hsm.Context, instance: "Interpretation", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, FailedEventData)


class Interpretation(ability.Ability[sensitivity.OutputData, cognition.InputData]):
    """Turns a scored sound into something the bot can judge, or into nothing.

    Input is a sound that has already been weighed against what the body predicted about it
    (:class:`~bot.abilities.listening.sensitivity.OutputData`), which is why this can take as
    long as it likes: the weighing already happened, at the moment the sound arrived.

    Stages (VAD, optional non-speech classification, optional diarization, optional STT) are
    internal. Success terminal is cognitive input whose stimulus is the original sound or decoded
    speech. No-voice audio is skipped unless an optional sound classifier assigns labels. Stage
    progress is carried only on typed private completion/failure events (HSM-COMPLETION-001).
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
    _attachment_group: attachment.Group

    def __init__(
        self,
        *,
        voice_detector: voice.detection.VoiceDetector,
        sound_classifier: sound.classification.SoundClassifier | None = None,
        speech_decoder: speech.SpeechDecoder | None = None,
        voice_diarizer: voice.diarization.VoiceDiarizer | None = None,
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
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> None:
        sensed = _listening_sensed(event)
        if sensed is None:
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="voice_detection",
                message="listening input is missing sound data.",
            )
            return
        try:
            voice_detection = await instance._voice_detection.classifier.classify(sensed.sound.audio)
        except Exception as error:
            _dispatch_stage_failure(
                ctx,
                instance,
                event,
                stage="voice_detection",
                message=str(error),
            )
            return
        completion = _VoiceDetectionCompletedEventData(sensed=sensed, voice_detection=voice_detection)
        _ = hsm.dispatch(
            ctx,
            instance,
            _listening_event_with_context(_VoiceDetectionCompletedEvent.with_data(completion), event),
        )

    @staticmethod
    async def _run_sound_classification(
        ctx: hsm.Context,
        instance: "Interpretation",
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
            classification = await sound_classification.classifier.classify(detection.sensed.sound)
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
            sensed=detection.sensed,
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
        instance: "Interpretation",
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
            diarization = await voice_diarization.classifier.classify(detection.sensed.sound.audio)
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
            sensed=detection.sensed,
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
        instance: "Interpretation",
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
        await Interpretation._decode_speech(
            ctx,
            instance,
            event,
            sensed=detection.sensed,
            voice_detection=detection.voice_detection,
            diarization=None,
        )

    @staticmethod
    async def _run_diarized_speech_decoding(
        ctx: hsm.Context,
        instance: "Interpretation",
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
        await Interpretation._decode_speech(
            ctx,
            instance,
            event,
            sensed=diarization.sensed,
            voice_detection=diarization.voice_detection,
            diarization=diarization.diarization,
        )

    @staticmethod
    async def _decode_speech(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
        *,
        sensed: sensitivity.OutputData,
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
            decoded = await speech_decoding.decoder.decode(sensed.sound.audio)
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
            sensed=sensed,
            voice_detection=voice_detection,
            speech=decoded,
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
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """No voice and no sound classifier: skip cognitive handoff."""

        return _has_detected_no_voice(ctx, instance, event) and instance._sound_classification is None

    @staticmethod
    def _has_detected_no_voice_with_sound_classification(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """No voice with a sound classifier: try non-speech acoustic labeling."""

        return _has_detected_no_voice(ctx, instance, event) and instance._sound_classification is not None

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
        return _has_voice_diarization_completion(ctx, instance, event) and instance._speech_decoding is not None

    @staticmethod
    def _has_voice_diarization_completion_without_speech_decoding(
        ctx: hsm.Context,
        instance: "Interpretation",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return _has_voice_diarization_completion(ctx, instance, event) and instance._speech_decoding is None

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
                hsm.target("/Interpretation/DetectingVoice"),
            ),
        ),
        hsm.state(
            "DetectingVoice",
            hsm.defer(input_event),
            hsm.activity(_run_voice_detection),
            hsm.transition(
                hsm.on(_VoiceDetectionCompletedEvent),
                hsm.guard(_has_detected_no_voice_without_sound_classification),
                hsm.target("/Interpretation/Idle"),
            ),
            hsm.transition(
                hsm.on(_VoiceDetectionCompletedEvent),
                hsm.guard(_has_detected_no_voice_with_sound_classification),
                hsm.target("/Interpretation/ClassifyingSound"),
            ),
            hsm.transition(
                hsm.on(_VoiceDetectionCompletedEvent),
                hsm.guard(_has_detected_voice),
                hsm.target("/Interpretation/RoutingDetectedVoice"),
            ),
            hsm.transition(
                hsm.on(_ListeningStageFailedEvent),
                hsm.guard(_has_listening_stage_failure),
                hsm.effect(_dispatch_listening_failure),
                hsm.target("/Interpretation/Idle"),
            ),
        ),
        hsm.state(
            "ClassifyingSound",
            hsm.defer(input_event),
            hsm.activity(_run_sound_classification),
            hsm.transition(
                hsm.on(_SoundClassificationCompletedEvent),
                hsm.guard(_has_labeled_sound),
                hsm.target("/Interpretation/HandingOff"),
            ),
            hsm.transition(
                hsm.on(_SoundClassificationCompletedEvent),
                hsm.guard(_has_unlabeled_sound),
                hsm.target("/Interpretation/Idle"),
            ),
            hsm.transition(
                hsm.on(_ListeningStageFailedEvent),
                hsm.guard(_has_listening_stage_failure),
                hsm.effect(_dispatch_listening_failure),
                hsm.target("/Interpretation/Idle"),
            ),
        ),
        hsm.choice(
            "RoutingDetectedVoice",
            hsm.transition(
                hsm.guard(_has_voice_diarization_ability),
                hsm.target("/Interpretation/DiarizingVoice"),
            ),
            hsm.transition(
                hsm.guard(_has_speech_decoding_ability),
                hsm.target("/Interpretation/DecodingSpeech/Detected"),
            ),
            hsm.transition(
                hsm.target("/Interpretation/HandingOff"),
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
                hsm.effect(_dispatch_product_cognition_input),
                hsm.target("/Interpretation/Idle"),
            ),
            hsm.transition(
                hsm.target("/Interpretation/Idle"),
            ),
        ),
        hsm.state(
            "DiarizingVoice",
            hsm.defer(input_event),
            hsm.activity(_run_voice_diarization),
            hsm.transition(
                hsm.on(_VoiceDiarizationCompletedEvent),
                hsm.guard(_has_voice_diarization_completion_and_speech_decoding),
                hsm.target("/Interpretation/DecodingSpeech/Diarized"),
            ),
            hsm.transition(
                hsm.on(_VoiceDiarizationCompletedEvent),
                hsm.guard(_has_voice_diarization_completion_without_speech_decoding),
                hsm.target("/Interpretation/HandingOff"),
            ),
            hsm.transition(
                hsm.on(_ListeningStageFailedEvent),
                hsm.guard(_has_listening_stage_failure),
                hsm.effect(_dispatch_listening_failure),
                hsm.target("/Interpretation/Idle"),
            ),
        ),
        hsm.state(
            "DecodingSpeech",
            hsm.initial(hsm.target("/Interpretation/DecodingSpeech/Detected")),
            hsm.transition(
                hsm.on(_SpeechDecodingCompletedEvent),
                hsm.guard(_has_speech_decoding_completion),
                hsm.target("/Interpretation/HandingOff"),
            ),
            hsm.transition(
                hsm.on(_ListeningStageFailedEvent),
                hsm.guard(_has_listening_stage_failure),
                hsm.effect(_dispatch_listening_failure),
                hsm.target("/Interpretation/Idle"),
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
]
