"""Listening: the ear. Hears a sound, weighs it against what the body predicted, hands it on.

Two things happen to an arriving sound and they happen on very different timescales. Weighing it
against what the body is currently doing is about a *moment* and costs nothing. Working out what
it was — voice detection, direct voice identification, speech decoding — takes seconds and is somebody else's job:
:class:`~mosfet.abilities.listening.interpretation.Interpretation` when the ear hears locally, or
:class:`~mosfet.abilities.listening.transcription.Transcription` when it streams to speech-to-text sessions.

Nothing sits between the ear and the prediction. That is the whole shape of this machine: a sound
arrives, it is scored while the thing that produced it is still producing, and it leaves already
scored. Interpretation queues behind that, never in front of it, so however long a decoder takes
it cannot change what an arrival was worth.
"""

from __future__ import annotations

from . import enrollment
from . import interpretation
from . import sensitivity
from . import transcription as streaming
from .. import ability
from .. import classifying
from ..hearing import sound
from ..hearing import speech
from ..hearing import voice

import dataclasses
import datetime
import typing
import uuid

import hsm

from mosfet.protocols import attachment

import mosfet
from mosfet.abilities import cognition
from mosfet.abilities import identification
from mosfet import telemetry
from mosfet.telemetry import span
from mosfet.environment import SoundData, SoundEvent
from ..speaking import EfferenceData, EfferenceEvent

_STAGE_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)
_SensitivityCompletedEvent = hsm.Event[sensitivity.OutputData](
    name="bot.ability.listening.sensitivity.completed",
    kind=hsm.CompletionEventKind,
    schema=sensitivity.OutputData,
)
_SensitivityFailedEvent = hsm.Event[interpretation.FailedEventData](
    name="bot.ability.listening.sensitivity.failed",
    kind=hsm.ErrorEventKind,
    schema=interpretation.FailedEventData,
)


def _has_listening_input(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, SoundData)


def _has_efference_copy(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, EfferenceData)


def _has_interpreted_product(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, cognition.InputData)


def _has_hearing_failure(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, interpretation.FailedEventData)


class Listening(ability.Ability[SoundData, cognition.InputData]):
    _attachment_group: attachment.Group | None
    """Sensory ability that may turn environment sound into ``cognition.InputEvent``.

    Public input is ``environment.sound``; the success terminal is cognitive input whose stimulus
    is the original sound or decoded speech. Scoring against the body's own predictions
    (:class:`~mosfet.abilities.listening.sensitivity.Sensitivity`) happens here, on arrival.
    Everything slower, including direct voice identification, is delegated to
    :class:`~mosfet.abilities.listening.interpretation.Interpretation` or, on the streaming path,
    :class:`~mosfet.abilities.listening.transcription.Transcription`, whose products come back out
    through this ability's own terminals.
    """

    input_event: typing.ClassVar[hsm.Event[SoundData]] = SoundEvent
    output_event: typing.ClassVar[hsm.Event[cognition.InputData]] = cognition.InputEvent
    failed_event: typing.ClassVar[hsm.Event[interpretation.FailedEventData]] = interpretation.ListeningFailedEvent
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _sensitivity: sensitivity.Sensitivity
    # The slow half: Interpretation (local voice detection and decoding) or Transcription (streaming sessions).
    _hearing: ability.Ability[sensitivity.OutputData, cognition.InputData]
    # The voice registry this ear owns (its group starts, attaches and stops it), or None when it does not
    # identify voices.
    _registry: identification.Registry | None

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        voice_activity_classifier: voice.detection.VoiceActivityClassifier | None = None,
        transcription: streaming.SessionFactory | None = None,
        sound_classifier: sound.classification.SoundClassifier | None = None,
        speech_decoder: speech.SpeechDecoder | None = None,
        voice_diarizer: voice.diarization.VoiceDiarizer | None = None,
        voice_classifier: classifying.Classifier[
            voice.identification.InputData,
            voice.identification.OutputData,
        ]
        | None = None,
        registry: identification.Registry | None = None,
        affect_classifier: voice.AffectClassifier | None = None,
        recognize_timeout: datetime.timedelta = interpretation.DEFAULT_RECOGNIZE_TIMEOUT,
        product_threshold_db: float = interpretation.DEFAULT_PRODUCT_THRESHOLD_DB,
        overlap_min_voiced: datetime.timedelta = interpretation.DEFAULT_OVERLAP_MIN_VOICED,
    ) -> None:
        """Hear through the given classifiers, or through streaming speech-to-text sessions.

        Exactly one hearing path is composed: ``voice_activity_classifier`` hears locally (voice
        detection, then the optional sound classifier, diarizer and speech decoder), or
        ``transcription`` makes one unstarted streaming session (``hearing.speech.stream``) per sound
        source and the hearing model's own end of turn makes each text speech product. The voice
        identification arguments serve either path.

        ``voice_classifier`` and ``registry`` identify voices and come together: ``registry`` is an
        unstarted ``identification.Registry`` constructed with ``kind="voice"``, which this ear owns
        from here on (its group starts it, attaches it for the ear's attached lifetime, and stops
        it). ``recognize_timeout`` bounds each wait for the registry's answer. ``overlap_min_voiced``
        is how long another voice must have been going while the bot's mouth is producing before the
        ear states it as a ``listening.OverlapEvent`` fact. ``affect_classifier`` hears the tone each
        text speech product was said in (``SpeechData.affect``); on the local path it needs
        ``speech_decoder``, since only decoded speech is text.

        Raises:
            ValueError: If not exactly one of ``voice_activity_classifier`` and ``transcription`` is
                given, ``transcription`` is combined with ``sound_classifier`` or ``speech_decoder``
                (local stages), only one of ``voice_classifier`` and ``registry`` is given, or as
                ``Interpretation`` or ``Transcription`` raises for its own arguments (including a
                non-positive ``overlap_min_voiced``).
        """

        super().__init__(id=id, clock=clock)
        if (voice_activity_classifier is None) == (transcription is None):
            raise ValueError("exactly one of voice_activity_classifier and transcription is required.")
        self._sensitivity = sensitivity.Sensitivity()
        if voice_activity_classifier is not None:
            self._hearing = interpretation.Interpretation(
                voice_activity_classifier=voice_activity_classifier,
                sound_classifier=sound_classifier,
                speech_decoder=speech_decoder,
                voice_diarizer=voice_diarizer,
                voice_classifier=voice_classifier,
                registry=registry,
                affect_classifier=affect_classifier,
                recognize_timeout=recognize_timeout,
                product_threshold_db=product_threshold_db,
                overlap_min_voiced=overlap_min_voiced,
            )
        else:
            assert transcription is not None
            if sound_classifier is not None or speech_decoder is not None:
                raise ValueError("sound_classifier and speech_decoder hear locally; transcription streams instead.")
            self._hearing = streaming.Transcription(
                session=transcription,
                voice_diarizer=voice_diarizer,
                voice_classifier=voice_classifier,
                registry=registry,
                affect_classifier=affect_classifier,
                recognize_timeout=recognize_timeout,
                product_threshold_db=product_threshold_db,
            )
        self._registry = registry
        owned = (self._sensitivity, self._hearing, *(() if registry is None else (registry,)))
        self._attachment_group = attachment.Group(*owned)

    @staticmethod
    def _has_sensed_sound(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, sensitivity.OutputData)

    @staticmethod
    def _has_sensitivity_failure(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, interpretation.FailedEventData)

    @staticmethod
    def _emit_sensitivity_failure(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, interpretation.FailedEventData)
        terminal = dataclasses.replace(
            instance.failed_event.with_data_and_id(data, event.id),
            source=hsm.id(instance),
            target=None,
            metadata=dict(event.metadata),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _forward_efference_copy(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Carry the mouth's report to the stage that holds the body's prediction.

        Wiring an effector's copy to the sensors is a nerve, not a judgment: nothing here reads
        the payload, ranks it, or decides anything about it.
        """

        target = instance._sensitivity
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(event, target=hsm.id(target), metadata=dict(event.metadata)),
        )

    @staticmethod
    def _has_enrollment(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return instance._registry is not None and isinstance(event.data, enrollment.EnrollData)

    @staticmethod
    def _forward_enrollment(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> None:
        """Carry the mouth's copy of the bot's own voice to the stage that identifies voices.

        A nerve like the efference copy: nothing here reads, ranks or decides anything about it.
        """

        target = instance._hearing
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(event, target=hsm.id(target), metadata=dict(event.metadata)),
        )

    @staticmethod
    async def _run_sensitivity(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Score an arriving sound against whatever the body predicts about it right now.

        The only thing this ability ever waits on, and it waits for microseconds: scoring is a
        transition effect on a machine holding one prediction, with no I/O in it. That is why the
        ear can afford to be busy here and nowhere else.
        """

        sound = event.data
        assert isinstance(sound, SoundData)
        with span.operation(
            "bot.listening.sense",
            scope="bot.abilities.listening",
            component="listening",
            stage="sensitivity",
            context=telemetry.event_context(event),
        ) as active:
            # The first span an arriving sound gets: a trace that has this and nothing after it
            # says the sound reached the ear and stopped there.
            active.set_attribute("bot.audio.byte.count", len(sound.audio))
            operation_id = event.id if event.id else uuid.uuid4().hex
            child_operation_id = f"{operation_id}:sensitivity:{uuid.uuid4().hex}"
            stage = instance._sensitivity
            try:
                terminal = await ability.run_terminal_operation(
                    instance.context(),
                    child=stage,
                    request=dataclasses.replace(
                        sensitivity.ScoreEvent.with_data_and_id(
                            sensitivity.ScoreData(stimulus=mosfet.StimulusData.from_event(event)),
                            child_operation_id,
                        ),
                        metadata=dict(event.metadata),
                    ),
                    terminals=(stage.output_event, stage.failed_event),
                    timeout=_STAGE_OPERATION_TIMEOUT,
                )
            except TimeoutError:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    telemetry.inject_context(
                        dataclasses.replace(
                            _SensitivityFailedEvent.with_data_and_id(
                                interpretation.FailedEventData(
                                    stage="sensitivity",
                                    message="Listening sensitivity timed out.",
                                ),
                                operation_id,
                            ),
                            source=event.source,
                            metadata=dict(event.metadata),
                        )
                    ),
                )
                return
            sensed = terminal.data
            if not isinstance(sensed, sensitivity.OutputData):
                failure = (
                    interpretation.FailedEventData.from_ability_failure(stage="sensitivity", failure=sensed)
                    if isinstance(sensed, ability.FailureData)
                    else interpretation.FailedEventData(
                        stage="sensitivity",
                        message="Listening sensitivity failed.",
                    )
                )
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    telemetry.inject_context(
                        dataclasses.replace(
                            _SensitivityFailedEvent.with_data_and_id(failure, operation_id),
                            source=event.source,
                            metadata=dict(event.metadata),
                        )
                    ),
                )
                return
            active.set_attribute("bot.sound.level.scored", sensed.perceived_level_db is not None)
            _ = hsm.dispatch(
                ctx,
                instance,
                telemetry.inject_context(
                    dataclasses.replace(
                        _SensitivityCompletedEvent.with_data_and_id(sensed, operation_id),
                        source=event.source,
                        metadata=dict(event.metadata),
                    )
                ),
            )

    @staticmethod
    def _hand_to_hearing(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Pass the scored sound on, and go back to listening.

        The transducer the sound came off rides along on the envelope: it is how the body works
        out which of its devices a sensory product belongs to, and nothing downstream can
        reconstruct it.
        """

        sensed = event.data
        assert isinstance(sensed, sensitivity.OutputData)
        target = instance._hearing
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                target.input_event.with_data_and_id(sensed, event.id or uuid.uuid4().hex),
                source=event.source,
                target=hsm.id(target),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _emit_interpreted_product(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        """What interpretation made of a sound is what this ability heard: re-emit it as ours.

        Interpretation is this ability's own slow half, not a peer with its own audience, so its
        products leave through the terminal the owner is already attached to.
        """

        _ = hsm.dispatch(
            ctx,
            instance,
            ability.TerminalOutputEvent.with_data(dataclasses.replace(event, target=None)),
        )

    @staticmethod
    def _has_learned_own_voice(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
        """The reply comes from this ability's own hearing stage, the one it handed the copy to."""

        del ctx
        return isinstance(event.data, enrollment.EnrolledData) and event.source == hsm.id(instance._hearing)

    @staticmethod
    def _tell_enroller(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> None:
        """Which id the bot's own voice has, from the stage that learned it, to whoever sent the copy.

        The stage is this ability's own and is not addressable from the enroller's scope, so its reply
        leaves through this ability's terminal, directed at the enroller it names.
        """

        _ = hsm.dispatch(
            ctx,
            instance,
            ability.TerminalOutputEvent.with_data(dataclasses.replace(event, metadata=dict(event.metadata))),
        )

    @staticmethod
    def _emit_hearing_failure(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        _ = hsm.dispatch(
            ctx,
            instance,
            ability.TerminalErrorEvent.with_data(dataclasses.replace(event, target=None)),
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
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
                hsm.target("/Listening/Perceiving"),
            ),
        ),
        hsm.state(
            "Perceiving",
            hsm.initial(hsm.target("/Listening/Perceiving/Listening")),
            # The body's report of what it is producing reaches the prediction from wherever
            # perception happens to be — a mouth does not wait for the ears to be free. Internal:
            # the report modulates perception, it does not interrupt it.
            hsm.transition(
                hsm.on(EfferenceEvent),
                hsm.guard(_has_efference_copy),
                hsm.effect(_forward_efference_copy),
            ),
            # The mouth's copy of the bot's own voice, to the stage that learns voices. An ear that
            # does not identify voices refuses it in the guard, so the copy is ignored.
            hsm.transition(
                hsm.on(enrollment.EnrollEvent),
                hsm.guard(_has_enrollment),
                hsm.effect(_forward_enrollment),
            ),
            # Interpretation's terminals, on their way out through this ability's own.
            hsm.transition(
                hsm.on(cognition.InputEvent),
                hsm.guard(_has_interpreted_product),
                hsm.effect(_emit_interpreted_product),
            ),
            hsm.transition(
                hsm.on(interpretation.ListeningFailedEvent),
                hsm.guard(_has_hearing_failure),
                hsm.effect(_emit_hearing_failure),
            ),
            # Which id the bot's own voice has, on its way to the enroller that sent the copy.
            hsm.transition(
                hsm.on(enrollment.EnrolledEvent),
                hsm.guard(_has_learned_own_voice),
                hsm.effect(_tell_enroller),
            ),
            hsm.state(
                "Listening",
                hsm.transition(
                    hsm.on(input_event),
                    hsm.guard(_has_listening_input),
                    hsm.target("/Listening/Perceiving/Sensing"),
                ),
            ),
            # The only stop between the ear and the prediction, and it is measured in
            # microseconds. A sound is worth whatever it was worth when it arrived; anything slow
            # standing here would make that a fact about the queue instead.
            hsm.state(
                "Sensing",
                hsm.defer(input_event),
                hsm.activity(_run_sensitivity),
                hsm.transition(
                    hsm.on(_SensitivityCompletedEvent),
                    hsm.guard(_has_sensed_sound),
                    hsm.effect(_hand_to_hearing),
                    hsm.target("/Listening/Perceiving/Listening"),
                ),
                hsm.transition(
                    hsm.on(_SensitivityFailedEvent),
                    hsm.guard(_has_sensitivity_failure),
                    hsm.effect(_emit_sensitivity_failure),
                    hsm.target("/Listening/Perceiving/Listening"),
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
                hsm.target("/Listening/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Listening/Perceiving"),
            ),
        ),
        hsm.state("degraded"),
    )


__all__ = [
    "Listening",
]
