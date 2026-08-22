"""Listening: the ear. Hears a sound, weighs it against what the body predicted, hands it on.

Two things happen to an arriving sound and they happen on very different timescales. Weighing it
against what the body is currently doing is about a *moment* and costs nothing. Working out what
it was — voice detection, direct voice identification, speech decoding — takes seconds and is somebody else's job
(:class:`~bot.abilities.listening.interpretation.Interpretation`).

Nothing sits between the ear and the prediction. That is the whole shape of this machine: a sound
arrives, it is scored while the thing that produced it is still producing, and it leaves already
scored. Interpretation queues behind that, never in front of it, so however long a decoder takes
it cannot change what an arrival was worth.
"""

from __future__ import annotations

from . import interpretation
from . import sensitivity
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

from bot.protocols import attachment

import bot
from bot.abilities import cognition
from bot import telemetry
from bot.telemetry import observer
from bot.telemetry import span
from bot.environment import SoundData, SoundEvent
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


def _has_interpretation_failure(ctx: hsm.Context, instance: "Listening", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, interpretation.FailedEventData)


class Listening(ability.Ability[SoundData, cognition.InputData]):
    """Sensory ability that may turn environment sound into ``cognition.InputEvent``.

    Public input is ``environment.sound``; the success terminal is cognitive input whose stimulus
    is the original sound or decoded speech. Scoring against the body's own predictions
    (:class:`~bot.abilities.listening.sensitivity.Sensitivity`) happens here, on arrival.
    Everything slower, including direct voice identification, is delegated to
    :class:`~bot.abilities.listening.interpretation.Interpretation`,
    whose products come back out through this ability's own terminals.
    """

    input_event: typing.ClassVar[hsm.Event[SoundData]] = SoundEvent
    output_event: typing.ClassVar[hsm.Event[cognition.InputData]] = cognition.InputEvent
    failed_event: typing.ClassVar[hsm.Event[interpretation.FailedEventData]] = interpretation.ListeningFailedEvent
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _sensitivity: sensitivity.Sensitivity
    _interpretation: interpretation.Interpretation
    _attachment_group: attachment.Group

    def __init__(
        self,
        *,
        voice_activity_classifier: voice.detection.VoiceActivityClassifier,
        sound_classifier: sound.classification.SoundClassifier | None = None,
        speech_decoder: speech.SpeechDecoder | None = None,
        voice_diarizer: voice.diarization.VoiceDiarizer | None = None,
        voice_classifier: classifying.Classifier[
            voice.identification.InputData,
            voice.identification.OutputData,
        ]
        | None = None,
        product_threshold_db: float = interpretation.DEFAULT_PRODUCT_THRESHOLD_DB,
    ) -> None:
        super().__init__()
        self._sensitivity = sensitivity.Sensitivity()
        self._interpretation = interpretation.Interpretation(
            voice_activity_classifier=voice_activity_classifier,
            sound_classifier=sound_classifier,
            speech_decoder=speech_decoder,
            voice_diarizer=voice_diarizer,
            voice_classifier=voice_classifier,
            product_threshold_db=product_threshold_db,
        )
        self._attachment_group = attachment.Group(self._sensitivity, self._interpretation)

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
                            sensitivity.ScoreData(stimulus=bot.StimulusData.from_event(event)),
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
    def _hand_to_interpretation(
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
        target = instance._interpretation
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
    def _emit_interpretation_failure(
        ctx: hsm.Context,
        instance: "Listening",
        event: hsm.Event[typing.Any],
    ) -> None:
        _ = hsm.dispatch(
            ctx,
            instance,
            ability.TerminalErrorEvent.with_data(dataclasses.replace(event, target=None)),
        )

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
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
            # Interpretation's terminals, on their way out through this ability's own.
            hsm.transition(
                hsm.on(cognition.InputEvent),
                hsm.guard(_has_interpreted_product),
                hsm.effect(_emit_interpreted_product),
            ),
            hsm.transition(
                hsm.on(interpretation.ListeningFailedEvent),
                hsm.guard(_has_interpretation_failure),
                hsm.effect(_emit_interpretation_failure),
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
                    hsm.effect(_hand_to_interpretation),
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
        hsm.observe(observer),
    )


__all__ = [
    "Listening",
]
