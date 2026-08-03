"""Sensitivity: what perception expects to hear while the body is producing sound.

This is the receiving half of an efference copy. The mouth reports what it was commanded to do
(:data:`~bot.abilities.speaking.EfferenceEvent`); this holds that report for as long as the
command lasts, and compares every sound that arrives during it against what the body predicts
its own production sounds like from where it stands.

The prediction is a *forward model* — a mapping from a command to its sensory consequences — and
it lives here rather than travelling on the copy, for the same reason it lives in the cerebellum
rather than in motor cortex: the command knows what was ordered, but only perception ever sees
what came back, so only perception can learn the mapping or notice it failing. The mouth is not
in a position to know how loud it is at the ears; that depends on the space, not on the mouth.

The quantity learned is the acoustic path gain: how much louder or quieter a sound is at the
bot's own ears than at its source. That is a property of the body's geometry and is invariant to
how loudly the bot chose to speak, which is exactly what a forward model should be. A residual —
own production arriving at a gain the model does not predict — is not noise to be discarded. It
is the bot noticing that its voice came back wrong, and it is allowed through.

What comes out is a level in dB: how loud the part of this arrival that the body could *not*
account for is. Nothing is decided here and nothing is dropped. Whether that level wins is one
comparison, made once, at the point perception becomes a product.
"""

from __future__ import annotations

from .. import ability
from .. import speaking

import dataclasses
import datetime
import typing

import hsm
import pydantic

from bot.environment import SoundData, SoundEvent
from bot.telemetry import observer

_PLAYOUT_SECONDS_ATTRIBUTE = "sensitivity_playout_seconds"
_PRODUCING_MOUTH_ATTRIBUTE = "sensitivity_producing_mouth"
_PATH_GAIN_DB_ATTRIBUTE = "sensitivity_path_gain_db"


class OutputData(pydantic.BaseModel):
    """A received sound, and how loud the part of it the body could not account for is."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "A received sound paired with its perceived level: the level, in dB, of whatever "
                "about this arrival the body's own predictions did not account for."
            ),
        },
    )

    sound: SoundData = pydantic.Field(description="The sound exactly as it was received. Never altered here.")
    perceived_level_db: float | None = pydantic.Field(
        default=None,
        description=(
            "Level in dB of the part of this arrival that nothing predicted — what perception is "
            "actually left with. With nothing predicting it, that is the whole sound, so this is "
            "the level it arrived at. Fully predicted, nothing is left and this is 0.0. Predicted "
            "but off by some margin, that margin is what survives. None means there was no level "
            "to work from, which is not the same as silence: an unmeasurable arrival cannot be "
            "shown to be quiet and must be treated as heard."
        ),
        examples=[76.5, 0.0, 12.0],
    )


OutputEvent = hsm.Event[OutputData](
    name="bot.ability.listening.sensitivity.output",
    schema=OutputData,
)


def _path_gain_db(sound: SoundData) -> float | None:
    """How much the space did to this sound between its source and here, in dB.

    Invariant to how loud the source was, which is what makes it a property of the path and
    therefore something a body can predict about its own mouth.
    """

    if sound.amplitude_db is None or sound.received_level_db is None:
        return None
    return sound.received_level_db - sound.amplitude_db


def _learned_path_gain_db(instance: "Sensitivity") -> float | None:
    value, ok = typing.cast(tuple[object, bool], instance.get(_PATH_GAIN_DB_ATTRIBUTE))
    return value if ok and isinstance(value, float) else None


def _producing_mouth(instance: "Sensitivity") -> str | None:
    value, ok = typing.cast(tuple[object, bool], instance.get(_PRODUCING_MOUTH_ATTRIBUTE))
    return value if ok and isinstance(value, str) and value else None


def _has_sound(ctx: hsm.Context, instance: "Sensitivity", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, SoundData)


def _has_efference_copy(ctx: hsm.Context, instance: "Sensitivity", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, speaking.EfferenceData)


class Sensitivity(ability.Ability[SoundData, OutputData]):
    """Front of the listening pipeline: holds the body's prediction and scores arrivals against it.

    Every sound handed here comes back out, always, with the prediction attached. Nothing is
    dropped and nothing is edited — deciding what a scored sound is worth belongs downstream,
    where perception turns into a product.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = SoundData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[SoundData]] = SoundEvent
    output_event: typing.ClassVar[hsm.Event[OutputData]] = OutputEvent

    @staticmethod
    def _remember_command(ctx: hsm.Context, instance: "Sensitivity", event: hsm.Event[typing.Any]) -> None:
        """Hold what the command commits to: which mouth, and for how long."""

        del ctx
        copy = event.data
        assert isinstance(copy, speaking.EfferenceData)
        _ = instance.set(_PLAYOUT_SECONDS_ATTRIBUTE, copy.duration)
        _ = instance.set(_PRODUCING_MOUTH_ATTRIBUTE, copy.mouth)

    @staticmethod
    def _playout_delay(
        ctx: hsm.Context,
        instance: "Sensitivity",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        """How long this window stays open: exactly as long as the commanded sound lasts.

        Read from the machine's own attribute rather than the event, because the event that
        fires this is the synthesized time event, not the copy that opened the window.
        """

        del ctx, event
        seconds, ok = typing.cast(tuple[object, bool], instance.get(_PLAYOUT_SECONDS_ATTRIBUTE))
        return datetime.timedelta(seconds=seconds if ok and isinstance(seconds, float) else 0.0)

    @staticmethod
    def _emit(
        ctx: hsm.Context,
        instance: "Sensitivity",
        event: hsm.Event[typing.Any],
        product: OutputData,
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(product),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _score_unproduced(ctx: hsm.Context, instance: "Sensitivity", event: hsm.Event[typing.Any]) -> None:
        """This arrival is no consequence of anything the body is doing, so all of it survives.

        Either nothing is being produced, or something is but this did not come off the mouth
        producing it. Own voice reaching here the first way means it took longer to come back
        than the command lasted — a delayed line, an echo, feedback — and that is real
        information about the call, which is the point of bounding the window rather than
        suppressing own audio outright.
        """

        sound = event.data
        assert isinstance(sound, SoundData)
        Sensitivity._emit(ctx, instance, event, OutputData(sound=sound, perceived_level_db=sound.received_level_db))

    @staticmethod
    def _score_against_prediction(ctx: hsm.Context, instance: "Sensitivity", event: hsm.Event[typing.Any]) -> None:
        """Compare what arrived against what the body predicts its own production sounds like.

        Only sound off the transducer the command was sent to is a candidate consequence of that
        command. Correlating the arrival against the commanded mouth is what the copy carries a
        mouth *for*, and it is not self-recognition: nothing here asks what the sound is like,
        only whether it came off the thing the body is currently driving. Anything else arriving
        while the bot talks — somebody cutting in, the far end still going — was never predicted
        and is not touched. It is also what keeps this honest about reflections and lines: a
        voice that came back off a wall or up a wire arrives from the wall or the wire, and is
        therefore news.

        For an arrival that *is* a candidate, three outcomes, and only one leaves a residual:

        * Nothing to compare (the sound carries no levels): the prediction covered every
          dimension there was to check, so nothing about it is unaccounted for.
        * Nothing learned yet: the model is uncalibrated, so it cannot be wrong. Calibrate it —
          which is what a body producing sound and listening to the result is for, and why
          suppression works on the very first utterance a body ever makes.
        * A model exists: what survives is the departure from it, in dB.

        The model is calibrated once and then left alone. Mouth-to-ear gain is a fact about the
        shape of the body, and bodies do not change shape between utterances — so a departure is
        always evidence about the world rather than about the body, and must never be allowed to
        teach the model that caught it. A body that really is rebuilt gets a new one of these.

        Do not simplify this to a tag. Carrying the command's operation id out through the mouth
        and matching it on the way back in would identify own audio in one comparison and look
        like an obvious cleanup. It would also destroy the residual: a voice off a wall or up a
        wire is the same audio and would arrive carrying the same tag, so every echo, delayed
        line, and feedback loop would be suppressed silently and there would be no way left to
        notice any of them. A tag says "this is mine". The window says "this is mine, arriving
        when mine should arrive" — and only the second can be violated. The violation is the
        information, which is the entire reason this is bounded in time rather than by identity.
        """

        sound = event.data
        assert isinstance(sound, SoundData)
        if event.source != _producing_mouth(instance):
            Sensitivity._score_unproduced(ctx, instance, event)
            return
        observed = _path_gain_db(sound)
        learned = _learned_path_gain_db(instance)
        if observed is None:
            perceived = 0.0
        elif learned is None:
            _ = instance.set(_PATH_GAIN_DB_ATTRIBUTE, observed)
            perceived = 0.0
        else:
            perceived = abs(observed - learned)
        Sensitivity._emit(ctx, instance, event, OutputData(sound=sound, perceived_level_db=perceived))

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Sensitivity",
        hsm.attribute(_PLAYOUT_SECONDS_ATTRIBUTE),
        hsm.attribute(_PRODUCING_MOUTH_ATTRIBUTE),
        hsm.attribute(_PATH_GAIN_DB_ATTRIBUTE),
        hsm.initial(hsm.target("/Sensitivity/quiet")),
        hsm.state(
            "quiet",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_sound),
                hsm.effect(_score_unproduced),
            ),
            hsm.transition(
                hsm.on(speaking.EfferenceEvent),
                hsm.guard(_has_efference_copy),
                hsm.effect(_remember_command),
                hsm.target("/Sensitivity/producing"),
            ),
        ),
        hsm.state(
            "producing",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_sound),
                hsm.effect(_score_against_prediction),
            ),
            # External self-transition: a second command while the first is still playing exits
            # and re-enters, which is what restarts the window on the new duration. Utterances
            # queue behind one another, so the later one is the one still to be heard.
            hsm.transition(
                hsm.on(speaking.EfferenceEvent),
                hsm.guard(_has_efference_copy),
                hsm.effect(_remember_command),
                hsm.target("/Sensitivity/producing"),
            ),
            # Ballistic: the window closes itself after exactly the sound it was opened for.
            # Nothing else can hold it open, so no failure here can leave a bot deaf.
            hsm.transition(
                hsm.after(_playout_delay),
                hsm.target("/Sensitivity/quiet"),
            ),
        ),
        hsm.observe(observer),
    )


__all__ = [
    "OutputData",
    "OutputEvent",
    "Sensitivity",
]
