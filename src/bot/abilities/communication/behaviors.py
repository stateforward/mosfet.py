"""Trusted Communication behavior seeds.

Communication's SpeechHeard seed is native HSM code because it wires a typed Listening
product into a typed Communication input. Learned behaviors remain Starlark inventory
records and are deliberately not used for this raw-media handoff.
"""

from __future__ import annotations

import bot
from bot.abilities import ability
from bot.abilities import cognition
from bot.abilities import listening
from bot.abilities import processing
from bot.behavior import seed

from . import communication

import dataclasses
import typing

import hsm


SPEECH_HEARD_NAME = "SpeechHeard"
SPEECH_HEARD_TRIGGERS: tuple[str, ...] = (listening.SpeechEvent.name,)

_SpeechHeardInputEvent = hsm.Event[cognition.InputData](
    name="bot.behavior.speech_heard.input",
    schema=cognition.InputData,
)
_SpeechHeardOutputEvent = hsm.Event[processing.OutputData](
    name="bot.behavior.speech_heard.output",
    schema=processing.OutputData,
)


def _speech_product(input_data: object) -> tuple[hsm.Event[typing.Any], listening.SpeechData] | None:
    if not isinstance(input_data, cognition.InputData):
        return None
    stimulus = input_data.stimulus
    if not isinstance(stimulus, hsm.Event) or not isinstance(stimulus.data, listening.SpeechData):
        return None
    speech = stimulus.data
    if not speech.source_ids:
        return None
    return stimulus, speech


class SpeechHeard(ability.Ability[cognition.InputData, processing.OutputData]):
    """Admit identified Listening speech to Communication through a typed HSM wire.

    The input is a ``cognition.InputData`` whose stimulus is ``listening.SpeechEvent``
    with non-empty ``source_ids``. The output is one typed ``communication.InputEvent``
    selection carrying the original audio/text, identity set, sample rate in hertz,
    channel count, and causal parent. The enclosing Autonomy owns this fresh Ability
    from candidate attach through detach; the model is reusable and deliberately does
    not install an observer. Callers opt into observation at their runtime boundary.
    Factory, adapter, attach, and detach failures are reported by the shared Autonomy
    lifecycle. The HSM is synchronous apart from the normal dispatch boundary and each
    materialization is single-run; concurrent runs must use separate instances.
    """

    input_event: typing.ClassVar[hsm.Event[cognition.InputData]] = _SpeechHeardInputEvent
    output_event: typing.ClassVar[hsm.Event[processing.OutputData]] = _SpeechHeardOutputEvent

    @staticmethod
    def _has_admissible_speech(
        ctx: hsm.Context,
        instance: "SpeechHeard",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return _speech_product(event.data) is not None

    @staticmethod
    def _emit_output(
        ctx: hsm.Context,
        instance: "SpeechHeard",
        event: hsm.Event[typing.Any],
        output: processing.OutputData,
    ) -> None:
        output_event = dataclasses.replace(
            instance.output_event.with_data(output),
            id=processing.active_operation_id(instance) or event.id or None,
            source=hsm.id(instance),
            target="",
            metadata=dict(event.metadata),
        )
        # The Ability terminal forwards the declared output event to the generic
        # CandidateRun owner with the normal attachment provenance.
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(output_event))

    @staticmethod
    def _emit_communication_input(
        ctx: hsm.Context,
        instance: "SpeechHeard",
        event: hsm.Event[typing.Any],
    ) -> None:
        product = _speech_product(event.data)
        assert product is not None
        stimulus, speech = product
        turn = communication.TurnData(
            source_ids=speech.source_ids,
            target_ids=frozenset(),
            content=speech.content,
            content_type=speech.content_type,
            sample_rate_hz=speech.sample_rate_hz,
            channels=speech.channels,
            parent=bot.StimulusData.from_event(stimulus),
        )
        SpeechHeard._emit_output(
            ctx,
            instance,
            event,
            processing.OutputData(
                events=(
                    processing.SelectedEvent(
                        event=communication.InputEvent.name,
                        data=turn,
                        reason="seeded speech admit via communication",
                    ),
                )
            ),
        )

    @staticmethod
    def _emit_unhandled(
        ctx: hsm.Context,
        instance: "SpeechHeard",
        event: hsm.Event[typing.Any],
    ) -> None:
        SpeechHeard._emit_output(ctx, instance, event, processing.OutputData(handled=False))

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        SPEECH_HEARD_NAME,
        hsm.initial(hsm.target(f"/{SPEECH_HEARD_NAME}/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(_SpeechHeardInputEvent),
                hsm.guard(_has_admissible_speech),
                hsm.effect(_emit_communication_input),
            ),
            hsm.transition(
                hsm.on(_SpeechHeardInputEvent),
                hsm.effect(_emit_unhandled),
            ),
        ),
    )


def _typed_input(input_data: object) -> object:
    return input_data


def speech_heard_seed() -> seed.Seed:
    """Return the trusted native wire for identified Listening speech.

    ``factory`` returns a fresh :class:`SpeechHeard` machine for each Autonomy candidate
    run and ``input_adapter`` preserves the typed ``cognition.InputData`` unchanged.
    A matching speech product must carry at least one ``source_id``; otherwise this seed
    completes unhandled and the learned inventory/next cognition stage may continue.
    The returned descriptor does not start or observe a machine. Autonomy owns each
    materialized instance, detaches it before retiring the candidate, and converts factory,
    input, or HSM failure into its typed failure terminal. No time or byte-size unit is
    implied by this factory; audio sample-rate and channel units remain on ``SpeechData``.
    """

    return seed.Seed(
        name=SPEECH_HEARD_NAME,
        triggers=SPEECH_HEARD_TRIGGERS,
        factory=SpeechHeard,
        input_adapter=_typed_input,
    )


__all__ = [
    "SPEECH_HEARD_NAME",
    "SPEECH_HEARD_TRIGGERS",
    "SpeechHeard",
    "speech_heard_seed",
]
