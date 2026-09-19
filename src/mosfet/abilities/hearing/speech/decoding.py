from ... import ability
from ... import decoding

import abc
import dataclasses
import typing

import hsm
import mosfet


class SpeechDecoder(decoding.Decoder[bytes, bytes], abc.ABC):
    """Decoder that converts speech input bytes into normalized speech bytes."""


_SpeechDecodingApplyCompletedEvent = hsm.Event[bytes](
    name="bot.ability.hearing.speech.decoding.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=bytes,
)
_SpeechDecodingApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.hearing.speech.decoding.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _has_speech_decoding_input(
    ctx: hsm.Context,
    instance: "SpeechDecoding",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, bytes)


def _has_speech_decoding_output(
    ctx: hsm.Context,
    instance: "SpeechDecoding",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, bytes)


def _has_invalid_speech_decoding_output(
    ctx: hsm.Context,
    instance: "SpeechDecoding",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, bytes)


def _has_speech_decoding_failure(
    ctx: hsm.Context,
    instance: "SpeechDecoding",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


class SpeechDecoding(decoding.Decoding[bytes, bytes]):
    """Ability to decode speech bytes."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = bytes
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = bytes
    input_event: typing.ClassVar[hsm.Event[bytes]] = hsm.Event[bytes](
        name="bot.ability.hearing.speech.decoding.input",
        schema=bytes,
    )
    output_event: typing.ClassVar[hsm.Event[bytes]] = hsm.Event[bytes](
        name="bot.ability.hearing.speech.decoding.output",
        schema=bytes,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _SpeechDecodingApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _SpeechDecodingApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_speech_decoding_output_failure(
        ctx: hsm.Context,
        instance: "SpeechDecoding",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(message="SpeechDecoding produced output that does not match its output schema.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "SpeechDecoding",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_speech_decoding_input),
                hsm.target("../applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(decoding.Decoding._run_behavior_activity),
            hsm.transition(
                hsm.on(_SpeechDecodingApplyCompletedEvent),
                hsm.guard(_has_speech_decoding_output),
                hsm.effect(decoding.Decoding._dispatch_decoding_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_SpeechDecodingApplyCompletedEvent),
                hsm.guard(_has_invalid_speech_decoding_output),
                hsm.effect(_dispatch_invalid_speech_decoding_output_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_SpeechDecodingApplyFailedEvent),
                hsm.guard(_has_speech_decoding_failure),
                hsm.effect(decoding.Decoding._dispatch_decoding_failure),
                hsm.target("../idle"),
            ),
        ),
    )


__all__ = ["SpeechDecoder", "SpeechDecoding"]
