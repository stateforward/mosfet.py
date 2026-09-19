from ... import ability
from ... import encoding

import dataclasses
import typing

import hsm
import mosfet


_SpeechEncodingApplyCompletedEvent = hsm.Event[bytes](
    name="bot.ability.vocal.speech.encoding.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=bytes,
)
_SpeechEncodingApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.vocal.speech.encoding.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _has_speech_encoding_input(
    ctx: hsm.Context,
    instance: "SpeechEncoding",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, bytes)


def _has_speech_encoding_output(
    ctx: hsm.Context,
    instance: "SpeechEncoding",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, bytes)


def _has_invalid_speech_encoding_output(
    ctx: hsm.Context,
    instance: "SpeechEncoding",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, bytes)


def _has_speech_encoding_failure(
    ctx: hsm.Context,
    instance: "SpeechEncoding",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


class SpeechEncoding(encoding.Encoding[bytes, bytes]):
    """Ability to encode bytes to verbal speech output."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = bytes
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = bytes
    input_event: typing.ClassVar[hsm.Event[bytes]] = hsm.Event[bytes](
        name="bot.ability.vocal.speech.encoding.input",
        schema=bytes,
    )
    output_event: typing.ClassVar[hsm.Event[bytes]] = hsm.Event[bytes](
        name="bot.ability.vocal.speech.encoding.output",
        schema=bytes,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _SpeechEncodingApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _SpeechEncodingApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_speech_encoding_output_failure(
        ctx: hsm.Context,
        instance: "SpeechEncoding",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(message="SpeechEncoding produced output that does not match its output schema.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "SpeechEncoding",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_speech_encoding_input),
                hsm.target("../applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(encoding.Encoding._run_behavior_activity),
            hsm.transition(
                hsm.on(_SpeechEncodingApplyCompletedEvent),
                hsm.guard(_has_speech_encoding_output),
                hsm.effect(encoding.Encoding._dispatch_encoding_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_SpeechEncodingApplyCompletedEvent),
                hsm.guard(_has_invalid_speech_encoding_output),
                hsm.effect(_dispatch_invalid_speech_encoding_output_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_SpeechEncodingApplyFailedEvent),
                hsm.guard(_has_speech_encoding_failure),
                hsm.effect(encoding.Encoding._dispatch_encoding_failure),
                hsm.target("../idle"),
            ),
        ),
    )
