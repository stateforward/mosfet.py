from mosfet import abilities
from mosfet.abilities.hearing import speech

import asyncio
import collections.abc
import datetime
import typing
from typing import override

import hsm
import pytest

from tests.bot.abilities.support import dispatch_ability_for_test, start_abilities_for_test
from tests.type_helpers import invalid_value


class FixedSpeechDecoder(speech.SpeechDecoder):
    @override
    async def decode(self, input: bytes) -> bytes:
        return input.upper()


class WrongSpeechDecoder(speech.SpeechDecoder):
    @override
    async def decode(self, input: bytes) -> bytes:
        del input
        return invalid_value(bytes, "not speech bytes")


class FailingSpeechDecoder(speech.SpeechDecoder):
    @override
    async def decode(self, input: bytes) -> bytes:
        del input
        raise RuntimeError("decoder unavailable")


class RecordingSpeechDecoding(speech.SpeechDecoding):
    outputs: list[bytes]
    failures: list[abilities.FailureData]

    def __init__(self, *, decoder: speech.SpeechDecoder) -> None:
        super().__init__(decoder=decoder)
        self.outputs = []
        self.failures = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, bytes)
            self.outputs.append(output)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, abilities.FailureData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0.001)


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


async def start_ability_tree(ctx: hsm.Context | None, ability: abilities.Ability[typing.Any, typing.Any]) -> None:
    await start_abilities_for_test(hsm.Context() if ctx is None else ctx, ability)


def test_speech_decoding_uses_injected_decoder() -> None:
    async def run() -> tuple[str, list[bytes]]:
        ability = RecordingSpeechDecoding(decoder=FixedSpeechDecoder())
        await start_ability_tree(None, ability)

        _ = await ability.apply(b"speech")
        await wait_until(lambda: bool(ability.outputs))
        return ability.state(), ability.outputs

    active_state, outputs = asyncio.run(run())

    assert active_state == "/RecordingSpeechDecodingLifecycle/attached/behavior/idle"
    assert outputs == [b"SPEECH"]


def test_speech_decoding_rejects_input_event_with_wrong_payload_type() -> None:
    async def run() -> tuple[str, list[bytes], list[abilities.FailureData]]:
        ability = RecordingSpeechDecoding(decoder=FixedSpeechDecoder())
        await start_ability_tree(None, ability)

        await hsm.dispatch(
            None,
            ability,
            ability.input_event.with_data("not speech bytes"),
        )
        return ability.state(), ability.outputs, ability.failures

    active_state, outputs, failures = asyncio.run(run())

    assert active_state == "/RecordingSpeechDecodingLifecycle/attached/behavior/idle"
    assert outputs == []
    assert failures == []


def test_speech_decoding_routes_wrong_output_type_to_failure() -> None:
    async def run() -> tuple[str, list[bytes], list[abilities.FailureData]]:
        ability = RecordingSpeechDecoding(decoder=WrongSpeechDecoder())
        await start_ability_tree(None, ability)

        with pytest.raises(RuntimeError, match="output schema"):
            _ = await dispatch_ability_for_test(ability, hsm.Context(), b"speech")
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.outputs, ability.failures

    active_state, outputs, failures = asyncio.run(run())

    assert active_state == "/RecordingSpeechDecodingLifecycle/attached/behavior/idle"
    assert outputs == []
    assert len(failures) == 1
    assert "output schema" in failures[0].message


def test_speech_decoding_directed_invalid_output_returns_to_terminal_operation() -> None:
    async def run() -> hsm.Event[typing.Any]:
        child = RecordingSpeechDecoding(decoder=WrongSpeechDecoder())
        ctx = hsm.Context()
        await start_ability_tree(ctx, child)
        return await abilities.run_terminal_operation(
            ctx,
            child=child,
            request=child.input_event.with_data_and_id(b"speech", "invalid-speech-decode"),
            terminals=(child.output_event, child.failed_event),
            timeout=datetime.timedelta(milliseconds=100),
        )

    terminal = asyncio.run(run())

    assert terminal.id == "invalid-speech-decode"
    assert isinstance(terminal.data, abilities.FailureData)
    assert "output schema" in terminal.data.message


def test_speech_decoding_routes_decoder_exception_to_failure() -> None:
    async def run() -> tuple[str, list[bytes], list[abilities.FailureData]]:
        ability = RecordingSpeechDecoding(decoder=FailingSpeechDecoder())
        await start_ability_tree(None, ability)

        with pytest.raises(RuntimeError, match="decoder unavailable"):
            _ = await dispatch_ability_for_test(ability, hsm.Context(), b"speech")
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.outputs, ability.failures

    active_state, outputs, failures = asyncio.run(run())

    assert active_state == "/RecordingSpeechDecodingLifecycle/attached/behavior/idle"
    assert outputs == []
    assert len(failures) == 1
    assert failures[0].message == "decoder unavailable"


def test_speech_decoding_is_concrete_ability() -> None:
    ability = speech.SpeechDecoding(decoder=FixedSpeechDecoder())

    assert isinstance(ability, abilities.Ability)
    assert issubclass(speech.SpeechDecoder, abilities.Decoder)
    assert speech.SpeechDecoding.input_data_type is bytes
    assert speech.SpeechDecoding.output_data_type is bytes
