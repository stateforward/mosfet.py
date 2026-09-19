from mosfet import abilities
from mosfet.abilities.vocal import speech

import asyncio
import collections.abc
import typing
from typing import override

import hsm
import pytest

from tests.bot.abilities.support import dispatch_ability_for_test, start_abilities_for_test
from tests.type_helpers import invalid_value


class FixedSpeechEncoder(abilities.Encoder[bytes, bytes]):
    @override
    async def encode(self, input: bytes) -> bytes:
        return b"encoded:" + input


class WrongSpeechEncoder(abilities.Encoder[bytes, bytes]):
    @override
    async def encode(self, input: bytes) -> bytes:
        del input
        return invalid_value(bytes, "not speech bytes")


class FailingSpeechEncoder(abilities.Encoder[bytes, bytes]):
    @override
    async def encode(self, input: bytes) -> bytes:
        del input
        raise RuntimeError("encoder unavailable")


class RecordingSpeechEncoding(speech.SpeechEncoding):
    outputs: list[bytes]
    failures: list[abilities.FailureData]

    def __init__(self, *, encoder: abilities.Encoder[bytes, bytes]) -> None:
        super().__init__(encoder=encoder)
        self.outputs = []
        self.failures = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
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


def test_speech_encoding_uses_injected_encoder() -> None:
    async def run() -> tuple[str, list[bytes]]:
        ability = RecordingSpeechEncoding(encoder=FixedSpeechEncoder())
        await start_ability_tree(None, ability)

        _ = await ability.apply(b"speech")
        await wait_until(lambda: bool(ability.outputs))
        return ability.state(), ability.outputs

    active_state, outputs = asyncio.run(run())

    assert active_state == "/RecordingSpeechEncodingLifecycle/attached/behavior/idle"
    assert outputs == [b"encoded:speech"]


def test_speech_encoding_rejects_input_event_with_wrong_payload_type() -> None:
    async def run() -> tuple[str, list[bytes], list[abilities.FailureData]]:
        ability = RecordingSpeechEncoding(encoder=FixedSpeechEncoder())
        await start_ability_tree(None, ability)

        await hsm.dispatch(
            None,
            ability,
            ability.input_event.with_data("not speech bytes"),
        )
        return ability.state(), ability.outputs, ability.failures

    active_state, outputs, failures = asyncio.run(run())

    assert active_state == "/RecordingSpeechEncodingLifecycle/attached/behavior/idle"
    assert outputs == []
    assert failures == []


def test_speech_encoding_routes_wrong_output_type_to_failure() -> None:
    async def run() -> tuple[str, list[bytes], list[abilities.FailureData]]:
        ability = RecordingSpeechEncoding(encoder=WrongSpeechEncoder())
        await start_ability_tree(None, ability)

        with pytest.raises(RuntimeError, match="output schema"):
            _ = await dispatch_ability_for_test(ability, hsm.Context(), b"speech")
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.outputs, ability.failures

    active_state, outputs, failures = asyncio.run(run())

    assert active_state == "/RecordingSpeechEncodingLifecycle/attached/behavior/idle"
    assert outputs == []
    assert len(failures) == 1
    assert "output schema" in failures[0].message


def test_speech_encoding_routes_encoder_exception_to_failure() -> None:
    async def run() -> tuple[str, list[bytes], list[abilities.FailureData]]:
        ability = RecordingSpeechEncoding(encoder=FailingSpeechEncoder())
        await start_ability_tree(None, ability)

        with pytest.raises(RuntimeError, match="encoder unavailable"):
            _ = await dispatch_ability_for_test(ability, hsm.Context(), b"speech")
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.outputs, ability.failures

    active_state, outputs, failures = asyncio.run(run())

    assert active_state == "/RecordingSpeechEncodingLifecycle/attached/behavior/idle"
    assert outputs == []
    assert len(failures) == 1
    assert failures[0].message == "encoder unavailable"


def test_speech_encoding_is_concrete_ability() -> None:
    ability = speech.SpeechEncoding(encoder=FixedSpeechEncoder())

    assert isinstance(ability, abilities.Ability)
    assert speech.SpeechEncoding.input_data_type is bytes
    assert speech.SpeechEncoding.output_data_type is bytes
