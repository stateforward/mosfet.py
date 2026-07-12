from bot import abilities
from bot.abilities.hearing import voice

import asyncio
import collections.abc
import typing
from typing import override

import hsm
import pytest

from tests.bot.abilities.support import start_abilities_for_test
from tests.type_helpers import invalid_value, model_view, object_dict

_VOICE_DETECTION_MODEL = "/VoiceDetectionLifecycle/attached/behavior"
_RECORDING_VOICE_DETECTION_MODEL = "/RecordingVoiceDetectionLifecycle/attached/behavior"

class FixedVoiceDetector(voice.detection.VoiceDetector):
    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        return voice.detection.OutputData(is_voice=True, confidence=0.87)

class NoVoiceDetector(voice.detection.VoiceDetector):
    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        return voice.detection.OutputData(is_voice=False, confidence=0.91)

class SlowVoiceDetector(voice.detection.VoiceDetector):
    release: asyncio.Event

    def __init__(self, release: asyncio.Event) -> None:
        self.release = release

    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        _ = await self.release.wait()
        return voice.detection.OutputData(is_voice=True, confidence=0.87)

class RecordingDelayedVoiceDetector(voice.detection.VoiceDetector):
    calls: list[bytes]
    release_first: asyncio.Event

    def __init__(self, calls: list[bytes], release_first: asyncio.Event) -> None:
        self.calls = calls
        self.release_first = release_first

    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        self.calls.append(input)
        if input == b"first":
            _ = await self.release_first.wait()
        return voice.detection.OutputData(is_voice=input != b"silence", confidence=0.87)

class WrongVoiceDetector(voice.detection.VoiceDetector):
    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        return invalid_value(voice.detection.OutputData, "not voice detection output")

class FailingVoiceDetector(voice.detection.VoiceDetector):
    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        raise RuntimeError("provider unavailable")

class VoiceThenInvalidOutputDetector(voice.detection.VoiceDetector):
    call_count: int

    def __init__(self) -> None:
        self.call_count = 0

    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        self.call_count += 1
        if self.call_count == 1:
            return voice.detection.OutputData(is_voice=True, confidence=0.87)
        return invalid_value(voice.detection.OutputData, "not voice detection output")

class VoiceThenFailingDetector(voice.detection.VoiceDetector):
    call_count: int

    def __init__(self) -> None:
        self.call_count = 0

    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        self.call_count += 1
        if self.call_count == 1:
            return voice.detection.OutputData(is_voice=True, confidence=0.87)
        raise RuntimeError("provider unavailable")

class RecordingVoiceDetection(voice.detection.VoiceDetection):
    outputs: list[voice.detection.OutputData]
    failures: list[abilities.FailureData]

    def __init__(self, *, classifier: voice.detection.VoiceDetector) -> None:
        super().__init__(classifier=classifier)
        self.outputs = []
        self.failures = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, voice.detection.OutputData)
            self.outputs.append(output)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, abilities.FailureData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)

async def await_voice_detection(output: collections.abc.Awaitable[voice.detection.OutputData]) -> voice.detection.OutputData:
    return await output

async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0)

def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model

async def start_ability_tree(ctx: hsm.Context | None, ability: abilities.Ability[typing.Any, typing.Any]) -> None:
    await start_abilities_for_test(hsm.Context() if ctx is None else ctx, ability)

def test_voice_detection_output_records_detection_and_confidence() -> None:
    detection = voice.detection.OutputData(is_voice=True, confidence=0.87)

    assert detection.is_voice is True
    assert detection.confidence == 0.87

def test_voice_detection_output_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        _ = voice.detection.OutputData(is_voice=True, confidence=1.1)

def test_voice_detection_uses_injected_detector() -> None:
    async def run() -> list[voice.detection.OutputData]:
        ability = RecordingVoiceDetection(classifier=FixedVoiceDetector())
        await start_ability_tree(None, ability)

        _ = await ability.apply(b"audio")
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs == [voice.detection.OutputData(is_voice=True, confidence=0.87)]

def test_voice_detection_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(voice.detection.VoiceDetection.input_event.schema)
    output_schema = object_dict(voice.detection.VoiceDetection.output_event.schema)

    assert voice.detection.VoiceDetection.input_event.name == "bot.ability.hearing.voice.detection.input"
    assert input_schema["type"] == "string"
    assert input_schema["format"] == "binary"
    assert input_schema["examples"] == ["base64-encoded audio bytes"]

    assert voice.detection.VoiceDetection.output_event.name == "bot.ability.hearing.voice.detection.output"
    assert output_schema == voice.detection.OutputData.model_json_schema()
    assert output_schema["description"]

def test_voice_detection_model_tracks_detection_lifecycle() -> None:
    model = model_view(require_model(voice.detection.VoiceDetection.model))

    assert model.qualified_name == "/VoiceDetectionLifecycle"
    assert model.initial == "/VoiceDetectionLifecycle/.initial"
    assert f"{_VOICE_DETECTION_MODEL}/NoVoiceDetected" in model.members
    assert f"{_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring" in model.members
    assert f"{_VOICE_DETECTION_MODEL}/NoVoiceDetected/Detecting" in model.members
    assert f"{_VOICE_DETECTION_MODEL}/VoiceDetected" in model.members
    assert f"{_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring" in model.members
    assert f"{_VOICE_DETECTION_MODEL}/VoiceDetected/Detecting" in model.members
    assert (
        "bot.ability.hearing.voice.detection.input"
        in model.transition_map[f"{_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"]
    )
    assert (
        "bot.ability.hearing.voice.detection.input"
        in model.transition_map[f"{_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"]
    )
    assert (
        "bot.ability.hearing.voice.detection.apply.completed"
        in model.transition_map[f"{_VOICE_DETECTION_MODEL}/NoVoiceDetected/Detecting"]
    )
    assert (
        "bot.ability.hearing.voice.detection.apply.completed"
        in model.transition_map[f"{_VOICE_DETECTION_MODEL}/VoiceDetected/Detecting"]
    )
    assert (
        "bot.ability.hearing.voice.detection.apply.failed"
        in model.transition_map[f"{_VOICE_DETECTION_MODEL}/NoVoiceDetected/Detecting"]
    )
    assert (
        "bot.ability.hearing.voice.detection.apply.failed"
        in model.transition_map[f"{_VOICE_DETECTION_MODEL}/VoiceDetected/Detecting"]
    )

def test_voice_detection_model_dispatches_output_and_tracks_voice_state() -> None:
    async def run() -> tuple[str, list[voice.detection.OutputData]]:
        ability = RecordingVoiceDetection(classifier=FixedVoiceDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(
            None,
            ability,
            voice.detection.VoiceDetection.input_event.with_data(b"audio"),
        )
        await wait_until(lambda: bool(ability.outputs))
        snapshot = hsm.take_snapshot(None, ability)
        return snapshot.state, ability.outputs

    active_state, outputs = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"
    assert outputs == [voice.detection.OutputData(is_voice=True, confidence=0.87)]

def test_voice_detection_model_dispatches_output_and_tracks_no_voice_state() -> None:
    async def run() -> tuple[str, list[voice.detection.OutputData]]:
        ability = RecordingVoiceDetection(classifier=NoVoiceDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"silence"))
        await wait_until(lambda: bool(ability.outputs))

        return ability.state(), ability.outputs

    active_state, outputs = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"
    assert outputs == [voice.detection.OutputData(is_voice=False, confidence=0.91)]

def test_voice_detection_public_output_event_does_not_complete_in_flight_detection() -> None:
    async def run() -> None:
        release = asyncio.Event()
        ability = RecordingVoiceDetection(classifier=SlowVoiceDetector(release))
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"audio"))
        await hsm.dispatch(
            None,
            ability,
            ability.output_event.with_data(voice.detection.OutputData(is_voice=False, confidence=0.1)),
        )

        assert ability.outputs == [voice.detection.OutputData(is_voice=False, confidence=0.1)]
        assert ability.state() == f"{_RECORDING_VOICE_DETECTION_MODEL}/NoVoiceDetected/Detecting"

        release.set()
        for _ in range(100):
            if ability.outputs[-1:] == [voice.detection.OutputData(is_voice=True, confidence=0.87)]:
                break
            await asyncio.sleep(0)

        assert ability.outputs == [
            voice.detection.OutputData(is_voice=False, confidence=0.1),
            voice.detection.OutputData(is_voice=True, confidence=0.87),
        ]
        assert ability.state() == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"

    asyncio.run(run())

def test_voice_detection_defers_repeated_input_while_detecting() -> None:
    async def run() -> tuple[list[bytes], list[voice.detection.OutputData], str]:
        release_first = asyncio.Event()
        calls: list[bytes] = []
        ability = RecordingVoiceDetection(classifier=RecordingDelayedVoiceDetector(calls, release_first))
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"first"))
        await hsm.dispatch(None, ability, ability.input_event.with_data(b"second"))

        assert calls == [b"first"]
        release_first.set()
        for _ in range(100):
            if calls == [b"first", b"second"] and len(ability.outputs) == 2:
                break
            await asyncio.sleep(0)

        return calls, ability.outputs, ability.state()

    calls, outputs, active_state = asyncio.run(run())

    assert calls == [b"first", b"second"]
    assert outputs == [
        voice.detection.OutputData(is_voice=True, confidence=0.87),
        voice.detection.OutputData(is_voice=True, confidence=0.87),
    ]
    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"

def test_voice_detection_rejects_input_event_with_wrong_payload_type() -> None:
    async def run() -> tuple[list[bytes], str]:
        release_first = asyncio.Event()
        calls: list[bytes] = []
        ability = RecordingVoiceDetection(classifier=RecordingDelayedVoiceDetector(calls, release_first))
        await start_ability_tree(None, ability)

        await hsm.dispatch(
            None,
            ability,
            ability.input_event.with_data("not audio bytes"),
        )
        return calls, ability.state()

    calls, active_state = asyncio.run(run())

    assert calls == []
    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"

def test_voice_detection_routes_wrong_output_type_to_failure() -> None:
    async def run() -> tuple[str, list[voice.detection.OutputData], list[abilities.FailureData]]:
        ability = RecordingVoiceDetection(classifier=WrongVoiceDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"audio"))
        for _ in range(100):
            if ability.failures:
                break
            await asyncio.sleep(0)

        return ability.state(), ability.outputs, ability.failures

    active_state, outputs, failures = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"
    assert outputs == []
    assert len(failures) == 1
    assert "output schema" in failures[0].message

def test_voice_detection_routes_detector_exception_to_failure() -> None:
    async def run() -> tuple[str, list[voice.detection.OutputData], list[abilities.FailureData]]:
        ability = RecordingVoiceDetection(classifier=FailingVoiceDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"audio"))
        for _ in range(100):
            if ability.failures:
                break
            await asyncio.sleep(0)

        return ability.state(), ability.outputs, ability.failures

    active_state, outputs, failures = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"
    assert outputs == []
    assert len(failures) == 1
    assert failures[0].message == "provider unavailable"

def test_voice_detection_preserves_present_state_after_wrong_output_type() -> None:
    async def run() -> tuple[str, list[voice.detection.OutputData], list[abilities.FailureData]]:
        ability = RecordingVoiceDetection(classifier=VoiceThenInvalidOutputDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"voice"))
        await wait_until(lambda: ability.state() == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring")
        await hsm.dispatch(None, ability, ability.input_event.with_data(b"bad-output"))
        await wait_until(lambda: bool(ability.failures))

        return ability.state(), ability.outputs, ability.failures

    active_state, outputs, failures = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"
    assert outputs == [voice.detection.OutputData(is_voice=True, confidence=0.87)]
    assert len(failures) == 1
    assert "output schema" in failures[0].message

def test_voice_detection_preserves_present_state_after_detector_exception() -> None:
    async def run() -> tuple[str, list[voice.detection.OutputData], list[abilities.FailureData]]:
        ability = RecordingVoiceDetection(classifier=VoiceThenFailingDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"voice"))
        await wait_until(lambda: ability.state() == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring")
        await hsm.dispatch(None, ability, ability.input_event.with_data(b"provider-failure"))
        await wait_until(lambda: bool(ability.failures))

        return ability.state(), ability.outputs, ability.failures

    active_state, outputs, failures = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"
    assert outputs == [voice.detection.OutputData(is_voice=True, confidence=0.87)]
    assert len(failures) == 1
    assert failures[0].message == "provider unavailable"

def test_voice_detection_is_concrete_ability() -> None:
    ability = voice.detection.VoiceDetection(classifier=FixedVoiceDetector())

    assert isinstance(ability, abilities.Ability)
    assert issubclass(voice.detection.VoiceDetector, abilities.Classifier)
    assert voice.detection.VoiceDetection.output_data_type is voice.detection.OutputData
