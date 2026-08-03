from bot import abilities
from bot.abilities import ability as ability_mod
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
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        return voice.detection.ApplyData(
            segments=(
                voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.0, confidence=0.87),
            )
        )


class NoVoiceDetector(voice.detection.VoiceDetector):
    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        return voice.detection.ApplyData(segments=())


class SlowVoiceDetector(voice.detection.VoiceDetector):
    release: asyncio.Event

    def __init__(self, release: asyncio.Event) -> None:
        self.release = release

    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        _ = await self.release.wait()
        return voice.detection.ApplyData(
            segments=(
                voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.0, confidence=0.87),
            )
        )


class RecordingDelayedVoiceDetector(voice.detection.VoiceDetector):
    calls: list[bytes]
    release_first: asyncio.Event

    def __init__(self, calls: list[bytes], release_first: asyncio.Event) -> None:
        self.calls = calls
        self.release_first = release_first

    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        self.calls.append(input)
        if input == b"first":
            _ = await self.release_first.wait()
        if input == b"silence":
            return voice.detection.ApplyData(segments=())
        return voice.detection.ApplyData(
            segments=(
                voice.detection.VoiceDetectionSegment(
                    start_seconds=0.0,
                    end_seconds=1.0,
                    confidence=0.87,
                ),
            )
        )


class WrongVoiceDetector(voice.detection.VoiceDetector):
    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        return invalid_value(voice.detection.ApplyData, "not voice detection apply data")


class FailingVoiceDetector(voice.detection.VoiceDetector):
    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        raise RuntimeError("provider unavailable")


class VoiceThenInvalidOutputDetector(voice.detection.VoiceDetector):
    call_count: int

    def __init__(self) -> None:
        self.call_count = 0

    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        self.call_count += 1
        if self.call_count == 1:
            return voice.detection.ApplyData(
                segments=(
                    voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.0, confidence=0.87),
                )
            )
        return invalid_value(voice.detection.ApplyData, "not voice detection apply data")


class VoiceThenFailingDetector(voice.detection.VoiceDetector):
    call_count: int

    def __init__(self) -> None:
        self.call_count = 0

    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        self.call_count += 1
        if self.call_count == 1:
            return voice.detection.ApplyData(
                segments=(
                    voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.0, confidence=0.87),
                )
            )
        raise RuntimeError("provider unavailable")


class RecordingVoiceDetection(voice.detection.VoiceDetection):
    starts: list[voice.detection.StartData]
    ends: list[voice.detection.EndData]
    failures: list[abilities.FailureData]
    # Terminal recorder may mirror Start into ``outputs``; keep a list for that seam.
    outputs: list[object]

    def __init__(self, *, classifier: voice.detection.VoiceDetector) -> None:
        super().__init__(classifier=classifier)
        self.starts = []
        self.ends = []
        self.failures = []
        self.outputs = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        # Public Start/End ride inside TerminalOutputEvent; also accept bare boundary events.
        public = event
        if event.name == ability_mod.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
            public = event.data
        if public.name == self.start_event.name:
            data = public.data
            assert isinstance(data, voice.detection.StartData)
            self.starts.append(data)
        if public.name == self.end_event.name:
            data = public.data
            assert isinstance(data, voice.detection.EndData)
            self.ends.append(data)
        if event.name == ability_mod.TerminalErrorEvent.name and isinstance(event.data, hsm.Event):
            failure = event.data.data
            if isinstance(failure, abilities.FailureData):
                self.failures.append(failure)
        elif event.name == self.failed_event.name and isinstance(event.data, abilities.FailureData):
            # Owner-mirrored failed event (avoid double-count when terminal already recorded).
            if not self.failures or self.failures[-1] != event.data:
                self.failures.append(event.data)
        return super().dispatch(ctx, event)


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


def test_voice_detection_apply_data_records_segments() -> None:
    detection = voice.detection.ApplyData(
        segments=(
            voice.detection.VoiceDetectionSegment(
                start_seconds=0.0,
                end_seconds=0.7,
                confidence=0.87,
            ),
        )
    )

    assert len(detection.segments) == 1
    assert detection.segments[0].start_seconds == 0.0
    assert detection.segments[0].end_seconds == 0.7
    assert detection.segments[0].confidence == 0.87


def test_voice_detection_segment_rejects_invalid_span() -> None:
    with pytest.raises(ValueError):
        _ = voice.detection.VoiceDetectionSegment(start_seconds=1.0, end_seconds=0.5)


def test_voice_detection_segment_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        _ = voice.detection.VoiceDetectionSegment(
            start_seconds=0.0,
            end_seconds=0.7,
            confidence=1.1,
        )


def test_voice_detection_emits_start_on_voice() -> None:
    async def run() -> tuple[str, list[voice.detection.StartData], list[voice.detection.EndData]]:
        ability = RecordingVoiceDetection(classifier=FixedVoiceDetector())
        await start_ability_tree(None, ability)

        _ = await ability.apply(b"audio")
        await wait_until(lambda: bool(ability.starts))
        return ability.state() or "", ability.starts, ability.ends

    state, starts, ends = asyncio.run(run())

    assert state == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"
    assert starts == [voice.detection.StartData(start_seconds=0.0, confidence=0.87)]
    assert ends == []


def test_voice_detection_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(voice.detection.VoiceDetection.input_event.schema)
    start_schema = object_dict(voice.detection.VoiceDetection.start_event.schema)
    end_schema = object_dict(voice.detection.VoiceDetection.end_event.schema)

    assert voice.detection.VoiceDetection.input_event.name == "bot.ability.hearing.voice.detection.input"
    assert input_schema["type"] == "string"
    assert input_schema["format"] == "binary"

    assert voice.detection.VoiceDetection.start_event.name == "bot.ability.hearing.voice.detection.start"
    assert start_schema == voice.detection.StartData.model_json_schema()
    assert voice.detection.VoiceDetection.end_event.name == "bot.ability.hearing.voice.detection.end"
    assert end_schema == voice.detection.EndData.model_json_schema()


def test_voice_detection_model_tracks_detection_lifecycle() -> None:
    model = model_view(require_model(voice.detection.VoiceDetection.model))

    assert model.qualified_name == "/VoiceDetectionLifecycle"
    assert model.initial == "/VoiceDetectionLifecycle/.initial"
    assert f"{_VOICE_DETECTION_MODEL}/NoVoiceDetected" in model.members
    assert f"{_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring" in model.members
    assert f"{_VOICE_DETECTION_MODEL}/VoiceDetected" in model.members
    assert f"{_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring" in model.members
    # One shared Detecting region (not nested under each presence composite).
    assert f"{_VOICE_DETECTION_MODEL}/Detecting" in model.members
    assert f"{_VOICE_DETECTION_MODEL}/NoVoiceDetected/Detecting" not in model.members
    assert f"{_VOICE_DETECTION_MODEL}/VoiceDetected/Detecting" not in model.members
    assert (
        "bot.ability.hearing.voice.detection.input"
        in model.transition_map[f"{_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"]
    )
    assert (
        "bot.ability.hearing.voice.detection.input"
        in model.transition_map[f"{_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"]
    )


def test_voice_detection_silence_emits_no_boundary_while_idle() -> None:
    async def run() -> tuple[str, list[voice.detection.StartData], list[voice.detection.EndData]]:
        ability = RecordingVoiceDetection(classifier=NoVoiceDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"silence"))
        await asyncio.sleep(0.05)
        return ability.state() or "", ability.starts, ability.ends

    active_state, starts, ends = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"
    assert starts == []
    assert ends == []


def test_voice_detection_emits_start_then_end_across_clips() -> None:
    async def run() -> tuple[str, list[voice.detection.StartData], list[voice.detection.EndData]]:
        ability = RecordingVoiceDetection(classifier=RecordingDelayedVoiceDetector([], asyncio.Event()))
        # Use detectors that alternate via input identity
        class VoiceThenSilence(voice.detection.VoiceDetector):
            @override
            async def classify(self, input: bytes) -> voice.detection.ApplyData:
                if input == b"silence":
                    return voice.detection.ApplyData(segments=())
                return voice.detection.ApplyData(
                    segments=(
                        voice.detection.VoiceDetectionSegment(
                            start_seconds=0.1, end_seconds=0.9, confidence=0.8
                        ),
                    )
                )

        ability = RecordingVoiceDetection(classifier=VoiceThenSilence())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"voice"))
        await wait_until(lambda: bool(ability.starts))
        await hsm.dispatch(None, ability, ability.input_event.with_data(b"silence"))
        await wait_until(lambda: bool(ability.ends))
        return ability.state() or "", ability.starts, ability.ends

    state, starts, ends = asyncio.run(run())
    assert state == f"{_RECORDING_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"
    assert starts == [voice.detection.StartData(start_seconds=0.1, confidence=0.8)]
    assert ends == [voice.detection.EndData(end_seconds=0.9, confidence=0.8)]


def test_voice_detection_does_not_reemit_start_while_voice_continues() -> None:
    async def run() -> tuple[str, list[voice.detection.StartData], list[voice.detection.EndData]]:
        ability = RecordingVoiceDetection(classifier=FixedVoiceDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"a"))
        await wait_until(lambda: bool(ability.starts))
        await hsm.dispatch(None, ability, ability.input_event.with_data(b"b"))
        await asyncio.sleep(0.05)
        return ability.state() or "", ability.starts, ability.ends

    state, starts, ends = asyncio.run(run())
    assert state == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"
    assert len(starts) == 1
    assert ends == []


def test_voice_detection_public_start_event_does_not_complete_in_flight_detection() -> None:
    async def run() -> None:
        release = asyncio.Event()
        ability = RecordingVoiceDetection(classifier=SlowVoiceDetector(release))
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"audio"))
        await hsm.dispatch(
            None,
            ability,
            ability.start_event.with_data(voice.detection.StartData(start_seconds=0.0)),
        )

        assert ability.starts == [voice.detection.StartData(start_seconds=0.0, confidence=None)]
        assert ability.state() == f"{_RECORDING_VOICE_DETECTION_MODEL}/Detecting"

        release.set()
        await wait_until(
            lambda: ability.starts[-1:]
            == [voice.detection.StartData(start_seconds=0.0, confidence=0.87)]
            or len(ability.starts) >= 2
        )

        assert ability.state() == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"
        assert voice.detection.StartData(start_seconds=0.0, confidence=0.87) in ability.starts

    asyncio.run(run())


def test_voice_detection_defers_repeated_input_while_detecting() -> None:
    async def run() -> tuple[list[bytes], list[voice.detection.StartData], str]:
        release_first = asyncio.Event()
        calls: list[bytes] = []
        ability = RecordingVoiceDetection(classifier=RecordingDelayedVoiceDetector(calls, release_first))
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"first"))
        await hsm.dispatch(None, ability, ability.input_event.with_data(b"second"))

        assert calls == [b"first"]
        release_first.set()
        for _ in range(100):
            if calls == [b"first", b"second"] and len(ability.starts) == 1:
                break
            await asyncio.sleep(0)

        return calls, ability.starts, ability.state() or ""

    calls, starts, active_state = asyncio.run(run())

    assert calls == [b"first", b"second"]
    # First clip opens voice (Start); second continues voice (no second Start).
    assert starts == [voice.detection.StartData(start_seconds=0.0, confidence=0.87)]
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
        return calls, ability.state() or ""

    calls, active_state = asyncio.run(run())

    assert calls == []
    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"


def test_voice_detection_routes_wrong_output_type_to_failure() -> None:
    async def run() -> tuple[str, list[voice.detection.StartData], list[abilities.FailureData]]:
        ability = RecordingVoiceDetection(classifier=WrongVoiceDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"audio"))
        for _ in range(100):
            if ability.failures:
                break
            await asyncio.sleep(0)

        return ability.state() or "", ability.starts, ability.failures

    active_state, starts, failures = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"
    assert starts == []
    assert len(failures) >= 1
    assert "apply schema" in failures[0].message


def test_voice_detection_routes_detector_exception_to_failure() -> None:
    async def run() -> tuple[str, list[voice.detection.StartData], list[abilities.FailureData]]:
        ability = RecordingVoiceDetection(classifier=FailingVoiceDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"audio"))
        for _ in range(100):
            if ability.failures:
                break
            await asyncio.sleep(0)

        return ability.state() or "", ability.starts, ability.failures

    active_state, starts, failures = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/NoVoiceDetected/Monitoring"
    assert starts == []
    assert len(failures) >= 1
    assert failures[0].message == "provider unavailable"


def test_voice_detection_preserves_present_state_after_wrong_output_type() -> None:
    async def run() -> tuple[str, list[voice.detection.StartData], list[abilities.FailureData]]:
        ability = RecordingVoiceDetection(classifier=VoiceThenInvalidOutputDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"voice"))
        await wait_until(lambda: ability.state() == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring")
        await hsm.dispatch(None, ability, ability.input_event.with_data(b"bad-output"))
        await wait_until(lambda: bool(ability.failures))

        return ability.state() or "", ability.starts, ability.failures

    active_state, starts, failures = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"
    assert starts == [voice.detection.StartData(start_seconds=0.0, confidence=0.87)]
    assert len(failures) >= 1
    assert "apply schema" in failures[0].message


def test_voice_detection_preserves_present_state_after_detector_exception() -> None:
    async def run() -> tuple[str, list[voice.detection.StartData], list[abilities.FailureData]]:
        ability = RecordingVoiceDetection(classifier=VoiceThenFailingDetector())
        await start_ability_tree(None, ability)

        await hsm.dispatch(None, ability, ability.input_event.with_data(b"voice"))
        await wait_until(lambda: ability.state() == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring")
        await hsm.dispatch(None, ability, ability.input_event.with_data(b"provider-failure"))
        await wait_until(lambda: bool(ability.failures))

        return ability.state() or "", ability.starts, ability.failures

    active_state, starts, failures = asyncio.run(run())

    assert active_state == f"{_RECORDING_VOICE_DETECTION_MODEL}/VoiceDetected/Monitoring"
    assert starts == [voice.detection.StartData(start_seconds=0.0, confidence=0.87)]
    assert len(failures) >= 1
    assert failures[0].message == "provider unavailable"


def test_voice_detection_is_concrete_ability() -> None:
    ability = voice.detection.VoiceDetection(classifier=FixedVoiceDetector())

    assert isinstance(ability, abilities.Ability)
    assert issubclass(voice.detection.VoiceDetector, abilities.Classifier)
    assert voice.detection.VoiceDetection.output_data_type is voice.detection.ApplyData
    assert voice.detection.VoiceDetection.output_event is voice.detection.OutputEvent
    assert voice.detection.VoiceDetection.start_event is voice.detection.StartEvent
    assert voice.detection.VoiceDetection.end_event is voice.detection.EndEvent
