from bot import abilities
from bot.abilities import cognition
from bot.abilities import listening
from bot.abilities.hearing import sound as sound_hearing
from bot.abilities.hearing import speech
from bot.abilities.hearing import voice

import asyncio
import collections.abc
import typing
from typing import override

import hsm
from tests.bot.abilities.support import dispatch_ability_for_test
import pytest

from bot.world import SoundData, SoundEvent
from tests.hsm_instance_state import start_ability_tree
from tests.type_helpers import model_view, object_dict


def sound(audio: bytes) -> SoundData:
    return SoundData(audio=audio, media_type="audio/pcm", sample_rate_hz=48_000, channels=1)

class FixedVoiceDetector(voice.detection.VoiceDetector):
    output: voice.detection.OutputData

    def __init__(self, output: voice.detection.OutputData) -> None:
        self.output = output

    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        return self.output

class FailingVoiceDetector(voice.detection.VoiceDetector):
    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        raise RuntimeError("voice detector offline")

class HangingVoiceDetector(voice.detection.VoiceDetector):
    cancelled: bool

    def __init__(self) -> None:
        self.cancelled = False

    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        del input
        try:
            _ = await asyncio.Event().wait()
        finally:
            self.cancelled = True
        raise AssertionError("unreachable")

class RecordingSpeechDecoder(speech.SpeechDecoder):
    calls: list[bytes]

    def __init__(self) -> None:
        self.calls = []

    @override
    async def decode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return b"decoded:" + input

class FixedVoiceDiarizer(voice.diarization.VoiceDiarizer):
    calls: list[bytes]

    def __init__(self) -> None:
        self.calls = []

    @override
    async def classify(self, input: bytes) -> voice.diarization.OutputData:
        self.calls.append(input)
        return voice.diarization.OutputData(
            segments=(
                voice.diarization.VoiceDiarizationSegment(
                    speaker_label="speaker_1",
                    start_seconds=0.0,
                    end_seconds=1.25,
                    confidence=0.87,
                ),
            )
        )


class FixedSoundClassifier(sound_hearing.classification.SoundClassifier):
    calls: list[bytes]
    labels: tuple[str, ...]

    def __init__(self, *, labels: tuple[str, ...] = ("alarm",)) -> None:
        self.calls = []
        self.labels = labels

    @override
    async def classify(self, input: bytes) -> sound_hearing.classification.OutputData:
        self.calls.append(input)
        return sound_hearing.classification.OutputData(labels=self.labels, confidence=0.9)


class EmptySoundClassifier(sound_hearing.classification.SoundClassifier):
    calls: list[bytes]

    def __init__(self) -> None:
        self.calls = []

    @override
    async def classify(self, input: bytes) -> sound_hearing.classification.OutputData:
        self.calls.append(input)
        return sound_hearing.classification.OutputData(labels=(), confidence=0.95)


class RecordingListening(listening.Listening):
    handoffs: list[cognition.InputData]
    failures: list[listening.FailedEventData]

    def __init__(
        self,
        *,
        voice_detector: voice.detection.VoiceDetector,
        sound_classifier: sound_hearing.classification.SoundClassifier | None = None,
        speech_decoder: speech.SpeechDecoder | None = None,
        voice_diarizer: voice.diarization.VoiceDiarizer | None = None,
    ) -> None:
        super().__init__(
            voice_detector=voice_detector,
            sound_classifier=sound_classifier,
            speech_decoder=speech_decoder,
            voice_diarizer=voice_diarizer,
        )
        self.handoffs = []
        self.failures = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == cognition.InputEvent.name:
            handoff = event.data
            assert isinstance(handoff, cognition.InputData)
            self.handoffs.append(handoff)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, listening.FailedEventData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)

async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0.001)

def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model

def _listening(
    *,
    is_voice: bool = True,
    diarizer: FixedVoiceDiarizer | None = None,
    decoder: RecordingSpeechDecoder | None | typing.Literal[False] = None,
    sound_classifier: sound_hearing.classification.SoundClassifier | None = None,
) -> tuple[RecordingListening, RecordingSpeechDecoder | None]:
    speech_decoder: RecordingSpeechDecoder | None
    if decoder is False:
        speech_decoder = None
    elif decoder is None:
        speech_decoder = RecordingSpeechDecoder()
    else:
        speech_decoder = decoder
    listening_ability = RecordingListening(
        voice_detector=FixedVoiceDetector(voice.detection.OutputData(is_voice=is_voice, confidence=0.91)),
        sound_classifier=sound_classifier,
        speech_decoder=speech_decoder,
        voice_diarizer=diarizer,
    )
    return listening_ability, speech_decoder

def _stimulus(handoff: cognition.InputData) -> hsm.Event[typing.Any]:
    stimulus = handoff.stimulus
    assert isinstance(stimulus, hsm.Event)
    return stimulus

def test_listening_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(listening.Listening.input_event.schema)
    failed_schema = object_dict(listening.Listening.failed_event.schema)

    assert listening.Listening.input_event is SoundEvent
    assert listening.Listening.input_event.name == "world.sound"
    assert input_schema == SoundData.model_json_schema()
    assert input_schema["properties"]["audio"]["format"] in {"binary", "base64", "base64url"}

    assert listening.Listening.output_event is cognition.InputEvent
    assert listening.Listening.output_event.name == "bot.ability.cognition.input"

    assert listening.Listening.failed_event.name == "bot.ability.listening.failed"
    assert failed_schema == listening.FailedEventData.model_json_schema()

def test_listening_defaults_to_voice_detection_without_requiring_speech_decoding() -> None:
    listening, decoder = _listening(decoder=False)
    fields = vars(listening)

    assert isinstance(fields["_voice_detection"], voice.detection.VoiceDetection)
    assert fields["_voice_detection"].classifier is not None
    assert fields["_speech_decoding"] is None
    assert decoder is None
    assert fields["_voice_diarization"] is None


def test_listening_accepts_optional_speech_decoding_ability() -> None:
    listening, decoder = _listening()
    fields = vars(listening)

    assert isinstance(fields["_speech_decoding"], speech.SpeechDecoding)
    assert fields["_speech_decoding"].decoder is decoder


def test_listening_accepts_optional_voice_diarization_ability() -> None:
    diarizer = FixedVoiceDiarizer()
    listening, _ = _listening(diarizer=diarizer)
    fields = vars(listening)

    assert isinstance(fields["_voice_diarization"], voice.diarization.VoiceDiarization)
    assert fields["_voice_diarization"].classifier is diarizer

def test_listening_apply_bridge_keeps_operation_state_out_of_instance() -> None:
    listening, _ = _listening()

    assert "_pending_apply_results" not in vars(listening)
    assert "_active_apply_operation_id" not in vars(listening)

def test_listening_apply_runs_voice_diarization_and_speech_decoding_pipeline() -> None:
    async def run() -> tuple[cognition.InputData, list[bytes], list[bytes]]:
        diarizer = FixedVoiceDiarizer()
        listening, decoder = _listening(diarizer=diarizer)
        await start_ability_tree(None, listening)

        output = await dispatch_ability_for_test(listening, hsm.Context(), sound(b"voice"))
        assert decoder is not None
        assert isinstance(output, cognition.InputData)
        return output, diarizer.calls, decoder.calls

    handoff, diarization_calls, decoder_calls = asyncio.run(run())
    stimulus = _stimulus(handoff)

    assert stimulus.name == speech.SpeechDecoding.output_event.name
    assert stimulus.data == b"decoded:voice"
    assert diarization_calls == [b"voice"]
    assert decoder_calls == [b"voice"]


def test_listening_model_tracks_detection_diarization_and_decoding_lifecycle() -> None:
    model = model_view(require_model(listening.Listening.model))

    assert model.qualified_name == "/ListeningLifecycle"
    assert model.initial == "/ListeningLifecycle/.initial"
    assert "/ListeningLifecycle/detached" in model.members
    assert "/ListeningLifecycle/attaching" in model.members
    assert "/ListeningLifecycle/attached" in model.members
    assert "/ListeningLifecycle/attached/behavior/initializing" in model.members
    assert "/ListeningLifecycle/attached/behavior/Listening" in model.members
    assert "/ListeningLifecycle/attached/behavior/DetectingVoice" in model.members
    assert "/ListeningLifecycle/attached/behavior/ClassifyingSound" in model.members
    assert "/ListeningLifecycle/attached/behavior/RoutingDetectedVoice" in model.members
    assert "/ListeningLifecycle/attached/behavior/DiarizingVoice" in model.members
    assert "/ListeningLifecycle/attached/behavior/DecodingSpeech" in model.members
    assert "/ListeningLifecycle/attached/behavior/DecodingSpeech/Detected" in model.members
    assert "/ListeningLifecycle/attached/behavior/DecodingSpeech/Diarized" in model.members
    assert "world.sound" in model.transition_map["/ListeningLifecycle/attached/behavior/Listening"]
    assert (
        "bot.ability.listening.voice_detection.completed"
        in model.transition_map["/ListeningLifecycle/attached/behavior/DetectingVoice"]
    )
    assert (
        "bot.ability.listening.voice_diarization.completed"
        in model.transition_map["/ListeningLifecycle/attached/behavior/DiarizingVoice"]
    )
    assert (
        "bot.ability.listening.speech_decoding.completed"
        in model.transition_map["/ListeningLifecycle/attached/behavior/DecodingSpeech"]
    )

def test_listening_skips_cognition_input_when_no_voice() -> None:
    async def run() -> tuple[list[cognition.InputData], list[bytes], str]:
        listening, decoder = _listening(is_voice=False)
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"silence"))
        await asyncio.sleep(0.05)

        assert decoder is not None
        return listening.handoffs, decoder.calls, listening.state()

    handoffs, decoder_calls, active_state = asyncio.run(run())

    assert handoffs == []
    assert decoder_calls == []
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Listening"


def test_listening_publishes_sound_cognition_input_when_sound_classifier_labels_nonvoice() -> None:
    async def run() -> tuple[list[cognition.InputData], list[bytes], list[bytes], str]:
        classifier = FixedSoundClassifier(labels=("alarm",))
        listening, decoder = _listening(is_voice=False, decoder=False, sound_classifier=classifier)
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"alarm-tone"))
        await wait_until(lambda: bool(listening.handoffs))

        return listening.handoffs, classifier.calls, [] if decoder is None else decoder.calls, listening.state()

    handoffs, classifier_calls, decoder_calls, active_state = asyncio.run(run())
    stimulus = _stimulus(handoffs[0])

    assert stimulus.name == SoundEvent.name
    assert isinstance(stimulus.data, SoundData)
    assert stimulus.data.audio == b"alarm-tone"
    assert classifier_calls == [b"alarm-tone"]
    assert decoder_calls == []
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Listening"


def test_listening_skips_cognition_input_when_sound_classifier_finds_no_labels() -> None:
    async def run() -> tuple[list[cognition.InputData], list[bytes], str]:
        classifier = EmptySoundClassifier()
        listening, _ = _listening(is_voice=False, decoder=False, sound_classifier=classifier)
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"ambient"))
        await asyncio.sleep(0.05)

        return listening.handoffs, classifier.calls, listening.state()

    handoffs, classifier_calls, active_state = asyncio.run(run())

    assert handoffs == []
    assert classifier_calls == [b"ambient"]
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Listening"


def test_listening_publishes_sound_cognition_input_when_speech_decoding_is_absent() -> None:
    async def run() -> tuple[list[cognition.InputData], str]:
        listening, _ = _listening(decoder=False)
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"voice"))
        await wait_until(lambda: bool(listening.handoffs))

        return listening.handoffs, listening.state()

    handoffs, active_state = asyncio.run(run())
    stimulus = _stimulus(handoffs[0])

    assert stimulus.name == SoundEvent.name
    assert isinstance(stimulus.data, SoundData)
    assert stimulus.data.audio == b"voice"
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Listening"


def test_listening_publishes_sound_after_diarization_when_speech_decoding_is_absent() -> None:
    async def run() -> tuple[list[cognition.InputData], list[bytes], str]:
        diarizer = FixedVoiceDiarizer()
        listening, _ = _listening(diarizer=diarizer, decoder=False)
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"voice"))
        await wait_until(lambda: bool(listening.handoffs))

        return listening.handoffs, diarizer.calls, listening.state()

    handoffs, diarizer_calls, active_state = asyncio.run(run())
    stimulus = _stimulus(handoffs[0])

    assert stimulus.name == SoundEvent.name
    assert isinstance(stimulus.data, SoundData)
    assert stimulus.data.audio == b"voice"
    assert diarizer_calls == [b"voice"]
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Listening"


def test_listening_publishes_speech_cognition_input_when_voice_is_detected() -> None:
    async def run() -> tuple[list[cognition.InputData], list[bytes], str]:
        listening, decoder = _listening()
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"voice"))
        await wait_until(lambda: bool(listening.handoffs))

        assert decoder is not None
        return listening.handoffs, decoder.calls, listening.state()

    handoffs, decoder_calls, active_state = asyncio.run(run())
    stimulus = _stimulus(handoffs[0])

    assert stimulus.name == speech.SpeechDecoding.output_event.name
    assert stimulus.data == b"decoded:voice"
    assert decoder_calls == [b"voice"]
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Listening"

def test_listening_runs_optional_diarization_before_decoding_speech() -> None:
    async def run() -> tuple[list[cognition.InputData], list[bytes], list[bytes]]:
        diarizer = FixedVoiceDiarizer()
        listening, decoder = _listening(diarizer=diarizer)
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"voice"))
        await wait_until(lambda: bool(listening.handoffs))

        assert decoder is not None
        return listening.handoffs, diarizer.calls, decoder.calls

    handoffs, diarizer_calls, decoder_calls = asyncio.run(run())
    stimulus = _stimulus(handoffs[0])

    assert stimulus.name == speech.SpeechDecoding.output_event.name
    assert stimulus.data == b"decoded:voice"
    assert diarizer_calls == [b"voice"]
    assert decoder_calls == [b"voice"]

def test_listening_detach_releases_owned_subabilities_while_detecting_voice() -> None:
    async def run() -> tuple[tuple[str, ...], str]:
        ctx = hsm.Context()
        listening_ability = RecordingListening(
            voice_detector=HangingVoiceDetector(),
            speech_decoder=RecordingSpeechDecoder(),
        )
        await start_ability_tree(ctx, listening_ability)
        _ = await listening_ability.apply(sound(b"voice"), ctx=ctx)
        await wait_until(lambda: listening_ability.state() == "/RecordingListeningLifecycle/attached/behavior/DetectingVoice")
        assert listening_ability.state() == "/RecordingListeningLifecycle/attached/behavior/DetectingVoice"
        subabilities = (
            typing.cast(abilities.Ability[typing.Any, typing.Any], vars(listening_ability)["_voice_detection"]),
            typing.cast(abilities.Ability[typing.Any, typing.Any], vars(listening_ability)["_speech_decoding"]),
        )

        _ = await listening_ability.detach(ctx=ctx)
        await wait_until(lambda: all(ability.state().endswith("/detached") for ability in subabilities))
        states = tuple(ability.state() for ability in subabilities)
        state = listening_ability.state()
        await listening_ability.stop(ctx)
        return states, state

    states, state = asyncio.run(run())

    assert all(child_state.endswith("/detached") for child_state in states)
    assert state == "/RecordingListeningLifecycle/detached"

def test_listening_dispatches_failure_when_detection_fails() -> None:
    async def run() -> tuple[list[listening.FailedEventData], list[cognition.InputData], str]:
        listening_ability = RecordingListening(
            voice_detector=FailingVoiceDetector(),
            speech_decoder=RecordingSpeechDecoder(),
        )
        await start_ability_tree(None, listening_ability)

        with pytest.raises(RuntimeError, match="voice detector offline"):
            _ = await dispatch_ability_for_test(listening_ability, hsm.Context(), sound(b"voice"))
        await wait_until(lambda: bool(listening_ability.failures))

        return listening_ability.failures, listening_ability.handoffs, listening_ability.state()

    failures, handoffs, active_state = asyncio.run(run())

    assert failures == [
        listening.FailedEventData(stage="voice_detection", message="voice detector offline"),
    ]
    assert handoffs == []
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Listening"

def test_listening_failure_payload_reuses_ability_failure_message_shape() -> None:
    failure = listening.FailedEventData.from_ability_failure(
        stage="speech_decoding",
        failure=abilities.FailureData(message="decoder timed out"),
    )

    assert failure == listening.FailedEventData(stage="speech_decoding", message="decoder timed out")
