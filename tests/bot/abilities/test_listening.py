from bot import StimulusData
from bot import abilities
from bot.abilities import cognition
from bot.abilities import listening
from bot.abilities.hearing import sound as sound_hearing
from bot.abilities.hearing import speech
from bot.abilities.hearing import voice
from bot.protocols import attachment

import asyncio
import collections.abc
import dataclasses
import datetime
import io
import pathlib
import typing
import wave
from typing import override

import hsm
import bot
from tests.bot.abilities.support import dispatch_ability_for_test
import pytest

from bot.abilities.speaking import EfferenceData, EfferenceEvent
from bot.environment import SoundData, SoundEvent
from tests.hsm_instance_state import ability_terminal_owner, start_ability_tree
from tests.type_helpers import model_view, object_dict


DIARIZED_AUDIO = b"\x00\x00" * 60_000


def sound(audio: bytes, *, received_level_db: float | None = None) -> SoundData:
    return SoundData(
        audio=audio,
        media_type="audio/pcm",
        sample_rate_hz=48_000,
        channels=1,
        received_level_db=received_level_db,
    )


def _score_request(sound: SoundData, operation_id: str) -> hsm.Event[typing.Any]:
    """The typed PRIVATE DIRECTED interface: a ScoreEvent carried as a terminal operation.

    Mirrors the listening pipeline (listening.py _run_sensitivity): the environment sound is
    wrapped in a ScoreData stimulus and dispatched as a directed request targeted at the stage.
    """

    stimulus = StimulusData.from_event(dataclasses.replace(SoundEvent.with_data(sound), source="mouth"))
    return listening.sensitivity.ScoreEvent.with_data_and_id(
        listening.sensitivity.ScoreData(stimulus=stimulus),
        operation_id,
    )


def test_sensitivity_directed_operations_keep_unique_child_envelopes() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any], str]:
        stage = listening.sensitivity.Sensitivity()
        await start_ability_tree(None, stage)
        first, second = await asyncio.gather(
            abilities.run_terminal_operation(
                stage.context(),
                child=stage,
                request=_score_request(sound(b"first"), "sensitivity:first"),
                terminals=(stage.output_event, stage.failed_event),
                timeout=datetime.timedelta.max,
            ),
            abilities.run_terminal_operation(
                stage.context(),
                child=stage,
                request=_score_request(sound(b"second"), "sensitivity:second"),
                terminals=(stage.output_event, stage.failed_event),
                timeout=datetime.timedelta.max,
            ),
        )
        return first, second, hsm.id(stage)

    first, second, stage_id = asyncio.run(run())

    assert (first.id, second.id) == ("sensitivity:first", "sensitivity:second")
    assert first.source == second.source == stage_id
    assert first.target and second.target and first.target != second.target
    assert isinstance(first.data, listening.sensitivity.OutputData)
    assert isinstance(second.data, listening.sensitivity.OutputData)


def wav_sound(
    pcm: bytes,
    *,
    sample_rate_hz: int = 16_000,
    channels: int = 1,
    received_level_db: float | None = None,
) -> SoundData:
    """Ambient person speech as phone_bot Person/SayEncoder emits it: RIFF/WAVE container."""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate_hz)
        stream.writeframes(pcm)
    return SoundData(
        audio=buffer.getvalue(),
        media_type="audio/wav",
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        received_level_db=received_level_db,
    )


class FixedVoiceActivityClassifier(voice.detection.VoiceActivityClassifier):
    output: voice.detection.ApplyData

    def __init__(self, output: voice.detection.ApplyData) -> None:
        self.output = output

    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        return self.output


class SequenceVoiceActivityClassifier(voice.detection.VoiceActivityClassifier):
    """Returns apply results in order, then repeats the last result."""

    outputs: list[voice.detection.ApplyData]
    index: int

    def __init__(self, *outputs: voice.detection.ApplyData) -> None:
        if not outputs:
            raise ValueError("SequenceVoiceActivityClassifier requires at least one ApplyData.")
        self.outputs = list(outputs)
        self.index = 0

    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        if self.index < len(self.outputs):
            result = self.outputs[self.index]
            self.index += 1
            return result
        return self.outputs[-1]


class FailingVoiceActivityClassifier(voice.detection.VoiceActivityClassifier):
    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        raise RuntimeError("voice activity classifier offline")


class HangingVoiceActivityClassifier(voice.detection.VoiceActivityClassifier):
    cancelled: bool
    entered: asyncio.Event

    def __init__(self) -> None:
        self.cancelled = False
        self.entered = asyncio.Event()

    @override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        del input
        self.entered.set()
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
    calls: list[voice.diarization.InputData]
    segment: voice.diarization.VoiceDiarizationSegment
    segments: tuple[voice.diarization.VoiceDiarizationSegment, ...]

    def __init__(
        self,
        *,
        segment: voice.diarization.VoiceDiarizationSegment | None = None,
        segments: tuple[voice.diarization.VoiceDiarizationSegment, ...] | None = None,
    ) -> None:
        self.calls = []
        self.segment = (
            voice.diarization.VoiceDiarizationSegment(
                audio=DIARIZED_AUDIO,
                media_type="audio/pcm",
                sample_rate_hz=48_000,
                channels=1,
                start_seconds=0.0,
                end_seconds=1.25,
                confidence=0.87,
            )
            if segment is None
            else segment
        )
        self.segments = (self.segment,) if segments is None else segments

    @override
    async def classify(self, input: voice.diarization.InputData) -> voice.diarization.OutputData:
        self.calls.append(input)
        return voice.diarization.OutputData(segments=self.segments)


class FixedVoiceClassifier(abilities.Classifier[voice.identification.InputData, voice.identification.OutputData]):
    calls: list[voice.identification.InputData]
    embeddings: tuple[voice.identification.VoiceEmbedding, ...]

    def __init__(self, *embeddings: voice.identification.VoiceEmbedding) -> None:
        self.calls = []
        self.embeddings = embeddings

    @override
    async def classify(self, input: voice.identification.InputData) -> voice.identification.OutputData:
        self.calls.append(input)
        return voice.identification.OutputData(embeddings=self.embeddings)


class FixedSoundClassifier(sound_hearing.classification.SoundClassifier):
    calls: list[SoundData]
    labels: tuple[str, ...]

    def __init__(self, *, labels: tuple[str, ...] = ("alarm",)) -> None:
        self.calls = []
        self.labels = labels

    @override
    async def classify(self, input: SoundData) -> sound_hearing.classification.OutputData:
        self.calls.append(input)
        return sound_hearing.classification.OutputData(labels=self.labels, confidence=0.9)


class EmptySoundClassifier(sound_hearing.classification.SoundClassifier):
    calls: list[SoundData]

    def __init__(self) -> None:
        self.calls = []

    @override
    async def classify(self, input: SoundData) -> sound_hearing.classification.OutputData:
        self.calls.append(input)
        return sound_hearing.classification.OutputData(labels=(), confidence=0.95)


class RecordingListening(listening.Listening):
    handoffs: list[cognition.InputData]
    failures: list[listening.FailedEventData]

    def __init__(
        self,
        *,
        voice_activity_classifier: voice.detection.VoiceActivityClassifier,
        sound_classifier: sound_hearing.classification.SoundClassifier | None = None,
        speech_decoder: speech.SpeechDecoder | None = None,
        voice_diarizer: voice.diarization.VoiceDiarizer | None = None,
        voice_classifier: abilities.Classifier[
            voice.identification.InputData,
            voice.identification.OutputData,
        ]
        | None = None,
        product_threshold_db: float | None = None,
    ) -> None:
        thresholds = {} if product_threshold_db is None else {"product_threshold_db": product_threshold_db}
        super().__init__(
            voice_activity_classifier=voice_activity_classifier,
            sound_classifier=sound_classifier,
            speech_decoder=speech_decoder,
            voice_diarizer=voice_diarizer,
            voice_classifier=voice_classifier,
            **thresholds,
        )
        # Filled by the terminal mirror from what this ability emits. Not from dispatch: the
        # interpretation half's products travel through this ability on their way out, so a
        # dispatch hook would count every product twice — once arriving, once leaving.
        self.handoffs = []
        self.failures = []


class ListeningAttachmentOwner(hsm.Instance):
    lifecycle: list[hsm.Event[typing.Any]]

    @staticmethod
    def _record(
        ctx: hsm.Context,
        instance: "ListeningAttachmentOwner",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        instance.lifecycle.append(event)

    model: typing.ClassVar[hsm.Model] = bot.define(
        "ListeningAttachmentOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(attachment.AttachCompleteEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.AttachFailedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.DetachedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.DetachFailedEvent), hsm.effect(_record)),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle = []


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0.001)


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


_VOICE_APPLY = voice.detection.ApplyData(
    segments=(
        voice.detection.VoiceDetectionSegment(
            start_seconds=0.0,
            end_seconds=1.0,
            confidence=0.91,
        ),
    )
)
_SILENCE_APPLY = voice.detection.ApplyData(segments=())


def _listening(
    *,
    is_voice: bool = True,
    diarizer: FixedVoiceDiarizer | None = None,
    decoder: RecordingSpeechDecoder | None | typing.Literal[False, True] = None,
    sound_classifier: sound_hearing.classification.SoundClassifier | None = None,
    product_threshold_db: float | None = None,
) -> tuple[RecordingListening, RecordingSpeechDecoder | None]:
    speech_decoder: RecordingSpeechDecoder | None
    if decoder is False:
        speech_decoder = None
    elif decoder is None or decoder is True:
        speech_decoder = RecordingSpeechDecoder()
    else:
        speech_decoder = decoder
    # Voice path uses Start then End (silence) so HearingSpeech can close the utterance.
    voice_activity_classifier: voice.detection.VoiceActivityClassifier
    if is_voice:
        voice_activity_classifier = SequenceVoiceActivityClassifier(_VOICE_APPLY, _SILENCE_APPLY)
    else:
        voice_activity_classifier = FixedVoiceActivityClassifier(_SILENCE_APPLY)
    listening_ability = RecordingListening(
        voice_activity_classifier=voice_activity_classifier,
        sound_classifier=sound_classifier,
        speech_decoder=speech_decoder,
        voice_diarizer=diarizer,
        product_threshold_db=(
            listening.interpretation.DEFAULT_PRODUCT_THRESHOLD_DB
            if product_threshold_db is None
            else product_threshold_db
        ),
    )
    return listening_ability, speech_decoder


async def _apply_utterance(
    listening_ability: RecordingListening,
    pcm: bytes = b"voice",
    *,
    silence: bytes = b"silence",
) -> None:
    """One voice frame then silence so VAD End closes HearingSpeech."""

    _ = await listening_ability.apply(sound(pcm))
    _ = await listening_ability.apply(sound(silence))


def _stimulus(handoff: cognition.InputData) -> hsm.Event[typing.Any]:
    stimulus = handoff.stimulus
    assert isinstance(stimulus, hsm.Event)
    return stimulus


def _is_silent_speech_handoff(handoff: cognition.InputData) -> bool:
    stimulus = _stimulus(handoff)
    return (
        stimulus.name == listening.SpeechEvent.name
        and isinstance(stimulus.data, listening.SpeechData)
        and not stimulus.data.voice_detection.segments
    )


def _is_decoded_speech_handoff(handoff: cognition.InputData) -> bool:
    """Decode rewrites the same SpeechEvent into words rather than minting a new envelope."""

    stimulus = _stimulus(handoff)
    return (
        stimulus.name == listening.SpeechEvent.name
        and isinstance(stimulus.data, listening.SpeechData)
        and stimulus.data.content_type == "text/plain"
    )


def test_listening_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(listening.Listening.input_event.schema)
    failed_schema = object_dict(listening.Listening.failed_event.schema)

    assert listening.Listening.input_event is SoundEvent
    assert listening.Listening.input_event.name == "environment.sound"
    assert input_schema == SoundData.model_json_schema()
    audio_schema = object_dict(object_dict(input_schema["properties"])["audio"])
    assert audio_schema["format"] in {"binary", "base64", "base64url"}

    assert listening.Listening.output_event is cognition.InputEvent
    assert listening.Listening.output_event.name == "bot.ability.cognition.input"

    assert listening.Listening.failed_event.name == "bot.ability.listening.failed"
    assert failed_schema == listening.FailedEventData.model_json_schema()
    assert listening.SpeechEvent.schema is listening.SpeechData
    speech_schema = object_dict(listening.SpeechData.model_json_schema())
    speech_properties = object_dict(speech_schema["properties"])
    assert "start_seconds" in speech_properties
    assert "end_seconds" in speech_properties
    assert "confidence" in speech_properties
    assert "voice_embedding" in speech_properties
    assert "source_ids" in speech_properties


def test_listening_pairs_diarized_segments_with_voice_embeddings_in_order() -> None:
    async def run() -> tuple[list[cognition.InputData], list[voice.identification.InputData]]:
        diarizer = FixedVoiceDiarizer(
            segments=(
                voice.diarization.VoiceDiarizationSegment(
                    audio=DIARIZED_AUDIO,
                    media_type="audio/pcm",
                    sample_rate_hz=48_000,
                    channels=1,
                    start_seconds=0.0,
                    end_seconds=1.25,
                ),
                voice.diarization.VoiceDiarizationSegment(
                    audio=b"\x01\x00" * 60_000,
                    media_type="audio/pcm",
                    sample_rate_hz=48_000,
                    channels=1,
                    start_seconds=1.25,
                    end_seconds=2.5,
                ),
            )
        )
        identifier = FixedVoiceClassifier(
            voice.identification.VoiceEmbedding(embedding=(0.1, 0.2), model="fixture"),
            voice.identification.VoiceEmbedding(embedding=(0.3, 0.4), model="fixture"),
        )
        listening_ability = RecordingListening(
            voice_activity_classifier=SequenceVoiceActivityClassifier(_VOICE_APPLY, _SILENCE_APPLY),
            voice_diarizer=diarizer,
            voice_classifier=identifier,
        )
        await start_ability_tree(None, listening_ability)
        await wait_until(lambda: (listening_ability.state() or "").endswith("/Listening"))
        await _apply_utterance(listening_ability, b"voice")
        await wait_until(lambda: len(listening_ability.handoffs) == 2)
        return listening_ability.handoffs, identifier.calls

    handoffs, identifier_inputs = asyncio.run(run())
    assert len(identifier_inputs) == 1
    assert len(identifier_inputs[0].segments) == 2
    assert all(not hasattr(segment, "speaker_label") for segment in identifier_inputs[0].segments)
    assert [segment.audio for segment in identifier_inputs[0].segments] == [
        DIARIZED_AUDIO,
        b"\x01\x00" * 60_000,
    ]
    observations = [_stimulus(handoff).data for handoff in handoffs]
    assert all(stimulus.name == listening.SpeechEvent.name for stimulus in [_stimulus(handoff) for handoff in handoffs])
    assert all(isinstance(observation, listening.SpeechData) for observation in observations)
    ordered_observations = sorted(
        (observation for observation in observations if isinstance(observation, listening.SpeechData)),
        key=lambda observation: observation.start_seconds or 0.0,
    )
    assert [observation.voice_embedding for observation in ordered_observations] == [
        voice.identification.VoiceEmbedding(embedding=(0.1, 0.2), model="fixture"),
        voice.identification.VoiceEmbedding(embedding=(0.3, 0.4), model="fixture"),
    ]
    assert [(observation.start_seconds, observation.end_seconds) for observation in ordered_observations] == [
        (0.0, 1.25),
        (1.25, 2.5),
    ]
    assert all(observation.source_ids for observation in ordered_observations)
    assert len({next(iter(observation.source_ids)) for observation in ordered_observations}) == len(
        ordered_observations
    )
    assert [next(iter(observation.source_ids)) for observation in ordered_observations] == [
        (0.1, 0.2),
        (0.3, 0.4),
    ]


def test_listening_identifies_first_vad_segment_and_carries_id_through_window() -> None:
    async def run() -> tuple[list[cognition.InputData], list[voice.identification.InputData]]:
        identifier = FixedVoiceClassifier(
            voice.identification.VoiceEmbedding(embedding=(0.1, 0.2), model="fixture", confidence=0.95),
        )
        listening_ability = RecordingListening(
            voice_activity_classifier=SequenceVoiceActivityClassifier(_VOICE_APPLY, _SILENCE_APPLY),
            voice_classifier=identifier,
        )
        await start_ability_tree(None, listening_ability)
        await _apply_utterance(listening_ability, b"\x01\x00" * 240, silence=b"\x00\x00" * 240)
        await wait_until(lambda: len(listening_ability.handoffs) == 1)
        return listening_ability.handoffs, identifier.calls

    handoffs, identifier_inputs = asyncio.run(run())
    assert len(identifier_inputs) == 1
    assert len(identifier_inputs[0].segments) == 1
    segment = identifier_inputs[0].segments[0]
    assert type(segment) is voice.VoiceSegment
    assert segment.audio == b"\x01\x00" * 240
    assert segment.start_seconds == 0.0
    assert segment.end_seconds == pytest.approx(240 / 48_000)
    assert not hasattr(segment, "confidence")

    stimulus = _stimulus(handoffs[0])
    assert stimulus.name == listening.SpeechEvent.name
    assert isinstance(stimulus.data, listening.SpeechData)
    assert stimulus.data.content == segment.audio
    assert stimulus.data.confidence is None
    assert stimulus.data.voice_detection.segments[0].confidence == 0.91
    assert stimulus.data.voice_embedding == voice.identification.VoiceEmbedding(
        embedding=(0.1, 0.2), model="fixture", confidence=0.95
    )
    assert stimulus.data.source_ids == frozenset({(0.1, 0.2)})


def test_listening_identifies_ambient_wav_sound_without_speech_decoder() -> None:
    """Person/SayEncoder ambient speech is audio/wav; identity still labels SpeechData."""

    pcm_voice = b"\x01\x00" * 240
    pcm_silence = b"\x00\x00" * 240
    sample_rate_hz = 16_000

    async def run() -> tuple[
        list[cognition.InputData], list[voice.identification.InputData], list[listening.FailedEventData]
    ]:
        identifier = FixedVoiceClassifier(
            voice.identification.VoiceEmbedding(embedding=(0.11, -0.07, 0.33), model="fixture", confidence=0.94),
        )
        listening_ability = RecordingListening(
            voice_activity_classifier=SequenceVoiceActivityClassifier(_VOICE_APPLY, _SILENCE_APPLY),
            voice_classifier=identifier,
            speech_decoder=None,
        )
        await start_ability_tree(None, listening_ability)
        _ = await listening_ability.apply(wav_sound(pcm_voice, sample_rate_hz=sample_rate_hz))
        _ = await listening_ability.apply(wav_sound(pcm_silence, sample_rate_hz=sample_rate_hz))
        await wait_until(lambda: len(listening_ability.handoffs) >= 1 or bool(listening_ability.failures))
        return listening_ability.handoffs, identifier.calls, listening_ability.failures

    handoffs, identifier_inputs, failures = asyncio.run(run())
    assert failures == []
    assert len(identifier_inputs) == 1
    segment = identifier_inputs[0].segments[0]
    assert type(segment) is voice.VoiceSegment
    assert segment.media_type == "audio/pcm"
    assert segment.sample_rate_hz == sample_rate_hz
    assert segment.channels == 1
    assert segment.audio == pcm_voice
    assert segment.end_seconds == pytest.approx(len(pcm_voice) / (2 * sample_rate_hz))

    speech_observations: list[listening.SpeechData] = []
    for handoff in handoffs:
        stimulus = _stimulus(handoff)
        if stimulus.name != listening.SpeechEvent.name:
            continue
        data = stimulus.data
        if isinstance(data, listening.SpeechData) and data.voice_detection.segments:
            speech_observations.append(data)
    assert speech_observations
    observation = speech_observations[0]
    assert observation.media_type == "audio/pcm"
    assert observation.sample_rate_hz == sample_rate_hz
    assert observation.channels == 1
    assert observation.content == pcm_voice
    assert observation.source_ids == frozenset({(0.11, -0.07, 0.33)})
    assert observation.voice_embedding == voice.identification.VoiceEmbedding(
        embedding=(0.11, -0.07, 0.33), model="fixture", confidence=0.94
    )


def test_listening_calls_direct_classifier_once_and_reuses_id_for_each_vad_product() -> None:
    async def run() -> tuple[list[cognition.InputData], list[voice.identification.InputData]]:
        identifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.1, 0.2), model="fixture"))
        listening_ability = RecordingListening(
            voice_activity_classifier=SequenceVoiceActivityClassifier(_VOICE_APPLY, _VOICE_APPLY, _SILENCE_APPLY),
            voice_classifier=identifier,
        )
        await start_ability_tree(None, listening_ability)
        for _ in range(3):
            await listening_ability.apply(sound(b"\x01\x00" * 240))
        await wait_until(lambda: len(listening_ability.handoffs) == 3)
        return listening_ability.handoffs, identifier.calls

    handoffs, calls = asyncio.run(run())
    observations = [_stimulus(handoff).data for handoff in handoffs]
    assert len(calls) == 1
    assert calls[0].segments[0].audio == b"\x01\x00" * 240
    assert all(isinstance(observation, listening.SpeechData) for observation in observations)
    speech = [observation for observation in observations if isinstance(observation, listening.SpeechData)]
    assert len({observation.source_ids for observation in speech}) == 1
    assert speech[0].source_ids == frozenset({(0.1, 0.2)})


def test_listening_direct_identification_uses_open_utterance_level_after_quiet_vad_end() -> None:
    async def run() -> list[cognition.InputData]:
        identifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.1, 0.2)))
        listening_ability = RecordingListening(
            voice_activity_classifier=SequenceVoiceActivityClassifier(_VOICE_APPLY, _SILENCE_APPLY),
            voice_classifier=identifier,
        )
        await start_ability_tree(None, listening_ability)
        await listening_ability.apply(sound(b"\x01\x00" * 240, received_level_db=6.0))
        await listening_ability.apply(sound(b"\x00\x00" * 240, received_level_db=0.0))
        await wait_until(lambda: len(listening_ability.handoffs) == 1)
        return listening_ability.handoffs

    handoffs = asyncio.run(run())

    assert len(handoffs) == 1
    stimulus = _stimulus(handoffs[0])
    assert stimulus.name == listening.SpeechEvent.name
    assert isinstance(stimulus.data, listening.SpeechData)


def test_listening_reports_typed_failure_for_invalid_direct_identification_pcm() -> None:
    async def run() -> tuple[
        list[listening.FailedEventData],
        list[cognition.InputData],
        list[voice.identification.InputData],
    ]:
        identifier = FixedVoiceClassifier(voice.identification.VoiceEmbedding(embedding=(0.1,)))
        listening_ability = RecordingListening(
            voice_activity_classifier=SequenceVoiceActivityClassifier(_VOICE_APPLY, _SILENCE_APPLY),
            voice_classifier=identifier,
        )
        await start_ability_tree(None, listening_ability)
        await _apply_utterance(listening_ability, b"voice", silence=b"silence!")
        await wait_until(lambda: bool(listening_ability.failures))
        return listening_ability.failures, listening_ability.handoffs, identifier.calls

    failures, handoffs, identifier_calls = asyncio.run(run())
    assert failures == [
        listening.FailedEventData(
            stage="voice_identification",
            message=(
                "Voice identification opening segment conversion failed: "
                "voice identification requires complete 16-bit PCM frames."
            ),
        )
    ]
    assert identifier_calls == []
    for handoff in handoffs:
        stimulus = _stimulus(handoff)
        assert not isinstance(stimulus.data, listening.SpeechData) or not stimulus.data.source_ids


def test_listening_rejects_voice_classifier_with_speech_decoder() -> None:
    with pytest.raises(ValueError, match="voice_classifier cannot be combined with speech_decoder"):
        _ = listening.Listening(
            voice_activity_classifier=FixedVoiceActivityClassifier(_VOICE_APPLY),
            speech_decoder=RecordingSpeechDecoder(),
            voice_classifier=FixedVoiceClassifier(
                voice.identification.VoiceEmbedding(embedding=(0.1, 0.2)),
            ),
        )


def test_listening_builds_one_group_for_minimum_and_maximum_children_and_waits_for_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[tuple[hsm.Instance, ...]], int, str]:
        groups: list[tuple[hsm.Instance, ...]] = []
        requests: list[hsm.Event[attachment.AttachData]] = []
        group_init = attachment.Group.__init__

        def record_group(group: attachment.Group, *members: hsm.Instance) -> None:
            groups.append(members)
            group_init(group, *members)

        async def hold_attach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            del group, ctx
            requests.append(event)

        monkeypatch.setattr(attachment.Group, "__init__", record_group)
        monkeypatch.setattr(attachment.Group, "attach", hold_attach)
        minimum = listening.Listening(
            voice_activity_classifier=FixedVoiceActivityClassifier(voice.detection.ApplyData(segments=()))
        )
        maximum = listening.Listening(
            voice_activity_classifier=FixedVoiceActivityClassifier(
                voice.detection.ApplyData(
                    segments=(
                        voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.0, confidence=0.9),
                    )
                )
            ),
            sound_classifier=FixedSoundClassifier(),
            voice_diarizer=FixedVoiceDiarizer(),
            speech_decoder=RecordingSpeechDecoder(),
        )
        owner = hsm.Instance()
        owner_model = bot.define(
            "ListeningAttachmentOwner",
            hsm.initial(hsm.target("ready")),
            hsm.state("ready"),
        )
        ctx = hsm.Context()
        _ = await bot.started(ctx, owner, owner_model)
        await minimum.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "listening-attach",
            ),
        )
        await wait_until(lambda: bool(requests) or minimum.state().endswith("/Listening"))
        state = minimum.state()
        await minimum.stop(minimum.context())
        del maximum
        return groups, len(requests), state

    groups, request_count, state = asyncio.run(run())

    # Two abilities, each building its own group: interpretation's first, because Listening
    # constructs it before grouping it. The ear's group never varies — a body always knows what
    # it is doing and always has something to interpret with. Configuration varies what
    # interpretation is made of, and that is the only thing it varies.
    minimum_stages, minimum_ear, maximum_stages, maximum_ear = groups
    assert len(groups) == 4
    assert [type(member) for member in minimum_ear] == [
        listening.sensitivity.Sensitivity,
        listening.interpretation.Interpretation,
    ]
    assert [type(member) for member in maximum_ear] == [
        listening.sensitivity.Sensitivity,
        listening.interpretation.Interpretation,
    ]
    assert len(minimum_stages) == 1
    assert isinstance(minimum_stages[0], voice.detection.VoiceDetection)
    assert len(maximum_stages) == 4
    assert isinstance(maximum_stages[0], voice.detection.VoiceDetection)
    assert isinstance(maximum_stages[1], sound_hearing.classification.SoundClassification)
    assert isinstance(maximum_stages[2], voice.diarization.VoiceDiarization)
    assert isinstance(maximum_stages[3], speech.SpeechDecoding)
    assert request_count == 1
    assert state == "/ListeningLifecycle/attached/behavior/initializing"


def test_listening_reports_aggregate_attachment_failure_and_accepts_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int, str]:
        requests: list[tuple[attachment.Group, hsm.Event[attachment.AttachData]]] = []

        async def hold_attach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            del ctx
            requests.append((group, event))

        monkeypatch.setattr(attachment.Group, "attach", hold_attach)
        ctx = hsm.Context()
        owner = ListeningAttachmentOwner()
        listening_ability = listening.Listening(
            voice_activity_classifier=FixedVoiceActivityClassifier(voice.detection.ApplyData(segments=()))
        )
        _ = await bot.started(ctx, owner, owner.model)
        await listening_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "listening-attach-failed",
            ),
        )
        await wait_until(lambda: len(requests) == 1)
        group, request = requests[0]
        request_data = request.data
        assert request_data is not None
        reply = request_data.reply_to
        assert reply is not None
        await hsm.dispatch(
            ctx,
            reply,
            dataclasses.replace(
                attachment.AttachFailedEvent.with_data(
                    attachment.FailedData(
                        actor=listening_ability,
                        kind=attachment.FailureKind.INITIALIZATION,
                        message="listening child failed",
                    )
                ),
                id=request.id,
                source=hsm.id(group),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        await wait_until(lambda: listening_ability.state().endswith("/detached"))
        await listening_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "listening-attach-retry",
            ),
        )
        await wait_until(lambda: len(requests) == 2)
        lifecycle = owner.lifecycle
        state = listening_ability.state()
        await listening_ability.stop(listening_ability.context())
        return lifecycle, len(requests), state

    lifecycle, request_count, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.AttachFailedEvent.name]
    assert lifecycle[0].id == "listening-attach-failed"
    assert isinstance(lifecycle[0].data, attachment.FailedData)
    assert lifecycle[0].data.kind is attachment.FailureKind.INITIALIZATION
    assert request_count == 2
    assert state == "/ListeningLifecycle/attached/behavior/initializing"


def test_listening_detaches_each_group_once_through_group_and_can_reattach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One detach from the owner, one detach per group, and the ability comes back afterwards."""

    async def run() -> tuple[list[hsm.Event[attachment.DetachData]], list[hsm.Event[typing.Any]], str]:
        requests: list[hsm.Event[attachment.DetachData]] = []
        group_detach = attachment.Group.detach

        async def record_detach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> None:
            requests.append(event)
            await group_detach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "detach", record_detach)
        ctx = hsm.Context()
        owner = ListeningAttachmentOwner()
        listening_ability = listening.Listening(
            voice_activity_classifier=FixedVoiceActivityClassifier(voice.detection.ApplyData(segments=()))
        )
        _ = await bot.started(ctx, owner, owner.model)
        await listening_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "listening-first-attach",
            ),
        )
        await wait_until(lambda: listening_ability.state().endswith("/Listening"))
        owner.lifecycle.clear()
        await listening_ability.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                "listening-detach",
            ),
        )
        await wait_until(lambda: listening_ability.state().endswith("/detached"))
        await listening_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "listening-second-attach",
            ),
        )
        await wait_until(lambda: listening_ability.state().endswith("/Listening"))
        lifecycle = owner.lifecycle
        state = listening_ability.state()
        await listening_ability.stop(listening_ability.context())
        return requests, lifecycle, state

    requests, lifecycle, state = asyncio.run(run())

    # Two groups now — the ear's and interpretation's — and the caller's one detach reaches each
    # exactly once. The ear's carries the caller's id; interpretation's is a detach of its own,
    # correlated by its own id, because it is releasing its own members and not the caller's.
    assert len(requests) == 2
    assert requests[0].id == "listening-detach"
    assert requests[1].id and requests[1].id != "listening-detach"
    assert [event.name for event in lifecycle] == [
        attachment.DetachedEvent.name,
        attachment.AttachCompleteEvent.name,
    ]
    assert [event.id for event in lifecycle] == ["listening-detach", "listening-second-attach"]
    assert state == "/ListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_apply_runs_voice_diarization_and_speech_decoding_pipeline() -> None:
    async def run() -> tuple[list[cognition.InputData], list[voice.diarization.InputData], list[bytes]]:
        diarizer = FixedVoiceDiarizer()
        listening, decoder = _listening(diarizer=diarizer)
        await start_ability_tree(None, listening)

        await _apply_utterance(listening, b"voice")
        await wait_until(lambda: any(_is_decoded_speech_handoff(h) for h in listening.handoffs))
        assert decoder is not None
        return list(listening.handoffs), diarizer.calls, decoder.calls

    handoffs, diarization_calls, decoder_calls = asyncio.run(run())
    stt = next(h for h in handoffs if _is_decoded_speech_handoff(h))
    stimulus = _stimulus(stt)

    assert isinstance(stimulus.data, listening.SpeechData)
    assert stimulus.data.content == "decoded:voicesilence"
    assert stimulus.data.content_type == "text/plain"
    assert [call.audio for call in diarization_calls] == [b"voicesilence"]
    assert [(call.media_type, call.sample_rate_hz, call.channels) for call in diarization_calls] == [
        ("audio/pcm", 48_000, 1)
    ]
    assert decoder_calls == [b"voicesilence"]


def test_listening_model_tracks_detection_diarization_and_decoding_lifecycle() -> None:
    model = model_view(require_model(listening.Listening.model))

    assert model.qualified_name == "/ListeningLifecycle"
    assert model.initial == "/ListeningLifecycle/.initial"
    assert "/ListeningLifecycle/detached" in model.members
    assert "/ListeningLifecycle/attaching" not in model.members
    assert "/ListeningLifecycle/attached" in model.members
    assert "/ListeningLifecycle/attached/behavior/initializing" in model.members
    assert "/ListeningLifecycle/attached/behavior/Perceiving" in model.members
    assert "/ListeningLifecycle/attached/behavior/Perceiving/Listening" in model.members
    assert "/ListeningLifecycle/attached/behavior/Perceiving/Sensing" in model.members
    assert "/ListeningLifecycle/attached/behavior/detaching" in model.members
    assert "/ListeningLifecycle/attached/behavior/degraded" in model.members
    # Nothing slow is in here. Every interpretation stage lives in its own machine, which is what
    # keeps a decoder from standing between a sound and the prediction it has to be scored
    # against; a stage reappearing here would put the queue back in front of the ear.
    members = model.members
    assert isinstance(members, collections.abc.Iterable)
    assert not [
        member
        for member in members
        if member.startswith("/ListeningLifecycle/attached/behavior/Perceiving/")
        and member.rsplit("/", 1)[-1]
        in {
            "DetectingVoice",
            "ClassifyingSound",
            "RoutingDetectedVoice",
            "DiarizingVoice",
            "DecodingSpeech",
            "HandingOff",
        }
    ]
    initializing_events = model.transition_map["/ListeningLifecycle/attached/behavior/initializing"]
    assert "bot.ability.attachment.terminal" in initializing_events
    assert "bot.ability.listening.children.attached" not in initializing_events
    assert "environment.sound" in model.transition_map["/ListeningLifecycle/attached/behavior/Perceiving/Listening"]
    assert (
        "bot.ability.listening.sensitivity.completed"
        in model.transition_map["/ListeningLifecycle/attached/behavior/Perceiving/Sensing"]
    )


def test_interpretation_model_tracks_detection_diarization_and_decoding_lifecycle() -> None:
    model = model_view(require_model(listening.interpretation.Interpretation.model))
    working = "/InterpretationLifecycle/attached/behavior/Working"

    assert model.qualified_name == "/InterpretationLifecycle"
    assert "/InterpretationLifecycle/attached/behavior/initializing" in model.members
    assert "/InterpretationLifecycle/attached/behavior/Idle" in model.members
    # Working owns the common sensitivity-output deferral. HearingSpeech has the only
    # leaf-specific input transition while the utterance is open.
    assert working in model.members
    assert f"{working}/HandingOff" in model.members
    assert f"{working}/DetectingVoice" in model.members
    assert f"{working}/HearingSpeech" in model.members
    assert f"{working}/FeedingSpeech" in model.members
    assert f"{working}/SpeechEnded" in model.members
    assert f"{working}/ClassifyingSound" in model.members
    assert f"{working}/DiarizingVoice" in model.members
    assert f"{working}/IdentifyingVoiceDirect" in model.members
    assert f"{working}/DecodingSpeech" in model.members
    assert f"{working}/DecodingSpeech/Detected" in model.members
    assert f"{working}/DecodingSpeech/Diarized" in model.members
    assert "/InterpretationLifecycle/attached/behavior/detaching" in model.members
    assert "/InterpretationLifecycle/attached/behavior/degraded" in model.members
    # Stage failure is owned once on Working, not on every leaf.
    working_events = typing.cast(collections.abc.Iterable[str], typing.cast(object, model.transition_map[working]))
    assert any("stage" in name and "failed" in name for name in working_events)
    # A scored sound is what comes in, so nothing here can be reached with an unscored one.
    assert (
        "bot.ability.listening.sensitivity.output"
        in model.transition_map["/InterpretationLifecycle/attached/behavior/Idle"]
    )
    assert "bot.ability.listening.voice_detection.completed" in model.transition_map[f"{working}/DetectingVoice"]
    assert "bot.ability.listening.sensitivity.output" in model.transition_map[f"{working}/HearingSpeech"]
    assert "bot.ability.listening.voice_diarization.completed" in model.transition_map[f"{working}/DiarizingVoice"]
    assert "bot.ability.listening.speech_decoding.completed" in model.transition_map[f"{working}/DecodingSpeech"]
    deferred_map = typing.cast(
        collections.abc.Mapping[str, collections.abc.Mapping[str, str]],
        typing.cast(object, getattr(model, "deferred_map")),
    )
    assert deferred_map[working]["bot.ability.listening.sensitivity.output"] == working
    for state in (
        "DetectingVoice",
        "FeedingSpeech",
        "ClassifyingSound",
        "DiarizingVoice",
        "IdentifyingVoiceDirect",
        "IdentifyingVoice",
        "DecodingSpeech",
    ):
        assert deferred_map[f"{working}/{state}"]["bot.ability.listening.sensitivity.output"] == working


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
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_publishes_sound_cognition_input_when_sound_classifier_labels_nonvoice() -> None:
    async def run() -> tuple[list[cognition.InputData], list[SoundData], list[bytes], str]:
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
    assert len(classifier_calls) == 1
    assert classifier_calls[0].audio == b"alarm-tone"
    assert decoder_calls == []
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_skips_cognition_input_when_sound_classifier_finds_no_labels_with_stt() -> None:
    """STT path keeps dropping unlabeled non-speech (no silence product for Conversation)."""

    async def run() -> tuple[list[cognition.InputData], list[SoundData], str]:
        classifier = EmptySoundClassifier()
        listening, _ = _listening(is_voice=False, decoder=True, sound_classifier=classifier)
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"ambient"))
        await wait_until(lambda: bool(classifier.calls))

        return listening.handoffs, classifier.calls, listening.state()

    handoffs, classifier_calls, active_state = asyncio.run(run())

    assert handoffs == []
    assert len(classifier_calls) == 1
    assert classifier_calls[0].audio == b"ambient"
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_publishes_silence_observation_when_classifier_finds_no_labels_without_stt() -> None:
    """Acoustic path: unlabeled no-voice still hands off a silence observation so sticky turns close."""

    async def run() -> tuple[list[cognition.InputData], list[SoundData], str]:
        classifier = EmptySoundClassifier()
        listening, _ = _listening(is_voice=False, decoder=False, sound_classifier=classifier)
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"ambient"))
        await wait_until(lambda: bool(listening.handoffs))

        return listening.handoffs, classifier.calls, listening.state()

    handoffs, classifier_calls, active_state = asyncio.run(run())
    stimulus = _stimulus(handoffs[0])

    assert stimulus.name == listening.SpeechEvent.name
    assert isinstance(stimulus.data, listening.SpeechData)
    assert stimulus.data.content == b"ambient"
    assert stimulus.data.voice_detection.segments == ()
    assert len(classifier_calls) == 1
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_kind_sound_classifier_labels_present_kind() -> None:
    async def run() -> sound_hearing.classification.OutputData:
        classifier = sound_hearing.classification.KindSoundClassifier()
        return await classifier.classify(SoundData(audio=b"ring-audio", kind="phone.ringing"))

    assert asyncio.run(run()) == sound_hearing.classification.OutputData(labels=("phone.ringing",), confidence=1.0)


def test_kind_sound_classifier_unlabeled_without_kind() -> None:
    async def run() -> sound_hearing.classification.OutputData:
        classifier = sound_hearing.classification.KindSoundClassifier()
        return await classifier.classify(SoundData(audio=b"noise"))

    assert asyncio.run(run()) == sound_hearing.classification.OutputData(labels=())


def test_listening_publishes_kind_labeled_nonvoice_sound() -> None:
    async def run() -> tuple[list[cognition.InputData], str]:
        listening, _ = _listening(
            is_voice=False,
            decoder=False,
            sound_classifier=sound_hearing.classification.KindSoundClassifier(),
        )
        await start_ability_tree(None, listening)
        _ = await listening.apply(SoundData(audio=b"ring-audio", media_type="audio/wav", kind="phone.ringing"))
        await wait_until(lambda: bool(listening.handoffs))
        return listening.handoffs, listening.state()

    handoffs, active_state = asyncio.run(run())
    stimulus = _stimulus(handoffs[0])
    assert stimulus.name == SoundEvent.name
    assert isinstance(stimulus.data, SoundData)
    assert stimulus.data.kind == "phone.ringing"
    assert stimulus.data.audio == b"ring-audio"
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


MOUTH = "the-mouth-that-is-producing"


async def start_producing(listening_ability: listening.Listening, *, duration: float = 1.0) -> None:
    """Tell this perception that the body has just commanded its mouth, as the body would."""

    await hsm.dispatch(
        hsm.Context(),
        listening_ability,
        EfferenceEvent.with_data(
            EfferenceData(mouth=MOUTH, duration=duration, media_type="audio/pcm", sample_rate_hz=16_000)
        ),
    )


async def arrives_from(listening_ability: listening.Listening, sound_data: SoundData, *, source: str) -> None:
    """Deliver a sound with the transducer that made it on the envelope, as the environment does."""

    await hsm.dispatch(
        hsm.Context(),
        listening_ability,
        dataclasses.replace(SoundEvent.with_data(sound_data), source=source),
    )


def own_production(audio: bytes, *, kind: str | None = None) -> SoundData:
    """A sound with the levels a mouth 15 cm from its own ears produces: +16.5 dB of path gain."""

    return SoundData(
        audio=audio,
        media_type="audio/pcm",
        sample_rate_hz=16_000,
        channels=1,
        kind=kind,
        amplitude_db=60.0,
        received_level_db=76.5,
    )


def test_listening_attenuates_a_labeled_nonvoice_sound_the_body_is_producing() -> None:
    """The sound is classified and labeled in full; it just does not become a turn.

    Attenuation lands at the handoff, not before it: the classifier still ran on the bot's own
    noise, which is what makes this attenuation rather than deafness.
    """

    async def run() -> tuple[list[cognition.InputData], list[SoundData], str]:
        classifier = FixedSoundClassifier(labels=("hum",))
        listening_ability, _ = _listening(is_voice=False, decoder=False, sound_classifier=classifier)
        await start_ability_tree(None, listening_ability)
        calls = classifier.calls

        await start_producing(listening_ability)
        await arrives_from(listening_ability, own_production(b"own-hum"), source=MOUTH)
        await wait_until(lambda: bool(calls))
        await asyncio.sleep(0.05)
        return listening_ability.handoffs, calls, listening_ability.state()

    handoffs, calls, active_state = asyncio.run(run())

    assert handoffs == []
    assert [call.audio for call in calls] == [b"own-hum"]
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_publishes_the_same_sound_once_the_body_has_stopped_producing() -> None:
    """Same sound, same levels, same mouth — arriving after the command it belonged to."""

    async def run() -> list[cognition.InputData]:
        listening_ability, _ = _listening(
            is_voice=False,
            decoder=False,
            sound_classifier=FixedSoundClassifier(labels=("hum",)),
        )
        await start_ability_tree(None, listening_ability)

        await start_producing(listening_ability, duration=0.05)
        await asyncio.sleep(0.2)
        await arrives_from(listening_ability, own_production(b"own-hum"), source=MOUTH)
        await wait_until(lambda: bool(listening_ability.handoffs))
        return listening_ability.handoffs

    handoffs = asyncio.run(run())

    stimulus = _stimulus(handoffs[0])
    assert isinstance(stimulus.data, SoundData)
    assert stimulus.data.audio == b"own-hum"


def test_the_product_threshold_is_the_one_place_a_level_becomes_a_yes_or_a_no() -> None:
    """Every route through perception is compared once, here, against one injected number.

    Nothing about this sound is predicted — nothing is being produced — so its full 76.5 dB
    survives scoring, and it is the threshold alone that decides. That is the seam a second
    contributor plugs into: reduce the perceived level and this comparison does the rest, with
    no second threshold and no second opinion about what quiet means.
    """

    async def run() -> tuple[list[cognition.InputData], list[cognition.InputData]]:
        deaf, _ = _listening(
            is_voice=False,
            decoder=False,
            sound_classifier=FixedSoundClassifier(labels=("hum",)),
            product_threshold_db=120.0,
        )
        await start_ability_tree(None, deaf)
        await arrives_from(deaf, own_production(b"hum"), source=MOUTH)
        await asyncio.sleep(0.05)
        nothing = list(deaf.handoffs)

        ordinary, _ = _listening(
            is_voice=False,
            decoder=False,
            sound_classifier=FixedSoundClassifier(labels=("hum",)),
        )
        await start_ability_tree(None, ordinary)
        await arrives_from(ordinary, own_production(b"hum"), source=MOUTH)
        await wait_until(lambda: bool(ordinary.handoffs))
        return nothing, list(ordinary.handoffs)

    nothing, something = asyncio.run(run())

    assert nothing == []
    assert len(something) == 1


def test_a_negative_product_threshold_is_refused() -> None:
    try:
        _ = listening.Listening(
            voice_activity_classifier=FixedVoiceActivityClassifier(
                voice.detection.ApplyData(
                    segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.0),)
                )
            )
        )
        _ = listening.Listening(
            voice_activity_classifier=FixedVoiceActivityClassifier(
                voice.detection.ApplyData(
                    segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.0),)
                )
            ),
            product_threshold_db=-1.0,
        )
    except ValueError as error:
        assert "product_threshold_db" in str(error)
    else:
        raise AssertionError("a negative product threshold must be refused.")


def test_listening_publishes_speech_when_speech_decoding_is_absent() -> None:
    async def run() -> tuple[list[cognition.InputData], str]:
        listening_ability, _ = _listening(decoder=False)
        await start_ability_tree(None, listening_ability)

        await _apply_utterance(listening_ability, b"voice")
        await wait_until(lambda: any(_is_silent_speech_handoff(h) for h in listening_ability.handoffs))

        return listening_ability.handoffs, listening_ability.state()

    handoffs, active_state = asyncio.run(run())
    observations = [_stimulus(h).data for h in handoffs if _stimulus(h).name == listening.SpeechEvent.name]
    assert isinstance(observations[0], listening.SpeechData)
    assert observations[0].content == b"voice"
    assert len(observations[0].voice_detection.segments) == 1
    assert observations[0].sample_rate_hz == 48_000
    assert observations[0].media_type == "audio/pcm"
    assert isinstance(observations[-1], listening.SpeechData)
    assert observations[-1].voice_detection.segments == ()
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_publishes_speech_after_diarization_when_speech_decoding_is_absent() -> None:
    async def run() -> tuple[list[cognition.InputData], list[voice.diarization.InputData], str]:
        diarizer = FixedVoiceDiarizer()
        listening_ability, _ = _listening(diarizer=diarizer, decoder=False)
        await start_ability_tree(None, listening_ability)

        await _apply_utterance(listening_ability, b"voice")
        await wait_until(lambda: bool(diarizer.calls) and bool(listening_ability.handoffs))
        await wait_until(
            lambda: listening_ability.state() == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"
        )

        return listening_ability.handoffs, diarizer.calls, listening_ability.state()

    handoffs, diarizer_calls, active_state = asyncio.run(run())
    observations = [_stimulus(h).data for h in handoffs if _stimulus(h).name == listening.SpeechEvent.name]
    assert isinstance(observations[0], listening.SpeechData)
    assert observations[0].content == DIARIZED_AUDIO
    assert observations[0].start_seconds == 0.0
    assert observations[0].end_seconds == 1.25
    assert observations[0].confidence == 0.87
    # Diarize runs on the full utterance (voice + silence bytes) after VAD End.
    assert [call.audio for call in diarizer_calls] == [b"voicesilence"]
    assert [(call.media_type, call.sample_rate_hz, call.channels) for call in diarizer_calls] == [
        ("audio/pcm", 48_000, 1)
    ]
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_preserves_non_zero_clipped_diarization_timing_relative_to_voice_span() -> None:
    clipped_audio = b"\x00\x00" * 24_000
    diarizer = FixedVoiceDiarizer(
        segment=voice.diarization.VoiceDiarizationSegment(
            audio=clipped_audio,
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            channels=1,
            start_seconds=0.25,
            end_seconds=0.75,
            confidence=0.87,
        )
    )

    async def run() -> list[cognition.InputData]:
        listening_ability, _ = _listening(diarizer=diarizer, decoder=False)
        await start_ability_tree(None, listening_ability)

        await _apply_utterance(listening_ability, b"voice")
        await wait_until(lambda: bool(listening_ability.handoffs))
        return listening_ability.handoffs

    handoffs = asyncio.run(run())
    observations = [_stimulus(h).data for h in handoffs if _stimulus(h).name == listening.SpeechEvent.name]
    assert len(observations) == 1
    observation = observations[0]
    assert isinstance(observation, listening.SpeechData)
    assert observation.content == clipped_audio
    assert observation.start_seconds == 0.25
    assert observation.end_seconds == 0.75
    assert observation.voice_detection.segments[0].start_seconds == 0.0
    assert observation.voice_detection.segments[0].end_seconds == pytest.approx(0.5)


def test_listening_reports_typed_failure_when_diarized_product_conversion_fails() -> None:
    diarizer = FixedVoiceDiarizer(
        segment=voice.diarization.VoiceDiarizationSegment(
            audio=b"\x00\x00",
            media_type="audio/pcm",
            sample_rate_hz=2,
            channels=1,
            start_seconds=0.25,
            end_seconds=0.75,
            confidence=0.87,
        )
    )

    async def run() -> tuple[list[listening.FailedEventData], list[cognition.InputData], str]:
        listening_ability, _ = _listening(diarizer=diarizer, decoder=False)
        await start_ability_tree(None, listening_ability)

        await _apply_utterance(listening_ability, b"voice")
        await wait_until(lambda: bool(listening_ability.failures))
        return listening_ability.failures, listening_ability.handoffs, listening_ability.state()

    failures, handoffs, active_state = asyncio.run(run())

    assert failures == [
        listening.FailedEventData(
            stage="voice_diarization",
            message="Voice diarization product conversion failed: voice segment audio format does not match its source audio.",
        )
    ]
    assert handoffs == []
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_publishes_silence_speech_when_speech_decoding_is_absent() -> None:
    """Empty VAD + no STT still hands off so sticky turns can close."""

    async def run() -> list[cognition.InputData]:
        listening, _ = _listening(decoder=False, is_voice=False)
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"quiet"))
        await wait_until(lambda: bool(listening.handoffs))
        return listening.handoffs

    handoffs = asyncio.run(run())
    stimulus = _stimulus(handoffs[0])
    assert stimulus.name == listening.SpeechEvent.name
    assert isinstance(stimulus.data, listening.SpeechData)
    assert stimulus.data.content == b"quiet"
    assert stimulus.data.voice_detection.segments == ()


def test_listening_publishes_speech_cognition_input_when_voice_is_detected() -> None:
    async def run() -> tuple[list[cognition.InputData], list[bytes], str]:
        listening, decoder = _listening()
        await start_ability_tree(None, listening)

        await _apply_utterance(listening, b"voice")
        await wait_until(lambda: any(_is_decoded_speech_handoff(h) for h in listening.handoffs))

        assert decoder is not None
        return listening.handoffs, decoder.calls, listening.state()

    handoffs, decoder_calls, active_state = asyncio.run(run())
    stt = next(h for h in handoffs if _is_decoded_speech_handoff(h))
    stimulus = _stimulus(stt)

    assert isinstance(stimulus.data, listening.SpeechData)
    assert stimulus.data.content == "decoded:voicesilence"
    assert stimulus.data.content_type == "text/plain"
    assert decoder_calls == [b"voicesilence"]
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_runs_optional_diarization_before_decoding_speech() -> None:
    async def run() -> tuple[
        list[cognition.InputData],
        list[voice.diarization.InputData],
        list[bytes],
    ]:
        diarizer = FixedVoiceDiarizer()
        listening, decoder = _listening(diarizer=diarizer)
        await start_ability_tree(None, listening)

        await _apply_utterance(listening, b"voice")
        await wait_until(lambda: any(_is_decoded_speech_handoff(h) for h in listening.handoffs))

        assert decoder is not None
        return listening.handoffs, diarizer.calls, decoder.calls

    handoffs, diarizer_calls, decoder_calls = asyncio.run(run())
    stt = next(h for h in handoffs if _is_decoded_speech_handoff(h))
    stimulus = _stimulus(stt)

    assert isinstance(stimulus.data, listening.SpeechData)
    assert stimulus.data.content == "decoded:voicesilence"
    assert stimulus.data.content_type == "text/plain"
    assert [call.audio for call in diarizer_calls] == [b"voicesilence"]
    assert [(call.media_type, call.sample_rate_hz, call.channels) for call in diarizer_calls] == [
        ("audio/pcm", 48_000, 1)
    ]
    assert decoder_calls == [b"voicesilence"]


@pytest.mark.live
def test_listening_decodes_real_wav_with_real_mlx_providers() -> None:
    """Exercise Listening through real MLX VAD and speech decoding providers."""
    mlx_audio = pytest.importorskip("bot.providers.mlx_audio")

    async def run() -> cognition.InputData:
        speech_wav_path = (
            pathlib.Path(__file__).resolve().parents[3] / "src/providers/mlx_audio/tests/assets/speech.wav"
        )
        speech_wav = speech_wav_path.read_bytes()
        with wave.open(io.BytesIO(speech_wav), "rb") as speech_file:
            sample_rate_hz = speech_file.getframerate()
            channels = speech_file.getnchannels()
            assert speech_file.getsampwidth() == 2

        silence_buffer = io.BytesIO()
        with wave.open(silence_buffer, "wb") as silence_file:
            silence_file.setnchannels(channels)
            silence_file.setsampwidth(2)
            silence_file.setframerate(sample_rate_hz)
            silence_file.writeframes(b"\x00\x00" * (sample_rate_hz // 2 * channels))

        voice_activity_classifier = mlx_audio.VoiceActivityClassifier(
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )
        speech_decoder = mlx_audio.SpeechDecoder()
        listening_ability = listening.Listening(
            voice_activity_classifier=voice_activity_classifier,
            speech_decoder=speech_decoder,
        )
        ctx = hsm.Context()
        await start_ability_tree(ctx, listening_ability)

        await listening_ability.apply(
            SoundData(
                audio=speech_wav,
                media_type="audio/wav",
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                kind="test.speech",
            ),
            ctx=ctx,
        )
        return await dispatch_ability_for_test(
            listening_ability,
            ctx,
            SoundData(
                audio=silence_buffer.getvalue(),
                media_type="audio/wav",
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                kind="test.silence",
            ),
            timeout=120.0,
        )

    handoff = asyncio.run(run())
    stimulus = _stimulus(handoff)
    assert stimulus.name == listening.SpeechEvent.name
    assert isinstance(stimulus.data, listening.SpeechData)
    assert stimulus.data.content_type == "text/plain"
    assert isinstance(stimulus.data.content, str)
    assert stimulus.data.content.strip()


def test_listening_detach_releases_owned_subabilities_while_detecting_voice() -> None:
    async def run() -> tuple[bool, str]:
        ctx = hsm.Context()
        voice_activity_classifier = HangingVoiceActivityClassifier()
        listening_ability = RecordingListening(
            voice_activity_classifier=voice_activity_classifier,
            speech_decoder=RecordingSpeechDecoder(),
        )
        await start_ability_tree(ctx, listening_ability)
        _ = await listening_ability.apply(sound(b"voice"), ctx=ctx)
        # Waited on through the classifier rather than through a state name: the ear hands the sound
        # on and goes straight back to listening, so it is never the thing sitting in detection.
        await wait_until(lambda: voice_activity_classifier.entered.is_set())
        assert listening_ability.state() == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"

        owner = ability_terminal_owner(listening_ability)
        assert owner is not None
        _ = await listening_ability.detach(
            ctx,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )
        await wait_until(lambda: listening_ability.state().endswith("/detached"))
        await wait_until(lambda: voice_activity_classifier.cancelled)
        state = listening_ability.state()
        await listening_ability.stop(ctx)
        return voice_activity_classifier.cancelled, state

    cancelled, state = asyncio.run(run())

    assert cancelled
    assert state == "/RecordingListeningLifecycle/detached"


def test_listening_dispatches_failure_when_detection_fails() -> None:
    async def run() -> tuple[list[listening.FailedEventData], list[cognition.InputData], str]:
        listening_ability = RecordingListening(
            voice_activity_classifier=FailingVoiceActivityClassifier(),
            speech_decoder=RecordingSpeechDecoder(),
        )
        await start_ability_tree(None, listening_ability)

        with pytest.raises(RuntimeError, match="voice activity classifier offline"):
            _ = await dispatch_ability_for_test(listening_ability, hsm.Context(), sound(b"voice"))
        await wait_until(lambda: bool(listening_ability.failures))

        return listening_ability.failures, listening_ability.handoffs, listening_ability.state()

    failures, handoffs, active_state = asyncio.run(run())

    assert failures == [
        listening.FailedEventData(stage="voice_detection", message="voice activity classifier offline"),
    ]
    assert handoffs == []
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_failure_payload_reuses_ability_failure_message_shape() -> None:
    failure = listening.FailedEventData.from_ability_failure(
        stage="speech_decoding",
        failure=abilities.FailureData(message="decoder timed out"),
    )

    assert failure == listening.FailedEventData(stage="speech_decoding", message="decoder timed out")


def test_interpretation_routes_child_output_and_preserves_wav_for_vad() -> None:
    """A public Interpretation run preserves WAV ingress while consuming typed child output."""

    from bot.abilities.hearing import voice
    from bot.abilities.listening import interpretation as interpretation_module
    from bot.abilities.listening import sensitivity

    class RecordingVoiceActivityClassifier(voice.detection.VoiceActivityClassifier):
        inputs: list[bytes]

        def __init__(self) -> None:
            self.inputs = []

        @override
        async def classify(self, input: bytes) -> voice.detection.ApplyData:
            self.inputs.append(input)
            return voice.detection.ApplyData(
                segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=0.1),)
            )

    async def run() -> tuple[bytes, cognition.InputData, bytes, bytes]:
        classifier = RecordingVoiceActivityClassifier()
        instance = interpretation_module.Interpretation(voice_activity_classifier=classifier)
        ctx = hsm.Context()
        await start_ability_tree(ctx, instance)

        pcm = bytes((0, 64)) * 1600
        wav = wav_sound(pcm, sample_rate_hz=16_000)
        sensed = sensitivity.OutputData(sound=wav, perceived_level_db=60.0)
        handoff = await dispatch_ability_for_test(instance, ctx, sensed)

        assert len(classifier.inputs) == 1
        return classifier.inputs[0], handoff, pcm, wav.audio

    vad_input, handoff, pcm, wav_audio = asyncio.run(run())

    assert vad_input == wav_audio
    assert vad_input.startswith(b"RIFF")
    assert vad_input[8:12] == b"WAVE"
    stimulus = _stimulus(handoff)
    assert stimulus.name == listening.SpeechEvent.name
    assert isinstance(stimulus.data, listening.SpeechData)
    assert stimulus.data.content == pcm
    assert stimulus.data.media_type == "audio/pcm"
