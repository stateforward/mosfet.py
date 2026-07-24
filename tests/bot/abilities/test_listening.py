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
import typing
from typing import override

import hsm
from tests.bot.abilities.support import dispatch_ability_for_test
import pytest

from bot.world import SoundData, SoundEvent
from tests.hsm_instance_state import ability_terminal_owner, start_ability_tree
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

    model: typing.ClassVar[hsm.Model] = hsm.define(
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
            voice_detector=FixedVoiceDetector(voice.detection.OutputData(is_voice=False, confidence=0.9))
        )
        maximum = listening.Listening(
            voice_detector=FixedVoiceDetector(voice.detection.OutputData(is_voice=True, confidence=0.9)),
            sound_classifier=FixedSoundClassifier(),
            voice_diarizer=FixedVoiceDiarizer(),
            speech_decoder=RecordingSpeechDecoder(),
        )
        owner = hsm.Instance()
        owner_model = hsm.define(
            "ListeningAttachmentOwner",
            hsm.initial(hsm.target("ready")),
            hsm.state("ready"),
        )
        ctx = hsm.Context()
        _ = await hsm.started(ctx, owner, owner_model)
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

    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert isinstance(groups[0][0], voice.detection.VoiceDetection)
    assert len(groups[1]) == 4
    assert isinstance(groups[1][0], voice.detection.VoiceDetection)
    assert isinstance(groups[1][1], sound_hearing.classification.SoundClassification)
    assert isinstance(groups[1][2], voice.diarization.VoiceDiarization)
    assert isinstance(groups[1][3], speech.SpeechDecoding)
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
            voice_detector=FixedVoiceDetector(voice.detection.OutputData(is_voice=False, confidence=0.9))
        )
        _ = await hsm.started(ctx, owner, owner.model)
        await listening_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "listening-attach-failed",
            ),
        )
        await wait_until(lambda: len(requests) == 1)
        group, request = requests[0]
        reply = request.data.reply_to
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


def test_listening_detaches_once_through_group_and_can_reattach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            voice_detector=FixedVoiceDetector(voice.detection.OutputData(is_voice=False, confidence=0.9))
        )
        _ = await hsm.started(ctx, owner, owner.model)
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

    assert len(requests) == 1
    assert requests[0].id == "listening-detach"
    assert [event.name for event in lifecycle] == [
        attachment.DetachedEvent.name,
        attachment.AttachCompleteEvent.name,
    ]
    assert [event.id for event in lifecycle] == ["listening-detach", "listening-second-attach"]
    assert state == "/ListeningLifecycle/attached/behavior/Listening"


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
    assert "/ListeningLifecycle/attaching" not in model.members
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
    assert "/ListeningLifecycle/attached/behavior/detaching" in model.members
    assert "/ListeningLifecycle/attached/behavior/degraded" in model.members
    initializing_events = model.transition_map["/ListeningLifecycle/attached/behavior/initializing"]
    assert "bot.ability.attachment.terminal" in initializing_events
    assert "bot.ability.listening.children.attached" not in initializing_events
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
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Listening"


def test_listening_skips_cognition_input_when_sound_classifier_finds_no_labels() -> None:
    async def run() -> tuple[list[cognition.InputData], list[SoundData], str]:
        classifier = EmptySoundClassifier()
        listening, _ = _listening(is_voice=False, decoder=False, sound_classifier=classifier)
        await start_ability_tree(None, listening)

        _ = await listening.apply(sound(b"ambient"))
        await asyncio.sleep(0.05)

        return listening.handoffs, classifier.calls, listening.state()

    handoffs, classifier_calls, active_state = asyncio.run(run())

    assert handoffs == []
    assert len(classifier_calls) == 1
    assert classifier_calls[0].audio == b"ambient"
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Listening"


def test_kind_sound_classifier_labels_present_kind() -> None:
    async def run() -> sound_hearing.classification.OutputData:
        classifier = sound_hearing.classification.KindSoundClassifier()
        return await classifier.classify(SoundData(audio=b"ring-clip", kind="phone.ringing"))

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
        _ = await listening.apply(SoundData(audio=b"ring-clip", media_type="audio/wav", kind="phone.ringing"))
        await wait_until(lambda: bool(listening.handoffs))
        return listening.handoffs, listening.state()

    handoffs, active_state = asyncio.run(run())
    stimulus = _stimulus(handoffs[0])
    assert stimulus.name == SoundEvent.name
    assert isinstance(stimulus.data, SoundData)
    assert stimulus.data.kind == "phone.ringing"
    assert stimulus.data.audio == b"ring-clip"
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
    async def run() -> tuple[bool, str]:
        ctx = hsm.Context()
        detector = HangingVoiceDetector()
        listening_ability = RecordingListening(
            voice_detector=detector,
            speech_decoder=RecordingSpeechDecoder(),
        )
        await start_ability_tree(ctx, listening_ability)
        _ = await listening_ability.apply(sound(b"voice"), ctx=ctx)
        await wait_until(
            lambda: listening_ability.state() == "/RecordingListeningLifecycle/attached/behavior/DetectingVoice"
        )
        assert listening_ability.state() == "/RecordingListeningLifecycle/attached/behavior/DetectingVoice"

        owner = ability_terminal_owner(listening_ability)
        assert owner is not None
        _ = await listening_ability.detach(
            ctx,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )
        await wait_until(lambda: listening_ability.state().endswith("/detached"))
        await wait_until(lambda: detector.cancelled)
        state = listening_ability.state()
        await listening_ability.stop(ctx)
        return detector.cancelled, state

    cancelled, state = asyncio.run(run())

    assert cancelled
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
