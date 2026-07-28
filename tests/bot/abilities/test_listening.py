from bot import abilities
from bot.abilities import cognition
from bot.abilities import listening
from bot.abilities import speaking
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

from bot.environment import SoundData, SoundEvent
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
    entered: asyncio.Event

    def __init__(self) -> None:
        self.cancelled = False
        self.entered = asyncio.Event()

    @override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
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
        product_threshold_db: float | None = None,
    ) -> None:
        thresholds = {} if product_threshold_db is None else {"product_threshold_db": product_threshold_db}
        super().__init__(
            voice_detector=voice_detector,
            sound_classifier=sound_classifier,
            speech_decoder=speech_decoder,
            voice_diarizer=voice_diarizer,
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
    product_threshold_db: float | None = None,
) -> tuple[RecordingListening, RecordingSpeechDecoder | None]:
    speech_decoder: RecordingSpeechDecoder | None
    if decoder is False:
        speech_decoder = None
    elif decoder is None:
        speech_decoder = RecordingSpeechDecoder()
    else:
        speech_decoder = decoder
    thresholds = {} if product_threshold_db is None else {"product_threshold_db": product_threshold_db}
    listening_ability = RecordingListening(
        voice_detector=FixedVoiceDetector(voice.detection.OutputData(is_voice=is_voice, confidence=0.91)),
        sound_classifier=sound_classifier,
        speech_decoder=speech_decoder,
        voice_diarizer=diarizer,
        **thresholds,
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
    assert listening.Listening.input_event.name == "environment.sound"
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
    assert "/ListeningLifecycle/attached/behavior/Perceiving" in model.members
    assert "/ListeningLifecycle/attached/behavior/Perceiving/Listening" in model.members
    assert "/ListeningLifecycle/attached/behavior/Perceiving/Sensing" in model.members
    assert "/ListeningLifecycle/attached/behavior/detaching" in model.members
    assert "/ListeningLifecycle/attached/behavior/degraded" in model.members
    # Nothing slow is in here. Every interpretation stage lives in its own machine, which is what
    # keeps a decoder from standing between a sound and the prediction it has to be scored
    # against; a stage reappearing here would put the queue back in front of the ear.
    assert not [
        member
        for member in model.members
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

    assert model.qualified_name == "/InterpretationLifecycle"
    assert "/InterpretationLifecycle/attached/behavior/initializing" in model.members
    assert "/InterpretationLifecycle/attached/behavior/Idle" in model.members
    assert "/InterpretationLifecycle/attached/behavior/HandingOff" in model.members
    assert "/InterpretationLifecycle/attached/behavior/DetectingVoice" in model.members
    assert "/InterpretationLifecycle/attached/behavior/ClassifyingSound" in model.members
    assert "/InterpretationLifecycle/attached/behavior/RoutingDetectedVoice" in model.members
    assert "/InterpretationLifecycle/attached/behavior/DiarizingVoice" in model.members
    assert "/InterpretationLifecycle/attached/behavior/DecodingSpeech" in model.members
    assert "/InterpretationLifecycle/attached/behavior/DecodingSpeech/Detected" in model.members
    assert "/InterpretationLifecycle/attached/behavior/DecodingSpeech/Diarized" in model.members
    assert "/InterpretationLifecycle/attached/behavior/detaching" in model.members
    assert "/InterpretationLifecycle/attached/behavior/degraded" in model.members
    # A scored sound is what comes in, so nothing here can be reached with an unscored one.
    assert (
        "bot.ability.listening.sensitivity.output"
        in model.transition_map["/InterpretationLifecycle/attached/behavior/Idle"]
    )
    assert (
        "bot.ability.listening.voice_detection.completed"
        in model.transition_map["/InterpretationLifecycle/attached/behavior/DetectingVoice"]
    )
    assert (
        "bot.ability.listening.voice_diarization.completed"
        in model.transition_map["/InterpretationLifecycle/attached/behavior/DiarizingVoice"]
    )
    assert (
        "bot.ability.listening.speech_decoding.completed"
        in model.transition_map["/InterpretationLifecycle/attached/behavior/DecodingSpeech"]
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
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


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
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


MOUTH = "the-mouth-that-is-producing"


async def start_producing(listening_ability: listening.Listening, *, duration: float = 1.0) -> None:
    """Tell this perception that the body has just commanded its mouth, as the body would."""

    await hsm.dispatch(
        hsm.Context(),
        listening_ability,
        speaking.EfferenceEvent.with_data(
            speaking.EfferenceData(mouth=MOUTH, duration=duration, media_type="audio/pcm", sample_rate_hz=16_000)
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
        _ = listening.Listening(voice_detector=FixedVoiceDetector(voice.detection.OutputData(is_voice=True)))
        _ = listening.Listening(
            voice_detector=FixedVoiceDetector(voice.detection.OutputData(is_voice=True)),
            product_threshold_db=-1.0,
        )
    except ValueError as error:
        assert "product_threshold_db" in str(error)
    else:
        raise AssertionError("a negative product threshold must be refused.")


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
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


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
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


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
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


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
        # Waited on through the detector rather than through a state name: the ear hands the sound
        # on and goes straight back to listening, so it is never the thing sitting in detection.
        await wait_until(lambda: detector.entered.is_set())
        assert listening_ability.state() == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"

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
    assert active_state == "/RecordingListeningLifecycle/attached/behavior/Perceiving/Listening"


def test_listening_failure_payload_reuses_ability_failure_message_shape() -> None:
    failure = listening.FailedEventData.from_ability_failure(
        stage="speech_decoding",
        failure=abilities.FailureData(message="decoder timed out"),
    )

    assert failure == listening.FailedEventData(stage="speech_decoding", message="decoder timed out")
