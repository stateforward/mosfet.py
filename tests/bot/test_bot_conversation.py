"""Bot-owned conversation contribution → cognition → Speaking product path."""

from __future__ import annotations

import asyncio
import dataclasses
import typing
import uuid

import hsm

from bot.abilities import cognition
from bot.abilities import communication
from bot.abilities.communication import conversation
from bot.abilities import decoding
from bot.abilities import encoding
from bot.abilities import listening
from bot.abilities.communication.conversation import turn_detector
from bot.abilities import processing
from bot.abilities import speaking
from bot.abilities.hearing import speech
from bot.abilities.hearing import voice as hearing_voice
from bot.bot import Bot
from bot.device import Device
from bot.environment import SoundData, SoundEvent, Environment
from tests.bot.test_bot import CapturingCognition, as_cognition


class RecordingEncoder(encoding.Encoder[bytes, bytes]):
    def __init__(self, audio: bytes = b"\x11\x22") -> None:
        self.calls: list[bytes] = []
        self._audio = audio

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return self._audio


class IdentityTextDecoder(decoding.Decoder[typing.Any, str]):
    @typing.override
    async def decode(self, input: typing.Any) -> str:
        if isinstance(input, turn_detector.TextStimulus):
            return input.content
        if isinstance(input, turn_detector.EventStimulus):
            text = input.payload.get("text")
            assert isinstance(text, str)
            return text
        raise AssertionError(f"unexpected stimulus {input!r}")


class SpeakFromContributionProcessor(processing.Processor):
    """Select speaking.input using contribution text product from conversation terminal stimulus."""

    inputs: list[processing.InputData]
    require_conversation_actor: bool

    def __init__(self, *, require_conversation_actor: bool = True) -> None:
        self.inputs = []
        self.require_conversation_actor = require_conversation_actor

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.inputs.append(input)
        names = {event.name for event in input.schemas}
        assert "bot.ability.speaking.input" in names
        assert "speaking" in input.actors
        if self.require_conversation_actor:
            assert any(isinstance(actor, conversation.Conversation) for actor in input.actors.values())
        stimulus = input.input
        assert isinstance(stimulus, hsm.Event)
        assert stimulus.name == conversation.OutputEvent.name
        response = stimulus.data
        assert isinstance(response, conversation.Response)
        assert isinstance(response.content, str) and response.content
        assert response.content_type.lower().startswith("text/")
        return (
            processing.SelectedEvent(
                event="bot.ability.speaking.input",
                data={"text": f"Heard: {response.content}"},
                target="speaking",
                reason="reply to conversation contribution",
            ),
        )


class CountingProcessor(processing.Processor):
    """Count cognition entries without selecting Speaking (idle after process)."""

    inputs: list[processing.InputData]

    def __init__(self) -> None:
        self.inputs = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.inputs.append(input)
        return ()


async def _wait_until(condition: typing.Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met")


def _text_conversation() -> conversation.Conversation:
    return conversation.Conversation(
        turn_detector=turn_detector.TurnDetector(decoder=IdentityTextDecoder()),
    )


def _text_communication(
    conversation_ability: conversation.Conversation | None = None,
) -> tuple[communication.Communication, conversation.Conversation]:
    conversation_ability = conversation_ability if conversation_ability is not None else _text_conversation()
    return (
        communication.Communication(active_conversation=conversation_ability),
        conversation_ability,
    )


def test_conversation_input_uses_identity_sets_and_content() -> None:
    message = conversation.ConversationInputData(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset({"bot"}),
        content="hello there",
        content_type="text/plain",
    )
    assert message.source_ids == frozenset({"caller"})
    assert message.target_ids == frozenset({"bot"})
    assert message.content == "hello there"
    assert message.content_type == "text/plain"


def test_bot_conversation_contribution_selects_speaking() -> None:
    """Message → Conversation contribution → Bot cognition (Speaking in actors) → Speaking."""

    async def run() -> tuple[list[bytes], list[processing.InputData], conversation.Response | None]:
        encoder = RecordingEncoder()
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        communication_ability, conversation_ability = _text_communication()
        processor = SpeakFromContributionProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                    acquired_abilities=(communication_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/inactive"))

        operation_id = uuid.uuid4().hex
        message = conversation.ConversationInputData(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content="hello from the room",
            content_type="text/plain",
        )
        _ = await hsm.dispatch(
            environment,
            conversation_ability,
            conversation_ability.input_event.with_data_and_id(message, operation_id),
        )
        await _wait_until(lambda: bool(encoder.calls), timeout=10.0)
        await _wait_until(
            lambda: (probe.state() or "").endswith("/unfocused") or (probe.state() or "").endswith("/focused")
        )
        stimulus = processor.inputs[0].input
        assert isinstance(stimulus, hsm.Event)
        response = stimulus.data
        assert isinstance(response, conversation.Response)
        return encoder.calls, processor.inputs, response

    calls, inputs, response = asyncio.run(run())
    assert calls == [b"Heard: hello from the room"]
    assert len(inputs) == 1
    assert response is not None
    assert response.content == "hello from the room"
    assert response.source_ids == frozenset({"caller"})
    assert response.target_ids == frozenset({"bot"})
    assert response.content_type == "text/plain"


def test_bot_ignores_host_encoded_conversation_response() -> None:
    """Host-encoded Response (content set) must not re-enter cognition."""

    async def run() -> tuple[list[processing.InputData], str]:
        processor = CountingProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))

        encoded = conversation.Response(
            source_ids=frozenset({"bot"}),
            target_ids=frozenset({"caller"}),
            content=b"already encoded",
            content_type="audio/raw",
            session_ref="support-call",
        )
        # Address this Bot by id (dispatch_to / envelope target) — still fail closed on payload shape.
        terminal = dataclasses.replace(
            conversation.OutputEvent.with_data(encoded),
            target=hsm.id(probe),
            id=uuid.uuid4().hex,
        )
        _ = await hsm.dispatch_to(environment, terminal, hsm.id(probe))
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        return processor.inputs, probe.state() or ""

    inputs, state = asyncio.run(run())
    assert inputs == []
    assert state.endswith("/unfocused")


def test_bot_ignores_response_without_text_product() -> None:
    """Response with no text product (content is not text) must not re-enter cognition."""

    async def run() -> list[processing.InputData]:
        processor = CountingProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))

        bare = conversation.Response(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content=None,
            content_type="text/plain",
            session_ref="support-call",
        )
        terminal = dataclasses.replace(
            conversation.OutputEvent.with_data(bare),
            target=hsm.id(probe),
            id=uuid.uuid4().hex,
        )
        _ = await hsm.dispatch_to(environment, terminal, hsm.id(probe))
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        return processor.inputs

    assert asyncio.run(run()) == []


def test_bot_accepts_contribution_with_empty_text_product() -> None:
    """Empty string text product (silence/partial) is still a contribution and re-enters cognition."""

    async def run() -> list[object]:
        processor = CountingProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))

        silence = conversation.Response(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content="",
            content_type="text/plain",
            session_ref="support-call",
        )
        response_event = conversation.OutputEvent.with_data(silence)
        handoff = dataclasses.replace(
            cognition.InputEvent.with_data(cognition.InputData(stimulus=response_event)),
            target=hsm.id(probe),
            id=uuid.uuid4().hex,
        )
        _ = await hsm.dispatch_to(environment, handoff, hsm.id(probe))
        await _wait_until(lambda: len(processor.inputs) == 1, timeout=2.0)
        texts: list[object] = []
        for item in processor.inputs:
            stimulus = item.input
            if isinstance(stimulus, hsm.Event) and isinstance(stimulus.data, conversation.Response):
                texts.append(stimulus.data.content)
        return texts

    assert asyncio.run(run()) == [""]


def test_bot_accepts_contribution_via_dispatch_to_id() -> None:
    """Contribution handoff addressed to this Bot by id enters body cognition (no source-id walk)."""

    async def run() -> list[bytes]:
        encoder = RecordingEncoder()
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        # Address-only path: no acquired Conversation actor required on the body.
        processor = SpeakFromContributionProcessor(require_conversation_actor=False)

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))

        contribution = conversation.Response(
            source_ids=frozenset({"caller"}),
            target_ids=frozenset({"bot"}),
            content="addressed by id",
            content_type="text/plain",
            session_ref="support-call",
        )
        response_event = conversation.OutputEvent.with_data(contribution)
        handoff = dataclasses.replace(
            cognition.InputEvent.with_data(cognition.InputData(stimulus=response_event)),
            source="conversation-actor-id",
            target=hsm.id(probe),
            id=uuid.uuid4().hex,
        )
        _ = await hsm.dispatch_to(environment, handoff, hsm.id(probe))
        await _wait_until(lambda: bool(encoder.calls), timeout=10.0)
        return encoder.calls

    assert asyncio.run(run()) == [b"Heard: addressed by id"]


def test_bot_product_path_does_not_use_host_turn() -> None:
    """Bot body bridge is the product path: one cognition entry without host_turn."""

    async def run() -> int:
        encoder = RecordingEncoder()
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        communication_ability, conversation_ability = _text_communication()
        processor = SpeakFromContributionProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                    acquired_abilities=(communication_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/inactive"))
        _ = await hsm.dispatch(
            environment,
            conversation_ability,
            conversation_ability.input_event.with_data_and_id(
                conversation.ConversationInputData(
                    source_ids=frozenset({"caller"}),
                    target_ids=frozenset({"bot"}),
                    content="only body path",
                    content_type="text/plain",
                ),
                uuid.uuid4().hex,
            ),
        )
        await _wait_until(lambda: bool(encoder.calls), timeout=10.0)
        return len(processor.inputs)

    assert asyncio.run(run()) == 1


def test_bot_defers_second_conversation_contribution_until_idle() -> None:
    """Second contribution while processing is deferred and runs after the first completes."""

    async def run() -> list[str]:
        encoder = RecordingEncoder()
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        communication_ability, conversation_ability = _text_communication()
        # Gate first process until second message is queued.
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        class GatedSpeakProcessor(processing.Processor):
            inputs: list[processing.InputData]

            def __init__(self) -> None:
                self.inputs = []

            @typing.override
            async def process(self, input: processing.InputData) -> processing.Events:
                self.inputs.append(input)
                first_started.set()
                if len(self.inputs) == 1:
                    await release_first.wait()
                stimulus = input.input
                assert isinstance(stimulus, hsm.Event)
                response = stimulus.data
                assert isinstance(response, conversation.Response)
                assert isinstance(response.content, str) and response.content
                return (
                    processing.SelectedEvent(
                        event="bot.ability.speaking.input",
                        data={"text": response.content},
                        target="speaking",
                        reason="deferred turn proof",
                    ),
                )

        processor = GatedSpeakProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                    acquired_abilities=(communication_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/inactive"))

        first_id = uuid.uuid4().hex
        _ = await hsm.dispatch(
            environment,
            conversation_ability,
            conversation_ability.input_event.with_data_and_id(
                conversation.ConversationInputData(
                    source_ids=frozenset({"caller-a"}),
                    target_ids=frozenset({"bot"}),
                    content="first",
                    content_type="text/plain",
                ),
                first_id,
            ),
        )
        await first_started.wait()
        await _wait_until(lambda: (probe.state() or "").endswith("/processing"))

        second_id = uuid.uuid4().hex
        # Relationship stays active; wait for turn idle (active/waiting) before second input.
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/active/waiting"))
        _ = await hsm.dispatch(
            environment,
            conversation_ability,
            conversation_ability.input_event.with_data_and_id(
                conversation.ConversationInputData(
                    source_ids=frozenset({"caller-b"}),
                    target_ids=frozenset({"bot"}),
                    content="second",
                    content_type="text/plain",
                ),
                second_id,
            ),
        )
        # Second contribution should reach Bot while still processing first → defer.
        await _wait_until(lambda: len(processor.inputs) >= 1)
        release_first.set()
        await _wait_until(lambda: len(encoder.calls) >= 2, timeout=10.0)
        await _wait_until(
            lambda: (probe.state() or "").endswith("/unfocused") or (probe.state() or "").endswith("/focused")
        )
        return [call.decode() for call in encoder.calls]

    texts = asyncio.run(run())
    assert texts == ["first", "second"]


def test_listening_speech_reaches_cognition_without_conversation_dispatch() -> None:
    """Listening keeps its cognition.InputEvent/SpeechEvent contract through the Bot body."""

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[processing.InputData], str, str, str]:
        communication_ability, conversation_ability = _text_communication()
        processor = CountingProcessor()
        cognitive = CapturingCognition(processor)
        listening_ability = listening.Listening(
            voice_detector=_VoiceThenSilence(),
            speech_decoder=None,
        )

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=cognitive,
                    input=(listening_ability,),
                    acquired_abilities=(communication_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/inactive"))
        await hsm.dispatch(
            environment,
            probe,
            SoundEvent.with_data(
                SoundData(audio=b"pcm-chunk", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
            ),
        )
        await _wait_until(lambda: len(cognitive.input_events) == 1)
        return (
            cognitive.input_events,
            processor.inputs,
            conversation_ability.state() or "",
            hsm.id(probe),
            hsm.id(cognitive),
        )

    cognition_events, processing_inputs, conversation_state, probe_id, cognition_id = asyncio.run(run())

    assert len(cognition_events) == 1
    assert len(processing_inputs) == 1
    cognition_event = cognition_events[0]
    assert cognition_event.source == probe_id
    cognition_input = cognition_event.data
    assert isinstance(cognition_input, cognition.InputData)
    assert cognition_event.target == cognition_id
    assert isinstance(cognition_input.stimulus, hsm.Event)
    assert cognition_input.stimulus.name == listening.SpeechEvent.name
    assert isinstance(cognition_input.stimulus.data, listening.SpeechData)
    assert isinstance(processing_inputs[0].input, hsm.Event)
    assert processing_inputs[0].input.name == listening.SpeechEvent.name
    assert conversation_state.endswith("/behavior/inactive")


def test_listening_stt_product_reaches_cognition_without_conversation_dispatch() -> None:
    """The former STT bridge also remains inside Cognition when Communication holds Conversation."""

    async def run() -> tuple[list[hsm.Event[typing.Any]], list[processing.InputData], list[bytes], str, str, str]:
        communication_ability, conversation_ability = _text_communication()
        processor = CountingProcessor()
        cognitive = CapturingCognition(processor)
        speech_decoder = _RecordingSpeechDecoder()
        listening_ability = listening.Listening(
            voice_detector=_VoiceThenSilence(),
            speech_decoder=speech_decoder,
        )

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=cognitive,
                    input=(listening_ability,),
                    acquired_abilities=(communication_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/inactive"))

        await hsm.dispatch(
            environment,
            probe,
            SoundEvent.with_data(
                SoundData(audio=b"pcm-chunk", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
            ),
        )
        await hsm.dispatch(
            environment,
            probe,
            SoundEvent.with_data(
                SoundData(audio=b"silence", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
            ),
        )
        await _wait_until(lambda: len(cognitive.input_events) == 1)
        return (
            cognitive.input_events,
            processor.inputs,
            speech_decoder.calls,
            conversation_ability.state() or "",
            hsm.id(probe),
            hsm.id(cognitive),
        )

    cognition_events, processing_inputs, decoder_calls, conversation_state, probe_id, cognition_id = asyncio.run(run())

    assert len(cognition_events) == 1
    assert len(processing_inputs) == 1
    assert decoder_calls == [b"pcm-chunksilence"]
    cognition_event = cognition_events[0]
    assert cognition_event.source == probe_id
    assert cognition_event.target == cognition_id
    cognition_input = cognition_event.data
    assert isinstance(cognition_input, cognition.InputData)
    assert isinstance(cognition_input.stimulus, hsm.Event)
    assert cognition_input.stimulus.name == listening.SpeechEvent.name
    assert isinstance(cognition_input.stimulus.data, listening.SpeechData)
    assert cognition_input.stimulus.data.content_type == "text/plain"
    assert cognition_input.stimulus.data.content == "decoded:pcm-chunksilence"
    assert conversation_state.endswith("/behavior/inactive")


class _RecordingSpeechDecoder(speech.SpeechDecoder):
    def __init__(self) -> None:
        self.calls: list[bytes] = []

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return b"decoded:" + input


class _VoiceThenSilence(hearing_voice.detection.VoiceDetector):
    def __init__(self) -> None:
        self.calls = 0

    @typing.override
    async def classify(self, input: bytes) -> hearing_voice.detection.ApplyData:
        del input
        self.calls += 1
        if self.calls == 1:
            return hearing_voice.detection.ApplyData(
                segments=(
                    hearing_voice.detection.VoiceDetectionSegment(
                        start_seconds=0.0,
                        end_seconds=0.05,
                        confidence=1.0,
                    ),
                )
            )
        return hearing_voice.detection.ApplyData(segments=())
