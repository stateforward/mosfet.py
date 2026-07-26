"""Bot-owned conversation contribution → cognition → Speaking product path."""

from __future__ import annotations

import asyncio
import dataclasses
import typing
import uuid

import hsm

from bot.abilities import cognition
from bot.abilities import conversation
from bot.abilities.conversation import voice as conversation_voice
from bot.abilities import decoding
from bot.abilities import listening
from bot.abilities import participating
from bot.abilities import processing
from bot.abilities import speaking
from bot.abilities.hearing import speech
from bot.abilities.hearing import voice as hearing_voice
from bot.bot import Bot
from bot.device import Device
from bot.environment import SoundData, SoundEvent, Environment
from tests.bot.test_bot import as_cognition


class RecordingEncoder:
    def __init__(self, audio: bytes = b"\x11\x22") -> None:
        self.calls: list[bytes] = []
        self._audio = audio

    async def encode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return self._audio


class IdentityTextDecoder(decoding.Decoder[participating.ParticipationStimulus, str]):
    @typing.override
    async def decode(self, input: participating.ParticipationStimulus) -> str:
        if isinstance(input, participating.TextStimulus):
            return input.content
        if isinstance(input, participating.EventStimulus):
            text = input.payload.get("text")
            assert isinstance(text, str)
            return text
        raise AssertionError(f"unexpected stimulus {input!r}")


class SpeakFromContributionProcessor(processing.Processor):
    """Select speaking.input using contribution decoded_text from conversation terminal stimulus."""

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
        assert response.content is None
        assert response.decoded_text
        return (
            processing.SelectedEvent(
                event="bot.ability.speaking.input",
                data={"text": f"Heard: {response.decoded_text}"},
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


def _text_conversation() -> conversation.TextConversation:
    return conversation.TextConversation(
        decoding=decoding.Decoding(decoder=IdentityTextDecoder()),
        participating=participating.Participating(),
    )


def test_text_turn_builder_defaults() -> None:
    message = conversation.text_turn("hello there")
    assert message.conversation_ref == "conversation"
    assert message.self_participant_ref == "bot"
    assert message.content.kind == "text"
    assert message.content.content == "hello there"
    assert {p.ref for p in message.participants} == {"bot", "caller"}


def test_bot_conversation_contribution_selects_speaking() -> None:
    """Message → Conversation contribution → Bot cognition (Speaking in actors) → Speaking."""

    async def run() -> tuple[list[bytes], list[processing.InputData], conversation.Response | None]:
        encoder = RecordingEncoder()
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        conversation_ability = _text_conversation()
        processor = SpeakFromContributionProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                    acquired_abilities=(conversation_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/silent"))

        operation_id = uuid.uuid4().hex
        message = conversation.text_turn(
            "hello from the room",
            conversation_ref="support-call",
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
    assert response.decoded_text == "hello from the room"
    assert response.content is None
    assert response.conversation_ref == "support-call"


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
            conversation_ref="support-call",
            self_participant_ref="bot",
            participants=conversation.default_pair_participants(),
            content=b"already encoded",
            decoded_text=None,
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


def test_bot_ignores_contribution_with_null_decoded_text() -> None:
    """Contribution-shaped Response with decoded_text=None must not re-enter cognition."""

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
            conversation_ref="support-call",
            self_participant_ref="bot",
            participants=conversation.default_pair_participants(),
            content=None,
            decoded_text=None,
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


def test_bot_accepts_contribution_with_empty_decoded_text() -> None:
    """Empty decoded_text (silence/partial) is still a contribution and re-enters cognition."""

    async def run() -> list[str | None]:
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
            conversation_ref="support-call",
            self_participant_ref="bot",
            participants=conversation.default_pair_participants(),
            content=None,
            decoded_text="",
        )
        terminal = dataclasses.replace(
            conversation.OutputEvent.with_data(silence),
            target=hsm.id(probe),
            id=uuid.uuid4().hex,
        )
        _ = await hsm.dispatch_to(environment, terminal, hsm.id(probe))
        await _wait_until(lambda: len(processor.inputs) == 1, timeout=2.0)
        texts: list[str | None] = []
        for item in processor.inputs:
            stimulus = item.input
            if isinstance(stimulus, hsm.Event) and isinstance(stimulus.data, conversation.Response):
                texts.append(stimulus.data.decoded_text)
        return texts

    assert asyncio.run(run()) == [""]


def test_bot_accepts_contribution_via_dispatch_to_id() -> None:
    """Contribution terminal addressed to this Bot by id enters body cognition (no source-id walk)."""

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
            conversation_ref="support-call",
            self_participant_ref="bot",
            participants=conversation.default_pair_participants(),
            content=None,
            decoded_text="addressed by id",
        )
        terminal = dataclasses.replace(
            conversation.OutputEvent.with_data(contribution),
            source="conversation-actor-id",
            target=hsm.id(probe),
            id=uuid.uuid4().hex,
        )
        _ = await hsm.dispatch_to(environment, terminal, hsm.id(probe))
        await _wait_until(lambda: bool(encoder.calls), timeout=10.0)
        return encoder.calls

    assert asyncio.run(run()) == [b"Heard: addressed by id"]


def test_bot_product_path_does_not_use_host_turn() -> None:
    """Bot body bridge is the product path: one cognition entry without host_turn."""

    async def run() -> int:
        encoder = RecordingEncoder()
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        conversation_ability = _text_conversation()
        processor = SpeakFromContributionProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                    acquired_abilities=(conversation_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/silent"))
        _ = await hsm.dispatch(
            environment,
            conversation_ability,
            conversation_ability.input_event.with_data_and_id(
                conversation.text_turn("only body path"),
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
        conversation_ability = _text_conversation()
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
                assert response.decoded_text
                return (
                    processing.SelectedEvent(
                        event="bot.ability.speaking.input",
                        data={"text": response.decoded_text},
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
                    acquired_abilities=(conversation_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/silent"))

        first_id = uuid.uuid4().hex
        _ = await hsm.dispatch(
            environment,
            conversation_ability,
            conversation_ability.input_event.with_data_and_id(
                conversation.text_turn("first", conversation_ref="call-a"),
                first_id,
            ),
        )
        await first_started.wait()
        await _wait_until(lambda: (probe.state() or "").endswith("/processing"))

        second_id = uuid.uuid4().hex
        # Conversation is single-flight; wait for it to return to silent before second Message.
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/silent"))
        _ = await hsm.dispatch(
            environment,
            conversation_ability,
            conversation_ability.input_event.with_data_and_id(
                conversation.text_turn("second", conversation_ref="call-b"),
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


def test_bot_bridges_listening_speech_to_conversation_then_speaking() -> None:
    """Listening speech product → Conversation Message → contribution → cognition → Speaking."""

    from bot.abilities.hearing import speech

    async def run() -> list[bytes]:
        encoder = RecordingEncoder()
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        conversation_ability = _text_conversation()
        processor = SpeakFromContributionProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                    acquired_abilities=(conversation_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/silent"))

        # Simulate Listening speech-decoding terminal: cognition.InputEvent with speech product bytes.
        speech_product = speech.SpeechDecoding.output_event.with_data("hello from listening".encode("utf-8"))
        handoff = cognition.InputEvent.with_data(cognition.InputData(stimulus=speech_product))
        handoff = dataclasses.replace(
            handoff,
            id=uuid.uuid4().hex,
            source="listening-ability",
            target=hsm.id(probe),
        )
        _ = await hsm.dispatch(environment, probe, handoff)
        await _wait_until(lambda: bool(encoder.calls), timeout=10.0)
        # Speech must not enter deliberative cognition as raw sensory stimulus when Conversation is acquired.
        for item in processor.inputs:
            stimulus = item.input
            assert isinstance(stimulus, hsm.Event)
            assert stimulus.name == conversation.OutputEvent.name
        return encoder.calls

    assert asyncio.run(run()) == [b"Heard: hello from listening"]


def test_bot_text_conversation_rejects_non_utf8_speech_product() -> None:
    """TextConversation bridge drops non-UTF-8 speech product; Bot stays live for a good turn."""

    from bot.abilities.hearing import speech

    async def run() -> list[bytes]:
        encoder = RecordingEncoder()
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        conversation_ability = _text_conversation()
        processor = SpeakFromContributionProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                    acquired_abilities=(conversation_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/silent"))

        bad_product = speech.SpeechDecoding.output_event.with_data(b"\xff\xfe not utf-8")
        bad_handoff = cognition.InputEvent.with_data(cognition.InputData(stimulus=bad_product))
        bad_handoff = dataclasses.replace(bad_handoff, id=uuid.uuid4().hex, target=hsm.id(probe))
        _ = await hsm.dispatch(environment, probe, bad_handoff)
        # Fail closed: no Conversation turn, no deliberative cognition, Bot remains active.
        await asyncio.sleep(0.05)
        assert len(processor.inputs) == 0
        assert (probe.state() or "").endswith("/unfocused")
        assert (conversation_ability.state() or "").endswith("/behavior/silent")

        # Recovery: valid UTF-8 STT product still completes Listening → Conversation → Speaking.
        good_product = speech.SpeechDecoding.output_event.with_data("hello after refuse".encode("utf-8"))
        good_handoff = cognition.InputEvent.with_data(cognition.InputData(stimulus=good_product))
        good_handoff = dataclasses.replace(good_handoff, id=uuid.uuid4().hex, target=hsm.id(probe))
        _ = await hsm.dispatch(environment, probe, good_handoff)
        await _wait_until(lambda: bool(encoder.calls), timeout=10.0)
        return encoder.calls

    assert asyncio.run(run()) == [b"Heard: hello after refuse"]


def test_bot_text_conversation_drops_empty_stt_without_bricking() -> None:
    """Empty UTF-8 STT product must drop without ValidationError and leave Bot live for recovery."""

    from bot.abilities.hearing import speech

    async def run() -> list[bytes]:
        encoder = RecordingEncoder()
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        conversation_ability = _text_conversation()
        processor = SpeakFromContributionProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                    acquired_abilities=(conversation_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/silent"))

        empty_product = speech.SpeechDecoding.output_event.with_data(b"")
        empty_handoff = cognition.InputEvent.with_data(cognition.InputData(stimulus=empty_product))
        empty_handoff = dataclasses.replace(empty_handoff, id=uuid.uuid4().hex, target=hsm.id(probe))
        _ = await hsm.dispatch(environment, probe, empty_handoff)
        await asyncio.sleep(0.05)
        assert len(processor.inputs) == 0
        assert (probe.state() or "").endswith("/unfocused")
        assert (conversation_ability.state() or "").endswith("/behavior/silent")

        good_product = speech.SpeechDecoding.output_event.with_data("after empty".encode("utf-8"))
        good_handoff = cognition.InputEvent.with_data(cognition.InputData(stimulus=good_product))
        good_handoff = dataclasses.replace(good_handoff, id=uuid.uuid4().hex, target=hsm.id(probe))
        _ = await hsm.dispatch(environment, probe, good_handoff)
        await _wait_until(lambda: bool(encoder.calls), timeout=10.0)
        return encoder.calls

    assert asyncio.run(run()) == [b"Heard: after empty"]


def test_voice_conversation_refuses_utf8_stt_transcript_as_audio() -> None:
    """VoiceConversation drops UTF-8 STT text; Bot stays live for a following acoustic turn."""

    from bot.abilities.hearing import speech

    async def run() -> tuple[int, str, int]:
        conversation_ability = conversation_voice.VoiceConversation(
            participating=participating.Participating(),
            decoder=_VoiceIdentityDecoder(),
            encoder=_VoiceStubEncoder(),
        )
        processor = CountingProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    acquired_abilities=(conversation_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/silent"))

        bad_product = speech.SpeechDecoding.output_event.with_data("hello transcript".encode("utf-8"))
        bad_handoff = cognition.InputEvent.with_data(cognition.InputData(stimulus=bad_product))
        bad_handoff = dataclasses.replace(bad_handoff, id=uuid.uuid4().hex, target=hsm.id(probe))
        _ = await hsm.dispatch(environment, probe, bad_handoff)
        await asyncio.sleep(0.05)
        assert len(processor.inputs) == 0
        assert (conversation_ability.state() or "").endswith("/behavior/silent")
        assert (probe.state() or "").endswith("/unfocused")

        # Recovery: non-UTF-8 acoustic product is a valid VoiceMessage; actor still accepts dispatch.
        acoustic = bytes([0xFF, 0xFE, 0x00, 0x01, 0x80])
        good_product = speech.SpeechDecoding.output_event.with_data(acoustic)
        good_handoff = cognition.InputEvent.with_data(cognition.InputData(stimulus=good_product))
        good_handoff = dataclasses.replace(good_handoff, id=uuid.uuid4().hex, target=hsm.id(probe))
        _ = await hsm.dispatch(environment, probe, good_handoff)
        await _wait_until(lambda: len(processor.inputs) >= 1, timeout=10.0)
        bot_state = probe.state() or ""
        return len(processor.inputs), bot_state, int(bool(processor.inputs))

    cognition_count, bot_state, recovered = asyncio.run(run())
    assert cognition_count >= 1
    assert recovered == 1
    assert "/active/" in bot_state


class _VoiceIdentityDecoder(conversation_voice.VoiceDecoder):
    @typing.override
    async def decode(self, input: participating.AudioStimulus) -> str:
        return "decoded"


class _VoiceStubEncoder(conversation_voice.VoiceEncoder):
    @typing.override
    async def encode(self, input: conversation_voice.EncodeData) -> bytes:
        del input
        return b"enc"


class _AlwaysVoice(hearing_voice.detection.VoiceDetector):
    @typing.override
    async def classify(self, input: bytes) -> hearing_voice.detection.OutputData:
        del input
        return hearing_voice.detection.OutputData(is_voice=True, confidence=1.0)


class _FixedTranscriptSpeechDecoder(speech.SpeechDecoder):
    """Decode any acoustic chunk to a fixed UTF-8 STT transcript product."""

    def __init__(self, transcript: str = "hello from listening machine") -> None:
        self.transcript = transcript
        self.calls: list[bytes] = []

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return self.transcript.encode("utf-8")


def test_environment_sound_through_listening_bridges_to_conversation_then_speaking() -> None:
    """Real Listening machine: environment.sound → STT product → Bot bridge → Conversation → Speaking."""

    async def run() -> list[bytes]:
        encoder = RecordingEncoder()
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        conversation_ability = _text_conversation()
        speech_decoder = _FixedTranscriptSpeechDecoder("hello from listening machine")
        listening_ability = listening.Listening(
            voice_detector=_AlwaysVoice(),
            speech_decoder=speech_decoder,
        )
        processor = SpeakFromContributionProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    input=(listening_ability,),
                    output=(speaking_ability,),
                    acquired_abilities=(conversation_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conversation_ability.state() or "").endswith("/behavior/silent"))

        sound = SoundEvent.with_data(
            SoundData(audio=b"pcm-chunk", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)
        )
        _ = await hsm.dispatch(environment, probe, sound)
        await _wait_until(lambda: bool(speech_decoder.calls), timeout=10.0)
        await _wait_until(lambda: bool(encoder.calls), timeout=10.0)
        # Deliberative cognition must only see Conversation contribution, not raw speech product.
        for item in processor.inputs:
            stimulus = item.input
            assert isinstance(stimulus, hsm.Event)
            assert stimulus.name == conversation.OutputEvent.name
        return encoder.calls

    assert asyncio.run(run()) == [b"Heard: hello from listening machine"]
