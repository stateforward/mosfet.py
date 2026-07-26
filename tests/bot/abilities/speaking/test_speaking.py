"""Speaking output ability: text → encoder → speaker environment elevation."""

from __future__ import annotations

import asyncio
import typing

import hsm
from bot.abilities import processing

from bot.abilities import speaking
from bot.devices import audio
from bot.environment import SoundData, SoundEvent, Environment
from tests.hsm_instance_state import device_bots, start_ability_tree


class RecordingEncoder:
    def __init__(self, audio: bytes = b"pcm-audio") -> None:
        self.calls: list[bytes] = []
        self._audio = audio

    async def encode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return self._audio


class FailingEncoder:
    async def encode(self, input: bytes) -> bytes:
        del input
        raise RuntimeError("tts failed")


async def _wait_until(condition: typing.Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met")


def test_speaking_input_is_call_event_for_cognition_selection() -> None:
    assert speaking.InputEvent.name == "bot.ability.speaking.input"
    assert speaking.InputEvent.kind == processing.EventKind


class SoundListener(hsm.Instance):
    """Environment participant that records the ``environment.sound`` a speaker transduces."""

    def __init__(self, sounds: list[SoundData]) -> None:
        super().__init__()
        self._sounds = sounds

    @staticmethod
    def _record(ctx: hsm.Context, instance: "SoundListener", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        if isinstance(data, SoundData):
            instance._sounds.append(data)

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "SoundListener",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(SoundEvent), hsm.effect(_record))),
    )


def test_speaking_encodes_text_and_elevates_to_environment_sound() -> None:
    async def run() -> tuple[list[bytes], list[SoundData], speaking.OutputData | None]:
        encoder = RecordingEncoder(audio=b"\x00\x01")
        speaker = audio.Speaker()
        speaking_ability = speaking.Speaking(
            encoder=encoder,
            speaker=speaker,
            sample_rate_hz=24_000,
            channels=1,
            media_type="audio/pcm",
        )
        environment = Environment()
        sounds: list[SoundData] = []
        # Listen the way anything in the environment does, rather than patching the speaker: the
        # speaker transduces signal into environment.sound and every participant hears it.
        listener = SoundListener(sounds)

        await start_ability_tree(environment, speaking_ability)
        _ = await hsm.started(environment, speaker, speaker.model)
        _ = await hsm.started(environment, listener, listener.model, hsm.Config(id="environment-ear"))
        environment.join(listener)

        outputs: list[speaking.OutputData] = []
        original = speaking_ability.dispatch

        def capture_terminal(ctx: hsm.Context, event: hsm.Event) -> typing.Awaitable[None]:
            from bot.abilities import ability

            if event.name == ability.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
                data = event.data.data
                if isinstance(data, speaking.OutputData):
                    outputs.append(data)
            return original(ctx, event)

        speaking_ability.dispatch = capture_terminal  # type: ignore[method-assign]
        try:
            _ = await speaking_ability.apply(speaking.InputData(text="  Hello there.  "), ctx=environment)
            await _wait_until(lambda: bool(outputs))
        finally:
            speaking_ability.dispatch = original  # type: ignore[method-assign]

        return encoder.calls, sounds, outputs[0] if outputs else None

    calls, sounds, product = asyncio.run(run())
    assert calls == [b"Hello there."]
    assert product is not None
    assert product.text == "Hello there."
    assert product.sample_rate_hz == 24_000
    assert product.media_type == "audio/pcm"
    assert len(sounds) == 1
    assert sounds[0].audio == b"\x00\x01"


def test_speaking_failure_surfaces_on_failed_event() -> None:
    async def run() -> str:
        speaking_ability = speaking.Speaking(encoder=FailingEncoder(), speaker=audio.Speaker())
        environment = Environment()
        await start_ability_tree(environment, speaking_ability)
        failures: list[str] = []
        original = speaking_ability.dispatch

        def capture(ctx: hsm.Context, event: hsm.Event) -> typing.Awaitable[None]:
            from bot.abilities import ability

            if event.name == ability.TerminalErrorEvent.name and isinstance(event.data, hsm.Event):
                data = event.data.data
                if hasattr(data, "message"):
                    failures.append(str(data.message))
            return original(ctx, event)

        speaking_ability.dispatch = capture  # type: ignore[method-assign]
        try:
            _ = await speaking_ability.apply(speaking.InputData(text="hi"), ctx=environment)
            await _wait_until(lambda: bool(failures))
        finally:
            speaking_ability.dispatch = original  # type: ignore[method-assign]
        return failures[0]

    message = asyncio.run(run())
    assert "tts failed" in message


def test_speaking_is_cognition_callable_output_ability() -> None:
    """Speaking is a CallEvent ability suitable for bot.output / cognition tool selection.

    Actor-map assembly is bot-private; public contract is event kind + wiring (see e2e test).
    """

    speaking_ability = speaking.Speaking(encoder=RecordingEncoder())
    assert speaking.InputEvent.kind == processing.EventKind
    assert speaking_ability.input_event is speaking.InputEvent
    assert speaking_ability.input_event.kind == processing.EventKind


def test_cognition_to_speaking_output_end_to_end() -> None:
    """Bot input event → cognition/intuition → select speaking.input → encoder runs."""

    import bot
    from bot.bot import Bot
    from bot.device import Device
    from bot.abilities import processing
    from bot.environment import Environment
    from tests.bot.test_bot import as_cognition

    class SpeakProcessor(processing.Processor):
        inputs: list[processing.InputData]

        def __init__(self) -> None:
            self.inputs = []

        @typing.override
        async def process(self, input: processing.InputData) -> processing.Events:
            self.inputs.append(input)
            names = {event.name for event in input.schemas}
            assert "bot.ability.speaking.input" in names
            assert "speaking" in input.actors
            return (
                processing.SelectedEvent(
                    event="bot.ability.speaking.input",
                    data={"text": "hi from cognition"},
                    target="speaking",
                    reason="test speak selection",
                ),
            )

    async def run() -> tuple[list[bytes], list[processing.InputData]]:
        encoder = RecordingEncoder(audio=b"\x11\x22")
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        processor = SpeakProcessor()

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
        assert (speaking_ability.state() or "").endswith("/idle")

        await probe.dispatch(
            environment,
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=0)),
        )
        await _wait_until(lambda: bool(encoder.calls))
        return encoder.calls, processor.inputs

    calls, inputs = asyncio.run(run())
    assert calls == [b"hi from cognition"]
    assert len(inputs) == 1
    assert "bot.ability.speaking.input" in {event.name for event in inputs[0].schemas}


def test_speaking_wires_the_speaker_once_and_releases_it_on_stop() -> None:
    """Acquire once, release on stop.

    The speaker is injected and often shared — in the phone_bot wiring it is the phone's own —
    so this ability must not accumulate an attachment per utterance, nor hold one after it stops.
    """

    async def run() -> tuple[int, int]:
        speaker = audio.Speaker()
        speaking_ability = speaking.Speaking(encoder=RecordingEncoder(audio=b"\x00\x01"), speaker=speaker)
        environment = Environment()
        await start_ability_tree(environment, speaking_ability)
        _ = await hsm.started(environment, speaker, speaker.model)

        # Sequential utterances: let each finish so this pins "wired once", not a race with a
        # deferred queue.
        for _ in range(3):
            _ = await speaking_ability.apply(speaking.InputData(text="Hello."), ctx=environment)
            await _wait_until(lambda: speaking_ability.state().endswith("/idle"))
        while_speaking = len(device_bots(speaker))

        await speaking_ability.stop(environment)
        await _wait_until(lambda: not device_bots(speaker))

        return while_speaking, len(device_bots(speaker))

    while_speaking, after_stop = asyncio.run(run())

    assert while_speaking == 1
    assert after_stop == 0
