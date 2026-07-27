"""Speaking: bot output ability that utters text via an encoder and speaker.

Model-facing contract is short text (not raw PCM). Internals encode speech and
play through a ``Speaker`` (environment elevation and/or device path). Conversation is
separate and may invoke Speaking later; cognition can select ``speaking.input``
directly as an output ability.
"""

from __future__ import annotations

from .. import ability
from .. import encoding

import dataclasses
import typing

import hsm
from bot import event_schema
import pydantic

from bot import lifecycle
from bot.protocols import attachment
from bot.telemetry import observer

if typing.TYPE_CHECKING:
    from bot.devices.audio.speaker import Speaker

_DEFAULT_SAMPLE_RATE_HZ = 24_000
_DEFAULT_CHANNELS = 1


class InputData(pydantic.BaseModel):
    """Model-facing speak request: utter this text.

    No example utterance anywhere on this model, deliberately. Free text has no format to
    illustrate, so an example here is not documentation — it is a complete, valid answer sitting
    in the only field there is, and a bot that is unsure what to say says the example out loud.
    The description carries the whole contract instead.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A request to speak. text is the entire payload: it is required, must be non-empty, and is "
                "synthesized and played immediately, so this request is never silent and never partial. "
                "The ability owns speech synthesis and playout, so nothing downstream supplies, completes, "
                "or edits the words. Speaking does not end a turn — it can accompany other events that keep "
                "running alongside it."
            ),
        },
    )

    text: str = pydantic.Field(
        min_length=1,
        description=(
            "The words to say, verbatim: exactly this string is spoken aloud in the bot's own voice to "
            "whoever is present, as soon as this is selected, and cannot be taken back once heard. It is "
            "plain spoken language — what a listener would hear, not markup, not a stage direction, not a "
            "label for an utterance and not a description of one, because every one of those is read out "
            "loud just as literally. Required and non-empty: there is no way to send this and stay silent, "
            "so the choice not to speak is made by not selecting speech at all."
        ),
    )


class OutputData(pydantic.BaseModel):
    """Result of a completed speak action."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": "Speak completed: original text plus audio format metadata used for playout.",
        },
    )

    text: str = pydantic.Field(min_length=1, description="Text that was spoken.")
    media_type: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Media type of the encoded audio (e.g. audio/pcm, audio/wav).",
        examples=["audio/pcm", "audio/wav"],
    )
    sample_rate_hz: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Sample rate of the encoded audio in hertz, when known.",
        examples=[24000],
    )
    channels: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Channel count of the encoded audio, when known.",
        examples=[1],
    )


_SpeakCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.speaking.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_SpeakFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.speaking.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)

_INPUT_EVENT = ability.ability_input_event(
    "bot.ability.speaking.input",
    InputData,
    description=(
        "Say something aloud, in the bot's own voice, now. This is the ability that turns chosen words "
        "into sound and it does nothing else: what to say, and whether to say anything at all, is the "
        "bot's own to decide, and this event only carries the words it settled on. Selecting it does not "
        "end the turn — it can accompany events that go on deliberating."
    ),
)
# Selectable by Processing / cognition: mark the ability's one front door offerable.
InputEvent = dataclasses.replace(_INPUT_EVENT, kind=event_schema.EventKind)

OutputEvent = ability.ability_output_event(
    "bot.ability.speaking.output",
    OutputData,
    description="Speak completed with text and audio format metadata.",
)


def _has_speak_input(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)


def _has_speak_completed(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, OutputData)


def _has_speak_failure(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)


class Speaking(ability.Ability[InputData, OutputData]):
    """Bot output ability: encode text to speech and play it on a speaker.

    Constructor-inject a TTS ``encoder`` and optional ``speaker``. When a speaker is
    provided, completed audio is elevated as ``environment.sound`` (and available for device
    uplink paths that watch speaker playout). Cognition selects ``bot.ability.speaking.input``.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[OutputData]] = OutputEvent

    _encoder: encoding.Encoder[bytes, bytes]
    _speaker: Speaker | None
    # Live record of an attachment this ability acquired, so stop can release exactly what start
    # of speech took. The speaker is injected and often shared, so it is never ours to power.
    _speaker_attached: bool
    _sample_rate_hz: int
    _channels: int
    _media_type: str

    def __init__(
        self,
        *,
        encoder: encoding.Encoder[bytes, bytes],
        speaker: Speaker | None = None,
        sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ,
        channels: int = _DEFAULT_CHANNELS,
        media_type: str = "audio/pcm",
    ) -> None:
        super().__init__()
        if sample_rate_hz < 1:
            raise ValueError("sample_rate_hz must be positive.")
        if channels < 1:
            raise ValueError("channels must be positive.")
        self._encoder = encoder
        self._speaker = speaker
        self._speaker_attached = False
        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._media_type = media_type

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Release the speaker this ability wired itself to, then stop."""

        speaker = self._speaker
        if speaker is not None and self._speaker_attached:
            self._speaker_attached = False
            if lifecycle.is_started(speaker):
                await speaker.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=self)))
        await super().stop(ctx)

    @staticmethod
    def _dispatch_output(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, OutputData)
        terminal = dataclasses.replace(
            instance.output_event.with_data(data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _dispatch_failure(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    async def _run_speak_activity(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, InputData)
        try:
            text = data.text.strip()
            if not text:
                raise ValueError("Speaking requires non-blank text.")
            # Lazy import: abilities package init must not import devices (cycle via device → abilities).
            from bot.devices import audio

            audio_bytes = await instance._encoder.encode(text.encode("utf-8"))
            if not audio_bytes:
                raise ValueError("Speaking encoder produced empty audio.")
            frame = audio.AudioOutputData(
                audio=audio_bytes,
                media_type=instance._media_type,
                sample_rate_hz=instance._sample_rate_hz,
                channels=instance._channels,
            )
            speaker = instance._speaker
            if speaker is not None:
                # This ability is the speaker's controller: wire to it once, then hand it signal.
                # The speaker is the transducer that turns that into environment.sound, which is how bot
                # input (and call uplink paths) hear playout. Wiring waits until first use because
                # an injected speaker may start after this ability does; stop releases it.
                if not instance._speaker_attached:
                    await speaker.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=instance)))
                    instance._speaker_attached = True
                await speaker.dispatch(
                    ctx,
                    dataclasses.replace(
                        audio.OutputEvent.with_data(frame),
                        source=hsm.id(instance),
                        metadata=dict(event.metadata),
                    ),
                )
            product = OutputData(
                text=text,
                media_type=instance._media_type,
                sample_rate_hz=instance._sample_rate_hz,
                channels=instance._channels,
            )
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _SpeakFailedEvent.with_data(ability.FailureData(message=str(error))),
                    id=event.id or None,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _SpeakCompletedEvent.with_data(product),
                id=event.id or None,
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Speaking",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(InputEvent),
                hsm.guard(_has_speak_input),
                hsm.target("../speaking"),
            ),
        ),
        hsm.state(
            "speaking",
            hsm.defer(InputEvent),
            hsm.activity(_run_speak_activity),
            hsm.transition(
                hsm.on(_SpeakCompletedEvent),
                hsm.guard(_has_speak_completed),
                hsm.effect(_dispatch_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_SpeakFailedEvent),
                hsm.guard(_has_speak_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("../idle"),
            ),
        ),
        hsm.observe(observer),
    )


__all__ = [
    "InputData",
    "InputEvent",
    "OutputData",
    "OutputEvent",
    "Speaking",
]
