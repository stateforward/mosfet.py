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

_LINEAR_PCM_BYTES_PER_SAMPLE = 2
"""Sample width, in bytes, of the linear PCM this ability plays out (16-bit)."""

_LINEAR_PCM_MEDIA_TYPES = frozenset({"audio/pcm", "audio/l16", "audio/wav", "audio/x-wav"})
"""Media types whose byte count is proportional to time, so playout duration is a measurement.

Container headers (WAV's 44 bytes) are left in the count: a fixed header against seconds of
audio is far below the tolerance any prediction here is compared at. A compressed form is not
in this set, because for those the byte count says nothing about duration and a guess would be
a fabricated prediction rather than a measured one.
"""


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


class EfferenceData(pydantic.BaseModel):
    """A copy of the motor command driving the mouth, issued as the command is issued.

    Not a description of the utterance and not a recording of it: the words are deliberately
    absent. Perception matching on words would be self-*recognition*, and people are famously
    bad at recognizing their own recorded voice while still responding to it — the thing that
    stops you hearing your own speech as an external event is knowing you are producing it right
    now, not knowing what it sounds like. So this carries only what the command commits to:
    which mouth, for how long, in what form.

    What the bot expects to *hear* as a result is not here either. That is a forward model — the
    mapping from a command to its sensory consequences — and it belongs to perception, which is
    the only thing positioned to learn it and the only thing that ever sees whether it held.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Copy of a motor command sent to the mouth, delivered to perception at the moment "
                "the command is issued and before the sound exists in the environment."
            ),
            "examples": [
                {
                    "mouth": "01J8ZQ2K7M0000000000000000",
                    "duration": 3.6,
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 24000,
                    "channels": 1,
                }
            ],
        },
    )

    mouth: str = pydantic.Field(
        min_length=1,
        description=(
            "Identity of the transducer this command was sent to. A body may have more than one "
            "mouth (a handset receiver and a loudspeaker are different sound sources on the same "
            "robot), so a copy says which one is producing."
        ),
        examples=["01J8ZQ2K7M0000000000000000"],
    )
    duration: float = pydantic.Field(
        gt=0.0,
        description=(
            "Seconds of sound this command commits to, measured from the encoded audio rather "
            "than estimated. This is how long the consequences of the command are expected to "
            "last; sound arriving after it is no longer a consequence of this command."
        ),
        examples=[3.6, 0.4],
    )
    media_type: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Media type of the audio this command plays out, when known.",
        examples=["audio/pcm"],
    )
    sample_rate_hz: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Sample rate of the audio this command plays out, in hertz, when known.",
        examples=[24000],
    )
    channels: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Channel count of the audio this command plays out, when known.",
        examples=[1],
    )


class _EncodedData(pydantic.BaseModel):
    """Synthesized speech waiting to be played: the encoding stage's product."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    text: str = pydantic.Field(min_length=1)
    audio: bytes = pydantic.Field(min_length=1)
    media_type: str = pydantic.Field(min_length=1)
    sample_rate_hz: int = pydantic.Field(ge=1)
    channels: int = pydantic.Field(ge=1)


def _playout_duration(encoded: _EncodedData) -> float | None:
    """Seconds of sound ``encoded`` will make, or ``None`` when the form does not say.

    For linear PCM the byte count *is* the duration, so this is a measurement of the command
    that was built, not a guess about it.
    """

    if encoded.media_type not in _LINEAR_PCM_MEDIA_TYPES:
        return None
    frame_bytes = encoded.sample_rate_hz * encoded.channels * _LINEAR_PCM_BYTES_PER_SAMPLE
    duration = len(encoded.audio) / frame_bytes
    return duration if duration > 0.0 else None


EfferenceEvent = hsm.Event[EfferenceData](
    name="bot.ability.speaking.efference",
    schema=EfferenceData,
)
"""Copy of the command to the mouth, routed to the body's input abilities.

Deliberately a plain event and never ``processing.EventKind``: this is a nerve, not a tool. A
bot does not decide to send an efference copy any more than it decides to send one to its own
cerebellum, and it must never appear in a menu of things a model can select.
"""

_SpeechEncodedEvent = hsm.Event[_EncodedData](
    name="bot.ability.speaking.encoded",
    kind=hsm.CompletionEventKind,
    schema=_EncodedData,
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


def _has_encoded_speech(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, _EncodedData)


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
    # Live record of a speaker this ability powered up itself, so stop powers down exactly that
    # and never a speaker someone else started.
    _speaker_started: bool
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
        self._speaker_started = False
        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._media_type = media_type

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        instance = await super().start(ctx, data)
        speaker = self._speaker
        if speaker is not None and not lifecycle.is_started(speaker):
            # The mouth is part of the bot: this ability powers it the way phone firmware powers
            # an earpiece. Parented under the ability's own durable context — never an activity
            # context — so the speaker outlives any single utterance (HSM-CONTEXT-001). An
            # injected speaker someone else already started is left alone, start and stop.
            model = type(speaker).model
            if model is None:
                raise RuntimeError(f"{type(speaker).__name__} has no lifecycle model.")
            _ = await hsm.started(self.context(), speaker, model)
            self._speaker_started = True
        return instance

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Release the speaker this ability wired itself to, power down only what it powered up, then stop."""

        speaker = self._speaker
        if speaker is not None and self._speaker_attached:
            self._speaker_attached = False
            if lifecycle.is_started(speaker):
                await speaker.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=self)))
        if speaker is not None and self._speaker_started:
            self._speaker_started = False
            if lifecycle.is_started(speaker):
                await speaker.stop(ctx)
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
    def _dispatch_speak_failure(
        ctx: hsm.Context,
        instance: "Speaking",
        event: hsm.Event[typing.Any],
        error: Exception,
    ) -> None:
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _SpeakFailedEvent.with_data(ability.FailureData(message=str(error))),
                id=event.id or None,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _run_encoding_activity(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Synthesize speech. No sound exists yet, and nothing has been committed to the air."""

        data = event.data
        assert isinstance(data, InputData)
        try:
            text = data.text.strip()
            if not text:
                raise ValueError("Speaking requires non-blank text.")
            audio_bytes = await instance._encoder.encode(text.encode("utf-8"))
            if not audio_bytes:
                raise ValueError("Speaking encoder produced empty audio.")
            encoded = _EncodedData(
                text=text,
                audio=audio_bytes,
                media_type=instance._media_type,
                sample_rate_hz=instance._sample_rate_hz,
                channels=instance._channels,
            )
        except Exception as error:
            Speaking._dispatch_speak_failure(ctx, instance, event, error)
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _SpeechEncodedEvent.with_data(encoded),
                id=event.id or None,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _issue_efference_copy(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Copy the command to the mouth to the body, on the way to the mouth.

        Entry, not activity: the copy has to leave before the signal does, which is the whole
        point of an efference copy. A command that reached the air first would arrive at
        perception as something that merely happened.

        Nothing is sent when there is no mouth or the mouth is unpowered (no signal will exist),
        when the encoded form does not give a duration (an unbounded window is worse than none),
        or when nothing owns this ability (nothing to route through). Silence here degrades to
        the untouched behaviour: the bot hears itself.
        """

        encoded = event.data
        assert isinstance(encoded, _EncodedData)
        speaker = instance._speaker
        if speaker is None or not lifecycle.is_started(speaker) or not instance._attachments:
            return
        duration = _playout_duration(encoded)
        if duration is None:
            return
        body = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            body,
            dataclasses.replace(
                EfferenceEvent.with_data(
                    EfferenceData(
                        mouth=hsm.id(speaker),
                        duration=duration,
                        media_type=encoded.media_type,
                        sample_rate_hz=encoded.sample_rate_hz,
                        channels=encoded.channels,
                    )
                ),
                id=event.id or None,
                source=hsm.id(instance),
                target=hsm.id(body),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _run_playout_activity(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Hand the encoded signal to the mouth. Completing means the act was committed."""

        encoded = event.data
        assert isinstance(encoded, _EncodedData)
        try:
            # Lazy import: abilities package init must not import devices (cycle via device → abilities).
            from bot.devices import audio

            frame = audio.AudioOutputData(
                audio=encoded.audio,
                media_type=encoded.media_type,
                sample_rate_hz=encoded.sample_rate_hz,
                channels=encoded.channels,
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
                text=encoded.text,
                media_type=encoded.media_type,
                sample_rate_hz=encoded.sample_rate_hz,
                channels=encoded.channels,
            )
        except Exception as error:
            Speaking._dispatch_speak_failure(ctx, instance, event, error)
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
                hsm.target("../encoding"),
            ),
        ),
        # Synthesis and playout are separate because they are separate in time and only the
        # second one makes a sound. Encoding a sentence takes seconds during which the bot is
        # silent — treating that as part of speaking would have the bot deaf through exactly the
        # pause a caller is most likely to speak into.
        hsm.state(
            "encoding",
            hsm.defer(InputEvent),
            hsm.activity(_run_encoding_activity),
            hsm.transition(
                hsm.on(_SpeechEncodedEvent),
                hsm.guard(_has_encoded_speech),
                hsm.target("../playing"),
            ),
            hsm.transition(
                hsm.on(_SpeakFailedEvent),
                hsm.guard(_has_speak_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("../idle"),
            ),
        ),
        hsm.state(
            "playing",
            hsm.defer(InputEvent),
            hsm.entry(_issue_efference_copy),
            hsm.activity(_run_playout_activity),
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
    "EfferenceData",
    "EfferenceEvent",
    "InputData",
    "InputEvent",
    "OutputData",
    "OutputEvent",
    "Speaking",
]
