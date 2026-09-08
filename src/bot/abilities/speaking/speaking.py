"""Speaking: bot output ability that utters text via an encoder.

Model-facing contract is short text (not raw PCM). Internals encode speech and
elevate playout as ``environment.sound``. An optional ``Speaker`` may still be
injected for mouth identity / later mouth wiring; playout does not require it.
On playout entry, Speaking delivers a motor-command copy (efference) directly to
registered Listening peers — not via body fan-out or environment broadcast.
Conversation is separate and may invoke Speaking later; cognition does not select
``speaking.input`` directly. A behavior/topology route owns that dispatch.
"""

from __future__ import annotations

from .. import ability
from .. import encoding
from .. import processing

import datetime
import collections.abc
import dataclasses
import typing
import uuid

import hsm
import bot
import pydantic

from bot import lifecycle
from bot import telemetry
from bot.protocols import attachment
from bot.telemetry import observer
from bot.telemetry import span

if typing.TYPE_CHECKING:
    from bot.devices.audio.speaker import Speaker

_DEFAULT_SAMPLE_RATE_HZ = 24_000
_DEFAULT_CHANNELS = 1
_RECORDING_TIMEOUT = datetime.timedelta(seconds=5)
_ENCODING_TIMEOUT = datetime.timedelta(seconds=5)
ConversationTarget: typing.TypeAlias = ability.Ability[typing.Any, typing.Any]

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
_ConversationRecordedEvent = hsm.Event[OutputData](
    name="bot.ability.speaking.conversation.recorded",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)

# Speaking is reached through behavior/topology wiring, not as a direct cognition tool.
InputEvent = hsm.Event[InputData](
    name="bot.ability.speaking.input",
    kind=hsm.EventKind,
    schema=InputData,
)

OutputEvent = hsm.Event[OutputData](
    name="bot.ability.speaking.output",
    schema=OutputData,
)


class EfferenceData(pydantic.BaseModel):
    """A copy of a motor command issued as the command is issued.

    Not a description of the utterance and not a recording of it: the words are deliberately
    absent. Perception matching on words would be self-*recognition*, and people are famously
    bad at recognizing their own recorded voice while still responding to it — the thing that
    stops you hearing your own speech as an external event is knowing you are producing it right
    now, not knowing what it sounds like. So this carries only what the command commits to:
    which mouth, for how long, in what form.

    What the bot expects to *hear* as a result is not here either. That is a forward model — the
    mapping from a command to its sensory consequences — and it belongs to perception, which is
    the only thing positioned to learn it and the only thing that ever sees whether it held.

    Owned by Speaking and delivered directly to linked Listening peers on playout entry — a nerve
    between effector and ear, not an environment stimulus and not body fan-out.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Copy of a motor command sent to a mouth, delivered to perception at the moment "
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


EfferenceEvent = hsm.Event[EfferenceData](
    name="bot.ability.speaking.efference",
    schema=EfferenceData,
)
"""Copy of a motor command, delivered Speaking → linked Listening on playout entry.

Deliberately a plain event and never a model-offerable tool: this is a nerve, not a selection.
A bot does not decide to send an efference copy any more than it decides to send one to its own
cerebellum, and it must never appear in a menu of things a model can select.
"""


def _normalize_efference_targets(
    listening: hsm.Instance | collections.abc.Sequence[hsm.Instance] | None,
) -> tuple[hsm.Instance, ...]:
    if listening is None:
        return ()
    if isinstance(listening, hsm.Instance):
        return (listening,)
    return tuple(listening)


def _normalize_conversation_targets(
    conversation: ConversationTarget | collections.abc.Sequence[ConversationTarget] | None,
) -> tuple[ConversationTarget, ...]:
    if conversation is None:
        return ()
    match = typing.cast(object, conversation)
    if isinstance(match, ability.Ability):
        targets: collections.abc.Sequence[ConversationTarget] = (typing.cast(ConversationTarget, conversation),)
    elif isinstance(match, collections.abc.Sequence):
        targets = typing.cast("collections.abc.Sequence[ConversationTarget]", conversation)
    else:
        raise TypeError("Speaking conversation targets must be Ability instances.")
    if any(not isinstance(typing.cast(object, target), ability.Ability) for target in targets):
        raise TypeError("Speaking conversation targets must be Ability instances.")
    return tuple(dict.fromkeys(targets))


def link_conversation(speaking: "Speaking", *conversation: ConversationTarget) -> None:
    """Register explicit Conversation sink(s) for trusted outbound history records."""

    speaking.link_conversation(*conversation)


def link_listening(speaking: "Speaking", *listening: hsm.Instance) -> None:
    """Register Listening peers that receive motor-command copies from ``speaking`` on playout.

    Composition-time wiring when both abilities already exist. Prefer
    ``Speaking(..., listening=...)`` at construction when the peer is available then.
    """

    speaking.link_listening(*listening)


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
    """Bot output ability: encode text to speech and elevate it as ``environment.sound``.

    Constructor-inject a TTS ``encoder``, optional ``speaker`` (mouth identity / future mouth
    wiring), and optional ``listening`` peer(s) for the motor-command copy on playout entry.
    Playout broadcasts ``environment.sound`` from this ability; efference is delivered only to
    registered Listening targets. Cognition does not select ``bot.ability.speaking.input`` directly;
    a behavior/topology route must dispatch it.
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
    # Listening peers that receive the playout-entry motor-command copy. Composition/DI only —
    # never walked from the actor graph, never the body attachment list.
    _efference_targets: tuple[hsm.Instance, ...]
    # Explicit trusted sink(s) for committed bot text; never discovered through actor graphs.
    _conversation_targets: tuple[ConversationTarget, ...]

    def __init__(
        self,
        *,
        encoder: encoding.Encoder[bytes, bytes],
        speaker: Speaker | None = None,
        listening: hsm.Instance | collections.abc.Sequence[hsm.Instance] | None = None,
        conversation: ConversationTarget | collections.abc.Sequence[ConversationTarget] | None = None,
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
        self._efference_targets = _normalize_efference_targets(listening)
        self._conversation_targets = _normalize_conversation_targets(conversation)

    def link_conversation(self, *conversation: ConversationTarget) -> None:
        """Register explicit Conversation sink(s) for trusted outbound history recording."""

        existing = list(self._conversation_targets)
        for target in _normalize_conversation_targets(conversation):
            if target not in existing:
                existing.append(target)
        self._conversation_targets = tuple(existing)

    def link_listening(self, *listening: hsm.Instance) -> None:
        """Register Listening peer(s) that receive motor-command copies on playout entry.

        Idempotent for already-registered targets. Does not mutate Listening fields.
        """

        if not listening:
            return
        existing = list(self._efference_targets)
        for target in listening:
            if target not in existing:
                existing.append(target)
        self._efference_targets = tuple(existing)

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
            _ = await bot.started(self.context(), speaker, model)
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
            target=event.source,
        )
        with span.operation(
            "bot.speaking.terminal",
            scope="bot.abilities.speaking",
            component="speaking",
            stage="terminal",
            context=telemetry.event_context(event),
        ):
            if event.source and event.source != hsm.id(instance):
                # Directed terminal reply to the caller, not a NACK proxy: delivery is the gate.
                _ = hsm.dispatch_to(instance.context(), terminal, event.source)
            else:
                _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    async def _record_output(
        ctx: hsm.Context,
        instance: "Speaking",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, OutputData)
        try:
            from ..communication import conversation

            for target in instance._conversation_targets:
                operation_id = f"{event.id or hsm.id(instance)}:conversation:{hsm.id(target)}"
                message = conversation.Message(
                    sequence=0,
                    direction="outbound",
                    source_ids=frozenset(),
                    target_ids=frozenset(),
                    content=data.text,
                    content_type="text/plain",
                    provenance=conversation.MessageProvenance(event=conversation.AppendEvent.name),
                )
                await hsm.dispatch(
                    ctx,
                    target,
                    dataclasses.replace(
                        conversation.AppendEvent.with_data(conversation.AppendData(message=message)),
                        id=operation_id,
                        source=hsm.id(instance),
                        target=hsm.id(target),
                        metadata=dict(event.metadata),
                    ),
                )
        except Exception as error:
            Speaking._dispatch_speak_failure(ctx, instance, event, error)
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ConversationRecordedEvent.with_data(data),
                id=event.id or None,
                source=event.source,
                target=event.target,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _recording_timeout_delay(
        ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, instance, event
        return _RECORDING_TIMEOUT

    @staticmethod
    def _dispatch_recording_timeout(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        failure = ability.FailureData(message="Conversation history recording timed out.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            source=hsm.id(instance),
            target=event.source,
            metadata=dict(event.metadata),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _dispatch_failure(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        if processing.active_operation(instance, event.id) is not None:
            processing.finish_operation(ctx, instance, event.id)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source,
        )
        with span.operation(
            "bot.speaking.terminal",
            scope="bot.abilities.speaking",
            component="speaking",
            stage="terminal",
            context=telemetry.event_context(event),
        ) as active:
            span.record_failure(active, "speaking_failed")
            if event.source and event.source != hsm.id(instance):
                # Directed terminal reply to the caller, not a NACK proxy: delivery is the gate.
                _ = hsm.dispatch_to(instance.context(), terminal, event.source)
            else:
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
                source=event.source,
                target=event.target,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _run_encoding_activity(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Synthesize speech. No sound exists yet, and nothing has been committed to the air."""

        data = event.data
        assert isinstance(data, InputData)
        operation_id = event.id or uuid.uuid4().hex
        _ = await processing.start_operation(instance, operation_id)
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
                id=operation_id,
                source=event.source,
                target=event.target,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _has_current_encoding(
        ctx: hsm.Context,
        instance: "Speaking",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return isinstance(event.data, _EncodedData) and processing.active_operation(instance, event.id) is not None

    @staticmethod
    def _finish_encoding(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        processing.finish_operation(ctx, instance, event.id)

    @staticmethod
    def _encoding_timeout_delay(
        ctx: hsm.Context,
        instance: "Speaking",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, instance, event
        return _ENCODING_TIMEOUT

    @staticmethod
    def _dispatch_encoding_timeout(
        ctx: hsm.Context,
        instance: "Speaking",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        operation_id = processing.active_operation_id(instance)
        if operation_id is None:
            return
        processing.finish_operation(ctx, instance, operation_id)
        failure = ability.FailureData(message="Speech encoding timed out.")
        with span.operation(
            "bot.speaking.terminal",
            scope="bot.abilities.speaking",
            component="speaking",
            stage="terminal",
        ) as active:
            span.record_failure(active, "encoding_timeout")
            terminal = dataclasses.replace(
                instance.failed_event.with_data_and_id(failure, operation_id),
                source=hsm.id(instance),
                target=hsm.id(instance),
            )
            _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _issue_efference_copy(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Copy the command to the mouth to linked Listening peers, on the way to the mouth.

        Entry, not activity: the copy has to leave before the signal does, which is the whole
        point of an efference copy. A command that reached the air first would arrive at
        perception as something that merely happened.

        Delivery is Speaking → Listening only (registered peers). Not body fan-out, not
        environment broadcast. Nothing is sent when there is no mouth or the mouth is unpowered
        (no signal will exist), when the encoded form does not give a duration (an unbounded
        window is worse than none), or when no Listening peer is linked. Silence here degrades
        to the untouched behaviour: the bot hears itself.
        """

        encoded = event.data
        assert isinstance(encoded, _EncodedData)
        speaker = instance._speaker
        if speaker is None or not lifecycle.is_started(speaker) or not instance._efference_targets:
            return
        duration = _playout_duration(encoded)
        if duration is None:
            return
        copy = EfferenceData(
            mouth=hsm.id(speaker),
            duration=duration,
            media_type=encoded.media_type,
            sample_rate_hz=encoded.sample_rate_hz,
            channels=encoded.channels,
        )
        for peer in instance._efference_targets:
            if not lifecycle.is_started(peer):
                continue
            _ = hsm.dispatch(
                ctx,
                peer,
                dataclasses.replace(
                    EfferenceEvent.with_data(copy),
                    id=event.id or None,
                    source=hsm.id(instance),
                    target=hsm.id(peer),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    async def _run_playout_activity(ctx: hsm.Context, instance: "Speaking", event: hsm.Event[typing.Any]) -> None:
        """Elevate encoded speech as ``environment.sound``. Completing means the act was committed."""

        encoded = event.data
        assert isinstance(encoded, _EncodedData)
        try:
            from bot.environment import Environment, SoundData, SoundEvent

            # Temporary: ability elevates sound directly. Mouth/Speaker transduction returns later.
            # When a mouth is injected, stamp its public amplitude and origin so distance
            # filtering still works (Person geometry, far/near ears).
            speaker = instance._speaker
            amplitude_db = None if speaker is None else speaker.amplitude_db
            placement = None if speaker is None else speaker.placement
            origin = None if placement is None else placement.position
            sound = dataclasses.replace(
                SoundEvent.with_data(
                    SoundData(
                        audio=encoded.audio,
                        media_type=encoded.media_type,
                        sample_rate_hz=encoded.sample_rate_hz,
                        channels=encoded.channels,
                        amplitude_db=amplitude_db,
                    )
                ),
                source=hsm.id(instance) if speaker is None else hsm.id(speaker),
                metadata=dict(event.metadata),
            )
            _ = await Environment.from_context(ctx).broadcast(sound, origin=origin)
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
                source=event.source,
                target=event.target,
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
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
                hsm.guard(_has_current_encoding),
                hsm.effect(_finish_encoding),
                hsm.target("../playing"),
            ),
            hsm.transition(
                hsm.on(_SpeakFailedEvent),
                hsm.guard(_has_speak_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.after(_encoding_timeout_delay),
                hsm.effect(_dispatch_encoding_timeout),
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
                hsm.target("../recording"),
            ),
            hsm.transition(
                hsm.on(_SpeakFailedEvent),
                hsm.guard(_has_speak_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("../idle"),
            ),
        ),
        hsm.state(
            "recording",
            hsm.defer(InputEvent),
            hsm.activity(_record_output),
            hsm.transition(
                hsm.on(_ConversationRecordedEvent),
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
            hsm.transition(
                hsm.after(_recording_timeout_delay),
                hsm.effect(_dispatch_recording_timeout),
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
    "link_listening",
    "link_conversation",
    "OutputData",
    "OutputEvent",
    "Speaking",
]
