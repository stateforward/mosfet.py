"""Thin conversation coordinator: decode → participate → contribution terminal.

Decide, memory, and response encoding are host-owned. Conversation tracks
participant/room state and normalizes a turn into a participated contribution.
"""

from __future__ import annotations

from .. import ability
from .. import decoding
from .. import participating

import abc
import asyncio
import dataclasses
import typing as typ

import hsm

from bot.protocols import attachment
import pydantic
from pydantic.config import JsonDict, JsonValue

from bot.telemetry import observer

Stage: typ.TypeAlias = typ.Literal["decoding", "participating"]
ConversationChildKind: typ.TypeAlias = typ.Literal["decoding", "participating"]
TContent = typ.TypeVar("TContent", covariant=True)
_CONVERSATION_TEXT_INPUT_EXAMPLE: JsonDict = {
    "conversation_ref": "support-call",
    "self_participant_ref": "bot",
    "participants": [
        {
            "ref": "bot",
            "kind": "bot",
            "state": {"presence": "present", "attention": "available", "turn": "listening"},
        },
        {
            "ref": "caller",
            "kind": "human",
            "state": {"presence": "present", "attention": "available", "turn": "holding"},
        },
    ],
    "content": {"kind": "text", "source_participant_ref": "caller", "content": "hello"},
}

_CONVERSATION_VOICE_INPUT_EXAMPLE: JsonDict = {
    "conversation_ref": "support-call",
    "self_participant_ref": "bot",
    "participants": [
        {
            "ref": "bot",
            "kind": "bot",
            "state": {"presence": "present", "attention": "available", "turn": "listening"},
        },
        {
            "ref": "caller",
            "kind": "human",
            "state": {"presence": "present", "attention": "available", "turn": "holding"},
        },
    ],
    "content": {"kind": "audio", "source_participant_ref": "caller", "content": "aGVsbG8="},
}


def _conversation_schema_extra(description: str, example: JsonDict) -> JsonDict:
    examples: list[JsonValue] = [example]
    return {"description": description, "examples": examples}


class Message(pydantic.BaseModel, typ.Generic[TContent]):
    """Message entering or leaving a conversation."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra=_conversation_schema_extra(
            (
                "Conversation message. It carries the full participant state known by the conversation owner plus "
                "the message content to coordinate."
            ),
            _CONVERSATION_TEXT_INPUT_EXAMPLE,
        ),
    )

    conversation_ref: str = pydantic.Field(
        min_length=1,
        description="Stable reference for the conversation being coordinated.",
        examples=["support-call"],
    )
    self_participant_ref: str = pydantic.Field(
        min_length=1,
        description="Participant reference this conversation ability acts from inside the shared conversation.",
        examples=["bot"],
    )
    participants: tuple[participating.ParticipantSnapshot, ...] = pydantic.Field(
        min_length=1,
        description=(
            "All participant states currently known to the conversation owner. This is shared conversation state, "
            "not just the local bot participant."
        ),
    )
    content: TContent = pydantic.Field(
        description="Message content to coordinate.",
    )

    @pydantic.model_validator(mode="after")
    def validate_participant_refs(self) -> typ.Self:
        """Require the conversation to know every participant by stable unique reference."""

        refs = [participant.ref for participant in self.participants]
        if len(set(refs)) != len(refs):
            raise ValueError("participant refs must be unique.")
        if self.self_participant_ref not in refs:
            raise ValueError("self_participant_ref must match one participant snapshot.")
        return self


class TextMessage(Message[participating.TextStimulus]):
    """Conversation input for a text stimulus turn."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra=_conversation_schema_extra(
            "Conversation input for a text stimulus turn.",
            _CONVERSATION_TEXT_INPUT_EXAMPLE,
        ),
    )


class VoiceMessage(Message[participating.AudioStimulus]):
    """Conversation input for an audio stimulus turn."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra=_conversation_schema_extra(
            "Conversation input for an audio stimulus turn.",
            _CONVERSATION_VOICE_INPUT_EXAMPLE,
        ),
    )


AnyMessage: typ.TypeAlias = Message[participating.ParticipationStimulus]
TAnyMessage = typ.TypeVar("TAnyMessage", bound=AnyMessage)


class Response(Message[str | bytes | None]):
    """Message produced after a conversation contribution or host-completed response turn."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "Response message for one completed turn. Contribution-only turns leave content unset and set "
                "decoded_text so a Bot can re-enter cognition; hosts that encode a reply set content to the channel "
                "payload."
            ),
            "examples": [
                {
                    "conversation_ref": "support-call",
                    "self_participant_ref": "bot",
                    "participants": [
                        {
                            "ref": "bot",
                            "kind": "bot",
                            "state": {"presence": "present", "attention": "available", "turn": "listening"},
                        }
                    ],
                    "content": None,
                    "decoded_text": "hello",
                },
                {
                    "conversation_ref": "support-call",
                    "self_participant_ref": "bot",
                    "participants": [
                        {
                            "ref": "bot",
                            "kind": "bot",
                            "state": {"presence": "present", "attention": "available", "turn": "listening"},
                        }
                    ],
                    "content": "SSBjYW4gaGVscCB3aXRoIHRoYXQu",
                    "decoded_text": None,
                },
            ],
        },
    )

    content: str | bytes | None = pydantic.Field(
        default=None,
        description="Encoded channel response when a host completed encoding; null for contribution-only turns.",
        examples=["SSBjYW4gaGVscCB3aXRoIHRoYXQu"],
    )
    decoded_text: str | None = pydantic.Field(
        default=None,
        description=(
            "Readable contribution text when this is a contribution-only terminal (content is null). "
            "Null when a host already encoded a channel reply into content."
        ),
        examples=["hello", "How can I help?"],
    )


TResponse = typ.TypeVar("TResponse", bound=Response)


class FailureData(pydantic.BaseModel):
    """FailureData signal produced when a conversation phase cannot complete."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "FailureData signal produced when a conversation phase cannot complete.",
            "examples": [{"stage": "decoding", "message": "Conversation decoding timed out."}],
        },
    )

    stage: Stage = pydantic.Field(
        description="Conversation phase that failed.",
        examples=["decoding"],
    )
    message: str = pydantic.Field(
        min_length=1,
        description="Human-readable failure message for the conversation phase.",
        examples=["Conversation decoding timed out."],
    )


class SnapshotRequest(pydantic.BaseModel):
    """Request for the conversation coordinator to publish its current tracked room snapshot."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Request for the conversation coordinator to publish the durable room state it currently tracks. "
                "Use this event instead of reading state-machine attributes directly."
            ),
            "examples": [{"request_ref": "operator-panel-refresh"}],
        },
    )

    request_ref: str = pydantic.Field(
        min_length=1,
        description="Caller-chosen correlation reference echoed in the snapshot output event.",
        examples=["operator-panel-refresh"],
    )


class Snapshot(pydantic.BaseModel):
    """Current durable conversation room state published in response to a snapshot request."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Current durable conversation room state published in response to a snapshot request. The snapshot "
                "can be empty before the conversation coordinator has accepted its first message."
            ),
            "examples": [
                {
                    "request_ref": "operator-panel-refresh",
                    "conversation_ref": "support-call",
                    "participants": [
                        {
                            "ref": "bot",
                            "kind": "bot",
                            "state": {"presence": "present", "attention": "available", "turn": "listening"},
                        }
                    ],
                }
            ],
        },
    )

    request_ref: str = pydantic.Field(
        min_length=1,
        description="Caller-chosen correlation reference from the snapshot request event.",
        examples=["operator-panel-refresh"],
    )
    conversation_ref: str | None = pydantic.Field(
        default=None,
        description="Stable reference for the currently tracked conversation, or null before one is tracked.",
        examples=["support-call"],
    )
    participants: tuple[participating.ParticipantSnapshot, ...] = pydantic.Field(
        default=(),
        description="All participant states currently tracked by the conversation coordinator.",
    )


class DecodedTurn(pydantic.BaseModel):
    """Private phase payload carrying the decoded conversation stimulus."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    input: pydantic.SkipValidation[AnyMessage] = pydantic.Field(
        description="Original message accepted by the conversation ability.",
    )
    stimulus: participating.ParticipationStimulus = pydantic.Field(
        description="Stimulus normalized by decoding so participating can emit a contribution.",
    )
    decoded_text: str = pydantic.Field(
        min_length=1,
        description="DecodedData readable content when the turn produced one.",
        examples=["hello"],
    )


class ParticipatedTurn(pydantic.BaseModel):
    """Participated contribution for one conversation turn.

    Exposed to hosts after contribution completes so they can run decide/memory/encode.
    """

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    input: pydantic.SkipValidation[AnyMessage] = pydantic.Field(
        description="Original message accepted by the conversation ability.",
    )
    stimulus: participating.ParticipationStimulus = pydantic.Field(
        description="Stimulus normalized by decoding so participating can emit a contribution.",
    )
    decoded_text: str = pydantic.Field(
        min_length=1,
        description="DecodedData readable content when the turn produced one.",
    )
    participation: participating.OutputData = pydantic.Field(
        description="Contribution emitted by the participating ability.",
    )


InputEvent = ability.ability_input_event(
    "bot.ability.conversation.input",
    AnyMessage,
)
OutputEvent = ability.ability_output_event(
    "bot.ability.conversation.output",
    Response,
)
FailedEvent = hsm.Event[FailureData](
    name="bot.ability.conversation.failed",
    schema=FailureData,
)
SnapshotRequestEvent = hsm.Event[SnapshotRequest](
    name="bot.ability.conversation.snapshot.request",
    schema=SnapshotRequest,
)
SnapshotOutputEvent = hsm.Event[Snapshot](
    name="bot.ability.conversation.snapshot.output",
    schema=Snapshot,
)
_ConversationDecodingCompletedEvent = hsm.Event[DecodedTurn](
    name="bot.ability.conversation.decoding.completed",
    kind=hsm.CompletionEventKind,
    schema=DecodedTurn,
)
_ConversationDecodingFailedEvent = hsm.Event[FailureData](
    name="bot.ability.conversation.decoding.failed",
    kind=hsm.ErrorEventKind,
    schema=FailureData,
)
_ConversationParticipatingCompletedEvent = hsm.Event[ParticipatedTurn](
    name="bot.ability.conversation.participating.completed",
    kind=hsm.CompletionEventKind,
    schema=ParticipatedTurn,
)
_ConversationParticipatingFailedEvent = hsm.Event[FailureData](
    name="bot.ability.conversation.participating.failed",
    kind=hsm.ErrorEventKind,
    schema=FailureData,
)
DECODING_FAILED_EVENT = _ConversationDecodingFailedEvent
ParticipatingFailedEvent = _ConversationParticipatingFailedEvent


def _conversation_operation_id(event: hsm.Event[typ.Any]) -> str | None:
    return event.id if event.id else None


def _has_conversation_operation_id(
    ctx: hsm.Context,
    instance: "Conversation[typ.Any, typ.Any]",
    event: hsm.Event[typ.Any],
) -> bool:
    del ctx, instance
    return _conversation_operation_id(event) is not None


def _conversation_event_with_operation(
    event: hsm.Event[typ.Any],
    source: hsm.Event[typ.Any],
    *,
    operation_id: str | None = None,
) -> hsm.Event[typ.Any]:
    resolved_operation_id = operation_id if operation_id is not None else _conversation_operation_id(source)
    if resolved_operation_id is not None:
        event = event.with_data_and_id(event.data, resolved_operation_id)
    # Telemetry only: never stage payloads or host Futures.
    return dataclasses.replace(event, metadata=dict(source.metadata))


conversation_event_with_operation = _conversation_event_with_operation


def _matches_active_turn(
    instance: "Conversation[typ.Any, typ.Any]",
    event: hsm.Event[typ.Any],
) -> bool:
    """True when envelope id names the conversation's single-flight active turn."""

    turn_id = event.id if event.id else None
    active = instance._active_turn_id
    return turn_id is not None and active is not None and turn_id == active


def _has_conversation_decoded(
    ctx: hsm.Context,
    instance: "Conversation[typ.Any, typ.Any]",
    event: hsm.Event[typ.Any],
) -> bool:
    """Single in-flight turn (inputs deferred); completion carries DecodedTurn (HSM-COMPLETION-001)."""

    del ctx
    return isinstance(event.data, DecodedTurn) and _matches_active_turn(instance, event)


def _has_conversation_participated(
    ctx: hsm.Context,
    instance: "Conversation[typ.Any, typ.Any]",
    event: hsm.Event[typ.Any],
) -> bool:
    del ctx
    return isinstance(event.data, ParticipatedTurn) and _matches_active_turn(instance, event)


def _matches_data_type(
    data: object,
    data_type: type[object] | tuple[type[object], ...] | None,
) -> bool:
    if data_type is None:
        return True
    return isinstance(data, data_type)


def _matches_conversation_output_contract(
    instance: "Conversation[typ.Any, typ.Any]",
    data: object,
) -> bool:
    return isinstance(data, Response) and _matches_data_type(data, instance.output_data_type)


matches_conversation_output_contract = _matches_conversation_output_contract


def _matches_message_contract(
    instance: "Conversation[typ.Any, typ.Any]",
    data: object,
) -> bool:
    return isinstance(data, Message) and _matches_data_type(data, instance.input_data_type)


def _has_conversation_failure(
    ctx: hsm.Context,
    instance: "Conversation[typ.Any, typ.Any]",
    event: hsm.Event[typ.Any],
) -> bool:
    del ctx
    if not isinstance(event.data, FailureData):
        return False
    # Private phase failures must name the active turn when one is live (HSM-CORRELATION-001).
    if instance._active_turn_id is None:
        return True
    return _matches_active_turn(instance, event)


def _has_conversation_snapshot_request(
    ctx: hsm.Context,
    instance: "Conversation[typ.Any, typ.Any]",
    event: hsm.Event[typ.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, SnapshotRequest)


def _dispatch_conversation_phase_failure(
    ctx: hsm.Context,
    instance: "Conversation[typ.Any, typ.Any]",
    source: hsm.Event[typ.Any],
    failure_event: hsm.Event[FailureData],
    *,
    kind: ConversationChildKind | None = None,
) -> None:
    del kind
    # Prefer the active turn id; fall back to source envelope for fail-closed paths before start.
    operation_id = instance._active_turn_id or _conversation_operation_id(source)
    _ = hsm.dispatch(
        ctx,
        instance,
        _conversation_event_with_operation(
            failure_event,
            source,
            operation_id=operation_id,
        ),
    )


def _dispatch_conversation_terminal_output(
    ctx: hsm.Context,
    instance: "Conversation[typ.Any, typ.Any]",
    event: hsm.Event[typ.Any],
    output: Response,
) -> None:
    terminal = _conversation_event_with_operation(instance.output_event.with_data(output), event)
    terminal = dataclasses.replace(terminal, source=hsm.id(instance))
    _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))


def _dispatch_conversation_terminal_failure(
    ctx: hsm.Context,
    instance: "Conversation[typ.Any, typ.Any]",
    source: hsm.Event[typ.Any],
    failure: FailureData,
) -> None:
    operation_id = _conversation_operation_id(source)
    if operation_id is not None:
        instance.fail_contribution_waiter(operation_id, RuntimeError(failure.message))
    terminal = _conversation_event_with_operation(instance.failed_event.with_data(failure), source)
    terminal = dataclasses.replace(terminal, source=hsm.id(instance))
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


class Conversation(
    ability.Ability[TAnyMessage, TResponse],
    abc.ABC,
    typ.Generic[TAnyMessage, TResponse],
):
    """Thin conversation coordinator: decode, participate, and publish contribution terminals.

    Decision inputs, cognition, memory, and response encoding are host-owned.
    """

    input_data_type: typ.ClassVar[type[object] | tuple[type[object], ...] | None] = Message
    output_data_type: typ.ClassVar[type[object] | tuple[type[object], ...] | None] = Response
    input_event: typ.ClassVar[hsm.Event[typ.Any]] = InputEvent
    output_event: typ.ClassVar[hsm.Event[typ.Any]] = OutputEvent
    failed_event: typ.ClassVar[hsm.Event[FailureData]] = FailedEvent
    snapshot_request_event: typ.ClassVar[hsm.Event[SnapshotRequest]] = SnapshotRequestEvent
    snapshot_output_event: typ.ClassVar[hsm.Event[Snapshot]] = SnapshotOutputEvent
    _composite_attachment_lifecycle: typ.ClassVar[bool] = True
    # Concrete Text/Voice conversations replace this via define_model; keep composite
    # detach vertices so Ability lifecycle validation succeeds on the abstract base.
    submodel: typ.ClassVar[hsm.Model | None] = hsm.define(
        "Conversation",
        hsm.initial(hsm.target("/Conversation/silent")),
        hsm.state("silent"),
        hsm.state(
            "detaching",
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Conversation/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Conversation/silent"),
            ),
        ),
        hsm.state("degraded"),
        hsm.observe(observer),
    )
    _decoding: decoding.Decoding[participating.ParticipationStimulus, str]
    _participating: participating.Participating
    _attachment_group: attachment.Group
    _conversation_ref: str | None
    _participants_by_ref: dict[str, participating.ParticipantSnapshot]
    # Single-flight turn correlation only (HSM-CORRELATION-001). Message and DecodedTurn
    # never live on the instance — phase activities hold them and private completions carry them.
    _active_turn_id: str | None
    _contribution_waiters: dict[str, asyncio.Future[object]]

    def __init__(
        self,
        *,
        decoding: decoding.Decoding[participating.ParticipationStimulus, str] | None,
        participating: participating.Participating | None,
    ) -> None:
        if decoding is None:
            raise ValueError("Conversation requires decoding.")
        if participating is None:
            raise ValueError("Conversation requires participating.")
        super().__init__()
        self._decoding = decoding
        self._participating = participating
        self._attachment_group = attachment.Group(self._decoding, self._participating)
        self._conversation_ref = None
        self._participants_by_ref = {}
        self._active_turn_id = None
        self._contribution_waiters = {}

    def register_contribution_waiter(self, operation_id: str, waiter: asyncio.Future[object]) -> None:
        """Register a host Future completed with ParticipatedTurn for ``operation_id``."""

        if not operation_id:
            raise ValueError("operation_id is required.")
        self._contribution_waiters[operation_id] = waiter

    def clear_contribution_waiter(self, operation_id: str) -> None:
        """Drop a host contribution waiter if it is still registered."""

        _ = self._contribution_waiters.pop(operation_id, None)

    def complete_contribution_waiter(self, operation_id: str, participated: ParticipatedTurn) -> None:
        """Complete a host contribution waiter with the participated turn."""

        waiter = self._contribution_waiters.pop(operation_id, None)
        if isinstance(waiter, asyncio.Future) and not waiter.done():
            waiter.set_result(participated)

    def fail_contribution_waiter(self, operation_id: str, error: BaseException) -> None:
        """Fail a host contribution waiter."""

        waiter = self._contribution_waiters.pop(operation_id, None)
        if isinstance(waiter, asyncio.Future) and not waiter.done():
            waiter.set_exception(error)

    @abc.abstractmethod
    def _conversation_kind(self) -> str:
        """Return the concrete conversation modality marker."""

    def child_for_kind(self, kind: ConversationChildKind) -> ability.Ability[typ.Any, typ.Any] | None:
        if kind == "decoding":
            return typ.cast(ability.Ability[typ.Any, typ.Any], self._decoding)
        if kind == "participating":
            return typ.cast(ability.Ability[typ.Any, typ.Any], self._participating)
        return None

    @staticmethod
    def stage_for_child_kind(kind: ConversationChildKind) -> Stage:
        if kind == "participating":
            return "participating"
        return "decoding"

    @staticmethod
    def failed_event_for_child_kind(kind: ConversationChildKind) -> hsm.Event[FailureData]:
        if kind == "participating":
            return _ConversationParticipatingFailedEvent
        return _ConversationDecodingFailedEvent

    @staticmethod
    def _build_contribution_response(
        instance: "Conversation[typ.Any, typ.Any]",
        participated: ParticipatedTurn,
    ) -> object:
        """Build the contribution-only terminal response (no host-encoded content)."""

        input = participated.input
        return Response(
            conversation_ref=input.conversation_ref,
            self_participant_ref=input.self_participant_ref,
            participants=tuple(instance._participants_by_ref.values()),
            content=None,
            decoded_text=participated.decoded_text,
        )

    @staticmethod
    def _clear_active_turn(
        instance: "Conversation[typ.Any, typ.Any]",
    ) -> None:
        """Drop single-flight turn correlation on terminal or detach."""

        instance._active_turn_id = None

    @staticmethod
    def _clear_active_turn_and_fail(
        ctx: hsm.Context,
        instance: "Conversation[typ.Any, typ.Any]",
        event: hsm.Event[typ.Any],
    ) -> None:
        """Clear active-turn correlation and publish the public failure terminal."""

        failure = event.data
        assert isinstance(failure, FailureData)
        Conversation._clear_active_turn(instance)
        _dispatch_conversation_terminal_failure(ctx, instance, event, failure)

    @staticmethod
    def _dispatch_contribution_output(
        ctx: hsm.Context,
        instance: "Conversation[typ.Any, typ.Any]",
        event: hsm.Event[typ.Any],
    ) -> None:
        """Complete a turn after participation; hosts may continue with decide/memory/encode."""

        participated = event.data
        assert isinstance(participated, ParticipatedTurn)
        Conversation._clear_active_turn(instance)
        output = Conversation._build_contribution_response(instance, participated)
        if not _matches_conversation_output_contract(instance, output):
            failure = FailureData(
                stage="participating",
                message="Conversation contribution output type does not match its output event.",
            )
            _dispatch_conversation_terminal_failure(ctx, instance, event, failure)
            return
        assert isinstance(output, Response)
        operation_id = _conversation_operation_id(event)
        if operation_id is not None:
            instance.complete_contribution_waiter(operation_id, participated)
        _dispatch_conversation_terminal_output(ctx, instance, event, output)

    @staticmethod
    def _record_participant_states(
        ctx: hsm.Context,
        instance: "Conversation[typ.Any, typ.Any]",
        event: hsm.Event[typ.Any],
    ) -> None:
        del ctx
        input = event.data
        assert isinstance(input, Message)
        instance._conversation_ref = input.conversation_ref
        instance._participants_by_ref = {participant.ref: participant for participant in input.participants}

    @staticmethod
    def _clear_prior_turn_on_detach(
        ctx: hsm.Context,
        instance: "Conversation[typ.Any, typ.Any]",
        event: hsm.Event[typ.Any],
    ) -> None:
        del ctx, event
        instance._conversation_ref = None
        instance._participants_by_ref = {}
        Conversation._clear_active_turn(instance)
        instance._contribution_waiters.clear()

    @staticmethod
    def _dispatch_conversation_snapshot(
        ctx: hsm.Context,
        instance: "Conversation[typ.Any, typ.Any]",
        event: hsm.Event[typ.Any],
    ) -> None:
        request = event.data
        assert isinstance(request, SnapshotRequest)
        snapshot = Snapshot(
            request_ref=request.request_ref,
            conversation_ref=instance._conversation_ref,
            participants=tuple(instance._participants_by_ref.values()),
        )
        _ = hsm.dispatch(ctx, instance, instance.snapshot_output_event.with_data(snapshot))

    @staticmethod
    async def _run_decoding_activity(
        ctx: hsm.Context,
        instance: "Conversation[typ.Any, typ.Any]",
        event: hsm.Event[typ.Any],
    ) -> None:
        """Decode phase: Message stays activity-local; private completion carries DecodedTurn."""

        message = event.data
        if not isinstance(message, Message):
            _dispatch_conversation_phase_failure(
                ctx,
                instance,
                event,
                _ConversationDecodingFailedEvent.with_data(
                    FailureData(
                        stage="decoding",
                        message="Conversation decoding started without a Message payload.",
                    )
                ),
                kind="decoding",
            )
            return
        operation_id = _conversation_operation_id(event)
        if operation_id is None:
            _dispatch_conversation_phase_failure(
                ctx,
                instance,
                event,
                _ConversationDecodingFailedEvent.with_data(
                    FailureData(
                        stage="decoding",
                        message="Conversation refused a turn without an envelope operation id.",
                    )
                ),
                kind="decoding",
            )
            return
        instance._active_turn_id = operation_id
        child = typ.cast(ability.Ability[typ.Any, typ.Any], instance._decoding)
        try:
            terminal = await ability.Ability.await_child_terminal(
                ctx,
                owner=instance,
                child=child,
                operation_id=operation_id,
                input=message.content,
                metadata=event.metadata,
            )
        except asyncio.CancelledError:
            Conversation._clear_active_turn(instance)
            raise
        if terminal.name == child.failed_event.name:
            message_text = getattr(terminal.data, "message", "Conversation decoding child failed.")
            _dispatch_conversation_phase_failure(
                ctx,
                instance,
                event,
                _ConversationDecodingFailedEvent.with_data(FailureData(stage="decoding", message=str(message_text))),
                kind="decoding",
            )
            return
        if not isinstance(terminal.data, str):
            _dispatch_conversation_phase_failure(
                ctx,
                instance,
                event,
                _ConversationDecodingFailedEvent.with_data(
                    FailureData(stage="decoding", message="Conversation decoding produced a non-text output.")
                ),
                kind="decoding",
            )
            return
        decoded_text = terminal.data.strip()
        if not decoded_text:
            _dispatch_conversation_phase_failure(
                ctx,
                instance,
                event,
                _ConversationDecodingFailedEvent.with_data(
                    FailureData(stage="decoding", message="Conversation decoding produced no decoded text.")
                ),
                kind="decoding",
            )
            return
        turn_input = typ.cast(AnyMessage, message)
        turn = DecodedTurn(
            input=turn_input,
            stimulus=participating.EventStimulus(
                source_participant_ref=turn_input.content.source_participant_ref,
                event="conversation.decoding",
                payload={
                    "source_kind": turn_input.content.kind,
                    "text": decoded_text,
                },
            ),
            decoded_text=decoded_text,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            _conversation_event_with_operation(
                _ConversationDecodingCompletedEvent.with_data(turn),
                event,
                operation_id=operation_id,
            ),
        )

    @staticmethod
    async def _run_participating_activity(
        ctx: hsm.Context,
        instance: "Conversation[typ.Any, typ.Any]",
        event: hsm.Event[typ.Any],
    ) -> None:
        """Participate phase: DecodedTurn is the trigger payload (completion event), not instance state."""

        # Entering transition already guards active turn + DecodedTurn; payload is local here.
        decoded = event.data
        assert isinstance(decoded, DecodedTurn)
        operation_id = instance._active_turn_id
        assert operation_id is not None
        child = typ.cast(ability.Ability[typ.Any, typ.Any], instance._participating)
        participating_input = participating.InputData(
            conversation_ref=decoded.input.conversation_ref,
            self_participant_ref=decoded.input.self_participant_ref,
            participants=decoded.input.participants,
            stimulus=decoded.stimulus,
        )
        try:
            terminal = await ability.Ability.await_child_terminal(
                ctx,
                owner=instance,
                child=child,
                operation_id=operation_id,
                input=participating_input,
                metadata=event.metadata,
            )
        except asyncio.CancelledError:
            Conversation._clear_active_turn(instance)
            raise
        if terminal.name == child.failed_event.name:
            message_text = getattr(terminal.data, "message", "Conversation participating child failed.")
            _dispatch_conversation_phase_failure(
                ctx,
                instance,
                event,
                _ConversationParticipatingFailedEvent.with_data(
                    FailureData(stage="participating", message=str(message_text))
                ),
                kind="participating",
            )
            return
        output = terminal.data
        if not isinstance(output, participating.OutputData):
            _dispatch_conversation_phase_failure(
                ctx,
                instance,
                event,
                _ConversationParticipatingFailedEvent.with_data(
                    FailureData(
                        stage="participating",
                        message="Conversation participation produced invalid output.",
                    )
                ),
                kind="participating",
            )
            return
        # decoded is still the activity-local completion payload — no instance stash.
        participated = ParticipatedTurn(
            input=decoded.input,
            stimulus=decoded.stimulus,
            decoded_text=decoded.decoded_text,
            participation=output,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            _conversation_event_with_operation(
                _ConversationParticipatingCompletedEvent.with_data(participated),
                event,
                operation_id=operation_id,
            ),
        )

    @classmethod
    def define_model(
        cls,
        root_name: str,
        *,
        input_event: hsm.Event[typ.Any],
        input_guard: typ.Callable[[hsm.Context, "Conversation[typ.Any, typ.Any]", hsm.Event[typ.Any]], bool],
    ) -> hsm.Model:
        root_path = f"/{root_name}"

        def _has_correlated_input(
            ctx: hsm.Context,
            instance: Conversation[typ.Any, typ.Any],
            event: hsm.Event[typ.Any],
        ) -> bool:
            return (
                _matches_message_contract(instance, event.data)
                and input_guard(ctx, instance, event)
                and _has_conversation_operation_id(ctx, instance, event)
            )

        return hsm.define(
            root_name,
            hsm.initial(hsm.target(f"{root_path}/initializing")),
            hsm.transition(
                hsm.on(cls.snapshot_request_event),
                hsm.guard(_has_conversation_snapshot_request),
                hsm.effect(cls._dispatch_conversation_snapshot),
            ),
            hsm.state(
                "initializing",
                hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
                hsm.activity(ability.Ability._attach_composite_group),
                hsm.transition(
                    hsm.on(ability.Ability._composite_attachment_terminal_event),
                    hsm.guard(ability.Ability._is_composite_attach_complete),
                    hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                    hsm.target(f"{root_path}/silent"),
                ),
            ),
            hsm.state(
                "silent",
                hsm.transition(
                    hsm.on(input_event),
                    hsm.guard(_has_correlated_input),
                    hsm.effect(cls._record_participant_states),
                    hsm.target(f"{root_path}/active/decoding"),
                ),
            ),
            hsm.state(
                "active",
                hsm.initial(hsm.target(f"{root_path}/active/decoding")),
                hsm.defer(input_event),
                hsm.transition(
                    hsm.on(
                        _ConversationDecodingFailedEvent,
                        _ConversationParticipatingFailedEvent,
                    ),
                    hsm.guard(_has_conversation_failure),
                    hsm.effect(cls._clear_active_turn_and_fail),
                    hsm.target(f"{root_path}/silent"),
                ),
                hsm.state(
                    "decoding",
                    # Message is activity-local; private DecodingCompleted carries DecodedTurn.
                    hsm.activity(cls._run_decoding_activity),
                    hsm.transition(
                        hsm.on(_ConversationDecodingCompletedEvent),
                        hsm.guard(_has_conversation_decoded),
                        hsm.target(f"{root_path}/active/participating"),
                    ),
                ),
                hsm.state(
                    "participating",
                    # DecodedTurn arrives as the DecodingCompleted trigger (completion event data).
                    hsm.activity(cls._run_participating_activity),
                    hsm.transition(
                        hsm.on(_ConversationParticipatingCompletedEvent),
                        hsm.guard(_has_conversation_participated),
                        hsm.effect(Conversation._dispatch_contribution_output),
                        hsm.target(f"{root_path}/silent"),
                    ),
                ),
            ),
            hsm.state(
                "detaching",
                hsm.entry(cls._clear_prior_turn_on_detach),
                hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
                hsm.activity(ability.Ability._detach_composite_group),
                hsm.transition(
                    hsm.on(ability.Ability._composite_attachment_terminal_event),
                    hsm.guard(ability.Ability._is_composite_rollback_failure),
                    hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                    hsm.target(f"{root_path}/degraded"),
                ),
                hsm.transition(
                    hsm.on(ability.Ability._composite_attachment_terminal_event),
                    hsm.guard(ability.Ability._is_composite_detach_failed),
                    hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                    hsm.target(f"{root_path}/silent"),
                ),
            ),
            hsm.state("degraded"),
            hsm.observe(observer),
        )


def define_conversation_model(
    root_name: str,
    *,
    input_event: hsm.Event[typ.Any],
    input_guard: typ.Callable[[hsm.Context, Conversation[typ.Any, typ.Any], hsm.Event[typ.Any]], bool],
) -> hsm.Model:
    return Conversation.define_model(root_name, input_event=input_event, input_guard=input_guard)


def default_pair_participants(
    *,
    self_participant_ref: str = "bot",
    source_participant_ref: str = "caller",
) -> tuple[participating.ParticipantSnapshot, ...]:
    """Minimal bot + remote participant snapshots for product ingress builders."""

    return (
        participating.ParticipantSnapshot(
            ref=self_participant_ref,
            kind="bot",
            state=participating.ParticipantStateSnapshot(
                presence="present",
                attention="available",
                turn="listening",
            ),
        ),
        participating.ParticipantSnapshot(
            ref=source_participant_ref,
            kind="human",
            state=participating.ParticipantStateSnapshot(
                presence="present",
                attention="available",
                turn="holding",
            ),
        ),
    )


def text_turn(
    text: str,
    *,
    conversation_ref: str = "conversation",
    self_participant_ref: str = "bot",
    source_participant_ref: str = "caller",
    participants: tuple[participating.ParticipantSnapshot, ...] | None = None,
) -> TextMessage:
    """Build a text conversation Message for Bot/product ingress (no I/O)."""

    room = participants or default_pair_participants(
        self_participant_ref=self_participant_ref,
        source_participant_ref=source_participant_ref,
    )
    return TextMessage(
        conversation_ref=conversation_ref,
        self_participant_ref=self_participant_ref,
        participants=room,
        content=participating.TextStimulus(
            source_participant_ref=source_participant_ref,
            content=text,
        ),
    )


def voice_turn(
    audio: bytes,
    *,
    conversation_ref: str = "conversation",
    self_participant_ref: str = "bot",
    source_participant_ref: str = "caller",
    participants: tuple[participating.ParticipantSnapshot, ...] | None = None,
) -> VoiceMessage:
    """Build a voice conversation Message for Bot/product ingress (no I/O)."""

    room = participants or default_pair_participants(
        self_participant_ref=self_participant_ref,
        source_participant_ref=source_participant_ref,
    )
    return VoiceMessage(
        conversation_ref=conversation_ref,
        self_participant_ref=self_participant_ref,
        participants=room,
        content=participating.AudioStimulus(
            source_participant_ref=source_participant_ref,
            content=audio,
        ),
    )


__all__ = [
    "FailedEvent",
    "InputEvent",
    "OutputEvent",
    "SnapshotOutputEvent",
    "SnapshotRequestEvent",
    "AnyMessage",
    "Conversation",
    "ConversationChildKind",
    "DecodedTurn",
    "FailureData",
    "Message",
    "ParticipatedTurn",
    "Response",
    "Snapshot",
    "SnapshotRequest",
    "Stage",
    "TextMessage",
    "VoiceMessage",
    "conversation_event_with_operation",
    "default_pair_participants",
    "define_conversation_model",
    "matches_conversation_output_contract",
    "text_turn",
    "voice_turn",
]
