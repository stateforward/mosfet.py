"""The ``Embedding`` ability: a bot's provider-neutral capacity to embed text and images.

Admission, the bounded FIFO queue, per-request timeouts, and stale-result handling live here, in
topology. The injected ``Embedder`` does the embedding; provider actors it names in
``attachments`` attach before this ability reports attached and detach after it stops serving.
Every request inside the attached ability is answered with exactly one typed terminal: an
``OutputEvent``, or a ``FailedEvent`` whose ``FailureData.kind`` is closed.

Telemetry (cardinality, METRICS > SPANS > Logs; every value comes from an enum in code):

* ``bot.ability.embedding.request.count`` — ``bot.outcome`` {ok, failed} × ``bot.failure.kind``
  (FailureKind or ``none``; ok pairs only with ``none``) × ``bot.embedding.role`` {query, document}
  × ``bot.embedding.modality`` {text, image, mixed}: at most 60 series, plus one series without
  role and modality for a request whose payload is not ``InputData``.
* ``bot.ability.embedding.request.duration`` — the embedder call, by outcome (whether it returned
  or raised) × modality × role: 12 series.
* ``bot.ability.embedding.item.count`` — items embedded, by item modality × role: 6 series.
* ``bot.ability.embedding.queue.depth`` — requests queued or in flight, no attributes: 1 series.
"""

import collections
import dataclasses
import datetime
import time
import typing
import uuid

import hsm
import mosfet
from opentelemetry import metrics

from mosfet.protocols import attachment

from .. import ability
from ..language import text
from . import embedder, payload

_Modality: typing.TypeAlias = typing.Literal["text", "image", "mixed"]

_DEFAULT_TIMEOUT: typing.Final = datetime.timedelta(seconds=30)
_DEFAULT_QUEUE_LIMIT: typing.Final = 16
_METER = metrics.get_meter("bot.abilities.embedding")
_REQUESTS = _METER.create_counter(
    "bot.ability.embedding.request.count",
    unit="{request}",
    description="Embedding requests answered, by outcome, failure kind, role, and request modality.",
)
_DURATION = _METER.create_histogram(
    "bot.ability.embedding.request.duration",
    unit="s",
    description="Duration of each embedder call, by whether it returned, role, and request modality.",
)
_ITEMS = _METER.create_counter(
    "bot.ability.embedding.item.count",
    unit="{item}",
    description="Items embedded successfully, by item modality and role.",
)
_QUEUE_DEPTH = _METER.create_up_down_counter(
    "bot.ability.embedding.queue.depth",
    unit="{request}",
    description="Embedding requests admitted and not yet answered, the in-flight request included.",
)

InputEvent = hsm.Event[payload.InputData](
    name="bot.ability.embedding.input",
    schema=payload.InputData,
)
OutputEvent = hsm.Event[payload.OutputData](
    name="bot.ability.embedding.output",
    schema=payload.OutputData,
)
FailedEvent = hsm.Event[payload.FailureData](
    name="bot.ability.embedding.failed",
    schema=payload.FailureData,
)
_AppliedEvent = hsm.Event[payload.OutputData](
    name="bot.ability.embedding.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=payload.OutputData,
)
_ApplyFailedEvent = hsm.Event[payload.FailureData](
    name="bot.ability.embedding.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=payload.FailureData,
)


def _item_modality(item: payload.Item) -> _Modality:
    images = sum(isinstance(part, text.ImagePart) for part in item.parts)
    if images == 0:
        return "text"
    return "image" if images == len(item.parts) else "mixed"


def _modality(data: payload.InputData) -> _Modality:
    modalities: set[_Modality] = {_item_modality(item) for item in data.items}
    return modalities.pop() if len(modalities) == 1 else "mixed"


def _rejection(capacity: embedder.Capacity, data: object) -> payload.FailureData | None:
    """Mechanical admission against ``capacity``; pure, so guards and effects share it.

    The ability never trims: a request outside the capacity fails closed as a whole.
    """

    if not isinstance(data, payload.InputData):
        return payload.FailureData(
            kind=payload.FailureKind.INVALID_INPUT,
            message=f"embedding request payload must be InputData, got {type(data).__name__}",
        )
    if len(data.items) > capacity.max_items:
        return payload.FailureData(
            kind=payload.FailureKind.UNSUPPORTED_INPUT,
            message=f"request has {len(data.items)} items; the embedder accepts at most {capacity.max_items}",
        )
    if not capacity.interleaved:
        for index, item in enumerate(data.items):
            if _item_modality(item) == "mixed":
                return payload.FailureData(
                    kind=payload.FailureKind.UNSUPPORTED_INPUT,
                    message="the embedder does not accept items that mix text and images",
                    item_index=index,
                )
    owners = [
        (index, part.image)
        for index, item in enumerate(data.items)
        for part in item.parts
        if isinstance(part, text.ImagePart)
    ]
    verdicts = capacity.images.admit([image for _, image in owners])
    for (index, _), verdict in zip(owners, verdicts, strict=True):
        if verdict is not None:
            return payload.FailureData(
                kind=payload.FailureKind.UNSUPPORTED_INPUT,
                message=f"an image is outside the embedder's image capacity ({verdict})",
                item_index=index,
            )
    return None


def _response_failure(
    capacity: embedder.Capacity,
    request: payload.InputData | None,
    output: object,
) -> payload.FailureData | None:
    """Whether an embedder's output answers ``request`` in the declared space; pure."""

    assert request is not None, "only admitted InputData requests are queued"
    if not isinstance(output, payload.OutputData):
        problem = f"the embedder returned {type(output).__name__}, not OutputData"
    elif output.space != capacity.space:
        problem = "the embedder returned vectors in a space other than its declared one"
    elif len(output.vectors) != len(request.items):
        problem = f"the embedder returned {len(output.vectors)} vectors for {len(request.items)} items"
    elif output.role is not request.role:
        problem = "the embedder returned a role other than the request's"
    else:
        return None
    return payload.FailureData(kind=payload.FailureKind.INVALID_RESPONSE, message=problem)


def _request_attributes(data: object) -> dict[str, str]:
    if not isinstance(data, payload.InputData):
        return {}
    return {"bot.embedding.role": data.role.value, "bot.embedding.modality": _modality(data)}


class Embedding(ability.Ability[payload.InputData, payload.OutputData]):
    """A bot's capacity to embed text and images into one declared vector space.

    Send an ``InputEvent`` (for example with ``ability.run_terminal_operation``); the answer is
    one ``OutputEvent`` or ``FailedEvent`` carrying the request's ``id``. Requests are admitted
    mechanically against the embedder's ``Capacity`` and served one at a time in FIFO order;
    at most ``queue_limit`` wait behind the one in flight. A request that does not finish within
    ``timeout`` fails with ``TIMEOUT`` and its late result is ignored. While the ability is not
    serving (attaching its provider actors, detaching, or degraded) requests fail with
    ``UNAVAILABLE``; a detach fails every queued request the same way before the owner is told.
    """

    input_event: typing.ClassVar[hsm.Event[payload.InputData]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[payload.OutputData]] = OutputEvent
    failed_event: typing.ClassVar[hsm.Event[payload.FailureData]] = FailedEvent
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = payload.InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = payload.OutputData
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _attachment_group: attachment.Group | None
    _embedder: embedder.Embedder
    _capacity: embedder.Capacity
    _timeout: datetime.timedelta
    _queue_limit: int
    # Machine-owned admission buffer (HSM-STATE-001): the head is the request in flight.
    _queue: collections.deque[hsm.Event[payload.InputData]]

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        embedder: embedder.Embedder,
        timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
        queue_limit: int = _DEFAULT_QUEUE_LIMIT,
    ) -> None:
        """Wrap ``embedder``; ``timeout`` bounds each request once it is in flight.

        Raises:
            ValueError: If ``timeout`` is not positive or ``queue_limit`` is negative.
        """

        if timeout <= datetime.timedelta(0):
            raise ValueError("Embedding timeout must be positive.")
        if queue_limit < 0:
            raise ValueError("Embedding queue_limit cannot be negative.")
        super().__init__(id=id)
        self._embedder = embedder
        self._capacity = embedder.capacity
        self._timeout = timeout
        self._queue_limit = queue_limit
        self._queue = collections.deque()
        self._attachment_group = attachment.Group(*embedder.attachments)

    @staticmethod
    def _reply(
        ctx: hsm.Context,
        instance: "Embedding",
        request: hsm.Event[typing.Any],
        data: payload.OutputData | payload.FailureData,
    ) -> None:
        """Emit the one terminal for ``request`` (decoding rule: target is the request's source)."""

        terminal_event: hsm.Event[typing.Any] = (
            instance.output_event if isinstance(data, payload.OutputData) else instance.failed_event
        )
        terminal = dataclasses.replace(
            terminal_event.with_data(data),
            id=request.id or None,
            metadata=dict(request.metadata),
            source=hsm.id(instance),
            target=request.source if request.target == hsm.id(instance) else "",
        )
        wrapper = ability.TerminalOutputEvent if isinstance(data, payload.OutputData) else ability.TerminalErrorEvent
        _ = hsm.dispatch(ctx, instance, wrapper.with_data(terminal))
        attributes = _request_attributes(request.data)
        if isinstance(data, payload.OutputData):
            _REQUESTS.add(1, {"bot.outcome": "ok", "bot.failure.kind": "none", **attributes})
            assert isinstance(request.data, payload.InputData)
            for item in request.data.items:
                _ITEMS.add(
                    1, {"bot.embedding.role": request.data.role.value, "bot.embedding.modality": _item_modality(item)}
                )
            return
        _REQUESTS.add(1, {"bot.outcome": "failed", "bot.failure.kind": data.kind.value, **attributes})

    @staticmethod
    def _dequeue(instance: "Embedding") -> hsm.Event[payload.InputData]:
        _QUEUE_DEPTH.add(-1)
        return instance._queue.popleft()

    @staticmethod
    def _is_rejected(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return _rejection(instance._capacity, event.data) is not None

    @staticmethod
    def _has_room(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return _rejection(instance._capacity, event.data) is None and len(instance._queue) <= instance._queue_limit

    @staticmethod
    def _is_overloaded(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return _rejection(instance._capacity, event.data) is None and len(instance._queue) > instance._queue_limit

    @staticmethod
    def _reply_rejection(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> None:
        failure = _rejection(instance._capacity, event.data)
        assert failure is not None
        Embedding._reply(ctx, instance, event, failure)

    @staticmethod
    def _reply_overloaded(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> None:
        failure = payload.FailureData(
            kind=payload.FailureKind.OVERLOADED,
            message=f"embedding queue is full ({instance._queue_limit} waiting behind the request in flight)",
        )
        Embedding._reply(ctx, instance, event, failure)

    @staticmethod
    def _reply_unavailable(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> None:
        failure = payload.FailureData(kind=payload.FailureKind.UNAVAILABLE, message="embedding ability is not serving")
        Embedding._reply(ctx, instance, event, failure)

    @staticmethod
    def _enqueue(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> None:
        del ctx
        # Admission correlates results by id (HSM-CORRELATION-001), so every queued request has one.
        request = event if event.id else dataclasses.replace(event, id=uuid.uuid4().hex)
        instance._queue.append(typing.cast(hsm.Event[payload.InputData], request))
        _QUEUE_DEPTH.add(1)

    @staticmethod
    async def _embed_head(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> None:
        """Run the embedder for the head request and return its outcome as a correlated self-event."""

        del event
        request = instance._queue[0]
        data = request.data
        assert data is not None, "only admitted InputData requests are queued"
        attributes = _request_attributes(data)
        started = time.monotonic()  # telemetry only; hsm.after owns the request deadline
        try:
            output = await instance._embedder.embed(data, ctx=ctx)
        except embedder.Error as error:
            outcome = "failed"
            completion = _ApplyFailedEvent.with_data(
                payload.FailureData(kind=error.kind, message=str(error), item_index=error.item_index)
            )
        except Exception as error:
            # Boundary: any other provider failure becomes the typed EMBED_FAILED terminal.
            outcome = "failed"
            completion = _ApplyFailedEvent.with_data(
                payload.FailureData(kind=payload.FailureKind.EMBED_FAILED, message=f"{type(error).__name__}: {error}")
            )
        else:
            outcome = "ok"
            completion = _AppliedEvent.with_data(output)
        _DURATION.record(time.monotonic() - started, {"bot.outcome": outcome, **attributes})
        _ = instance.dispatch(
            ctx,
            dataclasses.replace(
                completion,
                id=request.id,
                source=request.source,
                target=request.target,
                metadata=dict(request.metadata),
            ),
        )

    @staticmethod
    def _is_head(instance: "Embedding", event: hsm.Event[typing.Any]) -> bool:
        return bool(instance._queue) and event.id == instance._queue[0].id and event.source == instance._queue[0].source

    @staticmethod
    def _is_valid_output(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return (
            Embedding._is_head(instance, event)
            and _response_failure(instance._capacity, instance._queue[0].data, event.data) is None
        )

    @staticmethod
    def _is_invalid_output(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return (
            Embedding._is_head(instance, event)
            and _response_failure(instance._capacity, instance._queue[0].data, event.data) is not None
        )

    @staticmethod
    def _is_head_failure(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return Embedding._is_head(instance, event) and isinstance(event.data, payload.FailureData)

    @staticmethod
    def _reply_output(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> None:
        output = event.data
        assert isinstance(output, payload.OutputData)
        Embedding._reply(ctx, instance, Embedding._dequeue(instance), output)

    @staticmethod
    def _reply_invalid_output(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> None:
        request = Embedding._dequeue(instance)
        failure = _response_failure(instance._capacity, request.data, event.data)
        assert failure is not None
        Embedding._reply(ctx, instance, request, failure)

    @staticmethod
    def _reply_failure(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> None:
        failure = event.data
        assert isinstance(failure, payload.FailureData)
        Embedding._reply(ctx, instance, Embedding._dequeue(instance), failure)

    @staticmethod
    def _timeout_delay(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._timeout

    @staticmethod
    def _reply_timeout(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> None:
        del event
        failure = payload.FailureData(
            kind=payload.FailureKind.TIMEOUT,
            message=f"embedding did not complete within {instance._timeout.total_seconds():g}s",
        )
        Embedding._reply(ctx, instance, Embedding._dequeue(instance), failure)

    @staticmethod
    def _has_queued(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return bool(instance._queue)

    @staticmethod
    def _fail_queued(ctx: hsm.Context, instance: "Embedding", event: hsm.Event[typing.Any]) -> None:
        """Leaving ``serving``: answer every queued request, the in-flight head included, UNAVAILABLE."""

        del event
        failure = payload.FailureData(kind=payload.FailureKind.UNAVAILABLE, message="embedding ability is detaching")
        while instance._queue:
            Embedding._reply(ctx, instance, Embedding._dequeue(instance), failure)

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Embedding",
        hsm.initial(hsm.target("/Embedding/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(hsm.on(InputEvent), hsm.effect(_reply_unavailable)),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Embedding/serving"),
            ),
        ),
        hsm.state(
            "serving",
            hsm.exit(_fail_queued),
            hsm.initial(hsm.target("/Embedding/serving/idle")),
            hsm.state(
                "idle",
                hsm.transition(hsm.on(InputEvent), hsm.guard(_is_rejected), hsm.effect(_reply_rejection)),
                hsm.transition(
                    hsm.on(InputEvent),
                    hsm.guard(_has_room),
                    hsm.effect(_enqueue),
                    hsm.target("/Embedding/serving/embedding"),
                ),
            ),
            hsm.state(
                "embedding",
                hsm.activity(_embed_head),
                hsm.transition(hsm.on(InputEvent), hsm.guard(_is_rejected), hsm.effect(_reply_rejection)),
                hsm.transition(hsm.on(InputEvent), hsm.guard(_is_overloaded), hsm.effect(_reply_overloaded)),
                hsm.transition(hsm.on(InputEvent), hsm.guard(_has_room), hsm.effect(_enqueue)),
                hsm.transition(
                    hsm.on(_AppliedEvent),
                    hsm.guard(_is_valid_output),
                    hsm.effect(_reply_output),
                    hsm.target("/Embedding/serving/next"),
                ),
                hsm.transition(
                    hsm.on(_AppliedEvent),
                    hsm.guard(_is_invalid_output),
                    hsm.effect(_reply_invalid_output),
                    hsm.target("/Embedding/serving/next"),
                ),
                hsm.transition(
                    hsm.on(_ApplyFailedEvent),
                    hsm.guard(_is_head_failure),
                    hsm.effect(_reply_failure),
                    hsm.target("/Embedding/serving/next"),
                ),
                hsm.transition(
                    hsm.after(_timeout_delay),
                    hsm.effect(_reply_timeout),
                    hsm.target("/Embedding/serving/next"),
                ),
            ),
            hsm.choice(
                "next",
                hsm.transition(hsm.guard(_has_queued), hsm.target("/Embedding/serving/embedding")),
                hsm.transition(hsm.target("/Embedding/serving/idle")),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(hsm.on(InputEvent), hsm.effect(_reply_unavailable)),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Embedding/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Embedding/serving"),
            ),
        ),
        hsm.state(
            "degraded",
            hsm.transition(hsm.on(InputEvent), hsm.effect(_reply_unavailable)),
        ),
    )


__all__ = [
    "FailedEvent",
    "InputEvent",
    "OutputEvent",
    "Embedding",
]
