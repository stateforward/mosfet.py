"""``Model``: one llama.cpp model resident on one device, shared by every owner attached to it.

A ``Model`` loads on its first owner's attach and unloads on its last owner's detach. Native work runs one
job at a time on the model's single worker thread, and every job's result comes back as a typed completion
event carrying the job id and the origin of the request it answers (HSM-COMPLETION-001). A native call cannot
be interrupted, so the machine never frees a session a job may still be using: ``abandoning``, ``releasing``
and ``stalled`` wait for the job's own completion.

The composition root owns the lifetime. It starts the model in a long-lived, non-private scope (``start``
refuses a private one) and stops it after every owner has stopped. Owners only attach and detach.

Machine fields (each owned by this class, with its lifetime):

* ``_in_flight``: the job occupying the worker (its id, kind and future). Written by every effect that submits
  a job; cleared by the effect that consumes that job's completion. ``_is_in_flight`` correlates completions by
  its id, and ``stop`` reads its kind to close a session exactly once.
* ``_session``: the resident session. Written when a load completes in ``loading``; cleared when its close
  completes (or by ``stop``).
* ``_deadline``: the one pending request a timeout must answer, because ``hsm.after`` carries no data. Written by
  the effect entering ``loading`` (the AttachEvent) or ``unloading`` (the DetachEvent); read by the delay
  function and the timeout effect; cleared by the effect of every transition that leaves ``loading`` or
  ``unloading`` (after the timeout effect has read it). hsm runs a state's exit before the transition's
  effects, so it is never cleared on exit. It is ``None`` in every other state.
* ``_worker``: the single worker thread. Created by ``start``; shut down by ``stop`` after its last job (the
  composition root that started the model stops it, PY-ASYNC-001).

Telemetry (cardinality, METRICS > SPANS > Logs; every value comes from code):

* ``bot.provider.llama_cpp.model.load.duration`` (histogram, s): ``bot.outcome`` {ok, failed, timeout} ×
  ``bot.llama_cpp.accelerator`` {metal, cuda, cpu, unknown} (``unknown`` when no session loaded) ×
  ``bot.llama_cpp.purpose`` {embedding}: 5 series (ok pairs only with a known accelerator, failed and timeout
  only with ``unknown``).
* ``bot.provider.llama_cpp.model.resident`` (up-down counter): purpose × accelerator {metal, cuda, cpu}: 3.
* ``bot.provider.llama_cpp.run.duration`` (histogram, s): purpose × outcome {ok, failed} × accelerator: 6.
* ``bot.provider.llama_cpp.stall.count`` (counter): purpose: 1.
* span ``bot.provider.llama_cpp.load`` (component ``llama_cpp.model``, stage ``load``) around each load.
* logs: INFO once per load (native dimensions, accelerator, vision); WARNING on a stall and on entering
  ``abandoning`` or ``releasing``. Never paths, file names, text, vectors or request ids.
"""

from __future__ import annotations

import abc
import asyncio
import collections.abc
import concurrent.futures
import dataclasses
import datetime
import logging
import time
import typing
import uuid

import hsm
import pydantic
from opentelemetry import metrics
from pydantic.json_schema import SkipJsonSchema

# The provider's ``mosfet`` stub hides the core package's attributes from the type checker, so core modules are
# imported by their full names (and ``define``, whose module name the package rebinds to the function, directly).
import mosfet.lifecycle as lifecycle
import mosfet.scope as scope
from mosfet.abilities import embedding
from mosfet.define import define
from mosfet.protocols import attachment
from mosfet.telemetry import span
from mosfet.telemetry.hsm import Traced, deliver

from .runtime import EmbeddingSession

_LOG = logging.getLogger(__name__)
_SCOPE: typing.Final = "mosfet.providers.llama_cpp"
_METER = metrics.get_meter(_SCOPE)
_LOAD_DURATION = _METER.create_histogram(
    "bot.provider.llama_cpp.model.load.duration",
    unit="s",
    description="Duration of each model load, by outcome, accelerator, and purpose.",
)
_RESIDENT = _METER.create_up_down_counter(
    "bot.provider.llama_cpp.model.resident",
    unit="{model}",
    description="Models resident on a device, by purpose and accelerator.",
)
_RUN_DURATION = _METER.create_histogram(
    "bot.provider.llama_cpp.run.duration",
    unit="s",
    description="Duration of each native run, by purpose, outcome, and accelerator.",
)
_STALLS = _METER.create_counter(
    "bot.provider.llama_cpp.stall.count",
    unit="{stall}",
    description="Native runs that outlived the model's stall timeout, by purpose.",
)

_UNKNOWN_ACCELERATOR: typing.Final = "unknown"
_SHARED = "llama_cpp.Model is shared; the composition root starts it in a long-lived scope"
_NOT_STARTED = "llama_cpp.Model is not started; the composition root owns its lifetime"

_JobKind: typing.TypeAlias = typing.Literal["open", "run", "close"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Job:
    """The one native job occupying the worker: its correlation id, what it does, and its future."""

    id: str
    kind: _JobKind
    future: concurrent.futures.Future[object]
    started: float
    """``time.monotonic()`` at submission; telemetry only, never behavior."""


class _Origin(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    id: str = pydantic.Field(description="Id of the request event this origin answers; replies carry it.")
    metadata: dict[str, object] = pydantic.Field(
        default_factory=dict,
        description="Telemetry propagation copied from the request onto its reply; never read by a guard.",
    )


class _AttachOrigin(_Origin):
    """The AttachEvent a load answers."""

    request: attachment.AttachData = pydantic.Field(description="The attach request: actor, reply_to, timeout.")


class _DetachOrigin(_Origin):
    """The DetachEvent an unload answers."""

    request: attachment.DetachData = pydantic.Field(description="The detach request: actor, reply_to, timeout.")


class _WorkOrigin(_Origin):
    """The work event a run answers."""

    source: str = pydantic.Field(description="Address of the actor that sent the work; the terminal goes there.")


class _JobData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    job_id: str = pydantic.Field(description="Id of the job that completed; correlated with the job in flight.")


class _LoadedData(_JobData):
    """A load completed; the session travels on the completion so whichever state receives it can keep or close
    it."""

    origin: _AttachOrigin
    session: SkipJsonSchema[pydantic.SkipValidation[EmbeddingSession]] = pydantic.Field(
        description="The loaded native session (runtime only)."
    )


class _LoadFailedData(_JobData):
    """A load failed; the attach it answers fails with ``initialization``."""

    origin: _AttachOrigin
    message: str = pydantic.Field(min_length=1, description="Why the model did not load.")


class _ClosedData(_JobData):
    """A close completed; ``origin`` is the detach it answers, or ``None`` when it answers no request."""

    origin: _DetachOrigin | None = None


class _RanData(_JobData):
    """A run completed. ``result`` is the subclass's completion payload (the generic-envelope exception of
    HSM-SCHEMA-001: the subclass's ``_completed_event`` schema narrows it)."""

    origin: _WorkOrigin
    result: pydantic.SerializeAsAny[pydantic.BaseModel]


class _RunFailedData(_JobData):
    """A run failed with a typed failure."""

    origin: _WorkOrigin
    failure: embedding.FailureData


_LoadedEvent = hsm.Event[_LoadedData](
    name="bot.provider.llama_cpp.model.loaded", kind=hsm.CompletionEventKind, schema=_LoadedData
)
_LoadFailedEvent = hsm.Event[_LoadFailedData](
    name="bot.provider.llama_cpp.model.load.failed", kind=hsm.ErrorEventKind, schema=_LoadFailedData
)
_ClosedEvent = hsm.Event[_ClosedData](
    name="bot.provider.llama_cpp.model.closed", kind=hsm.CompletionEventKind, schema=_ClosedData
)
_RanEvent = hsm.Event[_RanData](
    name="bot.provider.llama_cpp.run.completed", kind=hsm.CompletionEventKind, schema=_RanData
)
_RunFailedEvent = hsm.Event[_RunFailedData](
    name="bot.provider.llama_cpp.run.failed", kind=hsm.ErrorEventKind, schema=_RunFailedData
)


class Model(Traced, attachment.Attachment, abc.ABC):
    """One llama.cpp model resident on one device, shared by any number of attached owners.

    Abstract: a subclass declares its work event, its terminals, how to open a session and how to run one
    request, and builds its topology with ``define_model``. ``EmbeddingModel`` is the embedding actor.

    Lifetime: the composition root starts it in a long-lived, non-private scope and stops it after every owner
    has stopped. ``attach`` and ``detach`` only deliver; they never start the model.
    """

    _attachment_limit: typing.ClassVar[int | None] = None
    _purpose: typing.ClassVar[typing.Literal["embedding"]]
    """``bot.llama_cpp.purpose`` of this model's telemetry."""
    _completed_event: typing.ClassVar[hsm.Event[typing.Any]]
    """Terminal answering a run that completed; its data is the run's result."""
    _failed_event: typing.ClassVar[hsm.Event[embedding.FailureData]]
    """Terminal answering a run that failed, or work the model cannot take now (``unavailable``)."""
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str
    _stall_timeout: datetime.timedelta
    _worker: concurrent.futures.ThreadPoolExecutor | None
    _in_flight: _Job | None
    _session: EmbeddingSession | None
    _deadline: _AttachOrigin | _DetachOrigin | None

    def __init__(self, *, stall_timeout: datetime.timedelta) -> None:
        """``stall_timeout`` is how long one native run may take before the model reports it stalled.

        Raises:
            ValueError: If ``stall_timeout`` is not positive.
        """

        if stall_timeout <= datetime.timedelta(0):
            raise ValueError("llama_cpp.Model stall_timeout must be positive.")
        super().__init__()
        self._attachments = []
        # Protocol defaults, as Device sets them; no transition of this machine reads them.
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self._stall_timeout = stall_timeout
        self._worker = None
        self._in_flight = None
        self._session = None
        self._deadline = None

    @abc.abstractmethod
    def _open(self) -> EmbeddingSession:
        """Load the session; blocking, on the worker thread. Reads only constructor configuration.

        Raises:
            Exception: Any failure fails the attach with ``initialization``.
        """

    @abc.abstractmethod
    def _run(self, session: EmbeddingSession, data: object) -> pydantic.BaseModel:
        """Run one request's data on ``session``; blocking, on the worker thread.

        Raises:
            embedding.Error: A typed failure; any other exception is ``embed_failed``.
        """

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        """Start the model in the root's long-lived scope and give it its worker thread.

        Raises:
            RuntimeError: If ``ctx`` is a private scope (an owner's attachment group auto-starting it), so the
                owner's attach fails with ``initialization`` instead of tying the shared model to that owner.
        """

        if scope.is_private(ctx):
            raise RuntimeError(_SHARED)
        if self._worker is None:
            self._worker = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama_cpp")
        return await super().start(ctx, data)

    @typing.override
    async def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> bool:
        """Deliver ``event``; a model that is not started answers ``AttachFailed(dispatch)`` and returns False."""

        if lifecycle.is_started(self):
            return await deliver(self, ctx, event)
        await Model._refuse(ctx, event, attachment.AttachFailedEvent)
        return False

    @typing.override
    async def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> bool:
        """Deliver ``event``; a model that is not started answers ``DetachFailed(dispatch)`` and returns False."""

        if lifecycle.is_started(self):
            return await deliver(self, ctx, event)
        await Model._refuse(ctx, event, attachment.DetachFailedEvent)
        return False

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop the machine, close any session exactly once, and let the worker thread exit; idempotent.

        Called by the composition root after every owner has stopped. Completions that arrive after the
        machine stopped are dropped, so the session is closed here: behind a run in flight, after a load in
        flight (taking the session that load returns), and never a second time while a close is in flight.
        The worker runs jobs in order, so the close is its last job, and the thread exits after it.
        """

        await hsm.Instance.stop(self, ctx)
        # A stopped model holds no relationship: owners that never detached are released with it.
        self._attachments.clear()
        worker = self._worker
        if worker is None:
            return
        job, session = self._in_flight, self._session
        self._worker = None
        self._in_flight = None
        self._session = None
        self._deadline = None
        if session is not None:
            _RESIDENT.add(-1, Model._attributes(self, session))
        if job is not None and job.kind == "open":
            _ = worker.submit(Model._close_opened, job.future)
        elif session is not None and (job is None or job.kind != "close"):
            _ = worker.submit(Model._close_quietly, session)
        worker.shutdown(wait=False)

    @staticmethod
    async def _refuse(
        ctx: hsm.Context,
        event: hsm.Event[typing.Any],
        failed_event: hsm.Event[attachment.FailedData],
    ) -> None:
        data = event.data
        if not isinstance(data, (attachment.AttachData, attachment.DetachData)):
            return
        reply_to = data.actor if data.reply_to is None else data.reply_to
        _ = await deliver(
            reply_to,
            ctx,
            dataclasses.replace(
                failed_event.with_data(
                    attachment.FailedData(actor=data.actor, kind=attachment.FailureKind.DISPATCH, message=_NOT_STARTED)
                ),
                id=event.id,
                source=event.target,
                target=attachment.Attachment._actor_id(reply_to),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _close_opened(future: concurrent.futures.Future[object]) -> None:
        """On the worker, after the load ``future`` finished: close the session it returned, if any."""

        if future.cancelled() or future.exception() is not None:
            return
        session = typing.cast(EmbeddingSession, future.result())
        Model._close_quietly(session)

    @staticmethod
    def _close_quietly(session: EmbeddingSession) -> None:
        try:
            session.close()
        except Exception as error:
            _LOG.warning("llama_cpp session close failed kind=%s", span.failure_kind(error, type(error).__name__))

    @staticmethod
    def _attributes(instance: Model, session: EmbeddingSession | None) -> dict[str, str]:
        accelerator = _UNKNOWN_ACCELERATOR if session is None else session.accelerator.value
        return {"bot.llama_cpp.purpose": type(instance)._purpose, "bot.llama_cpp.accelerator": accelerator}

    # -- worker jobs -------------------------------------------------------------------------------------------

    @staticmethod
    def _submit(
        instance: Model,
        kind: _JobKind,
        work: collections.abc.Callable[[], object],
        complete: collections.abc.Callable[[str, concurrent.futures.Future[object]], hsm.Event[typing.Any]],
    ) -> None:
        """Start ``work`` on the worker; its completion event comes back to this machine on the loop."""

        worker = instance._worker
        assert worker is not None, "a started model has a worker"
        loop = asyncio.get_running_loop()
        job_id = uuid.uuid4().hex

        def done(future: concurrent.futures.Future[object]) -> None:
            event = complete(job_id, future)
            try:
                _ = loop.call_soon_threadsafe(Model._deliver_completion, instance, event)
            except RuntimeError:
                # The loop closed: the process is exiting and nothing is left to answer.
                return

        future = worker.submit(span.bind(work))
        instance._in_flight = _Job(id=job_id, kind=kind, future=future, started=time.monotonic())
        future.add_done_callback(done)

    @staticmethod
    def _deliver_completion(instance: Model, event: hsm.Event[typing.Any]) -> None:
        # The machine's own context: the completion outlives the state or activity that started the job, and a
        # stopped machine drops it.
        _ = deliver(instance, instance.context(), event)

    @staticmethod
    def _submit_open(instance: Model, origin: _AttachOrigin) -> None:
        def work() -> object:
            with span.operation(
                "bot.provider.llama_cpp.load",
                scope=_SCOPE,
                component="llama_cpp.model",
                stage="load",
                attributes={"bot.llama_cpp.purpose": type(instance)._purpose},
            ) as active:
                session = instance._open()
                active.set_attribute("bot.llama_cpp.accelerator", session.accelerator.value)
                return session

        def complete(job_id: str, future: concurrent.futures.Future[object]) -> hsm.Event[typing.Any]:
            error = future.exception()
            if error is None:
                session = typing.cast(EmbeddingSession, future.result())
                return _LoadedEvent.with_data(_LoadedData(job_id=job_id, origin=origin, session=session))
            message = str(error) or type(error).__name__
            return _LoadFailedEvent.with_data(_LoadFailedData(job_id=job_id, origin=origin, message=message))

        Model._submit(instance, "open", work, complete)

    @staticmethod
    def _submit_close(instance: Model, session: EmbeddingSession, origin: _DetachOrigin | None) -> None:
        def work() -> object:
            Model._close_quietly(session)
            return None

        def complete(job_id: str, future: concurrent.futures.Future[object]) -> hsm.Event[typing.Any]:
            del future
            return _ClosedEvent.with_data(_ClosedData(job_id=job_id, origin=origin))

        Model._submit(instance, "close", work, complete)

    # -- guards ------------------------------------------------------------------------------------------------

    @staticmethod
    def _is_in_flight(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        job = instance._in_flight
        return isinstance(data, _JobData) and job is not None and data.job_id == job.id

    @staticmethod
    def _deadline_delay(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        deadline = instance._deadline
        assert deadline is not None, "loading and unloading are entered with their deadline armed"
        return deadline.request.timeout

    @staticmethod
    def _stall_delay(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._stall_timeout

    # -- replies -----------------------------------------------------------------------------------------------

    @staticmethod
    def _reply_attachment(
        ctx: hsm.Context,
        instance: Model,
        origin: _AttachOrigin | _DetachOrigin,
        data: attachment.AttachCompleteData | attachment.DetachedData | attachment.FailedData,
        reply_event: hsm.Event[typing.Any],
    ) -> None:
        request = origin.request
        target = request.actor if request.reply_to is None else request.reply_to
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                reply_event.with_data(data),
                id=origin.id,
                source=hsm.id(instance),
                target=attachment.Attachment._actor_id(target),
                metadata=dict(origin.metadata),
            ),
        )

    @staticmethod
    def _reply_work(
        ctx: hsm.Context,
        instance: Model,
        request_id: str,
        source: str,
        metadata: collections.abc.Mapping[str, object],
        terminal: hsm.Event[typing.Any],
    ) -> None:
        """Answer one work request at the address it came from (the run_terminal_operation reply actor)."""

        _ = hsm.dispatch_to(
            ctx,
            dataclasses.replace(
                terminal, id=request_id, source=hsm.id(instance), target=source, metadata=dict(metadata)
            ),
            source,
        )

    @staticmethod
    def _reply_unavailable(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        failure = embedding.FailureData(
            kind=embedding.FailureKind.UNAVAILABLE, message="llama.cpp model is not resident or is stalled"
        )
        Model._reply_work(
            ctx, instance, event.id, event.source, event.metadata, type(instance)._failed_event.with_data(failure)
        )

    # -- effects -----------------------------------------------------------------------------------------------

    @staticmethod
    def _arm_attach(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, attachment.AttachData)
        origin = _AttachOrigin(request=data, id=event.id, metadata=dict(event.metadata))
        instance._deadline = origin
        Model._submit_open(instance, origin)

    @staticmethod
    def _complete_load(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _LoadedData)
        job = instance._in_flight
        assert job is not None
        session = data.session
        attributes = Model._attributes(instance, session)
        _LOAD_DURATION.record(time.monotonic() - job.started, {"bot.outcome": span.Ok, **attributes})
        _RESIDENT.add(1, attributes)
        _LOG.info(
            "llama_cpp model loaded purpose=%s native_dimensions=%d accelerator=%s vision=%s",
            type(instance)._purpose,
            session.native_dimensions,
            session.accelerator.value,
            session.vision,
        )
        instance._in_flight = None
        instance._session = session
        instance._attachments.append(data.origin.request.actor)
        Model._reply_attachment(
            ctx,
            instance,
            data.origin,
            attachment.AttachCompleteData(actor=data.origin.request.actor, created=True),
            attachment.AttachCompleteEvent,
        )
        instance._deadline = None

    @staticmethod
    def _fail_load(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _LoadFailedData)
        job = instance._in_flight
        assert job is not None
        _LOAD_DURATION.record(
            time.monotonic() - job.started, {"bot.outcome": span.Failed, **Model._attributes(instance, None)}
        )
        instance._in_flight = None
        Model._reply_attachment(
            ctx,
            instance,
            data.origin,
            attachment.FailedData(
                actor=data.origin.request.actor, kind=attachment.FailureKind.INITIALIZATION, message=data.message
            ),
            attachment.AttachFailedEvent,
        )
        instance._deadline = None

    @staticmethod
    def _time_out_load(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        del event
        origin = instance._deadline
        assert isinstance(origin, _AttachOrigin), "loading is entered with an attach deadline"
        job = instance._in_flight
        assert job is not None
        _LOAD_DURATION.record(
            time.monotonic() - job.started, {"bot.outcome": "timeout", **Model._attributes(instance, None)}
        )
        timeout = origin.request.timeout.total_seconds()
        _LOG.warning("llama_cpp model load outlived its %gs attach timeout; abandoning it", timeout)
        # Nothing is popped: the owner is appended only when a load completes.
        Model._reply_attachment(
            ctx,
            instance,
            origin,
            attachment.FailedData(
                actor=origin.request.actor,
                kind=attachment.FailureKind.TIMEOUT,
                message=f"{type(instance).__name__} did not load within {timeout:g} seconds.",
            ),
            attachment.AttachFailedEvent,
        )
        instance._deadline = None

    @staticmethod
    def _release_abandoned(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _LoadedData)
        _LOG.warning("llama_cpp model loaded after its attach timed out; releasing it")
        Model._submit_close(instance, data.session, None)

    @staticmethod
    def _settle(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        """A completion that answers no request: the job is done."""

        del ctx, event
        instance._in_flight = None

    @staticmethod
    def _arm_detach(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, attachment.DetachData)
        session = instance._session
        assert session is not None, "a loaded model holds its session"
        origin = _DetachOrigin(request=data, id=event.id, metadata=dict(event.metadata))
        instance._deadline = origin
        Model._submit_close(instance, session, origin)

    @staticmethod
    def _drop_session(instance: Model) -> None:
        session = instance._session
        if session is not None:
            _RESIDENT.add(-1, Model._attributes(instance, session))
        instance._session = None
        instance._in_flight = None

    @staticmethod
    def _complete_unload(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ClosedData) and data.origin is not None
        Model._drop_session(instance)
        Model._reply_attachment(
            ctx,
            instance,
            data.origin,
            attachment.DetachedData(actor=data.origin.request.actor, removed=True),
            attachment.DetachedEvent,
        )
        instance._deadline = None

    @staticmethod
    def _time_out_unload(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        del event
        origin = instance._deadline
        assert isinstance(origin, _DetachOrigin), "unloading is entered with a detach deadline"
        timeout = origin.request.timeout.total_seconds()
        _LOG.warning("llama_cpp model close outlived its %gs detach timeout; releasing it", timeout)
        # Nothing is popped: the owner was already removed when the detach was accepted.
        Model._reply_attachment(
            ctx,
            instance,
            origin,
            attachment.FailedData(
                actor=origin.request.actor,
                kind=attachment.FailureKind.TIMEOUT,
                message=f"{type(instance).__name__} did not unload within {timeout:g} seconds.",
            ),
            attachment.DetachFailedEvent,
        )
        instance._deadline = None

    @staticmethod
    def _settle_release(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Model._drop_session(instance)

    @staticmethod
    def _submit_run(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        del ctx
        session = instance._session
        assert session is not None, "a loaded model holds its session"
        origin = _WorkOrigin(id=event.id, source=event.source, metadata=dict(event.metadata))
        data = event.data

        def work() -> object:
            return instance._run(session, data)

        def complete(job_id: str, future: concurrent.futures.Future[object]) -> hsm.Event[typing.Any]:
            error = future.exception()
            if error is None:
                result = typing.cast(pydantic.BaseModel, future.result())
                return _RanEvent.with_data(_RanData(job_id=job_id, origin=origin, result=result))
            if isinstance(error, embedding.Error):
                failure = embedding.FailureData(kind=error.kind, message=str(error), item_index=error.item_index)
            else:
                failure = embedding.FailureData(
                    kind=embedding.FailureKind.EMBED_FAILED, message=f"{type(error).__name__}: {error}"
                )
            return _RunFailedEvent.with_data(_RunFailedData(job_id=job_id, origin=origin, failure=failure))

        Model._submit(instance, "run", work, complete)

    @staticmethod
    def _finish_run(instance: Model, outcome: str) -> None:
        job = instance._in_flight
        assert job is not None
        _RUN_DURATION.record(
            time.monotonic() - job.started, {"bot.outcome": outcome, **Model._attributes(instance, instance._session)}
        )
        instance._in_flight = None

    @staticmethod
    def _reply_ran(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _RanData)
        Model._finish_run(instance, span.Ok)
        origin = data.origin
        terminal = type(instance)._completed_event.with_data(data.result)
        Model._reply_work(ctx, instance, origin.id, origin.source, origin.metadata, terminal)

    @staticmethod
    def _reply_run_failed(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _RunFailedData)
        Model._finish_run(instance, span.Failed)
        origin = data.origin
        terminal = type(instance)._failed_event.with_data(data.failure)
        Model._reply_work(ctx, instance, origin.id, origin.source, origin.metadata, terminal)

    @staticmethod
    def _record_stall(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _STALLS.add(1, {"bot.llama_cpp.purpose": type(instance)._purpose})
        _LOG.warning(
            "llama_cpp run outlived the %gs stall timeout; new work fails until it returns",
            instance._stall_timeout.total_seconds(),
        )

    # -- topology ----------------------------------------------------------------------------------------------

    @staticmethod
    def define_model(name: str, *, work: tuple[hsm.Event[typing.Any], ...]) -> hsm.Model:
        """Build the shared-model topology for a subclass named ``name`` whose requests are ``work``."""

        def path(*states: str) -> str:
            return "/" + "/".join((name, *states))

        return define(
            name,
            hsm.initial(hsm.target(path("detached"))),
            hsm.state(
                "detached",
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._can_attach),
                    hsm.effect(Model._arm_attach),
                    hsm.target(path("loading")),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._is_attach_request),
                    hsm.effect(attachment.Attachment._dispatch_attach_failed),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._is_detach_request),
                    hsm.effect(attachment.Attachment._dispatch_detach_complete_absent),
                ),
                hsm.transition(hsm.on(*work), hsm.effect(Model._reply_unavailable)),
            ),
            hsm.state(
                "loading",
                hsm.defer(attachment.AttachEvent, attachment.DetachEvent, *work),
                hsm.transition(
                    hsm.on(_LoadedEvent),
                    hsm.guard(Model._is_in_flight),
                    hsm.effect(Model._complete_load),
                    hsm.target(path("loaded")),
                ),
                hsm.transition(
                    hsm.on(_LoadFailedEvent),
                    hsm.guard(Model._is_in_flight),
                    hsm.effect(Model._fail_load),
                    hsm.target(path("detached")),
                ),
                hsm.transition(
                    hsm.after(Model._deadline_delay),
                    hsm.effect(Model._time_out_load),
                    hsm.target(path("abandoning")),
                ),
            ),
            hsm.state(
                "abandoning",
                hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
                hsm.transition(hsm.on(*work), hsm.effect(Model._reply_unavailable)),
                hsm.transition(
                    hsm.on(_LoadedEvent),
                    hsm.guard(Model._is_in_flight),
                    hsm.effect(Model._release_abandoned),
                    hsm.target(path("releasing")),
                ),
                hsm.transition(
                    hsm.on(_LoadFailedEvent),
                    hsm.guard(Model._is_in_flight),
                    hsm.effect(Model._settle),
                    hsm.target(path("detached")),
                ),
            ),
            hsm.state(
                "loaded",
                hsm.initial(hsm.target(path("loaded", "idle"))),
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._can_attach),
                    hsm.effect(attachment.Attachment._attach, attachment.Attachment._dispatch_attach_complete_created),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._is_attached),
                    hsm.effect(attachment.Attachment._dispatch_attach_complete_existing),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._is_attach_request),
                    hsm.effect(attachment.Attachment._dispatch_attach_failed),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._detach_would_leave_attachments),
                    hsm.effect(attachment.Attachment._detach, attachment.Attachment._dispatch_detach_complete_removed),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(Model._is_absent_detach),
                    hsm.effect(attachment.Attachment._dispatch_detach_complete_absent),
                ),
                hsm.state(
                    "idle",
                    hsm.transition(
                        hsm.on(attachment.DetachEvent),
                        hsm.guard(attachment.Attachment._detach_last_attachment),
                        hsm.effect(attachment.Attachment._detach, Model._arm_detach),
                        hsm.target(path("unloading")),
                    ),
                    hsm.transition(hsm.on(*work), hsm.target(path("loaded", "running"))),
                ),
                hsm.state(
                    "running",
                    hsm.entry(Model._submit_run),
                    hsm.defer(attachment.DetachEvent, *work),
                    hsm.transition(
                        hsm.on(_RanEvent),
                        hsm.guard(Model._is_in_flight),
                        hsm.effect(Model._reply_ran),
                        hsm.target(path("loaded", "idle")),
                    ),
                    hsm.transition(
                        hsm.on(_RunFailedEvent),
                        hsm.guard(Model._is_in_flight),
                        hsm.effect(Model._reply_run_failed),
                        hsm.target(path("loaded", "idle")),
                    ),
                    hsm.transition(
                        hsm.after(Model._stall_delay),
                        hsm.effect(Model._record_stall),
                        hsm.target(path("loaded", "stalled")),
                    ),
                ),
                hsm.state(
                    "stalled",
                    hsm.defer(attachment.DetachEvent),
                    hsm.transition(hsm.on(*work), hsm.effect(Model._reply_unavailable)),
                    hsm.transition(
                        hsm.on(_RanEvent),
                        hsm.guard(Model._is_in_flight),
                        hsm.effect(Model._reply_ran),
                        hsm.target(path("loaded", "idle")),
                    ),
                    hsm.transition(
                        hsm.on(_RunFailedEvent),
                        hsm.guard(Model._is_in_flight),
                        hsm.effect(Model._reply_run_failed),
                        hsm.target(path("loaded", "idle")),
                    ),
                ),
            ),
            hsm.state(
                "unloading",
                hsm.defer(attachment.AttachEvent),
                hsm.transition(hsm.on(*work), hsm.effect(Model._reply_unavailable)),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._is_detach_request),
                    hsm.effect(attachment.Attachment._dispatch_detach_complete_absent),
                ),
                hsm.transition(
                    hsm.on(_ClosedEvent),
                    hsm.guard(Model._is_in_flight),
                    hsm.effect(Model._complete_unload),
                    hsm.target(path("detached")),
                ),
                hsm.transition(
                    hsm.after(Model._deadline_delay),
                    hsm.effect(Model._time_out_unload),
                    hsm.target(path("releasing")),
                ),
            ),
            hsm.state(
                "releasing",
                hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
                hsm.transition(hsm.on(*work), hsm.effect(Model._reply_unavailable)),
                hsm.transition(
                    hsm.on(_ClosedEvent),
                    hsm.guard(Model._is_in_flight),
                    hsm.effect(Model._settle_release),
                    hsm.target(path("detached")),
                ),
            ),
        )

    @staticmethod
    def _is_absent_detach(ctx: hsm.Context, instance: Model, event: hsm.Event[typing.Any]) -> bool:
        return attachment.Attachment._is_detach_request(ctx, instance, event) and not (
            attachment.Attachment._is_attached(ctx, instance, event)
        )


__all__ = ["Model"]
