"""HSM start namespace for stateforward.mosfet.

``started`` and ``start`` wrap the HSM constructors and best-effort register an
active live model in the web studio from the public snapshot. Failures never
raise out of start.
"""

from __future__ import annotations

import logging
import typing

import hsm
from opentelemetry.trace import Span

from mosfet import lifecycle
from mosfet.define import Owner, owner_qualified_name, publish, studio_live_url, topology
from mosfet import address
from mosfet import scope
from mosfet.environment import Environment
from mosfet.telemetry import span
from mosfet.telemetry.configure import otlp_endpoint

_LOG = logging.getLogger(__name__)
_PUBLISH_SKIPPED = "skipped"
_PUBLISH_FAILED = "failed"
_TOPOLOGY_ATTR = "bot.start.topology"
_LIVE_ATTR = "bot.start.live"

TInstance = typing.TypeVar("TInstance", bound=hsm.Instance)


@typing.runtime_checkable
class _HasModel(typing.Protocol):
    """Typed surface for reading an instance's published lifecycle model without ``getattr``."""

    model: hsm.Model | None


class Live(typing.TypedDict):
    name: str
    component: str
    state: str
    live: bool
    owner: typing.NotRequired[str | None]


def live_payload(
    instance: hsm.Instance,
    *,
    owner: Owner | None = None,
    clear_owner: bool = False,
    model: hsm.Model | None = None,
) -> Live:
    """Return the low-cardinality live registration body.

    Schema::

        {"name": "/Demo", "component": "Demo", "state": "/Demo/idle", "live": true}

    Name and state come from the public snapshot. Component is the instance
    class name. No instance ids, payloads, or other high-cardinality data.
    """

    snapshot = lifecycle.snapshot_if_started(instance)
    if snapshot is None and model is not None:
        payload: Live = {
            "name": model.qualified_name,
            "component": type(instance).__name__,
            "state": "",
            "live": False,
        }
    else:
        if snapshot is None:
            snapshot = instance.take_snapshot()
        payload = {
            "name": snapshot.QualifiedName,
            "component": type(instance).__name__,
            "state": snapshot.State,
            "live": True,
        }
    if clear_owner:
        payload["owner"] = None
    else:
        owner_name = owner_qualified_name(owner)
        if owner_name is not None:
            payload["owner"] = owner_name
    return payload


def bound_model(instance: hsm.Instance) -> hsm.Model | None:
    """Return a public ``model`` attribute when it is an ``hsm.Model``."""

    if isinstance(instance, _HasModel):
        model = instance.model
        if isinstance(model, hsm.Model):
            return model
    return None


def publish_live(payload: Live) -> str:
    """Publish live state when an OTLP endpoint is configured.

    Returns ``skipped``, ``ok``, or ``failed``. Never raises.
    """

    endpoint = otlp_endpoint()
    if endpoint is None:
        return _PUBLISH_SKIPPED
    return publish(payload, studio_live_url(endpoint))


def _register(
    instance: hsm.Instance,
    model: hsm.Model | None,
    owner: Owner | None,
    ctx: hsm.Context | None,
    clear_owner: bool = False,
) -> tuple[str, str]:
    """Publish topology when known, register the actor's address, mark live. Never raises."""
    try:
        environment_root = owner is None and ctx is not None and Environment.from_context(ctx).contains(instance)
        publish_clear_owner = clear_owner or environment_root
        topology_outcome = (
            publish(topology(model, owner=owner, clear_owner=publish_clear_owner))
            if model is not None
            else _PUBLISH_SKIPPED
        )
        return topology_outcome, publish_live(
            live_payload(instance, owner=owner, clear_owner=publish_clear_owner, model=model)
        )
    except Exception as error:
        _LOG.warning("live register failed kind=%s", span.failure_kind(error, type(error).__name__))
        return _PUBLISH_FAILED, _PUBLISH_FAILED


def _register_address(instance: hsm.Instance, ctx: hsm.Context | None) -> None:
    """Register the started instance's runtime address. Never raises.

    The path is the environment's identity plus the actor's: ``/<env-id>/<actor-id>``.
    Only actors that are actually under an Environment scope register. A bare
    ``hsm.Context`` has no environment and therefore no addressable environment id;
    registering one would fabricate an identity the actor does not belong to.
    Private scopes inherit their parent environment's environment segment — visibility
    is decided at dispatch, not by hiding the address — so a private actor stays
    addressable by explicit dispatch and invisible to environment broadcast.
    """

    try:
        if ctx is None:
            return
        environment = Environment.reachable(ctx)
        if environment is None:
            return
        path = f"{environment.scope_path}/{hsm.id(instance)}"
        existing = address.resolve(path)
        if existing is not None and existing is not instance:
            # A stale registration must not shadow a restarted actor. Only a genuinely live
            # actor at the same path is an addressing bug; a stopped one is replaced so the
            # new actor becomes resolvable.
            if lifecycle.is_started(existing):
                raise RuntimeError(f"address {path} is already registered to a live instance.")
            address.unregister(path)
        address.register(path, instance)
        if scope.is_private(ctx):
            address.mark_private(path)
    except Exception as error:
        _LOG.warning("address register failed kind=%s", span.failure_kind(error, type(error).__name__))


def _record(active: Span, topology_outcome: str, live_outcome: str) -> None:
    active.set_attribute(_TOPOLOGY_ATTR, topology_outcome)
    active.set_attribute(_LIVE_ATTR, live_outcome)
    if topology_outcome == _PUBLISH_FAILED or live_outcome == _PUBLISH_FAILED:
        span.record_failure(active, "publish_failed")


async def started(
    ctx: hsm.Context | None,
    instance: TInstance,
    model: hsm.Model,
    config: hsm.Config | None = None,
    owner: Owner | None = None,
) -> TInstance:
    """Start a new instance through ``hsm.started`` and register it live."""

    with span.operation(
        "bot.started",
        scope="bot.start",
        component=type(instance).__name__,
        stage="start",
    ) as active:
        started_instance = await hsm.started(ctx, instance, model, config)
        _record(active, *_register(started_instance, model, owner, ctx))
        _register_address(started_instance, ctx)
        return started_instance


async def start(
    ctx: hsm.Context | None,
    instance: TInstance,
    data: object = None,
    *,
    owner: Owner | None = None,
) -> TInstance:
    """Start an already-bound instance through ``hsm.start`` and register it live."""

    with span.operation(
        "bot.start",
        scope="bot.start",
        component=type(instance).__name__,
        stage="start",
    ) as active:
        started_instance = await hsm.start(ctx, instance, data)
        _register_address(started_instance, ctx)
        _record(active, *_register(started_instance, bound_model(started_instance), owner, ctx))
        return started_instance


def register(
    instance: hsm.Instance,
    model: hsm.Model | None = None,
    *,
    owner: Owner | None = None,
    clear_owner: bool = False,
) -> tuple[str, str]:
    """Refresh topology/live registration for an already-started instance without restarting it."""

    resolved_model = model if model is not None else bound_model(instance)
    with span.operation(
        "bot.register",
        scope="bot.start",
        component=type(instance).__name__,
        stage="register",
    ) as active:
        _register_address(instance, instance.context())
        outcomes = _register(instance, resolved_model, owner, instance.context(), clear_owner)
        _record(active, *outcomes)
        return outcomes


__all__ = [
    "Live",
    "bound_model",
    "live_payload",
    "publish_live",
    "register",
    "start",
    "started",
]
