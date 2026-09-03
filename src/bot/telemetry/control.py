"""Subscribe to dashboard commands and dispatch them into a live environment."""

from __future__ import annotations

import asyncio
import collections.abc
import inspect
import json
import logging
import threading
import typing
from urllib.parse import urlsplit

import grpc
import hsm
from opentelemetry import metrics
from opentelemetry.trace import Span

from bot.telemetry import span
from bot.telemetry.configure import otlp_endpoint

_LOG = logging.getLogger(__name__)
_COMPONENT = "telemetry.control"
_SCOPE = "bot.telemetry.control"
_SUBSCRIBE_PATH = "/bot.control.v1.Control/Subscribe"

_METER = metrics.get_meter(_SCOPE)
_REJECTIONS = _METER.create_counter(
    "bot.telemetry.control.command.rejected.count",
    unit="{command}",
    description="Count of dashboard control commands dropped before dispatch.",
)

_LOCK = threading.Lock()
_SUBSCRIPTIONS: dict[int, _Subscription] = {}


class _Subscription:
    environment: hsm.Context
    refs: int
    target: str
    loop: asyncio.AbstractEventLoop | None
    stop: threading.Event
    thread: threading.Thread | None
    channel: grpc.Channel | None

    def __init__(
        self,
        environment: hsm.Context,
        target: str,
        loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        self.environment = environment
        self.refs = 1
        self.target = target
        self.loop = loop
        self.stop = threading.Event()
        self.thread = None
        self.channel = None


def grpc_target(endpoint: str) -> str:
    """Return ``host:port`` for a control Subscribe, ignoring a leftover HTTP path."""

    parsed = urlsplit(endpoint if "://" in endpoint else f"http://{endpoint}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 4317
    return f"{host}:{port}"


def _decode_varint(raw: bytes, index: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while index < len(raw):
        octet = raw[index]
        index += 1
        value |= (octet & 0x7F) << shift
        if octet & 0x80 == 0:
            return value, index
        shift += 7
        if shift > 70:
            break
    message = "truncated control protobuf varint"
    raise ValueError(message)


def _decode_command(raw: bytes) -> tuple[str, str]:
    event_name = ""
    data_json = ""
    index = 0
    while index < len(raw):
        key, index = _decode_varint(raw, index)
        field = key >> 3
        wire = key & 7
        if wire != 2:
            message = "unexpected control command wire type"
            raise ValueError(message)
        length, index = _decode_varint(raw, index)
        end = index + length
        if end > len(raw):
            message = "truncated control command field"
            raise ValueError(message)
        text = raw[index:end].decode("utf-8")
        index = end
        if field == 1:
            event_name = text
        elif field == 2:
            data_json = text
    return event_name, data_json


def _iter_commands(subscription: _Subscription) -> collections.abc.Iterator[tuple[str, str]]:
    """Yield ``(event_name, data_json)`` from ``Control.Subscribe``. Tests may replace this."""

    channel = grpc.insecure_channel(subscription.target)
    subscription.channel = channel
    try:
        subscribe = typing.cast(
            collections.abc.Callable[[object], collections.abc.Iterator[tuple[str, str]]],
            channel.unary_stream(
                _SUBSCRIBE_PATH,
                request_serializer=lambda _: b"",
                response_deserializer=_decode_command,
            ),
        )
        for event_name, data_json in subscribe(None):
            if subscription.stop.is_set():
                return
            yield (event_name, data_json)
    finally:
        channel.close()
        subscription.channel = None


def _parse_data(data_json: str) -> object | None:
    trimmed = data_json.strip()
    if trimmed == "":
        return None
    try:
        parsed = typing.cast(object, json.loads(trimmed))
    except json.JSONDecodeError:
        message = "command data_json is not JSON"
        raise ValueError(message) from None
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        message = "command data_json must be a JSON object or null"
        raise ValueError(message)
    payload: dict[str, object] = {}
    for key, value in typing.cast(collections.abc.Mapping[object, object], parsed).items():
        if isinstance(key, str):
            payload[key] = value
    return payload


def _deliver(environment: hsm.Context, event: hsm.Event[object], loop: asyncio.AbstractEventLoop | None) -> None:
    async def emit() -> None:
        result = hsm.dispatch_all(environment, event)
        if inspect.isawaitable(result):
            await result

    if loop is not None and loop.is_running():
        _ = asyncio.run_coroutine_threadsafe(emit(), loop)
        return
    result = hsm.dispatch_all(environment, event)
    if inspect.isawaitable(result):
        asyncio.run(emit())


def _run(subscription: _Subscription) -> None:
    with span.operation(
        "bot.telemetry.control",
        scope=_SCOPE,
        component=_COMPONENT,
        stage="subscribe",
    ) as active:
        try:
            for event_name, data_json in _iter_commands(subscription):
                if subscription.stop.is_set():
                    break
                _accept(subscription, event_name, data_json)
        except Exception as error:
            kind = span.failure_kind(error, "subscribe_failed")
            span.record_failure(active, kind)
            _LOG.warning(
                "control subscribe ended kind=%s endpoint=%s outcome=failed",
                kind,
                subscription.target,
            )


def _reject(active: Span, *, reason: str) -> None:
    """Record one dropped control command as metric + span + log (never span-only).

    A rejection carries no dispatchable event — an empty name names nothing and
    a non-object payload fits no schema — so the counter is the load-bearing
    signal and the log line is the human-readable one. ``reason`` is a closed
    vocabulary (``empty_name`` / ``invalid_data``); raw names and payloads are
    never logged or attributed.
    """

    span.record_failure(active, "invalid_response")
    try:
        _REJECTIONS.add(1, attributes={"bot.component.name": _COMPONENT, "bot.failure.kind": "invalid_response"})
    except Exception:
        pass
    _LOG.warning("control command rejected reason=%s outcome=rejected", reason)


def _accept(subscription: _Subscription, event_name: str, data_json: str) -> None:
    with span.operation(
        "bot.telemetry.control",
        scope=_SCOPE,
        component=_COMPONENT,
        stage="dispatch",
    ) as active:
        name = event_name.strip()
        if name == "":
            _reject(active, reason="empty_name")
            return
        try:
            data = _parse_data(data_json)
        except ValueError:
            _reject(active, reason="invalid_data")
            return
        event = hsm.Event[object](name=name, data=data)
        _deliver(subscription.environment, event, subscription.loop)
        active.set_attribute("bot.outcome", "accepted")


def listen(environment: hsm.Context) -> None:
    """Open one ``Control.Subscribe`` per environment when an OTLP endpoint is set.

    Idempotent for a given environment. Missing endpoint or grpc is a no-op.
    """

    endpoint = otlp_endpoint()
    if endpoint is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    key = id(environment)
    with _LOCK:
        existing = _SUBSCRIPTIONS.get(key)
        if existing is not None:
            existing.refs += 1
            return
        subscription = _Subscription(environment, grpc_target(endpoint), loop)
        _SUBSCRIPTIONS[key] = subscription
        worker = threading.Thread(
            target=span.bind(_run),
            args=(subscription,),
            name="bot-telemetry-control",
            daemon=True,
        )
        subscription.thread = worker
        worker.start()


def unlisten(environment: hsm.Context) -> None:
    """Drop one listen ref for ``environment`` and stop the stream at zero."""

    key = id(environment)
    with _LOCK:
        existing = _SUBSCRIPTIONS.get(key)
        if existing is None:
            return
        existing.refs -= 1
        if existing.refs > 0:
            return
        del _SUBSCRIPTIONS[key]
    existing.stop.set()
    channel = existing.channel
    if channel is not None:
        channel.close()
    thread = existing.thread
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=1.0)


def reset() -> None:
    """Stop every subscribe. Safe to call from ``configure.reset``."""

    with _LOCK:
        subscriptions = list(_SUBSCRIPTIONS.values())
        _SUBSCRIPTIONS.clear()
    for subscription in subscriptions:
        subscription.stop.set()
        channel = subscription.channel
        if channel is not None:
            channel.close()
        thread = subscription.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)


__all__ = ["grpc_target", "listen", "reset", "unlisten"]
