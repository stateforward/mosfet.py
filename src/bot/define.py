"""HSM construction namespace for stateforward.bot.

``define`` is the only hooked constructor: it calls ``hsm.define`` and best-effort
publishes the finalized topology to the web studio. Other names are pass-throughs
to ``hsm``.
"""

from __future__ import annotations

import collections.abc
import http.client
import json
import logging
import posixpath
import typing
import urllib.parse

import hsm

from bot.telemetry import span
from bot.telemetry.configure import otlp_endpoint


class State(typing.TypedDict):
    qualified_name: str
    parent: str
    initial: str


class Transition(typing.TypedDict):
    source: str
    target: str
    events: list[str]


class Topology(typing.TypedDict):
    name: str
    states: list[State]
    transitions: list[Transition]
    initial: str
    owner: typing.NotRequired[str | None]


Owner = hsm.Instance | str

_LOG = logging.getLogger(__name__)
_POST_TIMEOUT_SECONDS = 0.5
_STUDIO_PORT = 5173
_MODELS_PATH = "/v1/models"
_LIVE_PATH = "/v1/models/live"
_PUBLISH_ATTR = "bot.define.publish"
_PUBLISH_SKIPPED = "skipped"
_PUBLISH_OK = "ok"
_PUBLISH_FAILED = "failed"

activity = hsm.activity
after = hsm.after
choice = hsm.choice
defer = hsm.defer
effect = hsm.effect
entry = hsm.entry
exit = hsm.exit
final = hsm.final
guard = hsm.guard
initial = hsm.initial
observe = hsm.observe
on = hsm.on
source = hsm.source
state = hsm.state
target = hsm.target
transition = hsm.transition


def owner_qualified_name(owner: Owner | None) -> str | None:
    if owner is None:
        return None
    qualified_name = owner if isinstance(owner, str) else owner.take_snapshot().QualifiedName
    return qualified_name if qualified_name else None


def topology(
    model: hsm.Model,
    *,
    owner: Owner | None = None,
    clear_owner: bool = False,
) -> Topology:
    """Return the low-cardinality topology JSON published to the studio.

    Schema::

        {
          "name": "/Demo",
          "states": [
            {"qualified_name": "/Demo", "parent": "/", "initial": "/Demo/.initial"},
            {"qualified_name": "/Demo/idle", "parent": "/Demo", "initial": ""}
          ],
          "transitions": [
            {"source": "/Demo/idle", "target": "/Demo/run", "events": ["go"]}
          ],
          "initial": "/Demo/.initial"
        }

    States come from ``StateElement`` members. Transitions carry event name
    strings only. No payloads, instance ids, or other high-cardinality data.
    """

    states: list[State] = []
    transitions: list[Transition] = []
    for member in model.members.values():
        if isinstance(member, hsm.StateElement):
            qualified_name = member.qualified_name
            parent = "" if qualified_name in {"", "/"} else posixpath.dirname(qualified_name)
            states.append(
                {
                    "qualified_name": qualified_name,
                    "parent": parent,
                    "initial": member.initial,
                }
            )
            continue
        if isinstance(member, hsm.TransitionElement):
            events = [event_name for event_name in member.events if event_name]
            transitions.append(
                {
                    "source": member.source,
                    "target": member.target,
                    "events": events,
                }
            )
    payload: Topology = {
        "name": model.qualified_name,
        "states": states,
        "transitions": transitions,
        "initial": model.initial,
    }
    if clear_owner:
        payload["owner"] = None
    else:
        owner_name = owner_qualified_name(owner)
        if owner_name is not None:
            payload["owner"] = owner_name
    return payload


def studio_url(endpoint: str, path: str) -> str:
    """Derive ``http://<otlp-host>:5173<path>`` from an OTLP endpoint."""

    raw = endpoint if "://" in endpoint else f"http://{endpoint}"
    parsed = urllib.parse.urlparse(raw)
    host = parsed.hostname or "127.0.0.1"
    return f"http://{host}:{_STUDIO_PORT}{path}"


def studio_models_url(endpoint: str) -> str:
    """Derive ``http://<otlp-host>:5173/v1/models`` from an OTLP endpoint."""

    return studio_url(endpoint, _MODELS_PATH)


def studio_live_url(endpoint: str) -> str:
    """Derive ``http://<otlp-host>:5173/v1/models/live`` from an OTLP endpoint."""

    return studio_url(endpoint, _LIVE_PATH)


def post_model(payload: collections.abc.Mapping[str, object], url: str) -> None:
    """POST topology JSON to the studio. Callers treat failure as non-fatal."""

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if host is None:
        message = "model publish url missing host"
        raise ValueError(message)
    port = parsed.port if parsed.port is not None else _STUDIO_PORT
    path = parsed.path if parsed.path else _MODELS_PATH
    body = json.dumps(payload).encode("utf-8")
    connection = http.client.HTTPConnection(host, port, timeout=_POST_TIMEOUT_SECONDS)
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        _ = response.read()
    finally:
        connection.close()


def publish(payload: collections.abc.Mapping[str, object], url: str | None = None) -> str:
    """Publish topology or live state when an OTLP endpoint is configured.

    Returns ``skipped``, ``ok``, or ``failed``. Never raises.
    """

    endpoint = otlp_endpoint()
    if endpoint is None:
        return _PUBLISH_SKIPPED
    target = url if url is not None else studio_models_url(endpoint)
    try:
        post_model(payload, target)
    except (TimeoutError, OSError, ValueError) as error:
        _LOG.warning("model publish failed kind=%s", span.failure_kind(error, type(error).__name__))
        return _PUBLISH_FAILED
    except Exception as error:
        _LOG.warning("model publish failed kind=%s", span.failure_kind(error, type(error).__name__))
        return _PUBLISH_FAILED
    return _PUBLISH_OK


def define(name: str, *elements: hsm.Element) -> hsm.Model:
    """Define a model through ``hsm.define`` and publish its topology."""

    with span.operation(
        "bot.define",
        scope="bot.define",
        component="define",
        stage="construct",
    ) as active:
        model = hsm.define(name, *elements)
        outcome = publish(topology(model))
        active.set_attribute(_PUBLISH_ATTR, outcome)
        if outcome == _PUBLISH_FAILED:
            span.record_failure(active, "publish_failed")
        return model


__all__ = [
    "State",
    "Topology",
    "Transition",
    "activity",
    "after",
    "choice",
    "defer",
    "define",
    "effect",
    "entry",
    "exit",
    "final",
    "guard",
    "initial",
    "observe",
    "on",
    "Owner",
    "owner_qualified_name",
    "post_model",
    "publish",
    "source",
    "state",
    "studio_live_url",
    "studio_models_url",
    "studio_url",
    "target",
    "topology",
    "transition",
]
