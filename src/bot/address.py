"""Process-wide address registry for bot-scoped actors.

Every actor started through ``bot.start``/``bot.started`` inside an environment
scope registers here under its runtime path: ``/<environment-id>/<actor-id>``.
The registry is the addressing plane the environment dispatches through —
``Environment.broadcast`` resolves recipients by path prefix, and private scopes
register under a visibility marker that environment dispatch excludes.

Lifetime: values are weak, so a detached one-shot actor (reply, timer) whose
whole context chain dies vanishes from the registry on collection. That is the
*only* path weak cleanup serves: hsm's context chain (``Context._done`` futures
holding child-cancel callbacks, context values holding ``Keys.HSM``) keeps an
environment-scoped actor strongly alive for as long as its environment lives,
so environment children never GC out. The real cleanup for those is explicit:
bot stop paths call :func:`unregister` / :func:`unregister_instance` where they
previously scrubbed ``Keys.Instances`` entries. A leaked entry is therefore a
leaked *address string*, not a leaked actor — resolve() on it returns a live
actor only while that actor is genuinely still running elsewhere's scope.
"""

import weakref

import hsm

_REGISTRY: "weakref.WeakValueDictionary[str, hsm.Instance]" = weakref.WeakValueDictionary()
# Paths registered under a private scope: addressable by explicit dispatch, excluded
# from environment broadcast fan-out. Private actors keep their environment path segment,
# so the environment's identity never depends on visibility.
_PRIVATE: set[str] = set()


def register(path: str, instance: hsm.Instance) -> None:
    """Put ``instance`` at ``path``. Overwrites only on a true re-registration.

    A path collision with a different live instance is an addressing bug — two
    actors claiming one address — and raises rather than silently rebinding.
    """

    existing = _REGISTRY.get(path)
    if existing is not None and existing is not instance:
        raise RuntimeError(f"address {path} is already registered to another instance.")
    _REGISTRY[path] = instance


def unregister(path: str) -> None:
    """Remove ``path`` if it is still present. Idempotent."""

    _ = _REGISTRY.pop(path, None)
    _PRIVATE.discard(path)


def unregister_instance(instance: hsm.Instance, path: str | None = None) -> None:
    """Remove an instance's registration, deriving the path when not given."""

    if path is not None:
        if _REGISTRY.get(path) is instance:
            unregister(path)
        return
    for key, value in list(_REGISTRY.items()):
        if value is instance:
            unregister(key)


def _discard_stale_private(path: str) -> None:
    """Drop a private marker only for a registration that is no longer live."""

    if path in _PRIVATE and _REGISTRY.get(path) is None:
        _PRIVATE.discard(path)


def mark_private(path: str) -> None:
    """Mark ``path`` as a private-scope registration (excluded from broadcast).

    The marker is tied to the registered instance's lifetime: it is removed when
    the registration is explicitly unregistered, and also when the registered
    instance is collected, so a reused path cannot stay silently hidden under a
    marker left by an actor that is no longer live.
    """

    _PRIVATE.add(path)
    instance = _REGISTRY.get(path)
    if instance is not None:
        _ = weakref.finalize(instance, _discard_stale_private, path)


def is_private(path: str) -> bool:
    """Whether ``path`` was registered under a private scope."""

    return path in _PRIVATE


def resolve(path: str) -> hsm.Instance | None:
    """The live instance at ``path``, or ``None`` when absent or collected."""

    return _REGISTRY.get(path)


def under_prefix(prefix: str) -> list[tuple[str, hsm.Instance]]:
    """Every live ``(path, instance)`` under ``prefix`` (inclusive of ``prefix`` itself)."""

    if not prefix.endswith("/"):
        prefix = prefix + "/"
    return [(key, value) for key, value in _REGISTRY.items() if key == prefix.rstrip("/") or key.startswith(prefix)]


def under_prefix_public(prefix: str) -> list[tuple[str, hsm.Instance]]:
    """``under_prefix`` minus private-scope registrations — the broadcast candidate set."""

    return [(key, value) for key, value in under_prefix(prefix) if not is_private(key)]


def environment_path(environment_id: str, instance_id: str) -> str:
    """The canonical address of ``instance_id`` inside environment ``environment_id``.

    Matches ``Environment.scope_path``: ``/env/<environment-id>/<instance-id>``.
    """

    return f"/env/{environment_id}/{instance_id}"


def visibility_root(path: str) -> str:
    """The visibility root of ``path``: its first segment (the environment root)."""

    segments = path.split("/", 2)
    if len(segments) < 2 or not segments[1]:
        return path
    return f"/{segments[1]}"


__all__ = [
    "environment_path",
    "is_private",
    "mark_private",
    "register",
    "resolve",
    "under_prefix",
    "under_prefix_public",
    "unregister",
    "unregister_instance",
    "visibility_root",
]
