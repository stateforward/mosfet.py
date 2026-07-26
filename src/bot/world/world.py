import asyncio
import collections.abc
import typing
import weakref

import hsm

from bot import lifecycle


def _instance_scope(instance: hsm.Instance) -> object | None:
    return instance.context().value(hsm.Keys.Instances)


def _scope_map(existing: object | None) -> weakref.WeakValueDictionary[str, hsm.Instance]:
    """Reuse an inherited scope map, or seed a new one from a foreign mapping."""

    if isinstance(existing, weakref.WeakValueDictionary):
        return typing.cast(weakref.WeakValueDictionary[str, hsm.Instance], existing)
    scope: weakref.WeakValueDictionary[str, hsm.Instance] = weakref.WeakValueDictionary()
    if isinstance(existing, collections.abc.MutableMapping):
        mapped = typing.cast(collections.abc.MutableMapping[object, object], existing)
        for key, value in mapped.items():
            if isinstance(key, str) and isinstance(value, hsm.Instance):
                scope[key] = value
    return scope


class _Scope:
    """Context key carrying the World that published this context chain."""


class _Participants:
    """Context key for the world presence set (distinct from ``hsm.Keys.Instances``)."""


class World(hsm.Context):
    """HSM context that separates world *presence* from world *addressing*.

    World *is* an ``hsm.Context`` — pass it directly to ``hsm.started``, ``hsm.dispatch``,
    ``hsm.dispatch_to``, and ``hsm.dispatch_all``. Two planes live on it:

    * **Addressing/lifetime** is ``hsm.Keys.Instances``. Every started machine in the scope is
      in it, including device firmware, so it stays reachable and stays able to emit into the
      world. It is not a public registry or discovery surface.
    * **Presence** is the participant set. Only citizens admitted through :meth:`join` are in
      it, and only they receive :meth:`broadcast`. Device firmware is never a citizen: it
      consumes world stimuli through its shell's forward, so a broadcast reaches it once.

    Both planes propagate as context values, so a machine started under this world can
    broadcast *into* it without being a recipient *of* it. Neither key is readable from
    outside this module: World is a scope, not a registry.
    """

    _instances: weakref.WeakValueDictionary[str, hsm.Instance]
    _participants: weakref.WeakValueDictionary[str, hsm.Instance]

    def __init__(self, context: hsm.Context | None = None) -> None:
        self._instances = _scope_map(None if context is None else context.value(hsm.Keys.Instances))
        self._participants = _scope_map(None if context is None else context.value(_Participants))
        parent = None if context is None or context.is_done() else context
        super().__init__(
            parent=parent,
            values={
                hsm.Keys.Instances: self._instances,
                _Participants: self._participants,
                _Scope: self,
            },
        )

    @classmethod
    def from_context(cls, context: hsm.Context) -> "World":
        """Resolve the World owning ``context``, creating one only when none is reachable.

        Resolution is O(1) and allocation-free for any context started under a world, which
        matters on per-chunk paths: every constructed context registers a done-callback on its
        parent that is never removed.
        """

        if isinstance(context, cls):
            return context
        scope = context.value(_Scope)
        # hsm.dispatch_to returns early on a canceled context, so a canceled scope would go
        # permanently silent. Fall back to a live World over the *same* instance and participant
        # maps — _scope_map reuses both by reference — so delivery and presence are unchanged and
        # only the canceled context is left behind. This is also the one production reader of the
        # participants context value: without it, presence could be an instance field.
        if isinstance(scope, cls) and not scope.is_done():
            return scope
        return cls(context)

    def join(self, instance: hsm.Instance) -> None:
        """Admit a started instance as a world citizen. Idempotent.

        Presence is not addressing: firmware and privately scoped actors stay addressable
        without becoming independent broadcast recipients.
        """

        participant = type(instance).__name__
        if not lifecycle.is_started(instance):
            raise RuntimeError(f"{participant} must be started before it joins a world.")
        require_world_scope(self, instance, participant=participant)
        self._participants[hsm.id(instance)] = instance

    def leave(self, instance: hsm.Instance) -> None:
        """Remove a citizen from world presence. Idempotent; safe on a stopped instance."""

        for identifier, participant in list(self._participants.items()):
            if participant is instance:
                del self._participants[identifier]

    def broadcast(self, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
        """Deliver a world stimulus to every participant.

        ``hsm.dispatch_to`` with no ids means "every instance in scope", so an empty presence
        set short-circuits instead of degrading into an addressing-plane broadcast.
        """

        participants = tuple(self._participants.keys())
        # The POSITION of this guard is load-bearing, not just its presence. Any future
        # narrowing of recipients (attenuation, range, occlusion) MUST run before it, never
        # between it and dispatch_to: a filter that removes everyone would otherwise hand
        # dispatch_to an empty id list, which means "deliver to every instance in scope".
        if not participants:
            delivered = asyncio.get_running_loop().create_future()
            delivered.set_result(None)
            return delivered
        return hsm.dispatch_to(self, event, *participants)


def require_world_scope(world: World, instance: hsm.Instance, *, participant: str) -> None:
    """Reject attaching a running instance to a different world scope."""

    # Only enforce while the machine is started; a stopped instance has no world scope yet.
    if not lifecycle.is_started(instance):
        return
    if _instance_scope(instance) is world.value(hsm.Keys.Instances):
        return
    raise RuntimeError(f"{participant} is already started in another world.")
