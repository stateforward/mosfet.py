import asyncio
import collections.abc
import typing
import weakref

import hsm

from bot import lifecycle

from . import events
from . import space


def _instance_scope(instance: hsm.Instance) -> object | None:
    return instance.context().value(hsm.Keys.Instances)


class _Scope:
    """Context key carrying the Environment that published this context chain."""


class Environment(hsm.Context):
    """HSM context that separates environment *presence* from environment *addressing*.

    Environment *is* an ``hsm.Context`` — pass it directly to ``hsm.started``, ``hsm.dispatch``,
    ``hsm.dispatch_to``, and ``hsm.dispatch_all``. Two planes live on it:

    * **Addressing/lifetime** is ``hsm.Keys.Instances``. Every started machine in the scope is
      in it, including device firmware, so it stays reachable and stays able to emit into the
      environment. It is not a public registry or discovery surface.
    * **Presence** is the participant set, held on this object. Only citizens admitted through
      :meth:`join` are in it, and only they receive :meth:`broadcast`. Device firmware is never a
      citizen: it consumes environment stimuli through its shell's forward, so a broadcast reaches it
      once. A placement, when a citizen gives one, says where it is and how quiet a sound can get
      before it stops hearing it.

    The addressing map propagates as a context value, so a machine started under this environment can
    broadcast *into* it without being a recipient *of* it. The scope key is not readable from
    outside this module: Environment is a scope, not a registry.
    """

    _instances: weakref.WeakValueDictionary[str, hsm.Instance]
    _participants: weakref.WeakValueDictionary[str, hsm.Instance]
    # Keyed by instance, not id: a placement cannot outlive the thing it places, so leaving a
    # environment needs no placement bookkeeping and a stale placement is not representable.
    _placements: weakref.WeakKeyDictionary[hsm.Instance, space.Placement]

    def __init__(self, context: hsm.Context | None = None) -> None:
        # Reuse an inherited addressing map, or seed one from a foreign mapping.
        inherited = None if context is None else context.value(hsm.Keys.Instances)
        if isinstance(inherited, weakref.WeakValueDictionary):
            self._instances = typing.cast(weakref.WeakValueDictionary[str, hsm.Instance], inherited)
        else:
            self._instances = weakref.WeakValueDictionary()
            if isinstance(inherited, collections.abc.MutableMapping):
                mapped = typing.cast(collections.abc.MutableMapping[object, object], inherited)
                for key, value in mapped.items():
                    if isinstance(key, str) and isinstance(value, hsm.Instance):
                        self._instances[key] = value
        self._participants = weakref.WeakValueDictionary()
        self._placements = weakref.WeakKeyDictionary()
        parent = None if context is None or context.is_done() else context
        super().__init__(parent=parent, values={hsm.Keys.Instances: self._instances, _Scope: self})

    @classmethod
    def from_context(cls, context: hsm.Context) -> "Environment":
        """Resolve the Environment owning ``context``, creating one only when none is reachable.

        Resolution is O(1) and allocation-free for any context started under an environment, which
        matters on per-chunk paths: every constructed context registers a done-callback on its
        parent that is never removed.
        """

        if isinstance(context, cls):
            return context
        scope = context.value(_Scope)
        if isinstance(scope, cls) and not scope.is_done():
            return scope
        # hsm.dispatch_to returns early on a canceled context, so a canceled scope would go
        # permanently silent. Fall back to a live Environment over the same maps: the addressing map
        # comes off the context, and presence and placement are carried over from the canceled
        # Environment by reference, so joining or leaving either scope is visible in both.
        revived = cls(context)
        if isinstance(scope, cls):
            revived._participants = scope._participants
            revived._placements = scope._placements
        return revived

    def join(self, instance: hsm.Instance, *, placement: space.Placement | None = None) -> None:
        """Admit a started instance as an environment citizen. Idempotent.

        Presence is not addressing: firmware and privately scoped actors stay addressable
        without becoming independent broadcast recipients.

        ``placement`` says where this citizen is and how quiet a sound can get before it stops
        hearing it. Without one it hears every stimulus, which is what keeps geometry opt-in.
        """

        participant = type(instance).__name__
        if not lifecycle.is_started(instance):
            raise RuntimeError(f"{participant} must be started before it joins an environment.")
        require_environment_scope(self, instance, participant=participant)
        if placement is not None:
            placed = self._placements.get(instance)
            # One object cannot be in two places. Taking the newer one silently is how a shared
            # transducer ends up somewhere it is not, with no error and a wrong audience.
            if placed is not None and placed != placement:
                raise RuntimeError(f"{participant} is already placed elsewhere in this environment.")
            self._placements[instance] = placement
        self._participants[hsm.id(instance)] = instance

    def leave(self, instance: hsm.Instance) -> None:
        """Remove a citizen from environment presence. Idempotent; safe on a stopped instance."""

        for identifier, participant in list(self._participants.items()):
            if participant is instance:
                del self._participants[identifier]

    def _audible(
        self,
        participant: hsm.Instance,
        event: hsm.Event[typing.Any],
        origin: space.Position | None,
    ) -> bool:
        """Whether this stimulus still clears ``participant``'s threshold where it stands.

        True whenever any of the three inputs is missing — the emitter did not say where it was,
        or how loud, or the citizen never said where it is or how quiet it can hear. Geometry is
        opt-in on all three counts.
        """

        if origin is None:
            return True
        data = event.data
        amplitude_db = data.amplitude_db if isinstance(data, events.SoundData) else None
        if amplitude_db is None:
            return True
        placement = self._placements.get(participant)
        if placement is None or placement.threshold_db is None:
            return True
        return space.received_level_db(amplitude_db, origin.distance_to(placement.position)) >= placement.threshold_db

    def broadcast(
        self,
        event: hsm.Event[typing.Any],
        *,
        origin: space.Position | None = None,
    ) -> collections.abc.Awaitable[None]:
        """Deliver an environment stimulus to every participant it is still audible to.

        ``origin`` is where the sound came from. The environment computes what reaches each citizen,
        because how far a sound carries is a property of the space, not of the emitter — which is
        also why the emitter is a parameter here rather than something read off ``event.source``.

        ``hsm.dispatch_to`` with no ids means "every instance in scope", so an empty recipient
        list short-circuits instead of degrading into an addressing-plane broadcast.
        """

        # Narrowing runs BEFORE the guard, and the guard runs on the narrowed list. The POSITION
        # of this guard is load-bearing, not just its presence: a filter that removes everyone
        # would otherwise hand dispatch_to an empty id list, which means "deliver to every
        # instance in scope" — the defect the guard exists to prevent, in the case that hits most
        # often, a quiet sound in a large environment. Any future narrowing goes above this line.
        audible = [
            identifier
            for identifier, participant in self._participants.items()
            if self._audible(participant, event, origin)
        ]
        if not audible:
            delivered = asyncio.get_running_loop().create_future()
            delivered.set_result(None)
            return delivered
        return hsm.dispatch_to(self, event, *audible)


def require_environment_scope(environment: Environment, instance: hsm.Instance, *, participant: str) -> None:
    """Reject attaching a running instance to a different environment scope."""

    # Only enforce while the machine is started; a stopped instance has no environment scope yet.
    if not lifecycle.is_started(instance):
        return
    if _instance_scope(instance) is environment.value(hsm.Keys.Instances):
        return
    raise RuntimeError(f"{participant} is already started in another environment.")
