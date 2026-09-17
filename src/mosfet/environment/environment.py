import asyncio
import collections.abc
import dataclasses
import typing
import uuid
import weakref

import hsm

from mosfet import lifecycle

from . import events
from . import snapshot
from . import space


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
    # This environment's own identity for model-facing observation (the <environment id="…">
    # attribute) — never an HSM actor id (never Python id()). Stable for the life of this object;
    # a scope revived by from_context after cancellation carries it forward by assignment there,
    # the same way it carries presence and placement forward by reference.
    _id: str

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
        self._id = uuid.uuid4().hex
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
            revived._id = scope._id
        return revived

    @classmethod
    def reachable(cls, context: hsm.Context) -> "Environment | None":
        """Resolve the live Environment owning ``context`` without fabricating one.

        Returns ``None`` when ``context`` is not under an Environment scope at all
        (for example a bare ``hsm.Context``), so callers can skip environment-scoped
        registration instead of inventing an identity for an actor in no environment.
        Unlike :meth:`from_context`, it never synthesizes a new Environment.
        """

        if isinstance(context, cls):
            return context
        scope = context.value(_Scope)
        if isinstance(scope, cls):
            return scope
        return None

    @property
    def environment_id(self) -> str:
        """This environment's stable identity segment: ``<id>`` in ``/<id>/<actor-id>``.

        The same string the addressing registry uses as this environment's root, and the
        one model-facing observation renders as ``<environment id="…">``. Revival by
        :meth:`from_context` carries it forward, so an address survives scope cancellation.
        """
        return self._id

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

    def model_snapshot(self, perspective: hsm.Instance) -> str | None:
        """Compose this turn's model-facing world block from ``perspective``'s own live snapshot.

        The environment the bot is in takes the snapshot: this is the one place the world block
        is assembled, not cognition, because everything in it — the world's own identity, which
        actors are reachable, what a referenced device's snapshot says — is something only the
        environment knows first-hand. ``perspective`` says whose turn this is; the environment
        never invents "self" by scanning its own citizens, because which actor's viewpoint a turn
        is taken from is the caller's to say, not the environment's to guess.

        The root is ``<environment id="…">``, this environment's own identity (see ``_id``) —
        never an HSM actor id and never Python ``id()``. Its single child is ``<self>``:
        ``perspective``'s own snapshot, state as an attribute, its own snapshot attributes as
        children. A declared ``owned_devices`` reference→id map among those attributes resolves
        to full nested ``<device>`` elements by looking up each id in this environment's own
        addressing scope — never a caller-supplied actor map, which would make cognition the one
        deciding what the world contains.

        Returns ``None`` when ``perspective`` has no honest snapshot to render (not started, or
        started somewhere this environment cannot see): there is nothing true to put in the
        envelope, and an envelope around nothing would claim an observation the system does not
        have.
        """

        if not self.contains(perspective):
            return None
        self_element = snapshot.render_self(perspective, self._instances.get, 1)
        if self_element is None:
            return None
        return snapshot.render_environment(self._id, self_element)

    @property
    def scope_path(self) -> str:
        """The environment's own address: ``/env/<id>``.

        Built from the environment's stable identity (``_id``), never from any actor inside
        it, so the path says where this environment is without saying who is in it."""
        return f"/env/{self._id}"

    def contains(self, instance: hsm.Instance) -> bool:
        """Whether ``instance`` runs in this environment's own addressing scope.

        Map identity is the only honest test — a shared ancestor is not a shared scope — and an
        unstarted instance has no scope to test, so it is ``False``. Callers that must tell
        "not started yet" from "started elsewhere" check ``lifecycle.is_started`` first, the
        way ``require_environment_scope`` does."""
        return instance.context().value(hsm.Keys.Instances) is self._instances

    def path_of(self, instance: hsm.Instance) -> str | None:
        """``instance``'s path under this environment, or ``None`` when it is not a member.

        Derived on demand, never stored: a stored path would outlive the instance it named and
        address nothing — the same lie ``contains`` already refuses."""
        if not self.contains(instance):
            return None
        return f"{self.scope_path}/{hsm.id(instance)}"

    def _reception(
        self,
        participant: hsm.Instance,
        amplitude_db: float | None,
        origin: space.Position | None,
    ) -> tuple[bool, float | None]:
        """Whether this stimulus still clears ``participant``'s threshold, and how loud it arrives.

        The level is the quantity this whole model is for, so it is returned rather than reduced
        to the boolean and thrown away: a citizen is told how loud the sound was where it stands.

        A level of ``None`` means there was nothing to measure — the emitter did not say how loud
        it was, or where it was, or the citizen never said where it is. Geometry is opt-in on all
        three counts, and a citizen with nothing to measure hears the stimulus.
        """

        placement = self._placements.get(participant)
        if amplitude_db is None or origin is None or placement is None:
            return True, None
        level_db = space.received_level_db(amplitude_db, origin.distance_to(placement.position))
        if placement.threshold_db is None:
            return True, level_db
        return level_db >= placement.threshold_db, level_db

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

        Delivery is per recipient because the stimulus is not the same event for each of them: two
        citizens standing at two distances hear one sound at two levels, and each is handed the
        level it hears. ``amplitude_db`` is left exactly as the emitter set it.

        ``hsm.dispatch_to`` with no ids means "every instance in scope", so an empty recipient
        list short-circuits instead of degrading into an addressing-plane broadcast.
        """

        data = event.data
        sound = data if isinstance(data, events.SoundData) else None
        amplitude_db = None if sound is None else sound.amplitude_db
        # Narrowing runs BEFORE the guard, and the guard runs on the narrowed list. The POSITION
        # of this guard is load-bearing, not just its presence: a filter that removes everyone
        # would otherwise hand dispatch_to an empty id list, which means "deliver to every
        # instance in scope" — the defect the guard exists to prevent, in the case that hits most
        # often, a quiet sound in a large environment. Any future narrowing goes above this line.
        # Per-recipient dispatch below never reaches it with no ids, but that is a property of
        # this loop and not a reason to relax the guard.
        audible: list[tuple[str, float | None]] = []
        for identifier, participant in self._participants.items():
            heard, level_db = self._reception(participant, amplitude_db, origin)
            if heard:
                audible.append((identifier, level_db))
        if not audible:
            delivered = asyncio.get_running_loop().create_future()
            delivered.set_result(None)
            return delivered
        deliveries = [
            hsm.dispatch_to(
                self,
                event
                if sound is None or level_db is None
                else dataclasses.replace(event, data=sound.model_copy(update={"received_level_db": level_db})),
                identifier,
            )
            for identifier, level_db in audible
        ]

        async def delivered_to_all() -> None:
            _ = await asyncio.gather(*deliveries)

        # Eager, matching hsm.dispatch_to: the fire-and-forget callers that never await a
        # broadcast must still see it leave.
        return asyncio.Task(delivered_to_all(), loop=asyncio.get_running_loop(), eager_start=True)


def require_environment_scope(environment: Environment, instance: hsm.Instance, *, participant: str) -> None:
    """Reject attaching a running instance to a different environment scope."""

    # Only enforce while the machine is started; a stopped instance has no environment scope yet.
    if not lifecycle.is_started(instance):
        return
    if environment.contains(instance):
        return
    raise RuntimeError(f"{participant} is already started in another environment.")


def elevate_device_observation_to_input(
    ctx: hsm.Context,
    owner: hsm.Instance,
    observation: hsm.Event[typing.Any],
) -> None:
    """Elevate one typed device observation into body ``bot.input`` for ``owner``.

    Explicit boundary contract owned by environment: body and cognition consume the
    elevated ``bot.input`` form, never device event names. Coordinates via the typed
    device ``ObservationData.observation`` product — every observation elevates
    unconditionally and is never branched on for routing. Producers stamp identity and
    provenance at emission (observation ``id``/``source``/``metadata``); this
    preserves them onto the elevated envelope with ``target`` addressed to ``owner``.
    No attachment/device tree walk: the caller passes the explicit ``owner``.
    """

    from mosfet.device import ObservationData

    data = observation.data
    assert isinstance(data, ObservationData)
    import mosfet

    _ = hsm.dispatch(
        ctx,
        owner,
        dataclasses.replace(
            mosfet.InputEvent.with_data(
                mosfet.InputEventData(
                    priority=data.priority,
                    observation=data.observation,
                )
            ),
            id=observation.id or uuid.uuid4().hex,
            source=observation.source,
            target=hsm.id(owner),
            metadata=dict(observation.metadata),
        ),
    )
