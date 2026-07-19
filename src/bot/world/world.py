import collections.abc
import typing
import weakref

import hsm


def _instance_scope(instance: hsm.Instance) -> object | None:
    return instance.context().value(hsm.Keys.Instances)


class World(hsm.Context):
    """Broadcast scope backed by an HSM context.

    World intentionally exposes scope operations only. The internal instance map exists so the HSM runtime can
    deliver `dispatch_all`; it is not a public registry or discovery surface.
    """

    _instances: weakref.WeakValueDictionary[str, hsm.Instance]

    def __init__(self, context: hsm.Context | None = None) -> None:
        existing_instances = None if context is None else context.value(hsm.Keys.Instances)
        if isinstance(existing_instances, weakref.WeakValueDictionary):
            self._instances = typing.cast(weakref.WeakValueDictionary[str, hsm.Instance], existing_instances)
            parent = None if context is None or context.is_done() else context
            super().__init__(parent=parent, values={hsm.Keys.Instances: self._instances})
            return
        if isinstance(existing_instances, collections.abc.MutableMapping):
            self._instances = weakref.WeakValueDictionary()
            mapped_instances = typing.cast(collections.abc.MutableMapping[object, object], existing_instances)
            for key, value in mapped_instances.items():
                if isinstance(key, str) and isinstance(value, hsm.Instance):
                    self._instances[key] = value
        else:
            self._instances = weakref.WeakValueDictionary()
        parent = None if context is None or context.is_done() else context
        super().__init__(parent=parent, values={hsm.Keys.Instances: self._instances})

    @classmethod
    def from_context(cls, context: hsm.Context) -> "World":
        """Create a world view over an existing HSM context."""

        return cls(context)

    @property
    def context(self) -> typing.Self:
        """HSM context that carries this world's broadcast scope."""

        return self

    async def broadcast(self, event: hsm.Event[typing.Any]) -> None:
        """Broadcast an event to every started HSM instance in this world."""

        await hsm.dispatch_all(self, event)


def require_world_scope(world: World, instance: hsm.Instance, *, participant: str) -> None:
    """Reject attaching a running instance to a different world scope."""

    # Only enforce while the machine is started (hsm 1.3.2+: id fails after stop).
    try:
        _ = hsm.id(instance)
    except hsm.ErrorValidatingModel:
        return
    if _instance_scope(instance) is world.value(hsm.Keys.Instances):
        return
    raise RuntimeError(f"{participant} is already started in another world.")
