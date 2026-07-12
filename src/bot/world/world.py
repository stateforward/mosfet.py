import collections.abc
import typing
import weakref

import hsm

P = typing.ParamSpec("P")
TReturn = typing.TypeVar("TReturn", covariant=True)


@typing.runtime_checkable
class Attachment(typing.Protocol[P, TReturn]):
    """Participant that can enter and leave a World attachment scope."""

    def attach(self, world: "World", *args: P.args, **kwargs: P.kwargs) -> collections.abc.Awaitable[TReturn]:
        """Attach this participant to a world scope."""
        ...

    def detach(self, world: "World", *args: P.args, **kwargs: P.kwargs) -> collections.abc.Awaitable[TReturn]:
        """Detach this participant from a world scope."""
        ...


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

    async def attach(self, participant: Attachment[P, TReturn], *args: P.args, **kwargs: P.kwargs) -> TReturn:
        """Attach a participant through its own world-aware lifecycle contract."""

        return await participant.attach(self, *args, **kwargs)

    async def detach(self, participant: Attachment[P, TReturn], *args: P.args, **kwargs: P.kwargs) -> TReturn:
        """Detach a participant through its own world-aware lifecycle contract."""

        result = await participant.detach(self, *args, **kwargs)
        return result


def require_world_scope(world: World, instance: hsm.Instance, *, participant: str) -> None:
    """Reject attaching a running instance to a different world scope."""

    if not instance.state():
        return
    if _instance_scope(instance) is world.value(hsm.Keys.Instances):
        return
    raise RuntimeError(f"{participant} is already started in another world.")
