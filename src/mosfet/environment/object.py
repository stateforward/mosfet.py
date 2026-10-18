"""Physical things in the environment that other things sit on or in: a desk, a table, a shelf."""

import dataclasses

import pydantic

from . import space

_ID: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(space.ObjectId)
_ORIGIN = space.Position(0.0, 0.0)


@dataclasses.dataclass(frozen=True)
class Object:
    """A physical thing in the environment that other things sit on or in: a desk, a table, a shelf.

    It has an id and a place and does nothing: no power, no state, no machine, no snapshot of its
    own. It is never attached, started or joined to an environment; it is a placement anchor, built
    inline where the things on it are placed. It shows in the world block only through what sits on
    it: a device placed at ``desk.surface()`` renders ``on="desk"`` on its own element.

    >>> from mosfet import environment
    >>> desk = environment.Object(id="desk")
    >>> desk.surface().on
    'desk'

    Raises:
        pydantic.ValidationError: If ``id`` is not a valid :data:`~mosfet.environment.space.ObjectId`.
    """

    id: space.ObjectId
    placement: space.Placement = space.Placement(position=_ORIGIN)

    def __post_init__(self) -> None:
        _ = _ID.validate_python(self.id)

    def surface(self, offset: space.Position = _ORIGIN) -> space.Placement:
        """Where something set on this object, ``offset`` from its origin, is: ``on`` names this object.

        The placement keeps this object's own ``threshold_db``.
        """

        origin = self.placement.position
        return space.Placement(
            position=space.Position(origin.x + offset.x, origin.y + offset.y, origin.z + offset.z),
            threshold_db=self.placement.threshold_db,
            on=self.id,
        )


__all__ = ["Object"]
