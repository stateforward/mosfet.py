"""Where things are in the environment, and how loud a sound is by the time it gets there.

Levels are dB SPL referenced to :data:`D_REF`. An emitter declares how loud it is at that
reference distance; the environment works out what reaches each listener. A device cannot know how far
its own sound carries — that is a property of the space it is in, not of the device.
"""

import dataclasses
import math
import typing

import pydantic

D_REF = 1.0
"""Reference distance in metres. A declared amplitude is the level measured this far away."""

D_MIN = 0.01
"""Closest distance the propagation law is evaluated at, in metres.

Load-bearing, and not for an unlikely edge: a handset earpiece sits *at* the ear, so distance
zero is the single hottest path this model exists to describe. ``log10(0)`` is a domain error, so
without this clamp the earpiece case raises instead of returning a very loud level. Anything
nearer than this reads as this — 1 cm from a source is already 40 dB above its reference level.
"""


ObjectId = typing.Annotated[
    str,
    pydantic.Field(
        pattern=r"^[a-z][a-z0-9_-]{0,62}$",
        description=(
            "The id of a physical thing in the environment that other things sit on or in: a desk, a table, "
            "a shelf. A placement names it as what the placed thing rests on. Set when the object is built "
            "and never changed. 1 to 63 lower-case letters, digits, underscores or hyphens, starting with a "
            "letter."
        ),
        examples=["desk", "kitchen_counter", "shelf"],
    ),
]
_OBJECT_ID: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(ObjectId)


@dataclasses.dataclass(frozen=True)
class Position:
    """A point in the environment, in metres."""

    x: float
    y: float
    z: float = 0.0

    def distance_to(self, other: "Position") -> float:
        """Straight-line distance to ``other``, in metres."""

        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


@dataclasses.dataclass(frozen=True)
class Placement:
    """Where a participant is, how quiet a sound can get before it stops hearing it, and what it sits on.

    ``threshold_db`` of ``None`` means this participant is not selective: it receives every
    stimulus regardless of level, which is what keeps geometry opt-in. ``on`` is the
    :data:`ObjectId` of the object the participant rests on (``environment.Object.surface``
    sets it), or ``None`` when it stands free.

    Raises:
        pydantic.ValidationError: If ``on`` is not a valid :data:`ObjectId`.
    """

    position: Position
    threshold_db: float | None = None
    on: ObjectId | None = None

    def __post_init__(self) -> None:
        if self.on is not None:
            _ = _OBJECT_ID.validate_python(self.on)


def received_level_db(amplitude_db: float, distance_m: float) -> float:
    """Level in dB SPL of a sound of ``amplitude_db`` (at :data:`D_REF`) heard ``distance_m`` away.

    Free-field spherical spreading: −6 dB per doubling of distance.
    """

    return amplitude_db - 20.0 * math.log10(max(distance_m, D_MIN) / D_REF)
