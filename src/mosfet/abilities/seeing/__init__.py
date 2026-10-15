"""Seeing: the eye. Hands on measured visual change and looks on demand.

from mosfet.abilities import seeing
seeing.Seeing(change=vision.change.ByteIdentity())

Optional stages fill each view's objects and faces, and a face ``identification.Registry`` names
who each face is::

    seeing.Seeing(
        change=measure,
        object_detector=objects,
        face_detector=faces,
        faces=identification.Registry(identifier=matcher, kind="face", max_unlinked_age=age, max_ids=cap),
    )
"""

from . import seeing
from .seeing import (
    MAX_VIEWS,
    Cause,
    Face,
    LookData,
    LookEvent,
    Seeing,
    SightData,
    SightEvent,
    View,
)

__all__ = [
    "MAX_VIEWS",
    "Cause",
    "Face",
    "LookData",
    "LookEvent",
    "Seeing",
    "SightData",
    "SightEvent",
    "View",
    "seeing",
]
