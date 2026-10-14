"""Binary still codec: one camera stream message is one encoded image.

A message is one byte giving the media type's length, the ASCII media type (one of
``media.ImageMediaType``), then the encoded image file, at most ``media.MAX_IMAGE_BYTES``. It decodes
to ``devices.computer.camera.still`` carrying a validated ``media.Image``. The camera is inbound only,
so this codec encodes nothing.
"""

from .events import StillData, StillEvent

import typing

import hsm
import pydantic

from mosfet import media
from mosfet.protocols import transform

_MEDIA_TYPES: frozenset[str] = frozenset(typing.get_args(media.ImageMediaType))
_MAX_MEDIA_TYPE_BYTES = max(len(media_type) for media_type in _MEDIA_TYPES)


class StillCodec:
    """Camera stills from a stream to the camera port, validated into ``media.Image``."""

    name: str
    max_message_bytes: int
    encodes_any: bool

    def __init__(self) -> None:
        self.name = "still"
        self.max_message_bytes = 1 + _MAX_MEDIA_TYPE_BYTES + media.MAX_IMAGE_BYTES
        self.encodes_any = False

    def decode(self, message: bytes) -> hsm.Event[typing.Any]:
        """Return ``devices.computer.camera.still`` carrying the message's image.

        Raises:
            DecodeError: ``oversize`` past ``max_message_bytes``; ``malformed`` when the media-type header
                is empty, runs past the message, or is not ASCII; ``media_type`` for a media type outside
                ``media.ImageMediaType`` or bytes whose signature does not match it; ``schema_invalid`` for
                any other image the ``media.Image`` schema rejects (empty or over ``MAX_IMAGE_BYTES``).
        """

        if len(message) > self.max_message_bytes:
            raise transform.DecodeError("oversize", f"Still message exceeds {self.max_message_bytes} bytes.")
        if not message or message[0] == 0 or len(message) < 1 + message[0]:
            raise transform.DecodeError("malformed", "Still message needs a media-type length and media type.")
        end = 1 + message[0]
        try:
            media_type = message[1:end].decode("ascii")
        except UnicodeDecodeError as error:
            raise transform.DecodeError("malformed", "Still media type is not ASCII.") from error
        if media_type not in _MEDIA_TYPES:
            raise transform.DecodeError("media_type", "Still media type is not an accepted image type.")
        try:
            image = media.Image.model_validate({"content": message[end:], "media_type": media_type})
        except pydantic.ValidationError as error:
            # The image's own model check (no field location) is the signature-versus-media-type match.
            if all(not detail["loc"] for detail in error.errors()):
                raise transform.DecodeError("media_type", "Still bytes do not match their media type.") from error
            raise transform.DecodeError("schema_invalid", "Still bytes are not a valid image.") from error
        return StillEvent.with_data(StillData(image=image))

    def encodes(self, event: hsm.Event[typing.Any]) -> bool:
        """Always false: a camera only sends stills in."""

        del event
        return False

    def encode(self, event: hsm.Event[typing.Any]) -> bytes:
        """Refuse: nothing is sent to a camera.

        Raises:
            EncodeError: always, ``schema_invalid``.
        """

        raise transform.EncodeError("schema_invalid", f"StillCodec encodes nothing, not {event.name!r}.")


__all__ = ["StillCodec"]
