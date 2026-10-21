"""Typed media values that may reach a model, and the mechanical bounds on carrying them.

An image travels as one typed value, ``Image``: its encoded file bytes, a validated media type, and
an optional pixel size. ``ImageCapacity`` states what one model can take in one request and is
injected into the generator that fronts that model. ``ReleasedImage`` is the metadata-only stand-in
for an image whose bytes are not carried. Every bound here is mechanical and content-blind: it trims by
order, count, size, and media type, never by what an image shows.

Usage::

    from mosfet import media

    image = media.Image(content=png_bytes, media_type="image/png", width_px=640, height_px=480)
    media.ImageCapacity(max_images=1, max_total_bytes=len(image.content), media_types=frozenset({"image/png"}))
"""

from __future__ import annotations

import collections.abc
import typing

import pydantic

ImageMediaType = typing.Literal["image/png", "image/jpeg", "image/webp"]
"""Image media types every image-capable provider accepts. Producers transcode anything else."""

MAX_IMAGE_BYTES: typing.Final = 4 * 1024 * 1024
"""Largest encoded image accepted, in bytes (a boundary size limit, PY-DATA-002)."""

MAX_TURN_IMAGES: typing.Final = 8
"""Most images one conversation turn may carry (a boundary count limit, PY-DATA-002)."""

ReleaseReason = typing.Literal["memory"]
"""Why an image's bytes are absent: a memory read omitted them."""

ImageOmission = typing.Literal["capacity", "media_type", "memory"]
"""Why an image is shown to a model as a placeholder only."""

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"


class Image(pydantic.BaseModel):
    """One encoded image file, exactly as its producer captured or received it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [{"content": "iVBORw0KGgo=", "media_type": "image/png", "width_px": 640, "height_px": 480}],
        },
    )

    content: bytes = pydantic.Field(
        min_length=1,
        max_length=MAX_IMAGE_BYTES,
        description=(
            "Encoded image file bytes exactly as produced (PNG/JPEG/WebP). JSON carries base64. The leading "
            "signature must match media_type."
        ),
        examples=["iVBORw0KGgo="],
    )
    media_type: ImageMediaType = pydantic.Field(
        description="Media type of the encoded bytes; one of image/png, image/jpeg, or image/webp.",
        examples=["image/png"],
    )
    width_px: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Image width in pixels, stamped by the producer when known.",
        examples=[640],
    )
    height_px: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Image height in pixels, stamped by the producer when known.",
        examples=[480],
    )

    @pydantic.model_validator(mode="after")
    def _validate_signature(self) -> typing.Self:
        content = self.content
        if self.media_type == "image/png":
            matches = content.startswith(_PNG_SIGNATURE)
        elif self.media_type == "image/jpeg":
            matches = content.startswith(_JPEG_SIGNATURE)
        else:
            matches = content[:4] == b"RIFF" and content[8:12] == b"WEBP"
        if not matches:
            raise ValueError("image content does not match media_type")
        return self


class ReleasedImage(pydantic.BaseModel):
    """Metadata-only stand-in for an image whose bytes are not carried; it has no field that holds bytes."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "media_type": "image/png",
                    "width_px": 640,
                    "height_px": 480,
                    "byte_count": 48213,
                    "reason": "memory",
                }
            ],
        },
    )

    media_type: ImageMediaType = pydantic.Field(
        description="Media type of the released image.",
        examples=["image/png"],
    )
    width_px: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Width in pixels of the released image, when its producer stamped one.",
        examples=[640],
    )
    height_px: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Height in pixels of the released image, when its producer stamped one.",
        examples=[480],
    )
    byte_count: int = pydantic.Field(
        ge=1,
        description="Size in bytes of the released image's encoded content.",
        examples=[48213],
    )
    reason: ReleaseReason = pydantic.Field(
        description="Why the bytes are absent: 'memory' when a memory read returned the image without its bytes.",
        examples=["memory"],
    )

    @classmethod
    def of(cls, image: Image, *, reason: ReleaseReason) -> ReleasedImage:
        """Return the metadata-only stand-in for ``image``, stating why its bytes are absent."""

        return cls(
            media_type=image.media_type,
            width_px=image.width_px,
            height_px=image.height_px,
            byte_count=len(image.content),
            reason=reason,
        )


class ImageCapacity(pydantic.BaseModel):
    """What images one model can take in one request; a capacity, never a choice. The default is text-only."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "max_images": 8,
                    "max_total_bytes": 12 * 1024 * 1024,
                    "media_types": ["image/png", "image/jpeg", "image/webp"],
                }
            ],
        },
    )

    max_images: int = pydantic.Field(
        default=0,
        ge=0,
        description="Most images the model takes in one request; 0 means the model is text-only.",
        examples=[8],
    )
    max_total_bytes: int = pydantic.Field(
        default=0,
        ge=0,
        description="Most encoded image bytes, summed, the model takes in one request.",
        examples=[12 * 1024 * 1024],
    )
    media_types: frozenset[ImageMediaType] = pydantic.Field(
        default=frozenset(),
        description="Image media types the model accepts.",
        examples=[["image/png", "image/jpeg", "image/webp"]],
    )

    def admit(self, images: collections.abc.Sequence[Image]) -> tuple[ImageOmission | None, ...]:
        """Return one entry per image, in input order: ``None`` when attached, else why it is omitted.

        Images are considered from last to first, so the newest image in document or chronological
        order wins. An unsupported media type is ``"media_type"``; an image that would exceed the
        count or byte bound is ``"capacity"``.
        """

        verdicts: list[ImageOmission | None] = []
        admitted = 0
        admitted_bytes = 0
        for image in reversed(images):
            size = len(image.content)
            if image.media_type not in self.media_types:
                verdicts.append("media_type")
            elif admitted + 1 > self.max_images or admitted_bytes + size > self.max_total_bytes:
                verdicts.append("capacity")
            else:
                admitted += 1
                admitted_bytes += size
                verdicts.append(None)
        return tuple(reversed(verdicts))


__all__ = [
    "MAX_IMAGE_BYTES",
    "MAX_TURN_IMAGES",
    "Image",
    "ImageCapacity",
    "ImageMediaType",
    "ImageOmission",
    "ReleaseReason",
    "ReleasedImage",
]
