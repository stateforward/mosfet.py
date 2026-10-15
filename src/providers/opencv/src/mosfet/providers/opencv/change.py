"""Cheap change measures through OpenCV: a reduced-decode frame difference and a downscaling wrapper."""

from __future__ import annotations

import asyncio
import typing

import hsm
import numpy

import mosfet.media as media
from mosfet.abilities.vision import change

from .surface import Cv2Loader, Error, decode, import_cv2, measured

_FRAME_DIFF: typing.Final = "frame_diff"
_DEFAULT_SIZE: typing.Final = 32
_MIN_SIZE: typing.Final = 4
_MAX_SIZE: typing.Final = 256
_BLUR: typing.Final = (3, 3)
_DEFAULT_MAX_EDGE: typing.Final = 448
_MIN_MAX_EDGE: typing.Final = 16
_DEFAULT_QUALITY: typing.Final = 85
_JPEG: typing.Final = "image/jpeg"


class FrameDiff(change.ChangeMeasure):
    """The mean absolute difference of two small, blurred grayscale signatures, 0..1.

    Each measured frame is decoded already reduced and in grayscale (``IMREAD_REDUCED_GRAYSCALE_8``:
    a JPEG is decoded at 1/8 scale in the DCT domain; PNG and WebP decode at full size and are then
    reduced), resized to ``size × size`` with area averaging, and lightly blurred against sensor
    noise. The signature is those ``size²`` values scaled to 0..1. Cost: about 2–5 ms of CPU for a
    1280×720 JPEG, more for a full-size PNG; the comparison takes microseconds. The work runs in a
    worker thread.
    """

    _size: int
    _cv2_loader: Cv2Loader

    def __init__(self, *, size: int = _DEFAULT_SIZE, cv2_loader: Cv2Loader = import_cv2) -> None:
        """Raises ``ValueError`` unless ``size`` is in 4..256."""

        if not _MIN_SIZE <= size <= _MAX_SIZE:
            raise ValueError(f"FrameDiff size must be in {_MIN_SIZE}..{_MAX_SIZE}.")
        self._size = size
        self._cv2_loader = cv2_loader

    @typing.override
    async def measure(
        self, previous: change.Signature | None, image: media.Image, *, ctx: hsm.Context
    ) -> change.Measurement:
        del ctx
        if previous is not None and (previous.measure != _FRAME_DIFF or len(previous.values) != self._size**2):
            raise change.Error("previous signature is not a frame difference signature of this size")
        values = await asyncio.to_thread(self._signature, image.content)
        signature = change.Signature(measure=_FRAME_DIFF, values=values)
        if previous is None:
            return change.Measurement(change=None, signature=signature)
        difference = numpy.mean(numpy.abs(numpy.asarray(values) - numpy.asarray(previous.values)))
        return change.Measurement(change=min(1.0, max(0.0, float(difference))), signature=signature)

    def _signature(self, content: bytes) -> tuple[float, ...]:
        with measured("frame_diff", change.Error):
            cv2 = self._cv2_loader()
            reduced = decode(cv2, content, cv2.IMREAD_REDUCED_GRAYSCALE_8)
            small = cv2.resize(reduced, (self._size, self._size), interpolation=cv2.INTER_AREA)
            blurred = cv2.GaussianBlur(small, _BLUR, 0)
            return tuple(float(value) for value in (blurred.astype(numpy.float32) / 255.0).ravel())


class Downscaled(change.ChangeMeasure):
    """Hands ``inner`` a JPEG no larger than ``max_edge`` on its longer side; the measure for expensive measures.

    A frame larger than ``max_edge`` (or not a JPEG) is decoded, resized with area averaging and
    re-encoded as a JPEG at ``quality``; a JPEG that already fits is handed on unchanged. Signatures
    are ``inner``'s. Cost: about 3–8 ms of CPU, plus ``inner`` on a small frame. ``attachments`` are
    ``inner``'s.
    """

    _inner: change.ChangeMeasure
    _max_edge: int
    _quality: int
    _cv2_loader: Cv2Loader

    def __init__(
        self,
        *,
        inner: change.ChangeMeasure,
        max_edge: int = _DEFAULT_MAX_EDGE,
        quality: int = _DEFAULT_QUALITY,
        cv2_loader: Cv2Loader = import_cv2,
    ) -> None:
        """Raises ``ValueError`` unless ``max_edge`` is at least 16 and ``quality`` is in 1..100."""

        if max_edge < _MIN_MAX_EDGE:
            raise ValueError(f"Downscaled max_edge must be at least {_MIN_MAX_EDGE}.")
        if not 1 <= quality <= 100:
            raise ValueError("Downscaled quality must be in 1..100.")
        self._inner = inner
        self._max_edge = max_edge
        self._quality = quality
        self._cv2_loader = cv2_loader
        self.attachments: tuple[hsm.Instance, ...] = inner.attachments

    @typing.override
    async def measure(
        self, previous: change.Signature | None, image: media.Image, *, ctx: hsm.Context
    ) -> change.Measurement:
        small = await asyncio.to_thread(self._downscaled, image)
        return await self._inner.measure(previous, small, ctx=ctx)

    def _downscaled(self, image: media.Image) -> media.Image:
        with measured("downscale", change.Error):
            cv2 = self._cv2_loader()
            decoded = decode(cv2, image.content, cv2.IMREAD_COLOR)
            height, width = decoded.shape[:2]
            longer = max(height, width)
            if longer <= self._max_edge and image.media_type == _JPEG:
                return image
            if longer > self._max_edge:
                scale = self._max_edge / longer
                width, height = max(1, round(width * scale)), max(1, round(height * scale))
                decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_AREA)
            encoded, buffer = cv2.imencode(".jpg", decoded, (cv2.IMWRITE_JPEG_QUALITY, self._quality))
            if not encoded:
                raise Error("OpenCV could not encode the downscaled frame")
            return media.Image(content=buffer.tobytes(), media_type=_JPEG, width_px=width, height_px=height)


__all__ = ["Downscaled", "FrameDiff"]
