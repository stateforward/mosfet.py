from __future__ import annotations

import collections.abc
import contextlib
import os
import tempfile
import typing

import mosfet.media as media


_SUFFIXES: typing.Final[collections.abc.Mapping[media.ImageMediaType, str]] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


@contextlib.contextmanager
def temporary_image_file(image: media.Image) -> collections.abc.Generator[str, None, None]:
    """Write a typed image to a temporary file, named by its media type, for file-oriented MLX VLM APIs."""

    with tempfile.NamedTemporaryFile(suffix=_SUFFIXES[image.media_type]) as image_file:
        _ = image_file.write(image.content)
        image_file.flush()
        os.fsync(image_file.fileno())
        yield image_file.name
