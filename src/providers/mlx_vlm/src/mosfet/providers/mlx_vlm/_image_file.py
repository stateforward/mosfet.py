from __future__ import annotations

import collections.abc
import contextlib
import os
import tempfile


@contextlib.contextmanager
def temporary_image_file(image: bytes, *, suffix: str) -> collections.abc.Generator[str, None, None]:
    """Write in-memory image bytes to a temporary file for file-oriented MLX VLM APIs."""

    with tempfile.NamedTemporaryFile(suffix=suffix) as image_file:
        _ = image_file.write(image)
        image_file.flush()
        os.fsync(image_file.fileno())
        yield image_file.name
