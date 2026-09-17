from __future__ import annotations

import collections.abc
import contextlib
import os
import tempfile


@contextlib.contextmanager
def temporary_audio_file(audio: bytes, *, suffix: str) -> collections.abc.Generator[str, None, None]:
    """Write in-memory audio bytes to a temporary file for file-oriented MLX Audio APIs."""

    with tempfile.NamedTemporaryFile(suffix=suffix) as audio_file:
        _ = audio_file.write(audio)
        audio_file.flush()
        os.fsync(audio_file.fileno())
        yield audio_file.name
