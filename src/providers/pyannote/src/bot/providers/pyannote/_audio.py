from __future__ import annotations

import contextlib
import collections.abc
import pathlib
import tempfile


@contextlib.contextmanager
def temporary_audio_file(audio: bytes, *, suffix: str) -> collections.abc.Generator[pathlib.Path, None, None]:
    """Write encoded audio to a private temporary path for a provider call."""

    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        _ = handle.write(audio)
        handle.flush()
        yield pathlib.Path(handle.name)
