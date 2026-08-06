from __future__ import annotations

import contextlib
import collections.abc
import os
import pathlib
import tempfile


def configure_headless_matplotlib() -> None:
    """Force a noninteractive backend before importing pyannote's Matplotlib users."""

    # pyannote does not render plots. Override inherited notebook/GUI settings so this provider
    # remains usable in a headless phone process without an operator-managed environment export.
    os.environ["MPLBACKEND"] = "Agg"


@contextlib.contextmanager
def temporary_audio_file(audio: bytes, *, suffix: str) -> collections.abc.Generator[pathlib.Path, None, None]:
    """Write encoded audio to a private temporary path for a provider call."""

    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        _ = handle.write(audio)
        handle.flush()
        yield pathlib.Path(handle.name)
