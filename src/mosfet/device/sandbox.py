from dataclasses import dataclass


@dataclass(frozen=True)
class Sandbox:
    """Isolation boundary where a device service runs."""

    name: str
