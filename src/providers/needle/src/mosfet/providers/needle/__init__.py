"""Cactus Compute Needle provider: local tool calling for the Intuition reflex tier.

Needle 3 (Apache-2.0, 121M parameters laddered 2-20 layers, shipped as a 2-bit ``.cact``
archive of 8-29 MB) answers every turn as function calls under a byte-level grammar compiled
from the offered tool schemas, with one calibrated confidence per response. Unlike a label
tier it authors payloads: arguments are grammar-constrained to each event's JSON schema, and
the engine's own grounding gates withhold calls it cannot ground (``suppressed_calls``).

The processor offers each selectable event as one Needle tool, maps returned calls to
``SelectedEvent`` values, and returns Intuition's explicit unhandled envelope when the engine
calls nothing, so the host cascades to deliberative reasoning.
"""

from .engine import Engine, EngineError, LocalEngine
from .processing import ProcessingError, Processor

__version__ = "0.1.0"

__all__ = [
    "Engine",
    "EngineError",
    "LocalEngine",
    "ProcessingError",
    "Processor",
    "__version__",
]
