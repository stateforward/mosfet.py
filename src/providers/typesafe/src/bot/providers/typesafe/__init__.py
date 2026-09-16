"""TypeSafe AI provider: system_one label-tier selection for stateforward.bot.

System One answers label questions over a state document — Choice (one criterion),
Score (graded, anchored), and Noul (typed likelihood). It is not a tool-calling
models: it cannot author event payloads. The provider therefore maps the bot's
Intuition reflex tier (per-event Noul selection, aggregate confidence) onto
`system_one`, and stays out of payload-authoring tiers.

The selection budget is strict: the processor never authors payload fields. A
chosen event with required payload keys is dispatched with `data=None`, and the
host's dispatch validation is the honest boundary (typed rejection → cascade).
"""

from .client import AsyncSystemOneClient, SystemOneError
from .processing import Processor, ProcessingError

__version__ = "0.1.0"

__all__ = [
    "AsyncSystemOneClient",
    "ProcessingError",
    "Processor",
    "SystemOneError",
    "__version__",
]
