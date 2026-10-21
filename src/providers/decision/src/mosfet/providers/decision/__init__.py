"""Decision-model provider: reflex's decision layer decides on any server that speaks the decisions wire format.

A decision server answers named questions about one state document with calibrated
probabilities (``POST {base_url}/v1/systemone``). anygame's decide serves it locally, and a hosted
server with the same wire format works the same way, given its ``api_key``. It is not a
tool-calling model and never authors a payload, so `Processor` is a
`processing.DecisionProcessor`, the kind reflex's decision layer runs on: it chooses one of the offered events,
fills payload keys only by choosing among values the turn's own evidence supplies, and declines a
turn it cannot place to the next tier.

`Client` is the HTTP boundary, and `Request`, `Choice`, `Response` and `ChoiceAnswer` are the wire
models it speaks. Every client failure is an `Error`, which `Processor` raises as a
`ProcessingError`.
"""

from .client import Choice, ChoiceAnswer, Client, Error, Request, Response
from .processing import PayloadGroundingError, ProcessingError, Processor

__version__ = "0.1.0"

__all__ = [
    "Choice",
    "ChoiceAnswer",
    "Client",
    "Error",
    "PayloadGroundingError",
    "ProcessingError",
    "Processor",
    "Request",
    "Response",
    "__version__",
]
