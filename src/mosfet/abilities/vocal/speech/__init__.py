"""Speech production: streaming speech encoding of the bot's own voice."""

from . import encoding
from .encoding import Audio, Format, SpeechEncoder, Text

__all__ = ["Audio", "Format", "SpeechEncoder", "Text", "encoding"]
