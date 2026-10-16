"""Speech production: a reply as structured parts, and streaming speech encoding of the bot's own voice."""

from . import encoding, part
from .encoding import Audio, Format, SpeechEncoder, Text
from .part import Cue, Expression, Expressiveness, Locale, Part, Piece, Range, Sound, said

__all__ = [
    "Audio",
    "Cue",
    "Expression",
    "Expressiveness",
    "Format",
    "Locale",
    "Part",
    "Piece",
    "Range",
    "Sound",
    "SpeechEncoder",
    "Text",
    "encoding",
    "part",
    "said",
]
