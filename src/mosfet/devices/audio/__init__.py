"""Audio peripherals for stateforward.mosfet devices."""

from . import codec, microphone, speaker
from .codec import PcmCodec
from .events import (
    FlushData,
    FlushEvent,
    FlushedData,
    FlushedEvent,
    InputData,
    InputEvent,
    OutputData,
    OutputEvent,
    PlayedData,
    PlayedEvent,
)
from .microphone import Microphone
from .speaker import Speaker

__all__ = [
    "InputEvent",
    "OutputEvent",
    "FlushEvent",
    "PlayedEvent",
    "FlushedEvent",
    "InputData",
    "OutputData",
    "FlushData",
    "PlayedData",
    "FlushedData",
    "PcmCodec",
    "Microphone",
    "Speaker",
    "codec",
    "microphone",
    "speaker",
]
