"""Audio peripherals for stateforward.mosfet devices."""

from . import codec, microphone, speaker
from .codec import PcmCodec
from .events import InputEvent, OutputEvent, InputData, OutputData
from .microphone import Microphone
from .speaker import Speaker

__all__ = [
    "InputEvent",
    "OutputEvent",
    "InputData",
    "OutputData",
    "PcmCodec",
    "Microphone",
    "Speaker",
    "codec",
    "microphone",
    "speaker",
]
