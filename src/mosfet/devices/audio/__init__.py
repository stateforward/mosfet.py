"""Audio peripherals for stateforward.mosfet devices."""

from . import microphone, speaker
from .events import InputEvent, OutputEvent, InputData, OutputData
from .microphone import Microphone
from .speaker import Speaker

__all__ = [
    "InputEvent",
    "OutputEvent",
    "InputData",
    "OutputData",
    "Microphone",
    "Speaker",
    "microphone",
    "speaker",
]
