"""Audio peripherals for stateforward.bot devices."""

from . import microphone, speaker
from .events import InputEvent, OutputEvent, AudioInputData, AudioOutputData
from .microphone import Microphone
from .speaker import Speaker

__all__ = [
    "InputEvent",
    "OutputEvent",
    "AudioInputData",
    "AudioOutputData",
    "Microphone",
    "Speaker",
    "microphone",
    "speaker",
]
