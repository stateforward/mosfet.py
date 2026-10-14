"""Human Interface Device ports for stateforward.mosfet devices.

A keyboard and a pointer are remote-side transducers: each owns a stream that delivers what a person
did on the remote hardware, and the port hands it to whatever is attached to it. Neither hears the environment.
"""

from . import keyboard, pointer
from .events import KeyData, KeyEvent, PointerData, PointerEvent
from .keyboard import Keyboard
from .pointer import Pointer

__all__ = [
    "KeyData",
    "KeyEvent",
    "Keyboard",
    "PointerData",
    "PointerEvent",
    "Pointer",
    "keyboard",
    "pointer",
]
