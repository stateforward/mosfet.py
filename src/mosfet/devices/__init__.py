"""Concrete devices for stateforward.mosfet.

Import device domains and qualify symbols so the namespace is present:

    from mosfet.devices import audio, computer, hid, phone

    phone.Phone
    phone.ServiceMediaReadyEvent
    audio.Microphone
    hid.Keyboard
    computer.Computer
"""

from . import audio, computer, hid, phone, smart_phone

__all__ = ["audio", "computer", "hid", "phone", "smart_phone"]
