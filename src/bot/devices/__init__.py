"""Concrete devices for stateforward.bot.

Import device domains and qualify symbols so the namespace is present:

    from bot.devices import audio, phone

    phone.Phone
    phone.ServiceMediaReadyEvent
    audio.Microphone
"""

from . import audio, phone

__all__ = ["audio", "phone"]
