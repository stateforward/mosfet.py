"""The bot's voice: canonical voice schemas, and the voice a bot is given at inception and keeps for life.

Core names a voice by ``Ref`` and describes it with ``Entry``, ``Descriptors`` and ``Preview``; a source that designs
voices takes a ``DesignRequest`` and answers ``Designed``. ``Catalog`` and ``Designer`` are the collaborators a
provider implements. ``Inception`` gives a bot its voice the first time it starts, from its traits, and gives it the
same voice on every start after; it tells its owner ``OutputEvent`` (``VoiceData``).

    from mosfet.abilities import vocal

    vocal.voice.Inception(traits=..., languages=("en",), speaking=..., processor=..., catalog=..., memory=...)

``Inception`` (module ``inception``) is exported lazily: it reaches Speaking by message, and Speaking's own imports
reach Cognition, which knows the voice it is told by these schemas.
"""

import importlib
import typing

from . import catalog, conception, design, given, voice
from .catalog import Catalog, Descriptors, Entry, Error, FailureKind, Preview, Ref
from .conception import ConceiveData, ConceiveEvent, Conception, Described, InceptionInput, Shortlisted
from .design import DesignRequest, Designed, Designer
from .given import VOICE_QUERY, Given, given_insert_input, given_select_input, recalled_given
from .voice import OutputEvent, VoiceData

if typing.TYPE_CHECKING:
    from . import inception
    from .inception import Inception

_LAZY_EXPORT_MODULES = {"Inception": ".inception", "inception": ".inception"}

__all__ = [
    "VOICE_QUERY",
    "Catalog",
    "ConceiveData",
    "ConceiveEvent",
    "Conception",
    "DesignRequest",
    "Described",
    "Descriptors",
    "Designed",
    "Designer",
    "Entry",
    "Error",
    "FailureKind",
    "Given",
    "Inception",
    "InceptionInput",
    "OutputEvent",
    "Preview",
    "Ref",
    "Shortlisted",
    "VoiceData",
    "catalog",
    "conception",
    "design",
    "given",
    "given_insert_input",
    "given_select_input",
    "inception",
    "recalled_given",
    "voice",
]


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = module if name == "inception" else typing.cast(object, getattr(module, name))
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
