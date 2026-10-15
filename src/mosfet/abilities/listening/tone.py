"""How a heard voice sounds: the affect of a speech product's longest voice segment. Shared by both hearing paths.

The text speech products Listening hands on (decoded speech on the local path, a turn on the
streaming path) carry the tone they were said in when an affect classifier is composed
(``SpeechData.affect``). The classifier hears one segment: the longest voice in the product, which is
the whole utterance or turn unless a diarizer split it. The hearing child composes the classifier as a
classifying ability and asks it here by typed request and reply, bounded by a modeled timeout; a tone
not heard in time, or not heard at all, leaves the product without one. Nothing waits on it beyond that.

Telemetry: ``bot.listening.affect.count`` counts each classification by ``bot.outcome`` (``ok`` or
``failed``) and, when failed, ``bot.failure.kind``: ``timeout`` (not answered within the bound),
``stage_failed`` (the classifier answered with a failure) or ``invalid_response`` (it answered with
something that is not an affect). At most four series; never labels, scores or audio.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import datetime
import typing
import uuid

import hsm
from opentelemetry import metrics

from mosfet.telemetry import span

from .. import ability
from .. import classifying
from ..hearing import voice

_AFFECT = metrics.get_meter("bot.abilities.listening").create_counter(
    "bot.listening.affect.count",
    unit="{product}",
    description="Speech products whose tone was classified, by outcome and, when failed, failure kind.",
)

type Classification = classifying.Classifying[voice.VoiceSegment, voice.Affect]
"""The affect classifier ability a hearing child composes: one voice segment in, its affect out."""


def _count(kind: typing.Literal["timeout", "stage_failed", "invalid_response"] | None) -> None:
    _AFFECT.add(1, {"bot.outcome": span.Ok} if kind is None else {"bot.outcome": span.Failed, "bot.failure.kind": kind})


async def classify(
    context: hsm.Context,
    classification: Classification | None,
    segments: tuple[voice.VoiceSegment, ...],
    *,
    timeout: datetime.timedelta,
    operation_id: str,
    metadata: collections.abc.Mapping[str, typing.Any],
) -> voice.Affect | None:
    """The affect of the longest of ``segments`` (the first of equals), or ``None`` when it was not heard.

    ``None`` without asking when no affect classifier is composed (``classification`` is ``None``) or
    there are no segments; otherwise one request, answered within ``timeout`` or not at all.
    """

    if classification is None or not segments:
        return None
    longest = max(segments, key=lambda segment: segment.duration_seconds)
    try:
        terminal = await ability.run_terminal_operation(
            context,
            child=classification,
            request=dataclasses.replace(
                classification.input_event.with_data_and_id(longest, f"{operation_id}:affect:{uuid.uuid4().hex}"),
                metadata=dict(metadata),
            ),
            terminals=(classification.output_event, classification.failed_event),
            timeout=timeout,
        )
    except TimeoutError:
        _count("timeout")
        return None
    affect = typing.cast(object, terminal.data)
    if terminal.name == classification.failed_event.name:
        _count("stage_failed")
        return None
    if not isinstance(affect, voice.Affect):
        _count("invalid_response")
        return None
    _count(None)
    return affect


__all__ = ["Classification", "classify"]
