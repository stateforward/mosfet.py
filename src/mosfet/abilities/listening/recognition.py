"""Who a heard voice is: embed it, then ask the voice registry. Shared by both hearing paths.

Listening's local path (:class:`~mosfet.abilities.listening.interpretation.Interpretation`) and its
streaming path (:class:`~mosfet.abilities.listening.transcription.Transcription`) identify voices the
same way: the voice classifier ability embeds segments, and the ``identification.Registry`` the ear
owns says who each embedding is. Both are machines, reached here only by typed request/reply events
(``ability.run_terminal_operation``); the embeddings and prints never leave these functions.

Telemetry: ``bot.identification.count`` (``bot.identification.kind="voice"``, ``bot.outcome="failed"``,
``bot.failure.kind`` ``timeout`` or ``invalid_input``) counts the prints the ear could not have
identified: an embedding with no model or an invalid vector, or an identify the registry did not answer
in time. Failures the registry answers it counts itself. A segment the classifier could not fingerprint
(no voiceprint, W3-13) is not a failure: it is counted as ``bot.outcome="skipped"`` with
``bot.failure.kind="none"`` and stays unidentified.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import datetime
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from .. import ability
from ..hearing import voice
from ..identity import value
from . import enrollment
from mosfet.abilities import identification

_MAX_PRINTS = 16
MAX_PARTICIPANTS = 32
"""The most participants a listening product carries: the device-stamped ones, then the voices heard."""
_IDENTIFICATION = metrics.get_meter("bot.abilities.listening").create_counter(
    "bot.identification.count",
    unit="{print}",
    description="Prints identified or enrolled, by kind and outcome; failed requests count once.",
)

type Embedder = voice.identification.VoiceIdentification
"""The voice classifier ability a hearing child composes: segments in, one optional embedding per segment out."""


def merge(*groups: tuple[mosfet.Participant, ...]) -> tuple[mosfet.Participant, ...]:
    """Ordered union of stamped participants: first occurrence wins, bounded like a product's field."""

    return tuple(dict.fromkeys(participant for group in groups for participant in group))[:MAX_PARTICIPANTS]


def stamps(voices: tuple[voice.HeardVoice, ...]) -> tuple[frozenset[str], tuple[mosfet.Participant, ...]]:
    """What a product stamps for the voices heard in it, earliest first.

    Every voice is a ``voice:<id>`` participant, the bot's own included and stamped ``own``, so hearing
    itself resolves to itself; only the others are ``source_ids``, at most ``MAX_IDENTITY_SET_SIZE`` of
    them.
    """

    ordered = sorted(voices, key=lambda heard: heard.start_seconds)
    others = dict.fromkeys(
        identification.ref("voice", heard.recognition.id) for heard in ordered if not heard.recognition.own
    )
    participants = tuple(
        identification.participant("voice", heard.recognition.id, own=heard.recognition.own) for heard in ordered
    )
    return frozenset(tuple(others)[: value.MAX_IDENTITY_SET_SIZE]), participants


def unidentified(count: int, failure: identification.FailureKind) -> None:
    """Count prints the ear could not have identified; the registry counts the failures it answers."""

    if count:
        _IDENTIFICATION.add(
            count,
            {"bot.identification.kind": "voice", "bot.outcome": "failed", "bot.failure.kind": failure.value},
        )


def _skipped(count: int) -> None:
    """Count segments the classifier gave no voiceprint: unidentified by design, not by failure."""

    if count:
        _IDENTIFICATION.add(
            count, {"bot.identification.kind": "voice", "bot.outcome": "skipped", "bot.failure.kind": "none"}
        )


def _print(embedding: voice.identification.VoiceEmbedding) -> identification.Print | None:
    """The print an embedding identifies by, or ``None`` when it has no model or no valid vector."""

    if embedding.model is None:
        return None
    try:
        return identification.Print(vector=embedding.embedding, model=embedding.model)
    except pydantic.ValidationError:
        return None


async def embed(
    context: hsm.Context,
    embedder: Embedder,
    segments: tuple[voice.VoiceSegment, ...],
    *,
    timeout: datetime.timedelta,
    operation_id: str,
    metadata: collections.abc.Mapping[str, typing.Any],
) -> tuple[voice.identification.VoiceEmbedding | None, ...] | None:
    """One entry per segment, in order (``None`` for a segment with no voiceprint), or ``None`` when the
    classifier failed or did not answer one entry per segment.

    Raises:
        TimeoutError: The classifier did not answer within ``timeout``.
    """

    embedded = await ability.run_terminal_operation(
        context,
        child=embedder,
        request=dataclasses.replace(
            embedder.input_event.with_data_and_id(
                voice.identification.InputData(segments=segments), f"{operation_id}:embedding:{uuid.uuid4().hex}"
            ),
            metadata=dict(metadata),
        ),
        terminals=(embedder.output_event, embedder.failed_event),
        timeout=timeout,
    )
    output = embedded.data
    if not isinstance(output, voice.identification.OutputData) or len(output.embeddings) != len(segments):
        return None
    return output.embeddings


async def identify(
    context: hsm.Context,
    registry: identification.Registry,
    segments: tuple[voice.VoiceSegment, ...],
    embeddings: tuple[voice.identification.VoiceEmbedding | None, ...],
    *,
    match_only: bool,
    timeout: datetime.timedelta,
    operation_id: str,
    metadata: collections.abc.Mapping[str, typing.Any],
) -> tuple[voice.HeardVoice | None, ...]:
    """Who was heard in each segment, in order: one identify request to the registry, bounded by ``timeout``.

    A segment with no voiceprint, whose embedding cannot be printed, or that the registry does not answer
    in time, is ``None``. With ``match_only`` (an overlap: the bot's own voice is in that audio too), a voice
    nobody knows is ``None`` as well, and the registry makes no new voice from it.
    """

    fingerprinted = [(index, embedding) for index, embedding in enumerate(embeddings) if embedding is not None]
    _skipped(len(embeddings) - len(fingerprinted))
    printed = [(index, _print(embedding)) for index, embedding in fingerprinted]
    indexed = [(index, print_) for index, print_ in printed if print_ is not None][:_MAX_PRINTS]
    unidentified(len(fingerprinted) - len(indexed), identification.FailureKind.INVALID_INPUT)
    voices: list[voice.HeardVoice | None] = [None] * len(segments)
    if not indexed:
        return tuple(voices)
    try:
        terminal = await ability.run_terminal_operation(
            context,
            child=registry,
            request=dataclasses.replace(
                identification.IdentifyEvent.with_data_and_id(
                    identification.IdentifyData(
                        prints=tuple(print_ for _, print_ in indexed),
                        mint=(not match_only,) * len(indexed),
                    ),
                    f"{operation_id}:recognize:{uuid.uuid4().hex}",
                ),
                metadata=dict(metadata),
            ),
            terminals=(identification.IdentifiedEvent, identification.FailedEvent),
            timeout=timeout,
        )
    except TimeoutError:
        unidentified(len(indexed), identification.FailureKind.TIMEOUT)
        return tuple(voices)
    reply = terminal.data
    if (
        not isinstance(reply, identification.IdentifiedData)
        or reply.kind != "voice"
        or len(reply.recognitions) != len(indexed)
    ):
        # A FailedEvent was counted by the registry; a reply from a registry of another kind names nobody here.
        return tuple(voices)
    for (index, _), recognition in zip(indexed, reply.recognitions, strict=True):
        if recognition is not None:
            segment = segments[index]
            voices[index] = voice.HeardVoice(
                recognition=recognition, start_seconds=segment.start_seconds, end_seconds=segment.end_seconds
            )
    return tuple(voices)


async def enroll(
    context: hsm.Context,
    embedder: Embedder,
    registry: identification.Registry,
    copy: enrollment.EnrollData,
    *,
    timeout: datetime.timedelta,
    operation_id: str,
    metadata: collections.abc.Mapping[str, typing.Any],
) -> identification.Recognition | identification.FailureKind:
    """Embed the mouth's copy of the bot's own voice and enroll it as its own.

    The copy's ``voice_id``, once learned, is the id it refines; without one, identification matches
    the nearest own voice or mints one. Returns the registry's recognition, whose ``id`` is the bot's
    own voice id, or the failure kind.
    """

    try:
        embeddings = await embed(
            context,
            embedder,
            (copy.segment,),
            timeout=timeout,
            operation_id=f"{operation_id}:enroll",
            metadata=metadata,
        )
    except TimeoutError:
        return identification.FailureKind.TIMEOUT
    embedding = None if embeddings is None else embeddings[0]
    print_ = None if embedding is None else _print(embedding)
    if print_ is None:
        return identification.FailureKind.INVALID_INPUT
    try:
        enrolled = await ability.run_terminal_operation(
            context,
            child=registry,
            request=dataclasses.replace(
                identification.EnrollEvent.with_data_and_id(
                    identification.EnrollData(id=copy.voice_id, print=print_, own=True),
                    f"{operation_id}:enroll:{uuid.uuid4().hex}",
                ),
                metadata=dict(metadata),
            ),
            terminals=(identification.EnrolledEvent, identification.FailedEvent),
            timeout=timeout,
        )
    except TimeoutError:
        return identification.FailureKind.TIMEOUT
    reply = enrolled.data
    if isinstance(reply, identification.Recognition):
        return reply
    if isinstance(reply, identification.FailureData):
        return reply.kind
    return identification.FailureKind.STORE_FAILED


__all__ = ["MAX_PARTICIPANTS", "Embedder", "embed", "enroll", "identify", "merge", "stamps", "unidentified"]
