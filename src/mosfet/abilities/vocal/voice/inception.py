"""Inception: the voice a bot is given when it is made, kept for life.

A bot's voice is its makers' to give, not its own choice (P-6, P-14). The first time a bot starts there is no voice
recorded for its source, so Inception asks the inception model for **one** conception of it from the bot's traits
(P-11): a voice described for a source that designs voices, or one voice shortlisted from the catalog of a source that
does not. The source makes it permanent, the given voice is written down once (fail-closed), and Speaking is told to
speak in it (``speaking.VoiceEvent``); Speaking constructs its encoder from it exactly once (W3-22). What that voice
can do with an expression comes back with ``speaking.VoicedEvent`` and is told to the owner as :data:`OutputEvent`.

On every later start the voice is recalled and the source is checked to still hold it; no model is asked anything. A
voice recorded only for another source is **estranged** (P-13), and so is one the source no longer holds (it is
**lost**; the lab TTS silently falls back to another voice for an unknown one, which would change the bot's voice):
either is a failure, the bot stays unvoiced, and it is never re-made automatically. Nothing about the voice changes
after inception; how a reply is delivered within it does (``vocal.speech.Expression``).

A bot is born only from its **view** of the catalog. With a ``designer`` that is the one voice it designs itself,
which its memory keeps (the ``self_voice`` row, recalled on every start), and no voice of its source is ``allowed``
beside it. Without one it is the voices whose keys its owner ``allowed``; by default none, so such a bot must be
allowed a voice to be born at all. Nothing else its source holds is ever offered to it, such as the voices another
maker designed on a shared engine. The view bounds only what a bot is born as: a voice it was already given is
checked against the whole catalog, and stays its own for life.

A designing source's frame offers a private copy of :data:`conception.ConceiveEvent`, of the same name, whose described
description is bounded by the designer's ``max_description`` (D-R-35), so the limit is in the schema the model writes
against. The conception is mechanical to check and nothing else: a conception of the wrong kind, a description longer
than the source takes, or a ref outside the view the model was shown is refused (``invalid_response``). A refused
conception is conceived again at once, the model shown why (``InceptionInput.refused``, D-R-35), at most twice (the
D-R-12 repair bound); a third refusal fails the attempt, which is conceived afresh after ``retry_after`` like any other.
Inception enrolls nothing: the bot learns its own voice id by hearing itself (P-12).

A bot without memory remembers no voice, so every start of it is an inception.

Once Speaking has taken the voice it speaks in it for life, and a second ``speaking.VoiceEvent`` is only a counted drop:
an Inception attached again, or left attached by a detach that failed, keeps the voice it gave and asks nothing again.

Telemetry (scope ``bot.abilities.voice``; every attribute value comes from code):

* ``bot.voice.inception.count`` — ``bot.voice.origin`` ∈ {designed, catalog} × ``bot.outcome`` ∈ {made, failed}.
* ``bot.voice.inception.duration`` — histogram, ``bot.outcome`` ∈ {made, failed}.
* ``bot.voice.recall.count`` — ``bot.outcome`` ∈ {recalled, unmade, estranged, lost, failed}.
* spans ``bot.voice.conceive``, ``bot.voice.create``, ``bot.voice.check`` and ``bot.voice.voicing``, each with
  ``bot.failure.kind`` on failure. No attribute carries a ref, key, description or trait.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import time
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet.protocols import attachment
from mosfet.telemetry import span

from ... import ability
from ... import memory
from ... import persona
from ... import processing
from ... import speaking
from .. import speech
from . import catalog
from . import conception
from . import design
from . import given
from . import voice

_SCOPE = "bot.abilities.voice"
_COMPONENT = "voice.inception"
_DEFAULT_RETRY_AFTER = datetime.timedelta(seconds=30)
_DEFAULT_INCEPTION_TIMEOUT = datetime.timedelta(seconds=180)
# One recall question is a single-row read; a memory or speaker not up yet is asked again soon.
_RECALL_ATTEMPT_TIMEOUT = datetime.timedelta(seconds=1)
_RETRY_DELAY = datetime.timedelta(milliseconds=100)
# A refused conception is conceived again at once at most this many times (the D-R-12 repair bound).
_RECONCEPTIONS = 2

# The keys of a source's voices, as refs carry them.
_KEYS: pydantic.TypeAdapter[tuple[str, ...]] = pydantic.TypeAdapter(
    tuple[typing.Annotated[str, pydantic.StringConstraints(min_length=1, max_length=256)], ...]
)

_METER = metrics.get_meter(_SCOPE)
_INCEPTIONS = _METER.create_counter(
    "bot.voice.inception.count",
    unit="{inception}",
    description="Inception attempts by origin (designed or catalog) and outcome (made, or failed and retried).",
)
_INCEPTION_DURATION = _METER.create_histogram(
    "bot.voice.inception.duration",
    unit="s",
    description="How long one inception attempt took, from conceiving to the voice written down or failed.",
)
_RECALLS = _METER.create_counter(
    "bot.voice.recall.count",
    unit="{recall}",
    description=(
        "Voice recalls on starting by outcome: recalled (and still held), unmade (none yet), estranged (recorded only "
        "for another source), lost (no longer held by the source), or failed (asked again)."
    ),
)


class _RecalledData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    recalled: tuple[given.Given, ...] = ()


class _CheckedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    voice: given.Given
    present: bool


class _ConceivedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    conceived: conception.Conception
    # The catalog entry a shortlisted conception names; None for a described one.
    entry: catalog.Entry | None = None


class _GivenData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    voice: given.Given


class _VoicedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    expressiveness: speech.Expressiveness


class _FailedData(pydantic.BaseModel):
    """A private failure: why, for the owner (its closed kind is already on the step's span and counters)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    message: str


class _RefusedData(pydantic.BaseModel):
    """A refused conception: why, shown to the model when it conceives again, and how many tries were refused."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    message: str
    refusals: int = pydantic.Field(ge=1)


def _completion(name: str, schema: type[pydantic.BaseModel]) -> hsm.Event[typing.Any]:
    return hsm.Event[typing.Any](name=f"bot.ability.voice.{name}", kind=hsm.CompletionEventKind, schema=schema)


def _error(name: str) -> hsm.Event[typing.Any]:
    return hsm.Event[typing.Any](name=f"bot.ability.voice.{name}", kind=hsm.ErrorEventKind, schema=_FailedData)


_RecalledEvent = _completion("recalled", _RecalledData)
_RecallFailedEvent = _error("recall.failed")
_CheckedEvent = _completion("checked", _CheckedData)
_CheckFailedEvent = _error("check.failed")
_ConceivedEvent = _completion("conceived", _ConceivedData)
_CreatedEvent = _completion("created", _GivenData)
_RememberedEvent = _completion("remembered", _GivenData)
_AttemptFailedEvent = _error("attempt.failed")
_RefusedEvent = hsm.Event[_RefusedData](
    name="bot.ability.voice.conception.refused", kind=hsm.ErrorEventKind, schema=_RefusedData
)
_VoicedEvent = _completion("voiced", _VoicedData)
_SpeakingAbsentEvent = _error("speaking.absent")
_VoicingFailedEvent = _error("voicing.failed")


def _with_context(event: hsm.Event[typing.Any], source: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
    """Copy operation id, provenance, and metadata along the private event chain."""

    operation_id = source.id if source.id else uuid.uuid4().hex
    carried = event.with_data_and_id(event.data, operation_id)
    return dataclasses.replace(carried, source=source.source or event.source, metadata=dict(source.metadata))


def _payload_is(*types: type[pydantic.BaseModel]) -> collections.abc.Callable[..., bool]:
    """A pure guard: the event carries one of ``types``."""

    def carries(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, types)

    carries.__name__ = carries.__qualname__ = f"carries_{'_'.join(t.__name__.strip('_') for t in types)}"
    return carries


_carries_failure = _payload_is(_FailedData)


class Inception(ability.Ability[None, voice.VoiceData]):
    """Ability that gives a bot its voice when it is first made, and gives it the same voice on every start after.

    It takes no input: what it does is fixed by its makers' composition. ``catalog`` is the voice source; a
    ``designer`` of the same source makes the voice from a description, and without one the voice is shortlisted from
    the catalog. ``processor`` is the inception model, wrapped in a child ``processing.Processing`` with no
    instructions of its own. ``speaking`` and ``memory`` are peers reached only by message. A refused conception is
    conceived again at once, at most twice; each inception attempt that fails is retried after ``retry_after``;
    ``inception_timeout`` bounds each conception and voicing. ``allowed`` are the keys of the voices of a source
    without a ``designer`` that its owner allows it to be born as; none by default.

    Raises:
        ValueError: if ``designer`` names another source than ``catalog`` or is given with ``allowed`` keys, a
            language is not a locale, an allowed key is empty, or a timeout is not positive.
    """

    output_event: typing.ClassVar[hsm.Event[voice.VoiceData]] = voice.OutputEvent
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = voice.VoiceData
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _attachment_group: attachment.Group | None
    _traits: persona.Traits
    _languages: tuple[str, ...]
    _allowed: frozenset[str]
    _speaking: speaking.Speaking
    _processing: processing.Processing
    _catalog: catalog.Catalog
    _designer: design.Designer | None
    _memory: memory.MemoryStore | None
    _retry_after: datetime.timedelta
    _inception_timeout: datetime.timedelta
    # The voice the bot was given, once it is known: durable self-knowledge, fixed for life.
    _given: given.Given | None
    # Speaking took that voice: it speaks in it for life, so it is never given again.
    _voiced: bool
    # When the current inception attempt began (telemetry only).
    _attempt_started: float | None

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        traits: persona.Traits,
        languages: tuple[speech.Locale, ...],
        speaking: speaking.Speaking,
        processor: processing.Processor | collections.abc.Callable[[], processing.Processor],
        catalog: catalog.Catalog,
        designer: design.Designer | None = None,
        memory: memory.MemoryStore | None = None,
        retry_after: datetime.timedelta = _DEFAULT_RETRY_AFTER,
        inception_timeout: datetime.timedelta = _DEFAULT_INCEPTION_TIMEOUT,
        allowed: collections.abc.Iterable[str] = (),
    ) -> None:
        if designer is not None and designer.source != catalog.source:
            raise ValueError(
                f"Inception's designer makes voices for {designer.source!r}, not its catalog's {catalog.source!r}."
            )
        keys = frozenset(_KEYS.validate_python(tuple(allowed)))
        if designer is not None and keys:
            raise ValueError(
                "Inception's designer gives the bot the one voice it designs, so it is allowed no voice of its "
                + f"source beside that one, got {sorted(keys)}."
            )
        if retry_after <= datetime.timedelta(0):
            raise ValueError(f"Inception retry_after must be positive, got {retry_after}.")
        if inception_timeout <= datetime.timedelta(0):
            raise ValueError(f"Inception inception_timeout must be positive, got {inception_timeout}.")
        super().__init__(id=id, clock=clock)
        self._traits = traits
        self._languages = pydantic.TypeAdapter(tuple[speech.Locale, ...]).validate_python(languages)
        self._allowed = keys
        # Peers: Speaking and memory are attached by their own owners and reached only by message.
        self._speaking = speaking
        self._memory = memory
        leaf = processor if isinstance(processor, processing.Processor) else processor()
        # No instructions of its own: what the model is asked is the conceive schema and what it is shown.
        self._processing = processing.Processing(processor=leaf)
        self._attachment_group = attachment.Group(self._processing)
        self._catalog = catalog
        self._designer = designer
        self._retry_after = retry_after
        self._inception_timeout = inception_timeout
        self._given = None
        self._voiced = False
        self._attempt_started = None

    # --- telling -------------------------------------------------------------------------------

    @staticmethod
    def _fail(ctx: hsm.Context, instance: "Inception", source: hsm.Event[typing.Any], message: str) -> None:
        """Tell the owner one inception failure; the owner perceives it."""

        terminal = _with_context(instance.failed_event.with_data(ability.FailureData(message=message)), source)
        terminal = dataclasses.replace(terminal, source=hsm.id(instance))
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _step(
        ctx: hsm.Context,
        instance: "Inception",
        source: hsm.Event[typing.Any],
        step: hsm.Event[typing.Any],
        data: pydantic.BaseModel,
    ) -> None:
        _ = hsm.dispatch(ctx, instance, _with_context(step.with_data(data), source))

    @staticmethod
    def _origin(instance: "Inception") -> typing.Literal["designed", "catalog"]:
        return "catalog" if instance._designer is None else "designed"

    # --- recalling -----------------------------------------------------------------------------

    @staticmethod
    async def _recall(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        """Ask memory for every voice this bot was given; a bot with no memory remembers none."""

        store = instance._memory
        if store is None:
            _RECALLS.add(1, {"bot.outcome": "unmade"})
            Inception._step(ctx, instance, event, _RecalledEvent, _RecalledData())
            return
        try:
            output = await memory.transact(
                ctx, store, given.given_select_input(), metadata=dict(event.metadata), timeout=_RECALL_ATTEMPT_TIMEOUT
            )
        except memory.RequestError as error:
            _RECALLS.add(1, {"bot.outcome": "failed", "bot.failure.kind": "unavailable"})
            failure = _FailedData(message=f"Inception could not recall the bot's voice from memory: {error}")
            Inception._step(ctx, instance, event, _RecallFailedEvent, failure)
            return
        recalled = given.recalled_given(output)
        if not recalled:
            _RECALLS.add(1, {"bot.outcome": "unmade"})
        elif not any(row.ref.source == instance._catalog.source for row in recalled):
            _RECALLS.add(1, {"bot.outcome": "estranged"})
        Inception._step(ctx, instance, event, _RecalledEvent, _RecalledData(recalled=recalled))

    @staticmethod
    def _is_ours(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> bool:
        """A voice was recorded for this bot's own source."""

        del ctx
        data = event.data
        return isinstance(data, _RecalledData) and any(
            row.ref.source == instance._catalog.source for row in data.recalled
        )

    @staticmethod
    def _is_estranged(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> bool:
        """Voices were recorded, but only for other sources (checked after ``_is_ours``)."""

        del ctx, instance
        data = event.data
        return isinstance(data, _RecalledData) and bool(data.recalled)

    @staticmethod
    def _fail_estranged(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        Inception._fail(
            ctx,
            instance,
            event,
            (
                "Inception found the bot's voice recorded only for another voice source; it stays unvoiced and is "
                + "never made again automatically."
            ),
        )

    @staticmethod
    def _give_up_recalling(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        """The bring-up budget ran out with the voice still not recalled: say so, and keep asking."""

        _RECALLS.add(1, {"bot.outcome": "failed", "bot.failure.kind": "timeout"})
        Inception._fail(
            ctx,
            instance,
            event,
            "Inception could not recall the bot's voice within the bring-up budget; it is still asking.",
        )

    # --- checking ------------------------------------------------------------------------------

    @staticmethod
    async def _check(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        """Ask the source whether it still holds the recalled voice, so the voice can never change silently."""

        data = event.data
        if not isinstance(data, _RecalledData):
            # Unreachable: checking is entered only on a recall naming this source's voice.
            return
        recalled = [row for row in data.recalled if row.ref.source == instance._catalog.source][-1]
        with span.operation("bot.voice.check", scope=_SCOPE, component=_COMPONENT, stage="check") as active:
            try:
                entries = await instance._catalog.voices()
            except Exception as error:
                # A documented boundary: whatever the source raised, the voice is not known lost, only unchecked.
                kind = span.failure_kind(error, "unavailable")
                span.record_failure(active, kind)
                _RECALLS.add(1, {"bot.outcome": "failed", "bot.failure.kind": kind})
                failure = _FailedData(message=f"Inception could not check the bot's voice: {error}")
                Inception._step(ctx, instance, event, _CheckFailedEvent, failure)
                return
            present = any(entry.ref == recalled.ref for entry in entries)
            if not present:
                span.record_failure(active, "unsupported_input")
        _RECALLS.add(1, {"bot.outcome": "recalled" if present else "lost"})
        Inception._step(ctx, instance, event, _CheckedEvent, _CheckedData(voice=recalled, present=present))

    @staticmethod
    def _is_present(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _CheckedData) and data.present

    @staticmethod
    def _fail_lost(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        Inception._fail(
            ctx,
            instance,
            event,
            (
                "Inception found the bot's voice is no longer held by its voice source; it stays unvoiced and is "
                + "never made again automatically."
            ),
        )

    # --- conceiving, creating, remembering -----------------------------------------------------

    @staticmethod
    def _begin_attempt(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._attempt_started = time.monotonic()  # telemetry only

    @staticmethod
    def _settle_attempt(instance: "Inception", outcome: typing.Literal["made", "failed"]) -> None:
        _INCEPTIONS.add(1, {"bot.voice.origin": Inception._origin(instance), "bot.outcome": outcome})
        if instance._attempt_started is not None:
            _INCEPTION_DURATION.record(time.monotonic() - instance._attempt_started, {"bot.outcome": outcome})
            instance._attempt_started = None

    @staticmethod
    async def _shown(instance: "Inception", refused: str | None) -> conception.InceptionInput:
        """What the inception model is shown: the traits, the languages, the source's capacity, and why its last
        conception was refused when it was; nothing else.

        A catalog source's capacity is the voices of it the bot's owner allowed. Previews are never shown: a
        model-facing input carries no audio.
        """

        designer = instance._designer
        traits = instance._traits.traits
        if designer is not None:
            return conception.InceptionInput(
                traits=traits,
                languages=instance._languages,
                description_max_chars=designer.max_description,
                refused=refused,
            )
        listed = [
            entry
            for entry in await instance._catalog.voices(languages=instance._languages)
            if entry.ref.key in instance._allowed
        ]
        if not listed:
            raise catalog.Error(
                "unsupported_input", "The voice source's catalog holds no voices this bot may be born as."
            )
        shown = tuple(entry.model_copy(update={"preview": None}) for entry in listed[: conception.MAX_ENTRIES])
        return conception.InceptionInput(
            traits=traits,
            languages=instance._languages,
            entries=shown,
            omitted=len(listed) - len(shown),
            refused=refused,
        )

    @staticmethod
    def _conceived(shown: conception.InceptionInput, output: processing.OutputData) -> _ConceivedData:
        """The one conception the model selected, checked mechanically against what it was shown.

        Raises:
            catalog.Error: ``invalid_response`` for anything but exactly one conception of the kind and bounds asked.
        """

        selections = processing.coerce_event_selections(output) or ()
        if len(selections) != 1 or selections[0].event != conception.ConceiveEvent.name:
            raise catalog.Error("invalid_response", "The inception model must conceive exactly one voice.")
        selected = selections[0].data
        try:
            answer = (
                selected
                if isinstance(selected, conception.ConceiveData)
                else conception.ConceiveData.model_validate(selected or {})
            )
        except pydantic.ValidationError as error:
            raise catalog.Error(
                "invalid_response", f"The inception model conceived an invalid voice: {error}"
            ) from error
        conceived = answer.conception
        bound = shown.description_max_chars
        if bound is not None:
            if not isinstance(conceived, conception.Described):
                raise catalog.Error("invalid_response", "The voice source designs voices: the voice must be described.")
            if len(conceived.description) > bound:
                raise catalog.Error(
                    "invalid_response", f"The voice description is longer than the source's {bound} characters."
                )
            return _ConceivedData(conceived=conceived)
        if not isinstance(conceived, conception.Shortlisted):
            raise catalog.Error("invalid_response", "The voice source does not design voices: shortlist an entry.")
        entry = next((entry for entry in shown.entries if entry.ref == conceived.ref), None)
        if entry is None:
            raise catalog.Error("invalid_response", "The shortlisted voice is not one of the catalog entries shown.")
        return _ConceivedData(conceived=conceived, entry=entry)

    @staticmethod
    async def _conceive(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        """Ask the inception model for one conception of the bot's voice, from its traits alone.

        Conceiving again after a refusal (``_RefusedData``), the model is shown why its last conception was refused.
        """

        child = instance._processing
        operation_id = uuid.uuid4().hex
        prior = event.data if isinstance(event.data, _RefusedData) else None
        with span.operation("bot.voice.conceive", scope=_SCOPE, component=_COMPONENT, stage="conceive") as active:
            try:
                shown = await Inception._shown(instance, None if prior is None else prior.message)
                offered: hsm.Event[typing.Any] = conception.ConceiveEvent
                bound = shown.description_max_chars
                if bound is not None:
                    # D-R-35: a designing source's limit is in the schema the model writes against, not only a
                    # number it is shown: a private copy of the conceive event, its description bounded by it.
                    described = pydantic.create_model(
                        "Described",
                        __base__=conception.Described,
                        description=(
                            typing.Annotated[
                                str, conception.Described.model_fields["description"], pydantic.Field(max_length=bound)
                            ],
                            ...,
                        ),
                    )
                    offered = hsm.Event[conception.ConceiveData](
                        name=conception.ConceiveEvent.name,
                        kind=conception.ConceiveEvent.kind,
                        schema=pydantic.create_model(
                            "ConceiveData",
                            __base__=conception.ConceiveData,
                            conception=(
                                typing.Annotated[
                                    described | conception.Shortlisted, pydantic.Field(discriminator="kind")
                                ],
                                conception.ConceiveData.model_fields["conception"],
                            ),
                        ),
                    )
                frame = processing.InputData(input=shown, schemas=(offered,), actors={})
                terminal = await ability.run_terminal_operation(
                    ctx,
                    child=child,
                    request=dataclasses.replace(
                        child.input_event.with_data_and_id(frame, operation_id), metadata=dict(event.metadata)
                    ),
                    terminals=(child.output_event, child.failed_event),
                    timeout=instance._inception_timeout,
                )
                completed = terminal.data
                if not isinstance(completed, processing.CompletionData):
                    failure = completed.message if isinstance(completed, ability.FailureData) else "no answer"
                    raise catalog.Error("unavailable", f"The inception model could not conceive a voice: {failure}")
                try:
                    conceived = Inception._conceived(shown, completed.output)
                except catalog.Error as refusal:
                    # Refused, not failed: the model answered outside what it was asked, and is told why.
                    span.record_failure(active, refusal.kind)
                    refused = _RefusedData(message=str(refusal), refusals=1 if prior is None else prior.refusals + 1)
                    Inception._step(ctx, instance, event, _RefusedEvent, refused)
                    return
            except (asyncio.CancelledError, TimeoutError) as error:
                # Conceiving was left, or timed out, before the model answered: stop it, so a late conception
                # never lands.
                processing.dispatch_child_cancel(
                    instance.context(),
                    instance,
                    child,
                    event,
                    request_id=operation_id,
                    parent_operation_id=operation_id,
                    token=uuid.uuid4().hex,
                )
                if isinstance(error, asyncio.CancelledError):
                    raise
                span.record_failure(active, "timeout")
                failure = _FailedData(message="Inception could not conceive the bot's voice in time.")
                Inception._step(ctx, instance, event, _AttemptFailedEvent, failure)
                return
            except Exception as error:
                # A documented boundary: whatever the source or model raised, this attempt failed and is retried.
                kind = span.failure_kind(error, "unavailable")
                span.record_failure(active, kind)
                failure = _FailedData(message=f"Inception could not conceive the bot's voice: {error}")
                Inception._step(ctx, instance, event, _AttemptFailedEvent, failure)
                return
        Inception._step(ctx, instance, event, _ConceivedEvent, conceived)

    @staticmethod
    async def _create(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        """Have the source make the conceived voice permanent; a catalog voice already is."""

        data = event.data
        if not isinstance(data, _ConceivedData):
            # Unreachable: creating is entered only on a conception.
            return
        conceived = data.conceived
        traits = instance._traits.traits
        with span.operation("bot.voice.create", scope=_SCOPE, component=_COMPONENT, stage="create") as active:
            try:
                designer = instance._designer
                if isinstance(conceived, conception.Described):
                    assert designer is not None, "a described voice is accepted only from a designing source"
                    request = design.DesignRequest(
                        description=conceived.description,
                        descriptors=conceived.descriptors,
                        languages=conceived.languages or instance._languages,
                    )
                    designed = await designer.create(request)
                    if designed.ref.source != instance._catalog.source:
                        raise catalog.Error("invalid_response", "The designer made a voice for another source.")
                    made = given.Given(
                        ref=designed.ref,
                        origin="designed",
                        description=designed.description,
                        descriptors=conceived.descriptors,
                        languages=request.languages,
                        traits=traits,
                    )
                else:
                    entry = data.entry
                    assert entry is not None, "a shortlisted voice is accepted only with its catalog entry"
                    made = given.Given(
                        ref=entry.ref,
                        origin="catalog",
                        description=entry.description or entry.name,
                        descriptors=entry.descriptors,
                        languages=entry.languages or instance._languages,
                        traits=traits,
                    )
            except Exception as error:
                # A documented boundary: whatever the source raised, the voice was not made; the attempt is retried.
                kind = span.failure_kind(error, "design_failed")
                span.record_failure(active, kind)
                failure = _FailedData(message=f"Inception could not make the bot's voice: {error}")
                Inception._step(ctx, instance, event, _AttemptFailedEvent, failure)
                return
        Inception._step(ctx, instance, event, _CreatedEvent, _GivenData(voice=made))

    @staticmethod
    async def _remember(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        """Write the given voice down before the bot speaks in it (fail-closed)."""

        data = event.data
        if not isinstance(data, _GivenData):
            # Unreachable: remembering is entered only on a voice made.
            return
        store = instance._memory
        if store is not None:
            try:
                _ = await memory.transact(
                    ctx, store, given.given_insert_input(data.voice), metadata=dict(event.metadata)
                )
            except memory.RequestError as error:
                failure = _FailedData(message=f"Inception could not remember the bot's voice: {error}")
                Inception._step(ctx, instance, event, _AttemptFailedEvent, failure)
                return
        Inception._step(ctx, instance, event, _RememberedEvent, data)

    @staticmethod
    def _made(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Inception._settle_attempt(instance, "made")

    @staticmethod
    def _fail_attempt(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        failure = event.data
        if not isinstance(failure, _FailedData):
            # Unreachable: paired with hsm.guard(_carries_failure), which narrows this payload.
            return
        Inception._settle_attempt(instance, "failed")
        Inception._fail(ctx, instance, event, failure.message)

    @staticmethod
    def _may_conceive_again(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> bool:
        """A refused conception is still within the immediate re-conceptions allowed."""

        del ctx, instance
        data = event.data
        return isinstance(data, _RefusedData) and data.refusals <= _RECONCEPTIONS

    @staticmethod
    def _refuse(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        """A refused try is a failed one: counted, and told to the owner."""

        data = event.data
        if not isinstance(data, _RefusedData):
            # Unreachable: paired with guards that narrow this payload.
            return
        Inception._settle_attempt(instance, "failed")
        Inception._fail(ctx, instance, event, f"Inception could not conceive the bot's voice: {data.message}")

    @staticmethod
    def _fail_conceiving_timeout(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        Inception._settle_attempt(instance, "failed")
        Inception._fail(ctx, instance, event, "Inception could not conceive the bot's voice in time.")

    @staticmethod
    def _retry_delay(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._retry_after

    @staticmethod
    def _inception_delay(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._inception_timeout

    @staticmethod
    def _short_delay(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, instance, event
        return _RETRY_DELAY

    # --- voicing -------------------------------------------------------------------------------

    @staticmethod
    def _take_voice(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        """The voice the bot was given is now known: recalled and still held, or just made and written down."""

        del ctx
        data = event.data
        if not isinstance(data, _CheckedData | _GivenData):
            # Unreachable: paired with the guards on the transitions into voicing, which narrow this payload.
            return
        instance._given = data.voice

    @staticmethod
    async def _voice(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        """Tell Speaking, by message, to speak in the given voice for life, and wait for what it can do."""

        voiced = instance._given
        assert voiced is not None, "voicing is entered only once the bot's voice is known"
        peer = instance._speaking
        request = dataclasses.replace(
            speaking.VoiceEvent.with_data_and_id(
                speaking.VoiceData(voice=voiced.ref.key, description=voiced.description), uuid.uuid4().hex
            ),
            metadata=dict(event.metadata),
        )
        with span.operation("bot.voice.voicing", scope=_SCOPE, component=_COMPONENT, stage="voicing") as active:
            try:
                terminal = await ability.run_terminal_operation(
                    ctx,
                    child=peer,
                    request=request,
                    terminals=(speaking.VoicedEvent, peer.failed_event),
                    timeout=instance._inception_timeout,
                )
            except RuntimeError:
                # Speaking is not started yet: it is asked again shortly, within the voicing bound.
                span.record_failure(active, "unavailable")
                failure = _FailedData(message="Speaking is not started.")
                Inception._step(ctx, instance, event, _SpeakingAbsentEvent, failure)
                return
            except TimeoutError:
                span.record_failure(active, "timeout")
                failure = _FailedData(message="Speaking did not take the bot's voice in time.")
                Inception._step(ctx, instance, event, _VoicingFailedEvent, failure)
                return
            answer = terminal.data
            if not isinstance(answer, speaking.VoicedData):
                span.record_failure(active, "unavailable")
                reason = answer.message if isinstance(answer, ability.FailureData) else "no answer"
                failure = _FailedData(message=f"Speaking could not take the bot's voice: {reason}")
                Inception._step(ctx, instance, event, _VoicingFailedEvent, failure)
                return
        Inception._step(ctx, instance, event, _VoicedEvent, _VoicedData(expressiveness=answer.expressiveness))

    @staticmethod
    def _tell(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        """Speaking took the voice for life: tell the owner the voice, and what it can do with an expression."""

        data = event.data
        voiced = instance._given
        if not isinstance(data, _VoicedData) or voiced is None:
            # Unreachable: paired with hsm.guard(_payload_is(_VoicedData)), and voicing knows the voice.
            return
        instance._voiced = True
        told = voice.VoiceData(
            ref=voiced.ref,
            origin=voiced.origin,
            description=voiced.description,
            languages=voiced.languages,
            expressiveness=data.expressiveness,
        )
        terminal = dataclasses.replace(_with_context(voice.OutputEvent.with_data(told), event), source=hsm.id(instance))
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _is_voiced(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> bool:
        """Speaking already speaks in the voice it was given (read from this machine's own field)."""

        del ctx, event
        return instance._voiced

    @staticmethod
    def _fail_voicing(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        failure = event.data
        if not isinstance(failure, _FailedData):
            # Unreachable: paired with hsm.guard(_carries_failure), which narrows this payload.
            return
        Inception._fail(ctx, instance, event, f"{failure.message} The bot stays unvoiced.")

    @staticmethod
    def _fail_voicing_timeout(ctx: hsm.Context, instance: "Inception", event: hsm.Event[typing.Any]) -> None:
        Inception._fail(ctx, instance, event, "Speaking did not take the bot's voice in time. The bot stays unvoiced.")

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Inception",
        hsm.initial(hsm.target("/Inception/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Inception/resuming"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detached_terminal),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Inception/degraded"),
            ),
        ),
        # Attached (again), or still attached after a failed detach: a voice Speaking already took is kept as it is.
        hsm.choice(
            "resuming",
            hsm.transition(hsm.guard(_is_voiced), hsm.target("/Inception/made")),
            hsm.transition(hsm.target("/Inception/recalling")),
        ),
        hsm.state(
            "recalling",
            hsm.initial(hsm.target("/Inception/recalling/asking")),
            # Memory is a peer attached by its own owner, with the same bring-up budget to start answering.
            hsm.transition(
                hsm.after(attachment.Attachment._attachment_timeout_delay),
                hsm.effect(_give_up_recalling),
                hsm.target("/Inception/recalling/waiting"),
            ),
            hsm.state(
                "asking",
                hsm.activity(_recall),
                hsm.transition(hsm.on(_RecalledEvent), hsm.guard(_is_ours), hsm.target("/Inception/checking")),
                hsm.transition(
                    hsm.on(_RecalledEvent),
                    hsm.guard(_is_estranged),
                    hsm.effect(_fail_estranged),
                    hsm.target("/Inception/estranged"),
                ),
                hsm.transition(
                    hsm.on(_RecalledEvent), hsm.guard(_payload_is(_RecalledData)), hsm.target("/Inception/conceiving")
                ),
                hsm.transition(
                    hsm.on(_RecallFailedEvent), hsm.guard(_carries_failure), hsm.target("/Inception/recalling/waiting")
                ),
            ),
            hsm.state(
                "waiting",
                hsm.transition(hsm.after(_short_delay), hsm.target("/Inception/recalling/asking")),
            ),
        ),
        hsm.state(
            "checking",
            hsm.activity(_check),
            hsm.transition(
                hsm.on(_CheckedEvent), hsm.guard(_is_present), hsm.effect(_take_voice), hsm.target("/Inception/voicing")
            ),
            hsm.transition(
                hsm.on(_CheckedEvent),
                hsm.guard(_payload_is(_CheckedData)),
                hsm.effect(_fail_lost),
                hsm.target("/Inception/estranged"),
            ),
            hsm.transition(
                hsm.on(_CheckFailedEvent), hsm.guard(_carries_failure), hsm.target("/Inception/recalling/waiting")
            ),
        ),
        hsm.state(
            "conceiving",
            hsm.entry(_begin_attempt),
            hsm.activity(_conceive),
            hsm.transition(
                hsm.after(_inception_delay), hsm.effect(_fail_conceiving_timeout), hsm.target("/Inception/resting")
            ),
            hsm.transition(
                hsm.on(_ConceivedEvent), hsm.guard(_payload_is(_ConceivedData)), hsm.target("/Inception/creating")
            ),
            # Refused: conceived again at once with why, within the bound; past it, rested like any failed attempt.
            hsm.transition(
                hsm.on(_RefusedEvent),
                hsm.guard(_may_conceive_again),
                hsm.effect(_refuse),
                hsm.target("/Inception/conceiving"),
            ),
            hsm.transition(
                hsm.on(_RefusedEvent),
                hsm.guard(_payload_is(_RefusedData)),
                hsm.effect(_refuse),
                hsm.target("/Inception/resting"),
            ),
            hsm.transition(
                hsm.on(_AttemptFailedEvent),
                hsm.guard(_carries_failure),
                hsm.effect(_fail_attempt),
                hsm.target("/Inception/resting"),
            ),
        ),
        hsm.state(
            "creating",
            hsm.activity(_create),
            hsm.transition(
                hsm.on(_CreatedEvent), hsm.guard(_payload_is(_GivenData)), hsm.target("/Inception/remembering")
            ),
            hsm.transition(
                hsm.on(_AttemptFailedEvent),
                hsm.guard(_carries_failure),
                hsm.effect(_fail_attempt),
                hsm.target("/Inception/resting"),
            ),
        ),
        hsm.state(
            "remembering",
            hsm.activity(_remember),
            hsm.transition(
                hsm.on(_RememberedEvent),
                hsm.guard(_payload_is(_GivenData)),
                hsm.effect(_made, _take_voice),
                hsm.target("/Inception/voicing"),
            ),
            hsm.transition(
                hsm.on(_AttemptFailedEvent),
                hsm.guard(_carries_failure),
                hsm.effect(_fail_attempt),
                hsm.target("/Inception/resting"),
            ),
        ),
        hsm.state(
            "resting",
            hsm.transition(hsm.after(_retry_delay), hsm.target("/Inception/conceiving")),
        ),
        hsm.state(
            "voicing",
            hsm.initial(hsm.target("/Inception/voicing/asking")),
            hsm.transition(
                hsm.after(_inception_delay), hsm.effect(_fail_voicing_timeout), hsm.target("/Inception/made")
            ),
            hsm.transition(
                hsm.on(_VoicedEvent),
                hsm.guard(_payload_is(_VoicedData)),
                hsm.effect(_tell),
                hsm.target("/Inception/made"),
            ),
            hsm.transition(
                hsm.on(_VoicingFailedEvent),
                hsm.guard(_carries_failure),
                hsm.effect(_fail_voicing),
                hsm.target("/Inception/made"),
            ),
            hsm.state(
                "asking",
                hsm.activity(_voice),
                hsm.transition(
                    hsm.on(_SpeakingAbsentEvent), hsm.guard(_carries_failure), hsm.target("/Inception/voicing/waiting")
                ),
            ),
            hsm.state(
                "waiting",
                hsm.transition(hsm.after(_short_delay), hsm.target("/Inception/voicing/asking")),
            ),
        ),
        # The voice is the bot's for life: nothing moves it from here.
        hsm.state("made"),
        # Recorded only for another source, or lost by its own: the bot stays unvoiced, never made again.
        hsm.state("estranged"),
        hsm.state(
            "detaching",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Inception/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Inception/resuming"),
            ),
        ),
        hsm.state("degraded"),
    )


__all__ = ["Inception"]
