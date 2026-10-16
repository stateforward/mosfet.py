"""``Catalog`` and ``Designer``: the voices a speech server holds and makes, on its ``/v1/voices`` protocol.

Both are plain collaborators that core's ``vocal.voice`` composes (``Inception`` lists and designs through them); they
are not state machines. The protocol is the lab TTS engine's own (anygame's Breeze engine):

* ``GET /v1/voices`` answers ``{default, voices: [{id, description, custom?}], emotive_tags}``. Each voice becomes a
  ``voice.Entry`` named by its id under the composition-assigned ``source``; a designed (``custom``) voice is
  labelled ``{"custom": "true"}`` and carries its preview URL, ``/v1/voices/{id}/preview``. The server describes no
  languages or descriptors: an entry's languages are the ones composition declared for the engine.
* ``POST /v1/voices {description, count: 1}`` designs exactly one voice (the server keeps its reference clip, so the
  voice is permanent from its 201 answer), which becomes ``voice.Designed``. Descriptions are at most 300
  characters (``max_description``); a longer one is refused before any request.

Failures raise ``voice.Error`` in the canonical kinds: 404 (the engine does not hold or design voices) is
``unsupported_input``, 422 ``invalid_input``, 503 ``unavailable``, 507 (voice storage full) ``overloaded``, an
unreachable server ``unavailable``, a deadline ``timeout``, and an answer that is not the documented shape
``invalid_response``. Any other refusal of a listing is ``invalid_input`` (4xx) or ``unavailable`` (5xx); of a design,
``design_failed``. Raw httpx errors stay chained as the cause; messages never carry server text or descriptions.

Telemetry (cardinality, METRICS > SPANS > Logs; every value comes from code):

* ``bot.provider.speech_server.voices.duration`` (histogram, s): ``operation`` {list, design} × ``bot.outcome``
  {ok, failed}: at most 4 series. Never URLs, ids or descriptions.
"""

from __future__ import annotations

import dataclasses
import time
import typing
import urllib.parse

import httpx
import pydantic
from opentelemetry import metrics

from mosfet.abilities.vocal import speech, voice
from mosfet.telemetry import span

from .request import MAX_RESPONSE_BYTES

_DURATION = metrics.get_meter("mosfet.providers.speech_server").create_histogram(
    "bot.provider.speech_server.voices.duration",
    unit="s",
    description="Duration of each speech-server voices request, by operation (list, design) and outcome (ok, failed).",
)
_LOCALES: typing.Final = pydantic.TypeAdapter(tuple[speech.Locale, ...])


class _Voice(pydantic.BaseModel):
    """One voice as the server lists or makes it; the fields it adds beyond these are ignored."""

    id: str = pydantic.Field(min_length=1, max_length=256)
    description: str = ""
    custom: bool = False


class _Listing(pydantic.BaseModel):
    voices: tuple[_Voice, ...]


class _Made(pydantic.BaseModel):
    voices: tuple[_Voice]
    """Exactly the one voice asked for."""


@dataclasses.dataclass(frozen=True)
class Catalog(voice.Catalog):
    """The voices a speech server holds, listed from ``GET /v1/voices``.

    ``source`` is the name composition gives this server; every listed ref carries it. ``languages`` are the
    languages composition knows the engine speaks, given to every entry. ``client`` is an injected
    ``httpx.AsyncClient``; without one, each request opens and closes its own. ``timeout_seconds`` bounds it.

    Raises:
        ValueError: ``source`` is not a voice source name, or a language is not a locale.
    """

    base_url: str
    _: dataclasses.KW_ONLY
    source: str
    languages: tuple[speech.Locale, ...] = ()
    client: httpx.AsyncClient | None = None
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        try:
            # A source name is valid exactly when a ref can carry it.
            _ = voice.Ref(source=self.source, key="-")
            _ = _LOCALES.validate_python(self.languages)
        except pydantic.ValidationError as error:
            raise ValueError(
                f"speech_server.Catalog cannot name voices: {error.error_count()} invalid fields."
            ) from error

    @typing.override
    async def voices(
        self, *, languages: tuple[speech.Locale, ...] = (), descriptors: voice.Descriptors | None = None
    ) -> tuple[voice.Entry, ...]:
        """Every voice the server holds. The filters are hints the server has no query for, so all are listed.

        Raises:
            voice.Error: The server refused the listing, was unreachable or slow, or answered with something other
                than a bounded voice listing.
        """

        del languages, descriptors
        base_url = self.base_url.rstrip("/")
        started = time.monotonic()
        outcome = span.Failed
        try:
            sender = httpx.AsyncClient(timeout=self.timeout_seconds) if self.client is None else self.client
            try:
                response = await sender.get(f"{base_url}/v1/voices", timeout=self.timeout_seconds)
            except httpx.TimeoutException as error:
                raise voice.Error("timeout", "speech-server voice listing timed out.") from error
            except httpx.HTTPError as error:
                raise voice.Error(
                    "unavailable", f"speech-server voice listing failed: {type(error).__name__}."
                ) from error
            finally:
                if self.client is None:
                    await sender.aclose()
            status = response.status_code
            kinds: dict[int, voice.FailureKind] = {404: "unsupported_input", 507: "overloaded"}
            kind = kinds.get(status, "invalid_input" if 400 <= status < 500 else "unavailable")
            if status >= 400:
                raise voice.Error(kind, f"speech server refused the voice listing (HTTP {status}).")
            if not response.is_success or len(response.content) > MAX_RESPONSE_BYTES:
                raise voice.Error("invalid_response", f"speech-server voice listing is not usable (HTTP {status}).")
            try:
                listing = _Listing.model_validate_json(response.content)
                entries = tuple(
                    voice.Entry(
                        ref=voice.Ref(source=self.source, key=listed.id),
                        name=listed.id,
                        description=listed.description.strip() or None,
                        languages=self.languages,
                        preview=voice.Preview.model_validate(
                            {"url": f"{base_url}/v1/voices/{urllib.parse.quote(listed.id, safe='')}/preview"}
                        )
                        if listed.custom
                        else None,
                        labels={"custom": "true"} if listed.custom else {},
                    )
                    for listed in listing.voices
                )
            except pydantic.ValidationError as error:
                message = f"speech-server voice listing is invalid ({error.error_count()} errors)."
                raise voice.Error("invalid_response", message) from error
            outcome = span.Ok
            return entries
        finally:
            _DURATION.record(time.monotonic() - started, {"operation": "list", "bot.outcome": outcome})


@dataclasses.dataclass(frozen=True)
class Designer(voice.Designer):
    """Makes one voice from a description with ``POST /v1/voices``; the server keeps it from then on.

    ``source`` names the server the way its :class:`Catalog` does. ``max_description`` is the server's own bound,
    300 characters. ``client`` is an injected ``httpx.AsyncClient``; without one, each request opens and closes its
    own. ``timeout_seconds`` bounds the request, which renders the voice's reference clip.

    Raises:
        ValueError: ``source`` is not a voice source name.
    """

    base_url: str
    _: dataclasses.KW_ONLY
    source: str
    client: httpx.AsyncClient | None = None
    timeout_seconds: float = 60.0
    max_description: int = dataclasses.field(default=300, init=False)

    def __post_init__(self) -> None:
        try:
            # A source name is valid exactly when a ref can carry it.
            _ = voice.Ref(source=self.source, key="-")
        except pydantic.ValidationError as error:
            raise ValueError(
                f"speech_server.Designer cannot name voices: {error.error_count()} invalid fields."
            ) from error

    @typing.override
    async def create(self, request: voice.DesignRequest) -> voice.Designed:
        """Make exactly one voice from ``request.description``; its other fields are not part of this protocol.

        Raises:
            voice.Error: The description is longer than ``max_description`` (``invalid_input``, without a request),
                or the server refused, failed or answered with something other than one made voice.
        """

        if len(request.description) > self.max_description:
            raise voice.Error(
                "invalid_input", f"speech-server voice descriptions are at most {self.max_description} characters."
            )
        base_url = self.base_url.rstrip("/")
        started = time.monotonic()
        outcome = span.Failed
        try:
            sender = httpx.AsyncClient(timeout=self.timeout_seconds) if self.client is None else self.client
            try:
                response = await sender.post(
                    f"{base_url}/v1/voices",
                    json={"description": request.description, "count": 1},
                    timeout=self.timeout_seconds,
                )
            except httpx.TimeoutException as error:
                raise voice.Error("timeout", "speech-server voice design timed out.") from error
            except httpx.HTTPError as error:
                raise voice.Error(
                    "unavailable", f"speech-server voice design failed: {type(error).__name__}."
                ) from error
            finally:
                if self.client is None:
                    await sender.aclose()
            status = response.status_code
            kinds: dict[int, voice.FailureKind] = {
                404: "unsupported_input",
                422: "invalid_input",
                503: "unavailable",
                507: "overloaded",
            }
            if not response.is_success:
                raise voice.Error(kinds.get(status, "design_failed"), f"speech server did not design (HTTP {status}).")
            if len(response.content) > MAX_RESPONSE_BYTES:
                raise voice.Error("invalid_response", "speech-server voice design answer is too large.")
            try:
                (made,) = _Made.model_validate_json(response.content).voices
            except pydantic.ValidationError as error:
                message = f"speech-server voice design answer is invalid ({error.error_count()} errors)."
                raise voice.Error("invalid_response", message) from error
            outcome = span.Ok
            return voice.Designed(
                ref=voice.Ref(source=self.source, key=made.id),
                description=request.description,
                preview=voice.Preview.model_validate(
                    {"url": f"{base_url}/v1/voices/{urllib.parse.quote(made.id, safe='')}/preview"}
                ),
            )
        finally:
            _DURATION.record(time.monotonic() - started, {"operation": "design", "bot.outcome": outcome})


__all__ = ["Catalog", "Designer"]
