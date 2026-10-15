"""One request to a speech server: raw s16le PCM posted, a JSON object back, failures in a closed vocabulary.

Shared by the diarization, voice-embedding and affect clients. Failures use existing ``FAILURE_KINDS.md`` strings,
the same on the error and in telemetry: a 4xx answer is ``invalid_input``; a 5xx answer or an unreachable server is
``unavailable``; a deadline is ``timeout``; any other answer that is not a JSON object of at most
:data:`MAX_RESPONSE_BYTES` is ``invalid_response``. Raw httpx errors stay chained as the cause; messages name only the
operation and the status or exception type, never server text, audio or vectors.

Telemetry (cardinality, METRICS > SPANS > Logs; every value comes from code):

* ``bot.provider.speech_server.request.duration`` (histogram, s): ``operation`` {diarize, embed, emotion} ×
  ``bot.outcome`` {ok, failed}: at most 6 series. Never URLs, models, audio sizes or vectors.
"""

from __future__ import annotations

import collections.abc
import json
import time
import typing

import httpx
from opentelemetry import metrics

from mosfet.telemetry import span

MAX_RESPONSE_BYTES: typing.Final = 1 << 20
"""The largest answer read as JSON; the servers answer a few kilobytes."""

type Operation = typing.Literal["diarize", "embed", "emotion"]
"""Which speech-server endpoint a request is for; a metric dimension."""

type FailureKind = typing.Literal["invalid_input", "unavailable", "timeout", "invalid_response"]
"""Why a speech-server request failed, in existing ``FAILURE_KINDS.md`` strings."""

_DURATION = metrics.get_meter("mosfet.providers.speech_server").create_histogram(
    "bot.provider.speech_server.request.duration",
    unit="s",
    description=(
        "Duration of each speech-server request, by operation (diarize, embed, emotion) and outcome (ok, failed)."
    ),
)


class RequestError(RuntimeError):
    """A speech-server request failed; ``kind`` says why in a closed, low-cardinality vocabulary."""

    kind: FailureKind
    failure_kind: str
    """The same string as ``kind``, under the name ``telemetry.span.failure_kind`` reads."""

    def __init__(self, kind: FailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.failure_kind = kind


async def post(
    *,
    base_url: str,
    path: str,
    operation: Operation,
    content: bytes,
    params: collections.abc.Mapping[str, str | int] | None = None,
    client: httpx.AsyncClient | None,
    timeout_seconds: float,
) -> dict[str, object]:
    """POST ``content`` as ``application/octet-stream`` to ``base_url`` + ``path`` and return the JSON object answer.

    ``client`` is used when given; otherwise a client is opened for this request and closed after it.

    Raises:
        RequestError: The server refused the request, was unreachable or slow, or answered with something other
            than a bounded JSON object.
    """

    url = base_url.rstrip("/") + path
    started = time.monotonic()
    outcome = span.Failed
    try:
        sender = httpx.AsyncClient(timeout=timeout_seconds) if client is None else client
        try:
            response = await sender.post(
                url,
                content=content,
                params=None if params is None else dict(params),
                headers={"content-type": "application/octet-stream"},
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as error:
            raise RequestError("timeout", f"speech-server {operation} request timed out.") from error
        except httpx.HTTPError as error:
            raise RequestError(
                "unavailable", f"speech-server {operation} request failed: {type(error).__name__}."
            ) from error
        finally:
            if client is None:
                await sender.aclose()
        status = response.status_code
        if 400 <= status < 500:
            raise RequestError("invalid_input", f"speech server refused the {operation} request (HTTP {status}).")
        if status >= 500:
            raise RequestError("unavailable", f"speech server could not serve the {operation} request (HTTP {status}).")
        if not response.is_success or len(response.content) > MAX_RESPONSE_BYTES:
            raise RequestError("invalid_response", f"speech-server {operation} answer is not usable (HTTP {status}).")
        try:
            answer = typing.cast(object, json.loads(response.content))
        except ValueError as error:
            raise RequestError("invalid_response", f"speech-server {operation} answer is not JSON.") from error
        if not isinstance(answer, dict):
            raise RequestError("invalid_response", f"speech-server {operation} answer is not a JSON object.")
        outcome = span.Ok
        return {str(key): value for key, value in typing.cast(dict[object, object], answer).items()}
    finally:
        _DURATION.record(time.monotonic() - started, {"operation": operation, "bot.outcome": outcome})


__all__ = ["MAX_RESPONSE_BYTES", "FailureKind", "Operation", "RequestError", "post"]
