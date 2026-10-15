"""Speech over OpenAI-compatible audio endpoints: streamed speech encoding and batch transcription.

``SpeechEncoder`` is a ``vocal.speech.SpeechEncoder`` over ``POST /v1/audio/speech`` with ``response_format="pcm"``,
streamed through the SDK's ``with_streaming_response``. The endpoint takes one whole ``input``, so the encoder
reads the utterance's text to its end first (joining its pieces in order is its only buffering; there are no
chunking rules) and sends one request; closing it before the text ends sends none. The raw s16le body is yielded
as it arrives, re-cut to whole frames, with alignment emulated over the whole text: every chunk carries ``""`` but
the last, which carries the whole text, so one chunk is held until the next arrives or the body ends. Closing the
iterator closes the response, which is how a server such as Breeze cancels an encoding.
``SpeechDecoder`` is a ``hearing.speech.SpeechDecoder`` over ``POST /v1/audio/transcriptions``: raw PCM is wrapped
as a WAV file with the stdlib ``wave`` module and uploaded as multipart ``file``; the answer's ``text`` is returned
as UTF-8 bytes.

Failures use existing ``FAILURE_KINDS.md`` strings, the same on the error and on the span: a 4xx answer is
``invalid_input``, a 5xx answer or an unreachable server ``unavailable``, an SDK timeout ``timeout``, an answer the
contract does not allow ``invalid_response``; anything narrower-less is ``encode_failed`` (encoding) or
``decode_failed`` (transcription).

Telemetry (cardinality, METRICS > SPANS > Logs; every value comes from code). Speaking already counts and times each
encoding stream, so this provider adds spans only:

* ``bot.provider.openai_compat.speech.encode`` (component ``openai_compat.speech``, stage ``encode``), with
  the bounded counts ``bot.text.chars`` and ``bot.audio.bytes`` and ``bot.http.status_class``. It is not made the
  current span, because the stream yields to its consumer while it is open.
* ``bot.provider.openai_compat.speech.decode`` (stage ``decode``), with ``bot.audio.bytes``,
  ``bot.transcript.chars`` and ``bot.http.status_class``.

Never text, audio, transcripts, voices, URLs, model names or credentials.
"""

from __future__ import annotations

import asyncio
import collections.abc
import io
import typing
import wave

import openai
import pydantic

from mosfet.abilities import encoding, hearing, vocal
from mosfet.telemetry import span

from .client import AsyncOpenAIClient, jsonable, sdk

_SCOPE: typing.Final = "mosfet.providers.openai_compat"
_COMPONENT: typing.Final = "openai_compat.speech"
_ENCODE_SPAN: typing.Final = "bot.provider.openai_compat.speech.encode"
_DECODE_SPAN: typing.Final = "bot.provider.openai_compat.speech.decode"
_STATUS_CLASS_ATTR: typing.Final = "bot.http.status_class"
_OUTCOME_ATTR: typing.Final = "bot.outcome"
_SAMPLE_BYTES: typing.Final = 2
"""s16le: two bytes per sample per channel."""

type DecodingFailureKind = typing.Literal[
    "invalid_input", "unavailable", "timeout", "invalid_response", "decode_failed"
]
"""Why a transcription failed, in existing ``FAILURE_KINDS.md`` strings."""


class SpeechDecodingError(RuntimeError):
    """A transcription failed; ``kind`` says why in a closed, low-cardinality vocabulary."""

    kind: DecodingFailureKind
    failure_kind: str
    """The same string as ``kind``, under the name ``telemetry.span.failure_kind`` reads."""

    def __init__(self, kind: DecodingFailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.failure_kind = kind


def _request_failure(
    error: Exception, *, started: bool
) -> tuple[typing.Literal["invalid_input", "unavailable", "timeout"] | None, str]:
    """The failure kind an SDK or transport error names (None when none does) and its HTTP status class."""

    if isinstance(error, openai.APITimeoutError):
        return "timeout", "none"
    if isinstance(error, openai.APIStatusError):
        status = error.status_code
        if 400 <= status < 500:
            return "invalid_input", "4xx"
        return "unavailable", "5xx" if status >= 500 else "none"
    if isinstance(error, openai.APIConnectionError) or started:
        # After the response started, a broken body is the server no longer serving the stream.
        return "unavailable", "none"
    return None, "none"


def _client(
    client: AsyncOpenAIClient | None,
    *,
    base_url: str,
    api_key: pydantic.SecretStr | None,
    timeout_seconds: float,
    default_headers: collections.abc.Mapping[str, str],
) -> AsyncOpenAIClient:
    if client is not None:
        return client
    return typing.cast(
        AsyncOpenAIClient,
        sdk(
            base_url=base_url,
            api_key=None if api_key is None else api_key.get_secret_value(),
            timeout_seconds=timeout_seconds,
            default_headers=default_headers,
            asynchronous=True,
        ),
    )


class SpeechEncoder(vocal.speech.SpeechEncoder):
    """Streaming text-to-speech over an OpenAI-compatible ``/v1/audio/speech`` endpoint, as raw s16le PCM.

    ``voice`` is the provider voice name the text is spoken in; without one the request names no voice, so the
    server uses its default. ``sample_rate_hz`` and ``channels`` declare the PCM
    the server sends (``format``); a server that says otherwise in ``X-Sample-Rate`` or ``X-Sample-Format`` is an
    ``invalid_response``. ``client`` is an injected ``openai.AsyncOpenAI``; without one, it is built on first use
    from ``base_url``, ``api_key``, ``timeout_seconds`` and ``default_headers``.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        voice: str | None = None,
        api_key: pydantic.SecretStr | None = None,
        sample_rate_hz: int = 24_000,
        channels: int = 1,
        timeout_seconds: float = 60.0,
        default_headers: collections.abc.Mapping[str, str] | None = None,
        client: AsyncOpenAIClient | None = None,
    ) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=sample_rate_hz, channels=channels)
        self._model: str = model
        self._voice: str | None = voice
        self._base_url: str = base_url
        self._api_key: pydantic.SecretStr | None = api_key
        self._timeout_seconds: float = timeout_seconds
        self._default_headers: dict[str, str] = dict(default_headers or {})
        self._client: AsyncOpenAIClient | None = client

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[str]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece async for piece in input])
        request: dict[str, object] = {
            "model": self._model,
            "input": said,
            "voice": openai.omit if self._voice is None else self._voice,
            "response_format": "pcm",
        }
        active = span.tracer(_SCOPE).start_span(
            _ENCODE_SPAN,
            attributes={"bot.component.name": _COMPONENT, "bot.stage": "encode", "bot.text.chars": len(said)},
        )
        frame = _SAMPLE_BYTES * self.format.channels
        yielded = 0
        try:
            if not said.strip():
                raise encoding.Error("invalid_input", "OpenAI-compatible speech encoding requires non-blank text.")
            started = False
            pending = b""
            held: bytes | None = None
            try:
                self._client = _client(
                    self._client,
                    base_url=self._base_url,
                    api_key=self._api_key,
                    timeout_seconds=self._timeout_seconds,
                    default_headers=self._default_headers,
                )
                async with self._client.audio.speech.with_streaming_response.create(**request) as response:
                    active.set_attribute(_STATUS_CLASS_ATTR, "2xx")
                    self._check_format(response.headers)
                    started = True
                    async for chunk in response.iter_bytes():
                        pending += chunk
                        whole = len(pending) - len(pending) % frame
                        if whole:
                            if held is not None:
                                yielded += len(held)
                                yield vocal.speech.Audio(pcm=held, text="")
                            held, pending = pending[:whole], pending[whole:]
            except encoding.Error:
                raise
            except Exception as error:
                kind, status_class = _request_failure(error, started=started)
                if status_class != "none":
                    active.set_attribute(_STATUS_CLASS_ATTR, status_class)
                raise encoding.Error(
                    kind or "encode_failed",
                    f"OpenAI-compatible speech encoding failed: {type(error).__name__}.",
                ) from error
            if pending:
                raise encoding.Error(
                    "invalid_response", "OpenAI-compatible speech encoding returned PCM that ends mid-frame."
                )
            if held is None:
                raise encoding.Error("invalid_response", "OpenAI-compatible speech encoding returned no audio.")
            yielded += len(held)
            yield vocal.speech.Audio(pcm=held, text=said)
        except encoding.Error as error:
            span.record_failure(active, error.kind)
            raise
        except (GeneratorExit, asyncio.CancelledError):
            active.set_attribute(_OUTCOME_ATTR, span.Cancelled)
            raise
        else:
            active.set_attribute(_OUTCOME_ATTR, span.Ok)
        finally:
            active.set_attribute("bot.audio.bytes", yielded)
            active.end()

    def _check_format(self, headers: collections.abc.Mapping[str, str]) -> None:
        rate = headers.get("x-sample-rate")
        if rate is not None and rate.strip() != str(self.format.sample_rate_hz):
            raise encoding.Error(
                "invalid_response", "The speech server's X-Sample-Rate disagrees with the encoder's format."
            )
        sample_format = headers.get("x-sample-format")
        if sample_format is not None and sample_format.strip().lower() != "s16le":
            raise encoding.Error("invalid_response", "The speech server's X-Sample-Format is not s16le PCM.")


class SpeechDecoder(hearing.speech.SpeechDecoder):
    """Batch speech-to-text over an OpenAI-compatible ``/v1/audio/transcriptions`` endpoint.

    The input is raw s16le PCM at ``sample_rate_hz`` and ``channels``, wrapped as a WAV file before upload; an input
    that is already a WAV container is sent unchanged. ``language`` is sent when set. The returned bytes are the
    UTF-8 transcript, stripped. ``client`` is an injected ``openai.AsyncOpenAI``; without one, it is built on first
    use from ``base_url``, ``api_key`` and ``timeout_seconds``.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: pydantic.SecretStr | None = None,
        sample_rate_hz: int = 16_000,
        channels: int = 1,
        language: str | None = None,
        timeout_seconds: float = 60.0,
        client: AsyncOpenAIClient | None = None,
    ) -> None:
        if sample_rate_hz < 1 or channels < 1:
            raise ValueError("sample_rate_hz and channels must be positive.")
        self._model: str = model
        self._base_url: str = base_url
        self._api_key: pydantic.SecretStr | None = api_key
        self._sample_rate_hz: int = sample_rate_hz
        self._channels: int = channels
        self._language: str | None = language
        self._timeout_seconds: float = timeout_seconds
        self._client: AsyncOpenAIClient | None = client

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        """Transcribe ``input`` and return the transcript as UTF-8 bytes.

        Raises:
            SpeechDecodingError: The audio was empty or not whole frames, the server refused it, was unavailable
                or timed out, answered without text, or heard no words.
        """

        with span.operation(
            _DECODE_SPAN,
            scope=_SCOPE,
            component=_COMPONENT,
            stage="decode",
            attributes={"bot.audio.bytes": len(input)},
        ) as active:
            wav = self._wav(input)
            self._client = _client(
                self._client,
                base_url=self._base_url,
                api_key=self._api_key,
                timeout_seconds=self._timeout_seconds,
                default_headers={},
            )
            try:
                result = await self._client.audio.transcriptions.create(
                    file=("speech.wav", wav, "audio/wav"),
                    model=self._model,
                    response_format="json",
                    language=openai.omit if self._language is None else self._language,
                )
            except Exception as error:
                kind, status_class = _request_failure(error, started=False)
                active.set_attribute(_STATUS_CLASS_ATTR, status_class)
                raise SpeechDecodingError(
                    kind or "decode_failed",
                    f"OpenAI-compatible transcription failed: {type(error).__name__}.",
                ) from error
            active.set_attribute(_STATUS_CLASS_ATTR, "2xx")
            answer = jsonable(result)
            text = (
                typing.cast(collections.abc.Mapping[str, object], answer).get("text")
                if isinstance(answer, collections.abc.Mapping)
                else None
            )
            if not isinstance(text, str):
                raise SpeechDecodingError("invalid_response", "The transcription answer carries no text.")
            transcript = text.strip()
            active.set_attribute("bot.transcript.chars", len(transcript))
            if not transcript:
                raise SpeechDecodingError("decode_failed", "The transcription heard no words.")
            return transcript.encode("utf-8")

    def _wav(self, audio: bytes) -> bytes:
        if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
            return audio
        frame = _SAMPLE_BYTES * self._channels
        if not audio or len(audio) % frame:
            raise SpeechDecodingError("invalid_input", "Transcription needs whole, non-empty s16le PCM frames.")
        output = io.BytesIO()
        with wave.open(output, "wb") as stream:
            stream.setnchannels(self._channels)
            stream.setsampwidth(_SAMPLE_BYTES)
            stream.setframerate(self._sample_rate_hz)
            stream.writeframes(audio)
        return output.getvalue()


__all__ = ["DecodingFailureKind", "SpeechDecoder", "SpeechDecodingError", "SpeechEncoder"]
