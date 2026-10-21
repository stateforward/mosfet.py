"""Speech over OpenAI-compatible audio endpoints: streamed speech encoding and batch transcription.

``SpeechEncoder`` is a ``vocal.speech.SpeechEncoder`` over ``POST /v1/audio/speech`` with ``response_format="pcm"``,
streamed through the SDK's ``with_streaming_response``. The endpoint takes one whole ``input``, so the encoder says
the reply as one request per run of consecutive parts with the same direction (the provider merge rule): a run
closes when a part with another direction begins, or when the reply ends, and only then is its request sent, in
order; closing the stream before the first run closes sends none. Each request is built from typed data only: the
run's style tag, then each part's words with its sound tag before or after them, or the tag alone where a part
with no words sits (:class:`Markup`), one
``instructions`` rendered from the run's natural-language cues after the voice's description, and ``speed`` from a
declared pace range. Words are sent exactly as written, and nothing is ever searched for in them. The raw s16le body
is yielded as it arrives, re-cut to whole frames, with alignment emulated per request: every chunk carries ``""``
but the last, which carries the run's words (never its tags), so one chunk is held until the next arrives or the
reply ends. A part's pause closes the run before it and is said as ``format.silence`` between the requests, so
runs never merge across a pause. Closing the iterator closes the response, which is how a server such as Breeze
cancels an encoding.
``SpeechDecoder`` is a ``hearing.speech.SpeechDecoder`` over ``POST /v1/audio/transcriptions``: raw PCM is wrapped
as a WAV file with the stdlib ``wave`` module and uploaded as multipart ``file``; the answer's ``text`` is returned
as UTF-8 bytes.

Failures use existing ``FAILURE_KINDS.md`` strings, the same on the error and on the span: a 4xx answer is
``invalid_input``, a 5xx answer or an unreachable server ``unavailable``, an SDK timeout ``timeout``, an answer the
contract does not allow ``invalid_response``; anything narrower-less is ``encode_failed`` (encoding) or
``decode_failed`` (transcription).

Telemetry (cardinality, METRICS > SPANS > Logs; every value comes from code). Speaking already counts and times each
encoding stream, so this provider adds spans only:

* ``bot.provider.openai_compat.speech.encode`` (component ``openai_compat.speech``, stage ``encode``), one per
  stream, with the bounded counts ``bot.text.chars``, ``bot.audio.bytes`` and ``bot.speech.requests`` (synthesis
  requests sent) and the last ``bot.http.status_class``. It is not made the current span, because the stream
  yields to its consumer while it is open.
* ``bot.provider.openai_compat.speech.decode`` (stage ``decode``), with ``bot.audio.bytes``,
  ``bot.transcript.chars`` and ``bot.http.status_class``.

Never text, audio, transcripts, voices, descriptions, cue values, URLs, model names or credentials.
"""

from __future__ import annotations

import asyncio
import collections.abc
import contextlib
import dataclasses
import io
import typing
import wave

import openai
import pydantic
from opentelemetry import trace

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
_APPLICABLE: typing.Final[frozenset[vocal.speech.Cue]] = frozenset(
    {"instructions", "emotion", "intensity", "emphasis", "style", "sound", "pace"}
)
"""The cues this endpoint can carry: natural-language cues in ``instructions``, style and sound as inline markup,
and pace as ``speed``. Pitch and volume have no field, so declaring them would report them applied untruthfully."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class Markup:
    """How a server writes each declared sound and style inline in ``input``: the literal text of each tag.

    A tag is concatenated from a part's typed fields, before or after its words, and never searched for in them.
    """

    sounds: collections.abc.Mapping[vocal.speech.Sound, str] = dataclasses.field(
        default_factory=dict[vocal.speech.Sound, str]
    )
    styles: collections.abc.Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])


ANGLE_TAGS: typing.Final = Markup(
    sounds={"laugh": "<laugh>", "giggle": "<giggle>", "chuckle": "<chuckle>", "gasp": "<gasp>", "sigh": "<sigh>"},
    styles={"whisper": "<whisper>"},
)
"""The angle-tag markup of a lab speech engine that rewrites ``<laugh>`` and ``<whisper>`` style tags itself."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Direction:
    """How a run of parts is said, from the cues this encoder applies: equal directions share one request."""

    instructions: str | None = None
    emotion: str | None = None
    style: str | None = None
    intensity: float | None = None
    emphasis: bool = False
    speed: float | None = None


@dataclasses.dataclass(kw_only=True)
class _Said:
    """One part as read so far: the words written into it, and the expression it opened with."""

    expression: vocal.speech.Expression | None
    words: list[str] = dataclasses.field(default_factory=list[str])


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
    server uses its default. ``voice_description`` is the description the voice was made from: a server whose
    ``instructions`` replace the voice's stored description (as a lab engine's do) keeps it when a part is directed,
    because it is written before the direction. ``expressiveness`` declares what this server applies, and
    ``markup`` how it writes each declared sound and style inline; its keys must be exactly the declared sounds and
    styles. ``sample_rate_hz`` and ``channels`` declare the PCM the server sends (``format``); a server that says
    otherwise in ``X-Sample-Rate`` or ``X-Sample-Format`` is an ``invalid_response``. ``client`` is an injected
    ``openai.AsyncOpenAI``; without one, it is built on first use from ``base_url``, ``api_key``,
    ``timeout_seconds`` and ``default_headers``.

    Raises:
        ValueError: ``expressiveness`` declares pitch or volume, which the endpoint cannot carry, or ``markup`` does
            not write exactly the declared sounds and styles.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        voice: str | None = None,
        voice_description: str | None = None,
        expressiveness: vocal.speech.Expressiveness = vocal.speech.Expressiveness(),
        markup: Markup = Markup(),
        api_key: pydantic.SecretStr | None = None,
        sample_rate_hz: int = 24_000,
        channels: int = 1,
        timeout_seconds: float = 60.0,
        default_headers: collections.abc.Mapping[str, str] | None = None,
        client: AsyncOpenAIClient | None = None,
    ) -> None:
        unsupported = sorted(expressiveness.cues - _APPLICABLE)
        if unsupported:
            raise ValueError(f"An OpenAI-compatible speech endpoint cannot apply: {', '.join(unsupported)}.")
        if set(markup.sounds) != expressiveness.sounds or set(markup.styles) != set(expressiveness.styles):
            raise ValueError("The markup must write exactly the sounds and styles the expressiveness declares.")
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=sample_rate_hz, channels=channels)
        self.expressiveness: vocal.speech.Expressiveness = expressiveness
        self._model: str = model
        self._voice: str | None = voice
        self._voice_description: str | None = voice_description
        self._markup: Markup = markup
        self._base_url: str = base_url
        self._api_key: pydantic.SecretStr | None = api_key
        self._timeout_seconds: float = timeout_seconds
        self._default_headers: dict[str, str] = dict(default_headers or {})
        self._client: AsyncOpenAIClient | None = client

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        active = span.tracer(_SCOPE).start_span(
            _ENCODE_SPAN, attributes={"bot.component.name": _COMPONENT, "bot.stage": "encode"}
        )
        chars = 0
        requests = 0
        yielded = 0
        # The last chunk, held until the next one arrives or the reply ends: the last one of a request carries the
        # words that request said, and the reply's last one also carries any trailing words with nothing to say.
        held: vocal.speech.Audio | None = None
        # Blank words read before any chunk to carry them: they ride the next one.
        unsaid = ""
        try:
            async with contextlib.aclosing(self._runs(input)) as runs:
                async for segment in runs:
                    if isinstance(segment, str):
                        # A pause: silence of its own, between the requests around it.
                        if held is not None:
                            yielded += len(held.pcm)
                            yield held
                        held, unsaid = vocal.speech.Audio(pcm=self.format.silence(segment), text=unsaid), ""
                        continue
                    run, direction = segment
                    words = "".join(word for said in run for word in said.words)
                    chars += len(words)
                    if not self._sayable(run):
                        # A run with nothing to say ends the reply or comes before a pause: its words ride a chunk.
                        if held is None:
                            unsaid += words
                        else:
                            held = held.model_copy(update={"text": held.text + words})
                        continue
                    requests += 1
                    last: vocal.speech.Audio | None = None
                    async with contextlib.aclosing(self._say(run, direction, active)) as said:
                        async for pcm in said:
                            if held is not None:
                                yielded += len(held.pcm)
                                yield held
                            held = last = vocal.speech.Audio(pcm=pcm, text="")
                    if last is None:
                        raise encoding.Error("invalid_response", "OpenAI-compatible speech encoding returned no audio.")
                    held, unsaid = last.model_copy(update={"text": unsaid + words}), ""
            if held is None:
                raise encoding.Error("invalid_input", "OpenAI-compatible speech encoding requires non-blank text.")
            yielded += len(held.pcm)
            yield held
        except encoding.Error as error:
            span.record_failure(active, error.kind)
            raise
        except (GeneratorExit, asyncio.CancelledError):
            active.set_attribute(_OUTCOME_ATTR, span.Cancelled)
            raise
        else:
            active.set_attribute(_OUTCOME_ATTR, span.Ok)
        finally:
            active.set_attribute("bot.text.chars", chars)
            active.set_attribute("bot.speech.requests", requests)
            active.set_attribute("bot.audio.bytes", yielded)
            active.end()

    async def _runs(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncGenerator[tuple[list[_Said], _Direction] | vocal.speech.Pause, None]:
        """The reply's runs, each once it closes: consecutive parts said with one direction, in order, and each
        pause in its place between them.

        A run closes when a part with another direction begins, when a part opens with a pause (the pause follows
        it), or when the reply ends. A run with nothing to say yet (no words and no sound) takes the direction of
        the part that follows it, so a blank part never sends a request.

        Raises:
            encoding.Error: ``invalid_input`` when the pieces break the stream's invariants.
        """

        run: list[_Said] = []
        direction = _Direction()
        opened = 0
        async for piece in input:
            if piece.part == opened:
                opened += 1
                following = self._direction(piece.expression)
                pause = None if piece.expression is None else piece.expression.pause
                if run and pause is not None:
                    yield run, direction
                    run = []
                if pause is not None:
                    yield pause
                if self._sayable(run) and following != direction:
                    yield run, direction
                    run = []
                if not self._sayable(run):
                    direction = following
                run.append(_Said(expression=piece.expression))
            elif piece.part != opened - 1 or piece.expression is not None:
                raise encoding.Error("invalid_input", "Speech pieces must open each part in order, once.")
            run[-1].words.append(piece.words)
        if run:
            yield run, direction

    def _direction(self, expression: vocal.speech.Expression | None) -> _Direction:
        """The part's direction from the cues this encoder declares, values clamped to its ranges."""

        if expression is None:
            return _Direction()
        declared = self.expressiveness
        cues = declared.cues
        emotion = expression.emotion if "emotion" in cues else None
        if declared.emotions is not None and emotion not in declared.emotions:
            emotion = None
        pace = declared.pace if "pace" in cues else None
        return _Direction(
            instructions=expression.instructions if "instructions" in cues else None,
            emotion=emotion,
            style=expression.style if expression.style in declared.styles else None,
            intensity=expression.intensity if "intensity" in cues else None,
            emphasis=expression.emphasis and "emphasis" in cues,
            speed=None if pace is None or expression.pace is None else min(max(expression.pace, pace.min), pace.max),
        )

    async def _say(
        self, run: list[_Said], direction: _Direction, active: trace.Span
    ) -> collections.abc.AsyncGenerator[bytes, None]:
        """Send one run's request and yield its PCM as it arrives, in whole frames."""

        text: list[str] = [] if direction.style is None else [self._markup.styles[direction.style]]
        for said in run:
            words = "".join(said.words)
            tag = self._tag(said)
            if tag is None:
                text.append(words)
            elif typing.cast(vocal.speech.Expression, said.expression).sound_at == "after":
                text.extend((words, tag))
            else:
                text.extend((tag, words))
        cues = [
            *([] if direction.instructions is None else [direction.instructions]),
            *([] if direction.emotion is None else [f"Emotion: {direction.emotion}."]),
            *([] if direction.intensity is None else [f"Intensity: {direction.intensity:g} out of 1."]),
            *(["Stress these words."] if direction.emphasis else []),
        ]
        instructions = " ".join(cues) if cues else None
        if instructions is not None and self._voice_description:
            instructions = f"{self._voice_description} {instructions}"
        request: dict[str, object] = {
            "model": self._model,
            "input": "".join(text),
            "voice": openai.omit if self._voice is None else self._voice,
            "response_format": "pcm",
            "instructions": openai.omit if instructions is None else instructions,
            "speed": openai.omit if direction.speed is None else direction.speed,
        }
        frame = _SAMPLE_BYTES * self.format.channels
        started = False
        pending = b""
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
                        yield pending[:whole]
                        pending = pending[whole:]
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

    def _tag(self, said: _Said) -> str | None:
        """The markup of the sound ``said`` makes, when it makes one this server declares."""

        sound = None if said.expression is None else said.expression.sound
        return None if sound is None else self._markup.sounds.get(sound)

    def _sayable(self, run: collections.abc.Iterable[_Said]) -> bool:
        """Whether ``run`` has anything to say: words that are not blank, or a sound this server makes."""

        return any("".join(said.words).strip() or self._tag(said) is not None for said in run)

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


__all__ = ["ANGLE_TAGS", "DecodingFailureKind", "Markup", "SpeechDecoder", "SpeechDecodingError", "SpeechEncoder"]
