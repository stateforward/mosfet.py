"""MLX Audio text-to-speech as a streaming ``vocal.speech.SpeechEncoder``."""

from __future__ import annotations

import asyncio
import collections.abc
import concurrent.futures
import contextlib
import logging
import typing

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech

from ._mlx import (
    PcmConverter,
    SpeechEncodingModel,
    SpeechEncodingModelLoader,
    get_member,
    load_speech_encoding_model,
    pcm16,
)
from .speech_decoder import BoundedWorker

_LOG = logging.getLogger(__name__)
_APPLICABLE: typing.Final[frozenset[speech.Cue]] = frozenset(
    {"instructions", "emotion", "intensity", "emphasis", "pace"}
)
"""The cues a generation can carry: natural-language cues as ``instruct`` and pace as ``speed``."""

type _Direction = tuple[str | None, float | None]
"""How a run is generated: its ``instruct`` and its ``speed``, each ``None`` when the run sets none."""


class SpeechEncoder(speech.SpeechEncoder):
    """MLX Audio TTS, streamed: the model's generated chunks are yielded as mono s16le PCM as they are produced.

    ``generate`` takes whole text, so the reply is generated once per run of consecutive parts with the same
    direction (the provider merge rule): a run closes when a part with another direction begins, or when the reply
    ends, and only then is it generated, in order. ``expressiveness`` declares what the loaded checkpoint applies,
    as composition knows it: ``pace`` becomes the run's ``speed`` (clamped to the declared range), and the
    natural-language cues (``instructions``, ``emotion``, ``intensity``, ``emphasis``) become its ``instruct``.
    Sounds and every undeclared cue are ignored; a pause closes the run before it and is said as
    ``format.silence``. Alignment is emulated per generation: every chunk carries ``""``
    but the last, which carries the run's words, so one generated chunk is held back until the next arrives or the
    reply ends.
    Native work runs one call at a time on a bounded worker: loading the model, starting a generation, and
    pulling each chunk. Closing the stream between chunks stops the generation; a chunk already being generated
    finishes first, because MLX native work cannot be interrupted. The model's sample rate must match the
    declared ``sample_rate_hz``: raw PCM carries no header, so a mismatch is refused as ``invalid_response``
    rather than played at the wrong speed. ``voice`` is the model voice the text is spoken in.

    Raises:
        ValueError: ``expressiveness`` declares a cue a generation cannot carry (style, pitch, volume or sound).
    """

    def __init__(
        self,
        *,
        model_id: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        voice: str | None = None,
        expressiveness: speech.Expressiveness = speech.Expressiveness(),
        speed: float = 1.0,
        lang_code: str = "auto",
        sample_rate_hz: int = 24_000,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: bool = False,
        generate_kwargs: collections.abc.Mapping[str, object] | None = None,
        model: SpeechEncodingModel | None = None,
        load_model: SpeechEncodingModelLoader = load_speech_encoding_model,
        convert: PcmConverter = pcm16,
    ) -> None:
        unsupported = sorted(expressiveness.cues - _APPLICABLE)
        if unsupported:
            raise ValueError(f"An MLX Audio generation cannot apply: {', '.join(unsupported)}.")
        self.format: speech.Format = speech.Format(sample_rate_hz=sample_rate_hz, channels=1)
        self.expressiveness: speech.Expressiveness = expressiveness
        self._model_id: str = model_id
        self._voice: str | None = voice
        self._speed: float = speed
        self._lang_code: str = lang_code
        self._max_tokens: int | None = max_tokens
        self._temperature: float | None = temperature
        self._verbose: bool = verbose
        self._generate_kwargs: dict[str, object] = dict(generate_kwargs or {})
        self._model: SpeechEncodingModel | None = model
        self._load_model: SpeechEncodingModelLoader = load_model
        self._convert: PcmConverter = convert
        self._worker: BoundedWorker = BoundedWorker(name="mlx-audio-speech-encoder")

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[speech.Piece]
    ) -> collections.abc.AsyncIterator[speech.Audio]:
        # The last chunk, held until the next one arrives or the reply ends: the last one of a generation carries
        # the words it said, and the reply's last one also carries any trailing words with nothing to say.
        held: speech.Audio | None = None
        # Blank words read before any chunk to carry them: they ride the next one.
        unsaid = ""
        try:
            async with contextlib.aclosing(self._runs(input)) as runs:
                async for segment in runs:
                    if isinstance(segment, str):
                        # A pause: silence of its own, between the generations around it.
                        if held is not None:
                            yield held
                        held, unsaid = speech.Audio(pcm=self.format.silence(segment), text=unsaid), ""
                        continue
                    words, direction = segment
                    if not words.strip():
                        # A run with nothing to say ends the reply or comes before a pause: its words ride a chunk.
                        if held is None:
                            unsaid += words
                        else:
                            held = held.model_copy(update={"text": held.text + words})
                        continue
                    last: speech.Audio | None = None
                    async with contextlib.aclosing(self._generate(words, direction)) as generated:
                        async for pcm in generated:
                            if held is not None:
                                yield held
                            held = last = speech.Audio(pcm=pcm, text="")
                    if last is None:
                        raise encoding.Error("invalid_response", "MLX Audio generated no audio.")
                    held, unsaid = last.model_copy(update={"text": unsaid + words}), ""
            if held is None:
                raise encoding.Error("invalid_input", "MLX Audio speech encoding requires non-blank text.")
            yield held
        except encoding.Error:
            raise
        except Exception as error:
            raise encoding.Error("encode_failed", "MLX Audio speech encoding failed.") from error

    async def _runs(
        self, input: collections.abc.AsyncIterable[speech.Piece]
    ) -> collections.abc.AsyncGenerator[tuple[str, _Direction] | speech.Pause, None]:
        """The reply's runs, each once it closes: the joined words of consecutive parts with one direction, and
        each pause in its place between them.

        A part that opens with a pause closes the run before it; a run with no words to say yet takes the
        direction of the part that follows it.

        Raises:
            encoding.Error: ``invalid_input`` when the pieces break the stream's invariants.
        """

        words: list[str] = []
        direction: _Direction = (None, None)
        opened = 0
        async for piece in input:
            if piece.part == opened:
                opened += 1
                following = self._direction(piece.expression)
                pause = None if piece.expression is None else piece.expression.pause
                if pause is not None:
                    if words:
                        yield "".join(words), direction
                        words = []
                    yield pause
                spoken = bool("".join(words).strip())
                if spoken and following != direction:
                    yield "".join(words), direction
                    words, spoken = [], False
                if not spoken:
                    direction = following
            elif piece.part != opened - 1 or piece.expression is not None:
                raise encoding.Error("invalid_input", "Speech pieces must open each part in order, once.")
            words.append(piece.words)
        if words:
            yield "".join(words), direction

    def _direction(self, expression: speech.Expression | None) -> _Direction:
        """A part's ``instruct`` and ``speed`` from the cues this encoder declares, the pace clamped to its range."""

        if expression is None:
            return None, None
        cues = self.expressiveness.cues
        emotion = expression.emotion if "emotion" in cues else None
        if self.expressiveness.emotions is not None and emotion not in self.expressiveness.emotions:
            emotion = None
        directed = [
            *([expression.instructions] if expression.instructions is not None and "instructions" in cues else []),
            *([] if emotion is None else [f"Emotion: {emotion}."]),
            *(
                [f"Intensity: {expression.intensity:g} out of 1."]
                if expression.intensity is not None and "intensity" in cues
                else []
            ),
            *(["Stress these words."] if expression.emphasis and "emphasis" in cues else []),
        ]
        pace = self.expressiveness.pace if "pace" in cues else None
        speed = None if pace is None or expression.pace is None else min(max(expression.pace, pace.min), pace.max)
        return (" ".join(directed) or None), speed

    async def _generate(self, words: str, direction: _Direction) -> collections.abc.AsyncGenerator[bytes, None]:
        """Generate one run and yield its PCM as each chunk is produced; closing it stops the generation."""

        instruct, speed = direction
        kwargs = self._kwargs()
        if instruct is not None:
            kwargs["instruct"] = instruct
        if speed is not None:
            kwargs["speed"] = speed
        chunks: collections.abc.Iterator[object] | None = None
        try:
            stream = await self._run(lambda: iter(self._resolve_model().generate(text=words, **kwargs)))
            chunks = stream
            while (result := await self._run(lambda: next(stream, None))) is not None:
                pcm = self._pcm(result)
                if pcm:
                    yield pcm
        finally:
            if isinstance(chunks, collections.abc.Generator):
                generator = typing.cast(collections.abc.Generator[object], chunks)
                closed = self._worker.try_submit(generator.close)
                if closed is None:
                    _LOG.warning("MLX Audio generation could not be closed: the speech encoder worker is at capacity.")
                else:
                    _ = await asyncio.wrap_future(closed)

    async def _run[T](self, operation: collections.abc.Callable[[], T]) -> T:
        submitted: concurrent.futures.Future[T] | None = self._worker.try_submit(operation)
        if submitted is None:
            raise encoding.Error("unavailable", "MLX Audio speech encoder capacity is exhausted.")
        return await asyncio.wrap_future(submitted)

    def _resolve_model(self) -> SpeechEncodingModel:
        # Runs only on the worker thread, one call at a time, so the lazy load needs no lock.
        if self._model is None:
            self._model = self._load_model(self._model_id)
        return self._model

    def _kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "speed": self._speed,
            "lang_code": self._lang_code,
            "verbose": self._verbose,
            "stream": True,
        }
        if self._voice is not None:
            kwargs["voice"] = self._voice
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        kwargs.update(self._generate_kwargs)
        return kwargs

    def _pcm(self, result: object) -> bytes:
        audio = get_member(result, "audio")
        if audio is None:
            raise encoding.Error("invalid_response", "MLX Audio speech result is missing audio.")
        sample_rate = get_member(result, "sample_rate")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise encoding.Error("invalid_response", "MLX Audio speech result is missing an integer sample rate.")
        if sample_rate != self.format.sample_rate_hz:
            raise encoding.Error(
                "invalid_response",
                (
                    f"MLX Audio generated {sample_rate} Hz audio; this encoder declares "
                    + f"{self.format.sample_rate_hz} Hz. Configure sample_rate_hz={sample_rate}."
                ),
            )
        return self._convert(audio)


__all__ = ["SpeechEncoder"]
