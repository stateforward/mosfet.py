"""MLX Audio text-to-speech as a streaming ``vocal.speech.SpeechEncoder``."""

from __future__ import annotations

import asyncio
import collections.abc
import concurrent.futures
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


class SpeechEncoder(speech.SpeechEncoder):
    """MLX Audio TTS, streamed: the model's generated chunks are yielded as mono s16le PCM as they are produced.

    ``generate`` takes the whole text, so the text is read to its end first; joining its pieces in order is the
    only buffering. Alignment is emulated over the whole text: every chunk carries ``""`` but the last, which
    carries the whole text, so one generated chunk is held back until the next arrives or the generation ends.
    Native work runs one call at a time on a bounded worker: loading the model, starting a generation, and
    pulling each chunk. Closing the stream between chunks stops the generation; a chunk already being generated
    finishes first, because MLX native work cannot be interrupted. The model's sample rate must match the
    declared ``sample_rate_hz``: raw PCM carries no header, so a mismatch is refused as ``invalid_response``
    rather than played at the wrong speed. ``voice`` is the model voice the text is spoken in.
    """

    def __init__(
        self,
        *,
        model_id: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        voice: str | None = None,
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
        self.format: speech.Format = speech.Format(sample_rate_hz=sample_rate_hz, channels=1)
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
    async def encode(self, input: collections.abc.AsyncIterable[str]) -> collections.abc.AsyncIterator[speech.Audio]:
        said = "".join([piece async for piece in input])
        if not said.strip():
            raise encoding.Error("invalid_input", "MLX Audio speech encoding requires non-blank text.")
        chunks: collections.abc.Iterator[object] | None = None
        try:
            stream = await self._run(lambda: iter(self._resolve_model().generate(text=said, **self._kwargs())))
            chunks = stream
            held: bytes | None = None
            while (result := await self._run(lambda: next(stream, None))) is not None:
                pcm = self._pcm(result)
                if not pcm:
                    continue
                if held is not None:
                    yield speech.Audio(pcm=held, text="")
                held = pcm
            if held is None:
                raise encoding.Error("invalid_response", "MLX Audio generated no audio.")
            yield speech.Audio(pcm=held, text=said)
        except encoding.Error:
            raise
        except Exception as error:
            raise encoding.Error("encode_failed", "MLX Audio speech encoding failed.") from error
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
