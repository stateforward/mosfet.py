from __future__ import annotations

from bot.abilities.hearing import speech

import asyncio
import collections
import collections.abc
import concurrent.futures
import dataclasses
import functools
import inspect
import threading
import typing

from bot.telemetry import span

from ._audio_file import temporary_audio_file
from ._mlx import SpeechDecodingModel, SpeechDecodingModelLoader, load_speech_decoding_model

_SCOPE = "bot.providers.mlx_audio"
_COMPONENT = "mlx_audio.speech_decoder"
_CLOSE_TIMEOUT_SECONDS = 0.1
_MAX_ACCEPTED_WORK = 2


@dataclasses.dataclass(frozen=True)
class _OwnedModel:
    model: SpeechDecodingModel
    execution_guard: threading.Lock = dataclasses.field(default_factory=threading.Lock)


_T = typing.TypeVar("_T")


@dataclasses.dataclass
class _WorkItem(typing.Generic[_T]):
    operation: collections.abc.Callable[[], _T] | None
    result: concurrent.futures.Future[_T]
    released: bool = False


class BoundedWorker:
    """Run at most one native call while retaining at most one queued call.

    The worker thread is daemonized because Python cannot interrupt a native call that never
    returns. Cooperative calls are joined during close; a permanently blocked native call cannot
    make process shutdown or the provider's bounded terminal operation wait forever.
    """

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._condition = threading.Condition()
        self._pending: collections.deque[_WorkItem[object]] = collections.deque()
        self._accepted = 0
        self._closed = False
        self._thread: threading.Thread | None = None

    def try_submit(self, operation: collections.abc.Callable[[], _T]) -> concurrent.futures.Future[_T] | None:
        with self._condition:
            if self._closed or self._accepted >= _MAX_ACCEPTED_WORK:
                return None
            result: concurrent.futures.Future[_T] = concurrent.futures.Future()
            item = _WorkItem(operation=operation, result=result)
            self._accepted += 1
            self._pending.append(typing.cast(_WorkItem[object], item))
            result.add_done_callback(lambda completed: self._release_cancelled(item, completed))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                self._thread.start()
            self._condition.notify()
            return result

    def close(self, *, wait: bool) -> None:
        with self._condition:
            if not self._closed:
                self._closed = True
                pending = tuple(self._pending)
            else:
                pending = ()
            self._condition.notify_all()
            thread = self._thread
        for item in pending:
            _ = item.result.cancel()
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_CLOSE_TIMEOUT_SECONDS)

    def _release_cancelled(
        self,
        item: _WorkItem[_T],
        completed: concurrent.futures.Future[_T],
    ) -> None:
        if not completed.cancelled():
            return
        with self._condition:
            try:
                self._pending.remove(typing.cast(_WorkItem[object], item))
            except ValueError:
                return
            self._release_unlocked(typing.cast(_WorkItem[object], item))

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending:
                    return
                item = self._pending.popleft()
            if not item.result.set_running_or_notify_cancel():
                with self._condition:
                    self._release_unlocked(item)
                continue
            operation = item.operation
            try:
                if operation is None:
                    raise RuntimeError("native work was released before execution")
                item.result.set_result(operation())
            except BaseException as error:
                item.result.set_exception(error)
            finally:
                with self._condition:
                    self._release_unlocked(item)

    def _release_unlocked(self, item: _WorkItem[object]) -> None:
        if item.released:
            return
        item.released = True
        item.operation = None
        self._accepted -= 1
        self._condition.notify_all()


class SpeechDecodingError(RuntimeError):
    """Raised when MLX Audio speech decoding fails."""


def _empty_generate_kwargs() -> collections.abc.Mapping[str, object]:
    return {}


@dataclasses.dataclass(frozen=True, kw_only=True)
class SpeechDecoder(speech.SpeechDecoder):
    """Speech decoder backed by MLX Audio speech-to-text models.

    The decoder accepts encoded audio bytes, writes them to a temporary audio
    file, and returns the local STT transcript as UTF-8 bytes to match stateforward.bot's
    existing speech-decoding contract. Each decoder owns one worker and one model. It accepts at
    most one active and one queued call; further calls fail immediately with
    ``SpeechDecodingError`` instead of retaining unbounded audio.

    Cancelling a queued ``decode`` removes it before native execution and releases its audio.
    Cancelling an active ``decode`` cancels only the caller's wait because MLX native work is not
    interruptible; capacity remains occupied until that call returns. ``shutdown`` rejects new
    work and cancels queued work. ``close`` and ``aclose`` wait up to 100 ms for cooperative active
    work, then return with any permanently blocked call isolated to a daemon worker so it cannot
    hold process shutdown open. Cancelling ``aclose`` is reported after this bounded cleanup.
    """

    model_id: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "en"
    verbose: bool = False
    audio_file_suffix: str = ".wav"
    generate_kwargs: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_generate_kwargs)
    model: SpeechDecodingModel | None = None
    load_model: SpeechDecodingModelLoader = load_speech_decoding_model
    _worker: BoundedWorker | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _worker_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )
    _closed: bool = dataclasses.field(default=False, init=False, repr=False, compare=False)
    _owned_model: _OwnedModel | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _model_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        loop = asyncio.get_running_loop()
        with self._worker_lock:
            if self._closed:
                raise RuntimeError("MLX Audio speech decoder is closed.")
            worker = self._worker
            if worker is None:
                worker = BoundedWorker(name="mlx-audio-speech-decoder")
                object.__setattr__(self, "_worker", worker)
            result = worker.try_submit(functools.partial(span.bind(self._decode_blocking), input))
        if result is None:
            with span.operation(
                "bot.provider.mlx_audio.speech_decoder.admission",
                scope=_SCOPE,
                component=_COMPONENT,
                stage="admission",
            ):
                raise SpeechDecodingError("MLX Audio speech decoder capacity is exhausted.")
        return await asyncio.wrap_future(result, loop=loop)

    def shutdown(self) -> None:
        """Stop accepting work and cancel queued decoding calls."""

        worker = self._begin_shutdown()
        if worker is not None:
            worker.close(wait=False)

    def close(self) -> None:
        """Reject work, cancel queued calls, and wait at most 100 ms for active work."""

        worker = self._begin_shutdown()
        if worker is None:
            return
        with span.operation(
            "bot.provider.mlx_audio.speech_decoder.close",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="close",
        ):
            worker.close(wait=True)

    async def aclose(self) -> None:
        """Perform bounded terminal cleanup, deferring cancellation until it finishes."""

        worker = self._begin_shutdown()
        if worker is None:
            return
        with span.operation(
            "bot.provider.mlx_audio.speech_decoder.close",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="close",
        ):
            shutdown = asyncio.create_task(
                asyncio.to_thread(worker.close, wait=True),
                name="mlx-audio-speech-decoder-close",
            )
            try:
                await asyncio.shield(shutdown)
            except asyncio.CancelledError:
                await shutdown
                raise

    def _begin_shutdown(self) -> BoundedWorker | None:
        with self._worker_lock:
            object.__setattr__(self, "_closed", True)
            return self._worker

    def _decode_blocking(self, audio: bytes) -> bytes:
        with span.operation(
            "bot.provider.mlx_audio.speech_decoder.decode",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="decode",
        ) as active:
            try:
                cached, cache_result = self._resolve_model()
                active.set_attribute("bot.model.cache.result", cache_result)
                with cached.execution_guard:
                    with temporary_audio_file(audio, suffix=self.audio_file_suffix) as audio_path:
                        result = cached.model.generate(
                            str(audio_path),
                            **_accepted_generate_kwargs(
                                cached.model,
                                {
                                    "language": self.language,
                                    "verbose": self.verbose,
                                    **self.generate_kwargs,
                                },
                            ),
                        )
                return _transcription_text(result).encode("utf-8")
            except Exception as error:
                message = "MLX Audio speech decoding failed."
                raise SpeechDecodingError(message) from error

    def _resolve_model(self) -> tuple[_OwnedModel, str]:
        with self._model_lock:
            owned = self._owned_model
            if owned is not None:
                return owned, "hit"
            owned = _OwnedModel(model=self.model if self.model is not None else self.load_model(self.model_id))
            object.__setattr__(self, "_owned_model", owned)
            return owned, "injected" if self.model is not None else "miss"


def _accepted_generate_kwargs(
    model: SpeechDecodingModel,
    kwargs: collections.abc.Mapping[str, object],
) -> dict[str, object]:
    try:
        signature = inspect.signature(model.generate)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(kwargs)

    return {name: value for name, value in kwargs.items() if name in signature.parameters}


def _transcription_text(result: object) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, collections.abc.Mapping):
        result_mapping = typing.cast(collections.abc.Mapping[object, object], result)
        text = result_mapping.get("text")
        if isinstance(text, str):
            return text
        message = "MLX Audio speech decoding result is missing transcript text."
        raise TypeError(message)

    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text

    message = "MLX Audio speech decoding result is missing transcript text."
    raise TypeError(message)


__all__ = [
    "SpeechDecoder",
    "SpeechDecodingError",
]
