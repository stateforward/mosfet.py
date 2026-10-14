"""The pinned native llama.cpp library: the only code in this package that touches ``llama_cpp``.

``Library.load`` opens an embedding session on the exact native path proven by the spike: a non-causal model
with mean pooling and one ``n_ctx = n_batch = n_ubatch`` context, plus an optional mtmd projector. One item's text
tokens and image soft tokens are packed into a single ``llama_batch_ext`` and processed with one
``llama_process(DECODE)``, so attention and pooling see the whole input at once.

Process-wide native state: ``llama_backend_init`` (idempotent, never paired with ``llama_backend_free``) and one
log callback that forwards native WARN/ERROR lines to this module's logger and drops INFO/DEBUG. The callback is
created once per process and kept alive by ``_log_forwarder``'s cache, because native code may log at any time.

Telemetry: ``bot.provider.llama_cpp.tokens.count`` (counter): ``bot.llama_cpp.token.kind`` {text, image} ×
``bot.llama_cpp.purpose`` {embedding}: 2 series.
"""

from __future__ import annotations

import collections.abc
import ctypes
import functools
import importlib
import logging
import typing

from opentelemetry import metrics

from mosfet.abilities import embedding
from mosfet.abilities.language import text
from mosfet.protocols import attachment

from .runtime import Accelerator, EmbeddingContext, EmbeddingSession
from .weights import Weights

if typing.TYPE_CHECKING:
    from llama_cpp.llama_cpp import llama_context_p, llama_model_p
    from llama_cpp.mtmd_cpp import mtmd_context_p

_LOG = logging.getLogger(__name__)
_TOKENS = metrics.get_meter("mosfet.providers.llama_cpp").create_counter(
    "bot.provider.llama_cpp.tokens.count",
    unit="{token}",
    description="Tokens embedded, by token kind and purpose.",
)
_WARN: typing.Final = 3
"""``GGML_LOG_LEVEL_WARN``; ``ERROR`` is 4. Lower levels (DEBUG, INFO) and ``CONT`` (5) are dropped."""
_ERROR: typing.Final = 4
_POOLING_MEAN: typing.Final = 1

_BUILD = "0.3.36+mosfet.4fbc76de"
"""The one llama-cpp-python build whose ctypes bindings match the native library this package drives."""
_INSTALL = "uv sync --package mosfet-provider-llama-cpp"


class LibraryError(RuntimeError):
    """The native library is missing, is not the pinned build, or cannot be read; it fails the model load."""

    failure_kind: str = attachment.FailureKind.INITIALIZATION.value
    """Read by ``telemetry.span.failure_kind``: a library failure is always a failed initialization."""


class Library:
    """The pinned llama-cpp-python build, imported only when first used.

    Constructing a ``Library`` imports nothing, so a composition root can build one on the event loop. Every
    operation verifies that ``llama_cpp`` is exactly the pinned build ``0.3.36+mosfet.4fbc76de``. Any other
    build — the official 0.3.36 wheels, which vendor an older llama.cpp, or a newer upstream release — raises
    ``LibraryError`` instead of driving a native library whose struct layout the bindings were not proven against.
    Operations block (the first one loads the native library and its GPU backend), so callers run them off the
    event loop.
    """

    def __init__(self, *, module: object | None = None) -> None:
        """Use ``module`` as the ``llama_cpp`` bindings, or import the installed build on first use when ``None``."""
        self._module: object | None = module

    def load(self, weights: Weights, context: EmbeddingContext) -> EmbeddingSession:
        """Verify the pinned build and load ``weights`` into a fresh embedding session; blocking.

        Raises:
            LibraryError: The library is not the pinned build, a file does not load, the model's trained context is
                shorter than ``context.tokens``, the model does not mean-pool, or the projector cannot see images or
                needs M-RoPE positions.
        """

        accelerator = self.accelerator()
        from llama_cpp import llama_cpp as native
        from llama_cpp import mtmd_cpp as mtmd

        forwarder = _log_forwarder()
        native.llama_log_set(forwarder, ctypes.c_void_p())
        mtmd.mtmd_log_set(forwarder, ctypes.c_void_p())
        mtmd.mtmd_helper_log_set(forwarder, ctypes.c_void_p())
        native.llama_backend_init()
        session = _Session(accelerator=accelerator, tokens=context.tokens)
        try:
            model_params = native.llama_model_default_params()
            model_params.n_gpu_layers = context.gpu_layers
            model = native.llama_model_load_from_file(str(weights.model).encode(), model_params)
            if not model:
                raise LibraryError("llama.cpp could not load the model weights")
            session.model = model
            trained = native.llama_model_n_ctx_train(model)
            if context.tokens > trained:
                raise LibraryError(f"context tokens {context.tokens} exceed the model's trained context {trained}")
            context_params = native.llama_context_default_params()
            context_params.n_ctx = context_params.n_batch = context_params.n_ubatch = context.tokens
            context_params.n_seq_max = 1
            context_params.embeddings = True
            context_params.pooling_type = _POOLING_MEAN
            if context.threads is not None:
                context_params.n_threads = context_params.n_threads_batch = context.threads
            native_context = native.llama_init_from_model(model, context_params)
            if not native_context:
                raise LibraryError("llama.cpp could not create the embedding context")
            session.context = native_context
            if native.llama_pooling_type(native_context) != _POOLING_MEAN:
                raise LibraryError("the model does not mean-pool its embeddings")
            session.native_dimensions = native.llama_model_n_embd_out(model)
            session.input_dimensions = native.llama_model_n_embd_inp(model)
            if weights.projector is not None:
                projector_params = mtmd.mtmd_context_params_default()
                projector_params.use_gpu = context.gpu_layers != 0
                projector_params.print_timings = False
                projector_params.image_min_tokens = projector_params.image_max_tokens = context.image_tokens
                if context.threads is not None:
                    projector_params.n_threads = context.threads
                projector = mtmd.mtmd_init_from_file(str(weights.projector).encode(), model, projector_params)
                if not projector:
                    raise LibraryError("llama.cpp could not load the projector weights")
                session.projector = projector
                if not mtmd.mtmd_support_vision(projector):
                    raise LibraryError("the projector cannot see images")
                if mtmd.mtmd_decode_use_mrope(projector):
                    raise LibraryError("the projector needs M-RoPE positions, which embedding does not support")
                session.vision = True
            session.marker = mtmd.mtmd_default_marker().decode("utf-8")
        except BaseException:
            session.close()
            raise
        return session

    def accelerator(self) -> Accelerator:
        """Verify the pinned build and return the accelerator its registered backends compute on.

        Raises ``LibraryError`` when ``llama_cpp`` is not importable, is not the pinned build, or reports a system
        info string with no backend in it.
        """
        bindings = self._module
        if bindings is None:
            try:
                bindings = importlib.import_module("llama_cpp")
            except ImportError as error:
                raise LibraryError(f"llama_cpp is not installed; install the pinned build with `{_INSTALL}`") from error
        version = getattr(bindings, "__version__", None)
        if version != _BUILD:
            raise LibraryError(
                f"llama_cpp {version!r} is not the pinned build {_BUILD}; reinstall it with `{_INSTALL}`"
            )
        system_info = typing.cast(
            collections.abc.Callable[[], object], getattr(bindings, "llama_print_system_info", lambda: None)
        )()
        if not isinstance(system_info, bytes):
            raise LibraryError(f"llama.cpp system info is {type(system_info).__name__}, not bytes")
        # Each registered backend opens a "<NAME> : <feature> = <value>" segment; plain features have no " : ".
        segments = system_info.decode("utf-8", errors="replace").split(" | ")
        backends = {segment.split(" : ", 1)[0].strip() for segment in segments if " : " in segment}
        if not backends:
            raise LibraryError("llama.cpp system info names no backend; the native library did not initialize")
        if "CUDA" in backends:
            return Accelerator.CUDA
        if backends & {"MTL", "Metal"}:
            return Accelerator.METAL
        return Accelerator.CPU


@functools.cache
def _log_forwarder():  # the ctypes callback type is nameable only through private typeshed names
    """The one native log callback of this process: WARN/ERROR lines go to this module's logger.

    Cached so the ctypes callback outlives every native caller; it holds no state, so tests need no reset.
    """

    from llama_cpp import llama_cpp as native

    def forward(level: int, message: bytes | None, user_data: object) -> None:
        del user_data
        if level not in (_WARN, _ERROR) or not message:
            return
        line = message.decode("utf-8", errors="replace").rstrip()
        if line:
            _LOG.log(logging.ERROR if level == _ERROR else logging.WARNING, "llama.cpp: %s", line)

    return native.llama_log_callback(forward)


class _Session:
    """An ``EmbeddingSession`` over the native library; used only on its owning model's worker thread."""

    accelerator: Accelerator
    native_dimensions: int
    vision: bool
    input_dimensions: int
    marker: str
    model: llama_model_p | None
    context: llama_context_p | None
    projector: mtmd_context_p | None
    _tokens: int

    def __init__(self, *, accelerator: Accelerator, tokens: int) -> None:
        self.accelerator = accelerator
        self.native_dimensions = 0
        self.vision = False
        self.input_dimensions = 0
        self.marker = ""
        self.model = None
        self.context = None
        self.projector = None
        self._tokens = tokens

    def embed(self, parts: collections.abc.Sequence[text.ContentPart], *, item_index: int) -> tuple[float, ...]:
        """Embed one item in one batch and return its raw mean-pooled vector; blocking.

        Raises:
            embedding.Error: ``invalid_input`` for text holding the media marker or an undecodable image,
                ``unsupported_input`` for an image without a projector, ``too_long`` past ``tokens`` (never
                truncated), ``embed_failed`` when the native pass fails.
        """

        from llama_cpp import llama_cpp as native
        from llama_cpp import mtmd_cpp as mtmd

        context, projector = self.context, self.projector
        assert context is not None, "an open session has a context"
        pieces: list[str] = []
        images: list[bytes] = []
        for part in parts:
            if isinstance(part, text.TextPart):
                if self.marker and self.marker in part.text:
                    raise embedding.Error(
                        embedding.FailureKind.INVALID_INPUT,
                        "text contains the model's media marker",
                        item_index=item_index,
                    )
                pieces.append(part.text)
            else:
                if projector is None:
                    raise embedding.Error(
                        embedding.FailureKind.UNSUPPORTED_INPUT,
                        "the model has no projector loaded and cannot see images",
                        item_index=item_index,
                    )
                pieces.append(self.marker)
                images.append(part.image.content)
        prompt = "".join(pieces).encode("utf-8")
        batch = native.llama_batch_ext_init(context)
        if not batch:
            raise embedding.Error(embedding.FailureKind.EMBED_FAILED, "llama.cpp could not allocate a batch")
        bitmaps = (mtmd.mtmd_bitmap_p_ctypes * max(len(images), 1))()
        chunks = None
        decoded = 0
        try:
            position = 0
            text_tokens = 0
            image_tokens = 0

            def place(index: int) -> None:
                nonlocal position
                if index < 0:
                    raise embedding.Error(
                        embedding.FailureKind.EMBED_FAILED, f"llama.cpp batch rejected a row ({index})"
                    )
                positions = (native.llama_pos * 4)(position, position, position, 0)
                _ = native.llama_batch_ext_set_pos(
                    batch, index, ctypes.cast(positions, ctypes.POINTER(native.llama_pos))
                )
                _ = native.llama_batch_ext_set_output_embd(batch, index, True)
                position += 1

            if projector is None:
                tokens = self._tokenize(prompt, item_index)
                for token in tokens:
                    place(native.llama_batch_ext_add_token(batch, 0, token))
                text_tokens = len(tokens)
            else:
                for content in images:
                    buffer = (ctypes.c_uint8 * len(content)).from_buffer_copy(content)
                    bitmap = mtmd.mtmd_helper_bitmap_init_from_buf(projector, buffer, len(content), False)
                    if not bitmap:
                        raise embedding.Error(
                            embedding.FailureKind.INVALID_INPUT, "an image could not be decoded", item_index=item_index
                        )
                    bitmaps[decoded] = bitmap
                    decoded += 1
                chunks = mtmd.mtmd_input_chunks_init()
                if not chunks:
                    raise embedding.Error(embedding.FailureKind.EMBED_FAILED, "llama.cpp could not allocate chunks")
                prompt_text = mtmd.mtmd_input_text(
                    text=prompt, text_len=len(prompt), add_special=True, parse_special=False
                )
                if mtmd.mtmd_tokenize(projector, chunks, ctypes.pointer(prompt_text), bitmaps, decoded) != 0:
                    raise embedding.Error(
                        embedding.FailureKind.INVALID_INPUT, "the item could not be tokenized", item_index=item_index
                    )
                total = mtmd.mtmd_helper_get_n_tokens(chunks)
                if total > self._tokens:
                    raise embedding.Error(
                        embedding.FailureKind.TOO_LONG,
                        f"item uses {total} tokens; the model embeds at most {self._tokens}",
                        item_index=item_index,
                    )
                for chunk_index in range(mtmd.mtmd_input_chunks_size(chunks)):
                    chunk = mtmd.mtmd_input_chunks_get(chunks, chunk_index)
                    assert chunk is not None, "chunk indices come from mtmd_input_chunks_size"
                    kind = mtmd.mtmd_input_chunk_get_type(chunk)
                    if kind == mtmd.MTMD_INPUT_CHUNK_TYPE_TEXT:
                        count = ctypes.c_size_t()
                        tokens = mtmd.mtmd_input_chunk_get_tokens_text(chunk, ctypes.pointer(count))
                        assert tokens is not None
                        for offset in range(count.value):
                            place(native.llama_batch_ext_add_token(batch, 0, tokens[offset]))
                        text_tokens += count.value
                    elif kind == mtmd.MTMD_INPUT_CHUNK_TYPE_IMAGE:
                        if mtmd.mtmd_encode_chunk(projector, chunk) != 0:
                            raise embedding.Error(
                                embedding.FailureKind.EMBED_FAILED,
                                "llama.cpp could not encode an image",
                                item_index=item_index,
                            )
                        rows = mtmd.mtmd_input_chunk_get_n_tokens(chunk)
                        width = self.input_dimensions
                        # Own a copy: the projector's output buffer is overwritten by the next encode.
                        owned = (ctypes.c_float * (rows * width))()
                        source = mtmd.mtmd_get_output_embd(projector)
                        assert source is not None
                        _ = ctypes.memmove(owned, source, ctypes.sizeof(owned))
                        for row in range(rows):
                            pointer = ctypes.cast(
                                ctypes.byref(owned, row * width * ctypes.sizeof(ctypes.c_float)),
                                ctypes.POINTER(ctypes.c_float),
                            )
                            place(native.llama_batch_ext_add_embd(batch, 0, native.llama_embd(pointer, 1, width)))
                        image_tokens += rows
                    else:
                        raise embedding.Error(
                            embedding.FailureKind.UNSUPPORTED_INPUT,
                            "the item holds media other than text and images",
                            item_index=item_index,
                        )
            memory = native.llama_get_memory(context)
            if memory:
                native.llama_memory_clear(memory, True)
            if native.llama_process(context, native.LLAMA_PROCESS_TYPE_DECODE, batch) != 0:
                raise embedding.Error(
                    embedding.FailureKind.EMBED_FAILED, "llama.cpp could not process the item", item_index=item_index
                )
            pooled = native.llama_get_embeddings_seq(context, 0)
            if not pooled:
                raise embedding.Error(
                    embedding.FailureKind.EMBED_FAILED, "llama.cpp returned no pooled vector", item_index=item_index
                )
            _TOKENS.add(text_tokens, {"bot.llama_cpp.token.kind": "text", "bot.llama_cpp.purpose": "embedding"})
            _TOKENS.add(image_tokens, {"bot.llama_cpp.token.kind": "image", "bot.llama_cpp.purpose": "embedding"})
            return tuple(float(pooled[index]) for index in range(self.native_dimensions))
        finally:
            native.llama_batch_ext_free(batch)
            if chunks:
                mtmd.mtmd_input_chunks_free(chunks)
            for index in range(decoded):
                mtmd.mtmd_bitmap_free(bitmaps[index])

    def _tokenize(self, prompt: bytes, item_index: int) -> list[int]:
        """Text-only tokenization (no projector): special tokens added, never parsed from user text."""

        from llama_cpp import llama_cpp as native

        assert self.model is not None
        vocab = native.llama_model_get_vocab(self.model)
        if vocab is None:
            raise embedding.Error(
                embedding.FailureKind.EMBED_FAILED, "the model has no vocabulary", item_index=item_index
            )
        capacity = self._tokens + 1
        tokens = (native.llama_token * capacity)()
        count = native.llama_tokenize(vocab, prompt, len(prompt), tokens, capacity, True, False)
        if count < 0 or count > self._tokens:
            needed = -count if count < 0 else count
            raise embedding.Error(
                embedding.FailureKind.TOO_LONG,
                f"item uses {needed} tokens; the model embeds at most {self._tokens}",
                item_index=item_index,
            )
        return [int(tokens[index]) for index in range(count)]

    def close(self) -> None:
        """Free the projector, the context and the model; blocking and idempotent."""

        from llama_cpp import llama_cpp as native
        from llama_cpp import mtmd_cpp as mtmd

        projector, self.projector = self.projector, None
        context, self.context = self.context, None
        model, self.model = self.model, None
        if projector:
            mtmd.mtmd_free(projector)
        if context:
            native.llama_free(context)
        if model:
            native.llama_model_free(model)
