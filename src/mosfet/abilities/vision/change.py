"""How much a view changed: injected, device- and model-agnostic change measures.

A ``ChangeMeasure`` is a plain non-HSM collaborator the owning perceiving ability calls directly.
It compares a new image with the ``Signature`` of the last view it handed on and answers with a
``Measurement``: a change in 0..1 (``None`` when there was nothing to compare against) and the new
image's signature. What counts as change is the measure's: a byte digest, an embedding from any
image-capable or omni-modal model, or a cheap perceptual diff from a provider package. ``Cascade``
asks an expensive measure only when a cheap one sees change.

Measures here (providers add more, for example ``opencv.FrameDiff``):

* ``ByteIdentity`` — one SHA-256 per measured frame; for static sources such as screen captures.
* ``EmbeddingChange`` — one embedding forward pass; usually behind a downscaling measure and a
  ``Cascade``.
* ``Cascade`` — the cheap cost on every measured frame, plus the expensive cost only on change.

Signatures are opaque references to a perceived view. They are never placed in products, prompts
or telemetry. They are not biometric: a digest, a downscaled grid or an image embedding is not a
face print.

Telemetry (scope ``bot.abilities.vision``): ``bot.vision.change.cascade.count`` counts cascade
measurements by ``bot.outcome`` {cheap_only, expensive}: 2 series. It shows how much expensive
work a cascade saves.
"""

import abc
import asyncio
import hashlib
import math
import typing

import hsm
import pydantic
from opentelemetry import metrics

from mosfet import media

from .. import embedding
from ..language import text

_MAX_SIGNATURE_VALUES: typing.Final = 65_536
_MAX_SIGNATURE_PARTS: typing.Final = 4

_BYTE_IDENTITY: typing.Final = "byte_identity"
_EMBEDDING: typing.Final = "embedding"
_CASCADE: typing.Final = "cascade"
_JPEG: typing.Final = "image/jpeg"

_METER = metrics.get_meter("bot.abilities.vision")
_CASCADE_COUNT = _METER.create_counter(
    "bot.vision.change.cascade.count",
    unit="{measurement}",
    description="Cascade measurements, by whether the expensive measure was called (expensive) or not (cheap_only).",
)


class Signature(pydantic.BaseModel):
    """An opaque per-measure reference to a previously perceived view; never a product, prompt or telemetry value."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "measure": "byte_identity",
                    "digest": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                },
                {"measure": "embedding", "values": [0.6, 0.8]},
                {
                    "measure": "cascade",
                    "parts": [
                        {"measure": "frame_diff", "values": [0.25, 0.5]},
                        {"measure": "embedding", "values": [0.6, 0.8]},
                    ],
                },
            ],
        },
    )

    measure: str = pydantic.Field(
        min_length=1,
        max_length=100,
        description="Name of the measure that produced this signature; a measure compares only its own signatures.",
        examples=["byte_identity", "embedding", "cascade"],
    )
    values: tuple[typing.Annotated[float, pydantic.Field(allow_inf_nan=False)], ...] = pydantic.Field(
        default=(),
        max_length=_MAX_SIGNATURE_VALUES,
        description="Finite numbers the measure compares, such as a downscaled grayscale grid or an embedding.",
        examples=[[0.6, 0.8]],
    )
    digest: str | None = pydantic.Field(
        default=None,
        max_length=128,
        description="A content digest the measure compares for equality, such as a SHA-256 hex digest.",
        examples=["9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"],
    )
    parts: tuple["Signature", ...] = pydantic.Field(
        default=(),
        max_length=_MAX_SIGNATURE_PARTS,
        description="Signatures of the measures a composite measure combines, in its own order.",
        examples=[[{"measure": "frame_diff", "values": [0.25, 0.5]}, {"measure": "embedding", "values": [0.6, 0.8]}]],
    )


class Measurement(pydantic.BaseModel):
    """One measured view: how much it changed since the previous signature, and its own signature."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"change": 0.42, "signature": {"measure": "embedding", "values": [0.6, 0.8]}},
                {"change": None, "signature": {"measure": "embedding", "values": [1.0, 0.0]}},
            ],
        },
    )

    change: float | None = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="How much the view changed, 0 (same) to 1 (entirely different); null when there was no previous signature.",
        examples=[0.42, None],
    )
    signature: Signature = pydantic.Field(
        description="This view's signature: what the owner keeps if it hands the view on.",
        examples=[{"measure": "embedding", "values": [0.6, 0.8]}],
    )


class Error(RuntimeError):
    """A change measure could not measure a view; the owner counts it and hands nothing on."""


class ChangeMeasure(abc.ABC):
    """Measures how much an image changed since a previous signature; a plain collaborator, never an HSM.

    A measure is device- and model-agnostic: it knows nothing about cameras, screens or which model
    it wraps. ``attachments`` names provider actors the owning ability attaches for its whole
    attached lifetime (the same contract as ``embedding.Embedder.attachments``); each must already
    be started in a long-lived scope by the composition root.
    """

    attachments: tuple[hsm.Instance, ...] = ()

    @abc.abstractmethod
    async def measure(self, previous: Signature | None, image: media.Image, *, ctx: hsm.Context) -> Measurement:
        """Measure ``image`` against ``previous``, the signature of the last view handed on, if any.

        Returns ``change=None`` when ``previous`` is ``None``. Implementations do blocking work off
        the event loop and return promptly when ``ctx`` is cancelled. A provider call that cannot be
        cancelled keeps running in its worker thread, so providers bound their own work.

        Raises:
            Error: The view could not be measured, including a ``previous`` this measure cannot compare.
        """
        ...


class ByteIdentity(ChangeMeasure):
    """Changed (1.0) unless the encoded bytes are identical (0.0); one SHA-256 per measured frame.

    For static sources such as screen captures. On a live camera every encoded frame differs, so
    every measurement rates as changed: use a perceptual or embedding measure there instead.
    """

    @typing.override
    async def measure(self, previous: Signature | None, image: media.Image, *, ctx: hsm.Context) -> Measurement:
        del ctx
        if previous is not None and (previous.measure != _BYTE_IDENTITY or previous.digest is None):
            raise Error("previous signature is not a byte identity signature")
        hashed = await asyncio.to_thread(hashlib.sha256, image.content)
        signature = Signature(measure=_BYTE_IDENTITY, digest=hashed.hexdigest())
        if previous is None:
            return Measurement(change=None, signature=signature)
        return Measurement(change=0.0 if previous.digest == signature.digest else 1.0, signature=signature)


class EmbeddingChange(ChangeMeasure):
    """One minus the cosine similarity of image embeddings, clamped to 0..1; one forward pass per frame.

    Works with any image-capable embedder, including omni-modal ones. The embedder is a plain
    collaborator called directly; its provider actors are this measure's ``attachments``.
    """

    _embedder: embedding.Embedder

    def __init__(self, *, embedder: embedding.Embedder) -> None:
        """Raises ``ValueError`` unless ``embedder`` can take at least one ``image/jpeg`` image per request."""

        images = embedder.capacity.images
        if _JPEG not in images.media_types or images.max_images < 1 or images.max_total_bytes < 1:
            raise ValueError("EmbeddingChange needs an embedder whose image capacity admits image/jpeg.")
        self._embedder = embedder
        self.attachments: tuple[hsm.Instance, ...] = embedder.attachments

    @typing.override
    async def measure(self, previous: Signature | None, image: media.Image, *, ctx: hsm.Context) -> Measurement:
        if previous is not None and previous.measure != _EMBEDDING:
            raise Error("previous signature is not an embedding signature")
        if self._embedder.capacity.images.admit((image,)) != (None,):
            raise Error("the embedder's image capacity does not admit this image")
        request = embedding.InputData(
            role=embedding.Role.DOCUMENT,
            items=(embedding.Item(parts=(text.ImagePart(image=image),)),),
        )
        try:
            embedded = await self._embedder.embed(request, ctx=ctx)
        except embedding.Error as error:
            raise Error(f"the embedder failed: {error.failure_kind}") from error
        if len(embedded.vectors) != 1:
            raise Error("the embedder did not return exactly one vector")
        current = embedded.vectors[0].values
        signature = Signature(measure=_EMBEDDING, values=current)
        if previous is None:
            return Measurement(change=None, signature=signature)
        if len(previous.values) != len(current):
            raise Error("previous signature has a different dimension")
        norms = math.sqrt(math.fsum(value * value for value in previous.values)) * math.sqrt(
            math.fsum(value * value for value in current)
        )
        if norms == 0.0:
            raise Error("previous signature is a zero vector")
        cosine = math.fsum(left * right for left, right in zip(previous.values, current, strict=True)) / norms
        return Measurement(change=min(1.0, max(0.0, 1.0 - cosine)), signature=signature)


class Cascade(ChangeMeasure):
    """Asks ``expensive`` only when ``cheap`` sees change of at least ``cheap_threshold``.

    Below the cheap threshold the cascade answers ``change=0.0`` (never the cheap measure's own
    number, which is on a different scale than the owner's threshold) and keeps the previous
    expensive signature, so drift keeps accumulating against the last view handed on. A first view
    computes both signatures once. Cost: the cheap cost on every measured frame, plus the expensive
    cost only when the cheap measure sees change.
    """

    _cheap: ChangeMeasure
    _expensive: ChangeMeasure
    _cheap_threshold: float

    def __init__(self, *, cheap: ChangeMeasure, expensive: ChangeMeasure, cheap_threshold: float) -> None:
        """Raises ``ValueError`` unless ``0 < cheap_threshold <= 1``."""

        if not 0.0 < cheap_threshold <= 1.0:
            raise ValueError("cheap_threshold must be in (0, 1].")
        self._cheap = cheap
        self._expensive = expensive
        self._cheap_threshold = cheap_threshold
        self.attachments: tuple[hsm.Instance, ...] = cheap.attachments + expensive.attachments

    @typing.override
    async def measure(self, previous: Signature | None, image: media.Image, *, ctx: hsm.Context) -> Measurement:
        if previous is not None and (previous.measure != _CASCADE or len(previous.parts) != 2):
            raise Error("previous signature is not a cascade signature")
        cheap = await self._cheap.measure(None if previous is None else previous.parts[0], image, ctx=ctx)
        if previous is not None and cheap.change is not None and cheap.change < self._cheap_threshold:
            _CASCADE_COUNT.add(1, {"bot.outcome": "cheap_only"})
            return Measurement(
                change=0.0, signature=Signature(measure=_CASCADE, parts=(cheap.signature, previous.parts[1]))
            )
        expensive = await self._expensive.measure(None if previous is None else previous.parts[1], image, ctx=ctx)
        _CASCADE_COUNT.add(1, {"bot.outcome": "expensive"})
        return Measurement(
            change=expensive.change,
            signature=Signature(measure=_CASCADE, parts=(cheap.signature, expensive.signature)),
        )


__all__ = [
    "ByteIdentity",
    "Cascade",
    "ChangeMeasure",
    "EmbeddingChange",
    "Error",
    "Measurement",
    "Signature",
]
