from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import typing

import bot.abilities

from ._image_file import temporary_image_file
from ._mlx import (
    ChatTemplateApplier,
    ConfigLoader,
    ImageGenerator,
    ModelLoader,
    apply_chat_template,
    generate,
    load_config,
    load_model,
)


class ImageDecodingError(RuntimeError):
    """Raised when MLX VLM image decoding fails."""


def _default_generate_kwargs() -> collections.abc.Mapping[str, object]:
    return {"max_tokens": 512, "temperature": 0.0}


@dataclasses.dataclass(frozen=True, kw_only=True)
class ImageDecoder(bot.abilities.Decoder[bytes, str]):
    """Image decoder backed by an MLX vision-language model.

    The decoder accepts encoded image bytes, writes them to a temporary image
    file, and asks an injected or loaded MLX VLM model to return only readable
    text from the image.
    """

    model_id: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    prompt: str = "Read all visible text in this image. Return only the text."
    image_file_suffix: str = ".png"
    verbose: bool = False
    generate_kwargs: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_default_generate_kwargs)
    model: object | None = None
    processor: object | None = None
    config: object | None = None
    load_model: ModelLoader = load_model
    load_config: ConfigLoader = load_config
    apply_chat_template: ChatTemplateApplier = apply_chat_template
    generate: ImageGenerator = generate

    @typing.override
    async def decode(self, input: bytes) -> str:
        return await asyncio.to_thread(self._decode_blocking, input)

    def _decode_blocking(self, image: bytes) -> str:
        try:
            model, processor = self._model_and_processor()
            config = self._config(model)
            prompt = self.apply_chat_template(processor, config, self.prompt, num_images=1)
            with temporary_image_file(image, suffix=self.image_file_suffix) as image_path:
                output = self.generate(
                    model,
                    processor,
                    prompt,
                    [image_path],
                    verbose=self.verbose,
                    generate_kwargs=self.generate_kwargs,
                )
            return _generated_text(output)
        except Exception as error:
            message = "MLX VLM image decoding failed."
            raise ImageDecodingError(message) from error

    def _model_and_processor(self) -> tuple[object, object]:
        if self.model is not None and self.processor is not None:
            return self.model, self.processor

        loaded_model, loaded_processor = self.load_model(self.model_id)
        model = self.model if self.model is not None else loaded_model
        processor = self.processor if self.processor is not None else loaded_processor
        return model, processor

    def _config(self, model: object) -> object:
        if self.config is not None:
            return self.config
        model_config = typing.cast(object, getattr(model, "config", None))
        if model_config is not None:
            return model_config
        return self.load_config(self.model_id)


def _generated_text(output: object) -> str:
    if isinstance(output, str):
        return output.strip()

    if isinstance(output, collections.abc.Mapping):
        output_mapping = typing.cast(collections.abc.Mapping[object, object], output)
        text = output_mapping.get("text")
        if isinstance(text, str):
            return text.strip()
        message = "MLX VLM image decoding result is missing generated text."
        raise TypeError(message)

    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text.strip()

    message = "MLX VLM image decoding result is missing generated text."
    raise TypeError(message)


__all__ = ["ImageDecoder", "ImageDecodingError"]
