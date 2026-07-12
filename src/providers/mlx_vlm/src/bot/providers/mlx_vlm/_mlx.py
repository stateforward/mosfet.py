from __future__ import annotations

import collections.abc
import importlib
import typing


ModelLoader = collections.abc.Callable[[str], tuple[object, object]]
ConfigLoader = collections.abc.Callable[[str], object]


class ChatTemplateApplier(typing.Protocol):
    def __call__(self, processor: object, config: object, prompt: str, *, num_images: int) -> str:
        """Format a multimodal prompt for a model and processor."""
        ...


class ImageGenerator(typing.Protocol):
    def __call__(
        self,
        model: object,
        processor: object,
        prompt: str,
        images: collections.abc.Sequence[str],
        *,
        verbose: bool,
        generate_kwargs: collections.abc.Mapping[str, object],
    ) -> object:
        """Generate text for a prompt with image paths."""
        ...


def load_model(model_id: str) -> tuple[object, object]:
    module = importlib.import_module("mlx_vlm")
    load = typing.cast(collections.abc.Callable[[str], tuple[object, object]], getattr(module, "load"))
    return load(model_id)


def load_config(model_id: str) -> object:
    module = importlib.import_module("mlx_vlm.utils")
    load = typing.cast(collections.abc.Callable[[str], object], getattr(module, "load_config"))
    return load(model_id)


def apply_chat_template(processor: object, config: object, prompt: str, *, num_images: int) -> str:
    module = importlib.import_module("mlx_vlm.prompt_utils")
    apply = typing.cast(
        collections.abc.Callable[..., str],
        getattr(module, "apply_chat_template"),
    )
    return apply(processor, config, prompt, num_images=num_images)


def generate(
    model: object,
    processor: object,
    prompt: str,
    images: collections.abc.Sequence[str],
    *,
    verbose: bool,
    generate_kwargs: collections.abc.Mapping[str, object],
) -> object:
    module = importlib.import_module("mlx_vlm")
    generate_text = typing.cast(collections.abc.Callable[..., object], getattr(module, "generate"))
    return generate_text(model, processor, prompt, list(images), verbose=verbose, **dict(generate_kwargs))
