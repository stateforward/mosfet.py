from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import pathlib
import sys
import types

import pytest

from bot import abilities
from bot.providers.mlx_vlm import ImageDecoder, ImageDecodingError


@dataclasses.dataclass
class FakeRuntime:
    generated_text: object = "Total due: $14.21"
    loaded_model_ids: list[str] = dataclasses.field(default_factory=list)
    config_model_ids: list[str] = dataclasses.field(default_factory=list)
    template_calls: list[dict[str, object]] = dataclasses.field(default_factory=list)
    generate_calls: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def load_model(self, model_id: str) -> tuple[object, object]:
        self.loaded_model_ids.append(model_id)
        return "model", "processor"

    def load_config(self, model_id: str) -> object:
        self.config_model_ids.append(model_id)
        return "config"

    def apply_chat_template(self, processor: object, config: object, prompt: str, *, num_images: int) -> str:
        self.template_calls.append(
            {
                "processor": processor,
                "config": config,
                "prompt": prompt,
                "num_images": num_images,
            }
        )
        return f"formatted:{prompt}"

    def generate(
        self,
        model: object,
        processor: object,
        prompt: str,
        images: collections.abc.Sequence[str],
        *,
        verbose: bool,
        generate_kwargs: collections.abc.Mapping[str, object],
    ) -> object:
        image_path = pathlib.Path(images[0])
        self.generate_calls.append(
            {
                "model": model,
                "processor": processor,
                "prompt": prompt,
                "images": tuple(pathlib.Path(image_path) for image_path in images),
                "image_bytes": image_path.read_bytes(),
                "verbose": verbose,
                "generate_kwargs": dict(generate_kwargs),
            }
        )
        return self.generated_text


class FailingRuntime(FakeRuntime):
    def generate(
        self,
        model: object,
        processor: object,
        prompt: str,
        images: collections.abc.Sequence[str],
        *,
        verbose: bool,
        generate_kwargs: collections.abc.Mapping[str, object],
    ) -> object:
        del model, processor, prompt, images, verbose, generate_kwargs
        raise RuntimeError("mlx-vlm unavailable")


class FailingLoadModelRuntime(FakeRuntime):
    def load_model(self, model_id: str) -> tuple[object, object]:
        del model_id
        raise RuntimeError("mlx-vlm model unavailable")


class FailingLoadConfigRuntime(FakeRuntime):
    def load_config(self, model_id: str) -> object:
        del model_id
        raise RuntimeError("mlx-vlm config unavailable")


class FailingChatTemplateRuntime(FakeRuntime):
    def apply_chat_template(self, processor: object, config: object, prompt: str, *, num_images: int) -> str:
        del processor, config, prompt, num_images
        raise RuntimeError("mlx-vlm template unavailable")


async def await_image_decoding(output: collections.abc.Awaitable[str]) -> str:
    return await output


def test_image_decoder_uses_injected_runtime() -> None:
    runtime = FakeRuntime()
    decoder = ImageDecoder(
        model_id="local/qwen2-vl",
        prompt="Read the text.",
        verbose=True,
        generate_kwargs={"max_tokens": 64},
        load_model=runtime.load_model,
        load_config=runtime.load_config,
        apply_chat_template=runtime.apply_chat_template,
        generate=runtime.generate,
    )

    output = asyncio.run(await_image_decoding(decoder.decode(b"image bytes")))

    assert output == "Total due: $14.21"
    assert runtime.loaded_model_ids == ["local/qwen2-vl"]
    assert runtime.config_model_ids == ["local/qwen2-vl"]
    assert runtime.template_calls == [
        {
            "processor": "processor",
            "config": "config",
            "prompt": "Read the text.",
            "num_images": 1,
        }
    ]
    assert len(runtime.generate_calls) == 1
    images = runtime.generate_calls[0]["images"]
    assert isinstance(images, tuple)
    image_path = images[0]
    assert isinstance(image_path, pathlib.Path)
    assert not image_path.exists()
    assert runtime.generate_calls[0] == {
        "model": "model",
        "processor": "processor",
        "prompt": "formatted:Read the text.",
        "images": runtime.generate_calls[0]["images"],
        "image_bytes": b"image bytes",
        "verbose": True,
        "generate_kwargs": {"max_tokens": 64},
    }
    assert isinstance(decoder, abilities.Decoder)


def test_image_decoder_default_constructor_uses_mlx_vlm_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_model_ids: list[str] = []
    config_model_ids: list[str] = []
    template_calls: list[dict[str, object]] = []
    generate_calls: list[dict[str, object]] = []
    mlx_vlm_module = types.ModuleType("mlx_vlm")
    utils_module = types.ModuleType("mlx_vlm.utils")
    prompt_utils_module = types.ModuleType("mlx_vlm.prompt_utils")

    def load(model_id: str) -> tuple[object, object]:
        loaded_model_ids.append(model_id)
        return "model", "processor"

    def load_config(model_id: str) -> object:
        config_model_ids.append(model_id)
        return "config"

    def apply_chat_template(processor: object, config: object, prompt: str, *, num_images: int) -> str:
        template_calls.append(
            {
                "processor": processor,
                "config": config,
                "prompt": prompt,
                "num_images": num_images,
            }
        )
        return f"formatted:{prompt}"

    def generate(
        model: object,
        processor: object,
        prompt: str,
        images: collections.abc.Sequence[str],
        *,
        verbose: bool,
        **generate_kwargs: object,
    ) -> object:
        image_path = pathlib.Path(images[0])
        generate_calls.append(
            {
                "model": model,
                "processor": processor,
                "prompt": prompt,
                "images": tuple(pathlib.Path(path) for path in images),
                "image_bytes": image_path.read_bytes(),
                "verbose": verbose,
                "generate_kwargs": dict(generate_kwargs),
            }
        )
        return {"text": "default image text"}

    setattr(mlx_vlm_module, "load", load)
    setattr(mlx_vlm_module, "generate", generate)
    setattr(utils_module, "load_config", load_config)
    setattr(prompt_utils_module, "apply_chat_template", apply_chat_template)
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils_module)
    decoder = ImageDecoder()

    output = asyncio.run(await_image_decoding(decoder.decode(b"image bytes")))

    assert output == "default image text"
    assert loaded_model_ids == ["mlx-community/Qwen2-VL-2B-Instruct-4bit"]
    assert config_model_ids == ["mlx-community/Qwen2-VL-2B-Instruct-4bit"]
    assert template_calls == [
        {
            "processor": "processor",
            "config": "config",
            "prompt": "Read all visible text in this image. Return only the text.",
            "num_images": 1,
        }
    ]
    assert len(generate_calls) == 1
    images = generate_calls[0]["images"]
    assert isinstance(images, tuple)
    image_path = images[0]
    assert isinstance(image_path, pathlib.Path)
    assert not image_path.exists()
    assert generate_calls[0] == {
        "model": "model",
        "processor": "processor",
        "prompt": "formatted:Read all visible text in this image. Return only the text.",
        "images": generate_calls[0]["images"],
        "image_bytes": b"image bytes",
        "verbose": False,
        "generate_kwargs": {"max_tokens": 512, "temperature": 0.0},
    }


def test_image_decoder_uses_injected_model_processor_and_config() -> None:
    runtime = FakeRuntime(generated_text={"text": "serial number 1234"})
    decoder = ImageDecoder(
        model_id="local/qwen2-vl",
        model="model",
        processor="processor",
        config="config",
        load_model=runtime.load_model,
        load_config=runtime.load_config,
        apply_chat_template=runtime.apply_chat_template,
        generate=runtime.generate,
    )

    output = asyncio.run(await_image_decoding(decoder.decode(b"image bytes")))

    assert output == "serial number 1234"
    assert runtime.loaded_model_ids == []
    assert runtime.config_model_ids == []


def test_image_decoder_is_awaitable() -> None:
    runtime = FakeRuntime()
    decoder = ImageDecoder(
        load_model=runtime.load_model,
        load_config=runtime.load_config,
        apply_chat_template=runtime.apply_chat_template,
        generate=runtime.generate,
    )

    output = decoder.decode(b"image bytes")

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output) == "Total due: $14.21"


def test_image_decoder_wraps_provider_errors() -> None:
    runtime = FailingRuntime()
    decoder = ImageDecoder(
        load_model=runtime.load_model,
        load_config=runtime.load_config,
        apply_chat_template=runtime.apply_chat_template,
        generate=runtime.generate,
    )

    with pytest.raises(ImageDecodingError) as error:
        _ = asyncio.run(await_image_decoding(decoder.decode(b"image bytes")))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_image_decoder_wraps_model_loader_errors() -> None:
    runtime = FailingLoadModelRuntime()
    decoder = ImageDecoder(
        load_model=runtime.load_model,
        load_config=runtime.load_config,
        apply_chat_template=runtime.apply_chat_template,
        generate=runtime.generate,
    )

    with pytest.raises(ImageDecodingError) as error:
        _ = asyncio.run(await_image_decoding(decoder.decode(b"image bytes")))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_image_decoder_wraps_config_loader_errors() -> None:
    runtime = FailingLoadConfigRuntime()
    decoder = ImageDecoder(
        load_model=runtime.load_model,
        load_config=runtime.load_config,
        apply_chat_template=runtime.apply_chat_template,
        generate=runtime.generate,
    )

    with pytest.raises(ImageDecodingError) as error:
        _ = asyncio.run(await_image_decoding(decoder.decode(b"image bytes")))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_image_decoder_wraps_chat_template_errors() -> None:
    runtime = FailingChatTemplateRuntime()
    decoder = ImageDecoder(
        load_model=runtime.load_model,
        load_config=runtime.load_config,
        apply_chat_template=runtime.apply_chat_template,
        generate=runtime.generate,
    )

    with pytest.raises(ImageDecodingError) as error:
        _ = asyncio.run(await_image_decoding(decoder.decode(b"image bytes")))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_image_decoder_rejects_missing_text() -> None:
    runtime = FakeRuntime(generated_text={"segments": []})
    decoder = ImageDecoder(
        load_model=runtime.load_model,
        load_config=runtime.load_config,
        apply_chat_template=runtime.apply_chat_template,
        generate=runtime.generate,
    )

    with pytest.raises(ImageDecodingError) as error:
        _ = asyncio.run(await_image_decoding(decoder.decode(b"image bytes")))

    assert isinstance(error.value.__cause__, TypeError)
