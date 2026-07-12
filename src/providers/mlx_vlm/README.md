# bot-provider-mlx-vlm

MLX VLM reading provider package for `bot`. This package owns the `mlx-vlm`
dependency for local Apple Silicon image reading while core `bot` stays
provider-agnostic.

Run its tests from the workspace root:

```sh
uv run --package bot-provider-mlx-vlm --group dev python -m pytest src/providers/mlx_vlm/tests
```

These tests use injected fake MLX VLM runtimes so they can run without
downloading models. A live image-reading smoke test requires macOS on Apple
Silicon, the Darwin-only `mlx-vlm` dependency, and first-run access to download
the configured local model. Live MLX VLM smoke is not part of the default pytest
suite; run it explicitly with a local image file:

```sh
BOT_MLX_VLM_SMOKE_IMAGE=/path/to/image.png uv run --package bot-provider-mlx-vlm python - <<'PY'
import asyncio
import os
import pathlib

from bot.providers.mlx_vlm import ImageDecoder


async def main() -> None:
    image = pathlib.Path(os.environ["BOT_MLX_VLM_SMOKE_IMAGE"]).read_bytes()
    print(await ImageDecoder().decode(image))


asyncio.run(main())
PY
```

The package exposes provider operations for the reading ability:

```python
from bot.abilities import Reading
from bot.providers.mlx_vlm import (
    ImageDecoder,
    ReadingOutputEncoder,
    TextDecoder,
    VisualClassifier,
)

reading = Reading(
    visual_classifier=VisualClassifier(),
    text_decoder=TextDecoder(),
    image_decoder=ImageDecoder(),
    output_encoder=ReadingOutputEncoder(),
)
```
