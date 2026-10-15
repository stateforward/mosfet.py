# mosfet-provider-mlx-vlm

MLX VLM reading provider package for `mosfet`. This package owns the `mlx-vlm`
dependency for local Apple Silicon image reading while core `mosfet` stays
provider-agnostic.

Run its tests from the workspace root:

```sh
uv run --package mosfet-provider-mlx-vlm --group dev python -m pytest src/providers/mlx_vlm/tests
```

These tests use injected fake MLX VLM runtimes so they can run without
downloading models. A live image-reading smoke test requires macOS on Apple
Silicon, the Darwin-only `mlx-vlm` dependency, and first-run access to download
the configured local model. Live MLX VLM smoke is not part of the default pytest
suite; run it explicitly with a local PNG, JPEG, or WebP file. The decoder takes a
typed `mosfet.media.Image` and names its temporary file by the image's media type:

```sh
BOT_MLX_VLM_SMOKE_IMAGE=/path/to/image.png uv run --package mosfet-provider-mlx-vlm python - <<'PY'
import asyncio
import os
import pathlib

from mosfet import media
from mosfet.providers.mlx_vlm import ImageDecoder

MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


async def main() -> None:
    path = pathlib.Path(os.environ["BOT_MLX_VLM_SMOKE_IMAGE"])
    image = media.Image(content=path.read_bytes(), media_type=MEDIA_TYPES[path.suffix.lower()])
    print(await ImageDecoder().decode(image))


asyncio.run(main())
PY
```

The package exposes provider operations for the reading ability, which reads
`reading.InputData(content="text")` or `reading.InputData(content=media.Image(...))`:

```python
from mosfet.abilities.reading import Reading
from mosfet.providers.mlx_vlm import (
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
