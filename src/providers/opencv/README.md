# mosfet-provider-opencv

OpenCV provider package for `mosfet` vision. It owns the `opencv-python` and `numpy`
dependencies while core `mosfet` stays provider-agnostic, and serves:

| Adapter | Serves | Model |
| --- | --- | --- |
| `opencv.FaceDetector` | `vision.faces.FaceDetector`: face boxes, plus an `identification.Print` per face when SFace is configured | YuNet + SFace |
| `opencv.ObjectDetector` | `vision.detection.ObjectDetector`: COCO-80 objects | YOLOX-S |
| `opencv.FrameDiff` | `vision.change.ChangeMeasure`: a cheap perceptual difference | none |
| `opencv.Downscaled` | `vision.change.ChangeMeasure`: hands an expensive measure a small JPEG | none |

The face and object adapters are per-role adapters over one `opencv.Models` backend, which loads
each network lazily, once, and serializes calls into it. `cv2` is imported only when first needed.
Every call runs in a worker thread.

## Installing

The package depends on `opencv-python==4.13.0.92` (the full wheel, which `mlx-vlm` also uses) and
`numpy>=2.4,<3`. In a Docker image with no GUI libraries you may swap in
`opencv-python-headless==4.13.0.92` at image build time; do not install both in one environment,
because they ship the same `cv2` module.

## Model files

Nothing is downloaded at import or in tests: model paths come from your configuration. Fetch the
files from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (Git LFS) once:

| File | Zoo directory | License |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | `models/face_detection_yunet` | MIT |
| `face_recognition_sface_2021dec.onnx` | `models/face_recognition_sface` | Apache-2.0 (training-data provenance is not stated; check it before commercial use) |
| `object_detection_yolox_2022nov.onnx` | `models/object_detection_yolox` | Apache-2.0 |

```sh
git clone --depth 1 https://github.com/opencv/opencv_zoo.git
cd opencv_zoo && git lfs pull --include "models/face_detection_yunet/face_detection_yunet_2023mar.onnx,models/face_recognition_sface/face_recognition_sface_2021dec.onnx,models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
mkdir -p /path/to/opencv-models
cp models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
   models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
   models/object_detection_yolox/object_detection_yolox_2022nov.onnx /path/to/opencv-models/
```

Prints are tagged `opencv-sface-2021dec`; an identity store keeps one vector space per model, so a
different face model starts a new gallery.

## Per-frame cost

`seeing.Seeing` measures at most once per `measure_every` (500 ms by default) whatever the camera
rate, so these are the costs of one measured frame. The OpenCV numbers below were measured with
`tests/test_live.py` on Apple Silicon for a 1280×720 JPEG; run it on your hardware for your own.

| Measure | Work per measured frame | Cost |
| --- | --- | --- |
| `opencv.FrameDiff()` | 1/8-scale grayscale DCT decode, 32×32 area resize, 3×3 blur, mean absolute difference | ≈ 0.6 ms (≈ 10–20 ms for a full-size PNG, which cannot be decoded reduced) |
| `opencv.Downscaled(inner=…)` | decode, area resize to 448 px, JPEG re-encode | ≈ 5 ms, plus `inner` on the small frame |
| `vision.change.EmbeddingChange` | one embedding forward pass | the embedder's |
| `vision.change.ByteIdentity()` | one SHA-256 | ≈ 1–3 ms for a 1–4 MiB still; static sources only |
| `vision.change.Cascade(cheap, expensive)` | `cheap` every time, `expensive` only when `cheap` sees change | as combined |

`bot.opencv.call.duration{operation}` and `bot.seeing.measure.duration` report the live cost, and
`bot.vision.change.cascade.count{bot.outcome}` shows how often a cascade skipped its expensive
measure.

## Composing an eye

Every model-specific number below (the face `MatchPolicy`, `cheap_threshold`, retention) is a
starting point for these models, not a default anyone else applies: tune it on your own cameras.

```python
import datetime
import pathlib

from mosfet.abilities import identification, seeing, vision
from mosfet.providers import opencv, sqlite_memory

zoo = pathlib.Path(config.opencv_models)  # from configuration
models = opencv.Models(
    yunet=zoo / "face_detection_yunet_2023mar.onnx",
    sface=zoo / "face_recognition_sface_2021dec.onnx",
    yolox=zoo / "object_detection_yolox_2022nov.onnx",
)

# Face prints persist beside memory: one file database, shared with memory.Memory.
engine = sqlite_memory.open_engine(database_path=config.memory_database)
faces = identification.Registry(
    identifier=identification.Matcher(
        store=sqlite_memory.IdentityStore(engine=engine),
        # SFace's published cosine threshold is 0.363.
        policy=identification.MatchPolicy(threshold=0.363, margin=0.05, refine_threshold=0.5),
    ),
    kind="face",
    max_unlinked_age=datetime.timedelta(days=30),
    max_ids=10_000,
)

eye = seeing.Seeing(
    change=vision.change.Cascade(
        cheap=opencv.FrameDiff(),
        # Any image-capable (or omni-modal) embedding.Embedder; see vision.change.EmbeddingChange.
        expensive=opencv.Downscaled(inner=vision.change.EmbeddingChange(embedder=embedder)),
        cheap_threshold=0.03,
    ),
    object_detector=opencv.ObjectDetector(models=models),
    face_detector=opencv.FaceDetector(models=models),
    faces=faces,
    stage_timeout=datetime.timedelta(seconds=10),
)
```

- `faces` is passed unstarted: the eye starts, attaches and stops it. Keep the reference to erase
  an id with `identification.ForgetEvent.with_data(identification.ForgetData(id=...))`, dispatched to the registry.
- `IdentityStore` refuses an in-memory (`:memory:`) engine. For a composition without a database
  file use `identification.InProcessStore(max_ids=...)` instead.
- For a static source such as a screen, `change=vision.change.ByteIdentity()` is enough. On a live
  camera every encoded frame differs, so it rates every frame as changed.
- A cheap measure alone (`change=opencv.FrameDiff()`) works too; its numbers are on its own scale,
  so set `change_threshold` for it (start near 0.03).

### Sizing `stage_timeout`

Each stage serves one request at a time, and a look interprets every fresh view (up to
`seeing.MAX_VIEWS`, 4) in parallel. Requests for later views wait at the stage while their
deadlines already run, so size `stage_timeout` to cover `MAX_VIEWS` times the stage's slowest
healthy call. Local OpenCV stages answer in tens of milliseconds; a remote describer that takes 5 s
needs at least 20 s. A `stage_timeout` that is too short sends a healthy stage through recovery:
`bot.seeing.stage.restart.count` shows it.

### Camera resolution and memory

Every sight the eye hands on is remembered with its image bytes. Memory keeps a stimulus's images
up to one turn's worth (`media.MAX_TURN_IMAGES` images of at most `media.MAX_IMAGE_BYTES` each).
Have camera clients send JPEG with the long edge at or under 1280 px, so that each remembered view
stays small. Neither the eye nor this provider prunes the image bytes of old sight episodes: that
retention belongs to the memory lane.

## Tests

The default tests use a fake `cv2` surface (`tests/opencv_fakes.py` at the workspace root) and
load no model:

```sh
uv run --package mosfet-provider-opencv --group dev python -m pytest src/providers/opencv/tests -m "not live"
```

The live smoke uses the real `cv2` and the Zoo files and prints per-frame times as a report:

```sh
MOSFET_LIVE_OPENCV_MODELS=/path/to/opencv-models \
MOSFET_LIVE_OPENCV_IMAGE=/path/to/photo.jpg \
  uv run --package mosfet-provider-opencv --group dev python -m pytest \
  src/providers/opencv/tests/test_live.py -m live -v -s
```

`MOSFET_LIVE_OPENCV_IMAGE` is optional; without it the detectors run on a synthetic frame.
