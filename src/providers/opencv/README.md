# mosfet-provider-opencv

OpenCV provider package for `mosfet` vision. It owns the `opencv-python` and `numpy`
dependencies while core `mosfet` stays provider-agnostic, and serves:

- `opencv.FaceDetector` — YuNet faces with SFace `identification.Print` face prints
  (a `vision.faces.FaceDetector`);
- `opencv.ObjectDetector` — YOLOX-S COCO-80 objects (a `vision.detection.ObjectDetector`);
- `opencv.FrameDiff` — a cheap `vision.change.ChangeMeasure`;
- `opencv.Downscaled` — hands an expensive change measure a small JPEG.

The face and object adapters share one `opencv.Models` backend. `cv2` is imported lazily, and
model files come from configuration: nothing is downloaded.

Run its tests from the workspace root (they use a fake `cv2` surface and load no model):

```sh
uv run --package mosfet-provider-opencv --group dev python -m pytest src/providers/opencv/tests
```
