"""``opencv.FrameDiff`` and ``opencv.Downscaled``: cheap change measures over a fake ``cv2``."""

import asyncio
import threading
import typing

import hsm
import numpy
import pytest

import mosfet.media as media
from mosfet.abilities.vision import change
from mosfet.providers import opencv

from tests import opencv_fakes

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _jpeg(
    fake: opencv_fakes.FakeCv2, name: str, pixels: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint8]]
) -> media.Image:
    return media.Image(content=fake.frame(opencv_fakes.JPEG_SIGNATURE + name.encode(), pixels), media_type="image/jpeg")


def _gray(value: int, size: int = 8) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint8]]:
    return numpy.full((size, size), value, dtype=numpy.uint8)


def _measure(
    measure: change.ChangeMeasure, previous: change.Signature | None, image: media.Image
) -> change.Measurement:
    return asyncio.run(measure.measure(previous, image, ctx=hsm.Context()))


# FrameDiff


def test_frame_diff_signs_a_reduced_blurred_grayscale_grid() -> None:
    fake = opencv_fakes.FakeCv2()
    measure = opencv.FrameDiff(size=4, cv2_loader=opencv_fakes.Loader(fake))

    first = _measure(measure, None, _jpeg(fake, "a", _gray(51)))

    assert first.change is None
    assert first.signature.measure == "frame_diff"
    assert first.signature.values == pytest.approx((0.2,) * 16)
    assert [call.args for call in fake.named("imdecode")] == [(fake.IMREAD_REDUCED_GRAYSCALE_8,)]
    assert [call.args for call in fake.named("resize")] == [((4, 4), fake.INTER_AREA)]
    assert [call.args for call in fake.named("GaussianBlur")] == [((3, 3), 0)]


def test_frame_diff_is_the_mean_absolute_difference_clamped() -> None:
    fake = opencv_fakes.FakeCv2()
    measure = opencv.FrameDiff(size=4, cv2_loader=opencv_fakes.Loader(fake))
    previous = _measure(measure, None, _jpeg(fake, "a", _gray(0))).signature
    half = _gray(0)
    half[:, 4:] = 255

    same = _measure(measure, previous, _jpeg(fake, "b", _gray(0)))
    changed = _measure(measure, previous, _jpeg(fake, "c", half))
    inverted = _measure(measure, previous, _jpeg(fake, "d", _gray(255)))

    assert same.change == 0.0
    assert changed.change == pytest.approx(0.5)
    assert inverted.change == pytest.approx(1.0)


def test_frame_diff_refuses_a_foreign_or_differently_sized_signature() -> None:
    fake = opencv_fakes.FakeCv2()
    measure = opencv.FrameDiff(size=4, cv2_loader=opencv_fakes.Loader(fake))
    image = _jpeg(fake, "a", _gray(0))

    for previous in (
        change.Signature(measure="embedding", values=(0.0,) * 16),
        change.Signature(measure="frame_diff", values=(0.0,) * 9),
    ):
        with pytest.raises(change.Error, match="frame difference"):
            _ = _measure(measure, previous, image)


def test_frame_diff_failures_are_change_errors_raised_off_the_event_loop() -> None:
    fake = opencv_fakes.FakeCv2()
    measure = opencv.FrameDiff(cv2_loader=opencv_fakes.Loader(fake))

    with pytest.raises(change.Error, match="could not decode"):
        _ = _measure(measure, None, media.Image(content=opencv_fakes.JPEG_SIGNATURE + b"?", media_type="image/jpeg"))
    assert threading.get_ident() not in {call.thread for call in fake.calls}


def test_frame_diff_size_is_bounded() -> None:
    for size in (3, 257):
        with pytest.raises(ValueError, match="size"):
            _ = opencv.FrameDiff(size=size)
    assert isinstance(opencv.FrameDiff(), change.ChangeMeasure)
    assert opencv.FrameDiff().attachments == ()


# Downscaled


class RecordingMeasure(change.ChangeMeasure):
    seen: list[tuple[change.Signature | None, media.Image]]

    def __init__(self, attachments: tuple[hsm.Instance, ...] = ()) -> None:
        self.seen = []
        self.attachments: tuple[hsm.Instance, ...] = attachments

    @typing.override
    async def measure(
        self, previous: change.Signature | None, image: media.Image, *, ctx: hsm.Context
    ) -> change.Measurement:
        del ctx
        self.seen.append((previous, image))
        return change.Measurement(
            change=None if previous is None else 0.25, signature=change.Signature(measure="inner", values=(1.0,))
        )


def test_downscaled_hands_inner_a_jpeg_no_larger_than_max_edge() -> None:
    fake = opencv_fakes.FakeCv2()
    inner = RecordingMeasure()
    measure = opencv.Downscaled(inner=inner, max_edge=100, quality=70, cv2_loader=opencv_fakes.Loader(fake))
    previous = change.Signature(measure="inner", values=(0.5,))

    measured = _measure(measure, previous, _jpeg(fake, "big", numpy.zeros((300, 400, 3), dtype=numpy.uint8)))

    assert measured == change.Measurement(change=0.25, signature=change.Signature(measure="inner", values=(1.0,)))
    ((handed_previous, small),) = inner.seen
    assert handed_previous == previous
    assert small.media_type == "image/jpeg" and (small.width_px, small.height_px) == (100, 75)
    assert [call.args for call in fake.named("resize")] == [((100, 75), fake.INTER_AREA)]
    assert [call.args for call in fake.named("imencode")] == [(".jpg", (75, 100, 3), (fake.IMWRITE_JPEG_QUALITY, 70))]
    assert [call.args for call in fake.named("imdecode")] == [(fake.IMREAD_COLOR,)]


def test_downscaled_hands_on_a_jpeg_that_already_fits_unchanged() -> None:
    fake = opencv_fakes.FakeCv2()
    inner = RecordingMeasure()
    measure = opencv.Downscaled(inner=inner, max_edge=448, cv2_loader=opencv_fakes.Loader(fake))
    image = _jpeg(fake, "small", numpy.zeros((300, 400, 3), dtype=numpy.uint8))

    _ = _measure(measure, None, image)

    assert inner.seen == [(None, image)]
    assert fake.named("imencode") == [] and fake.named("resize") == []


def test_downscaled_re_encodes_a_small_png_as_jpeg() -> None:
    fake = opencv_fakes.FakeCv2()
    inner = RecordingMeasure()
    measure = opencv.Downscaled(inner=inner, cv2_loader=opencv_fakes.Loader(fake))
    content = fake.frame(_PNG_SIGNATURE + b"screen", numpy.zeros((30, 40, 3), dtype=numpy.uint8))

    _ = _measure(measure, None, media.Image(content=content, media_type="image/png"))

    ((_, small),) = inner.seen
    assert small.media_type == "image/jpeg" and (small.width_px, small.height_px) == (40, 30)
    assert fake.named("resize") == []
    assert [call.args[0] for call in fake.named("imencode")] == [".jpg"]


def test_downscaled_carries_inner_attachments_and_checks_its_settings() -> None:
    probe = hsm.Instance()
    inner = RecordingMeasure(attachments=(probe,))

    assert opencv.Downscaled(inner=inner).attachments == (probe,)
    with pytest.raises(ValueError, match="max_edge"):
        _ = opencv.Downscaled(inner=inner, max_edge=15)
    for quality in (0, 101):
        with pytest.raises(ValueError, match="quality"):
            _ = opencv.Downscaled(inner=inner, quality=quality)


def test_downscaled_failures_are_change_errors_and_inner_is_not_asked() -> None:
    fake = opencv_fakes.FakeCv2()
    inner = RecordingMeasure()
    measure = opencv.Downscaled(inner=inner, cv2_loader=opencv_fakes.Loader(fake))

    with pytest.raises(change.Error, match="could not decode"):
        _ = _measure(measure, None, media.Image(content=opencv_fakes.JPEG_SIGNATURE + b"?", media_type="image/jpeg"))
    assert inner.seen == []


def test_a_cascade_of_frame_diff_and_a_downscaled_expensive_measure_composes() -> None:
    fake = opencv_fakes.FakeCv2()
    loader = opencv_fakes.Loader(fake)
    expensive = RecordingMeasure()
    cascade = change.Cascade(
        cheap=opencv.FrameDiff(size=4, cv2_loader=loader),
        expensive=opencv.Downscaled(inner=expensive, max_edge=100, cv2_loader=loader),
        cheap_threshold=0.03,
    )
    still = numpy.zeros((300, 400, 3), dtype=numpy.uint8)

    measured = _measure(cascade, None, _jpeg(fake, "first", still))
    again = _measure(cascade, measured.signature, _jpeg(fake, "again", still))

    # A first view computes both; an unchanged one is answered by the cheap measure alone.
    assert measured.change is None and again.change == 0.0
    assert len(expensive.seen) == 1
    assert [part.measure for part in again.signature.parts] == ["frame_diff", "inner"]
