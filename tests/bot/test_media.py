import base64

import mosfet.event
import pydantic
import pytest

from mosfet import media

_PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"
_JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-body"
_WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"webp-body"


def _png(size: int) -> media.Image:
    return media.Image(content=_PNG + b"x" * (size - len(_PNG)), media_type="image/png")


@pytest.mark.parametrize(
    ("content", "media_type"),
    [(_PNG, "image/png"), (_JPEG, "image/jpeg"), (_WEBP, "image/webp")],
)
def test_image_accepts_matching_signatures(content: bytes, media_type: media.ImageMediaType) -> None:
    image = media.Image(content=content, media_type=media_type, width_px=4, height_px=3)

    assert image.content == content
    assert image.media_type == media_type
    assert (image.width_px, image.height_px) == (4, 3)


@pytest.mark.parametrize(
    ("content", "media_type"),
    [(_PNG, "image/jpeg"), (_JPEG, "image/webp"), (_WEBP, "image/png"), (b"RIFF\x00\x00\x00\x00WAVE", "image/webp")],
)
def test_image_rejects_signature_that_does_not_match_media_type(content: bytes, media_type: str) -> None:
    with pytest.raises(pydantic.ValidationError, match="image content does not match media_type"):
        _ = media.Image.model_validate({"content": content, "media_type": media_type})


def test_image_rejects_unsupported_media_type_and_empty_content() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = media.Image.model_validate({"content": b"GIF89a", "media_type": "image/gif"})
    with pytest.raises(pydantic.ValidationError):
        _ = media.Image.model_validate({"content": b"", "media_type": "image/png"})


def test_image_rejects_content_over_max_image_bytes() -> None:
    assert media.MAX_IMAGE_BYTES == 4 * 1024 * 1024
    _ = _png(media.MAX_IMAGE_BYTES)
    with pytest.raises(pydantic.ValidationError):
        _ = _png(media.MAX_IMAGE_BYTES + 1)


def test_image_json_round_trips_content_as_base64() -> None:
    image = media.Image(content=_PNG, media_type="image/png", width_px=2, height_px=2)

    encoded = image.model_dump_json()

    assert base64.b64encode(_PNG).decode("ascii") in encoded
    assert media.Image.model_validate_json(encoded) == image


def test_event_json_value_omits_image_content() -> None:
    projected = mosfet.event.event_json_value(media.Image(content=_PNG, media_type="image/png", width_px=2))

    assert projected == {"media_type": "image/png", "width_px": 2, "height_px": None}


def test_image_capacity_admits_newest_first_within_count_cap() -> None:
    capacity = media.ImageCapacity(max_images=2, max_total_bytes=1_000, media_types=frozenset({"image/png"}))

    assert capacity.admit((_png(20), _png(20), _png(20))) == ("capacity", None, None)


def test_image_capacity_admits_newest_first_within_byte_cap() -> None:
    capacity = media.ImageCapacity(max_images=8, max_total_bytes=100, media_types=frozenset({"image/png"}))

    assert capacity.admit((_png(40), _png(80), _png(30))) == (None, "capacity", None)


def test_image_capacity_omits_unsupported_media_types() -> None:
    capacity = media.ImageCapacity(max_images=8, max_total_bytes=1_000, media_types=frozenset({"image/png"}))
    jpeg = media.Image(content=_JPEG, media_type="image/jpeg")

    assert capacity.admit((jpeg, _png(20))) == ("media_type", None)


def test_default_image_capacity_is_text_only() -> None:
    assert media.ImageCapacity().admit((_png(20), _png(20))) == ("media_type", "media_type")
    assert media.ImageCapacity().admit(()) == ()


def test_released_image_carries_metadata_without_bytes() -> None:
    image = media.Image(content=_PNG, media_type="image/png", width_px=5, height_px=6)

    released = media.ReleasedImage.of(image, reason="memory")

    assert released == media.ReleasedImage(
        media_type="image/png", width_px=5, height_px=6, byte_count=len(_PNG), reason="memory"
    )
    assert "content" not in media.ReleasedImage.model_fields
    with pytest.raises(pydantic.ValidationError):
        _ = media.ReleasedImage.model_validate(
            {"media_type": "image/png", "byte_count": 1, "reason": "memory", "content": _PNG}
        )
    with pytest.raises(pydantic.ValidationError):
        _ = media.ReleasedImage.model_validate({"media_type": "image/png", "byte_count": 1})


def test_media_limits() -> None:
    assert media.MAX_TURN_IMAGES == 8
