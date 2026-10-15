"""Tests for the transform instruments a codec host records."""

import pathlib
import typing

import hsm
import pydantic
import pytest

from mosfet.protocols import transform
from tests import metric_reader


class _ColumnsData(pydantic.BaseModel):
    columns: int


_ColumnsEvent = hsm.Event[_ColumnsData](name="test.transform.columns", schema=_ColumnsData)

_INSTRUMENTS = (
    "bot.transform.message.count",
    "bot.transform.transfer.size",
    "bot.transform.decode.failed.count",
    "bot.transform.dropped.count",
)


def _changes(
    before: dict[str, tuple[metric_reader.Point, ...]], after: dict[str, tuple[metric_reader.Point, ...]]
) -> dict[str, dict[tuple[tuple[str, object], ...], float]]:
    changed: dict[str, dict[tuple[tuple[str, object], ...], float]] = {}
    for name, points in after.items():
        earlier = {tuple(sorted(point.attributes.items())): point.value for point in before[name]}
        for point in points:
            key = tuple(sorted(point.attributes.items()))
            delta = point.value - earlier.get(key, 0.0)
            if delta:
                changed.setdefault(name, {})[key] = delta
    return changed


def test_a_host_counts_messages_bytes_decode_failures_and_drops_by_codec(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codec = transform.JsonCodec(_ColumnsEvent, max_message_bytes=64)

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = {name: reader.points(name) for name in _INSTRUMENTS}
        transform.telemetry.transferred(codec, "in", 21)
        transform.telemetry.transferred(codec, "in", 9)
        transform.telemetry.transferred(codec, "out", 40)
        transform.telemetry.decode_failed(codec, "schema_invalid")
        transform.telemetry.dropped(codec, "unowned")
        transform.telemetry.dropped(codec, "flow_control")
        transform.telemetry.dropped(codec, "encode_failed")
        after = {name: reader.points(name) for name in _INSTRUMENTS}

    assert _changes(before, after) == {
        "bot.transform.message.count": {
            (("codec", "json"), ("direction", "in")): 2.0,
            (("codec", "json"), ("direction", "out")): 1.0,
        },
        "bot.transform.transfer.size": {
            (("codec", "json"), ("direction", "in")): 30.0,
            (("codec", "json"), ("direction", "out")): 40.0,
        },
        "bot.transform.decode.failed.count": {(("codec", "json"), ("reason", "schema_invalid")): 1.0},
        "bot.transform.dropped.count": {
            (("codec", "json"), ("reason", "unowned")): 1.0,
            (("codec", "json"), ("reason", "flow_control")): 1.0,
            (("codec", "json"), ("reason", "encode_failed")): 1.0,
        },
    }


def test_instrument_attributes_are_closed_labels_only() -> None:
    assert typing.get_args(typing.cast(object, transform.telemetry.Direction.__value__)) == ("in", "out")
    assert typing.get_args(typing.cast(object, transform.telemetry.DropReason.__value__)) == (
        "unowned",
        "flow_control",
        "encode_failed",
    )
    assert typing.get_args(typing.cast(object, transform.DecodeReason.__value__)) == (
        "malformed",
        "unknown_event",
        "schema_invalid",
        "oversize",
        "media_type",
    )
    assert typing.get_args(typing.cast(object, transform.EncodeReason.__value__)) == ("schema_invalid", "oversize")
