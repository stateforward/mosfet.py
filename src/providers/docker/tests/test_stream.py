import pytest

from mosfet.providers.docker import stream


def _frame(kind: int, payload: bytes) -> bytes:
    return bytes((kind, 0, 0, 0)) + len(payload).to_bytes(4, "big") + payload


def test_frames_are_split_into_stdout_and_stderr() -> None:
    demux = stream.Demux(limit=1024)

    demux.feed(_frame(1, b"out 1\n") + _frame(2, b"err\n") + _frame(1, b"out 2\n"))
    demux.close()

    assert demux.stdout.text(1024) == ("out 1\nout 2\n", False)
    assert demux.stderr.text(1024) == ("err\n", False)
    assert (demux.stdout.total, demux.stderr.total) == (12, 4)


@pytest.mark.parametrize("size", [1, 3, 7, 8, 9, 13])
def test_frames_split_anywhere_across_chunks_are_reassembled(size: int) -> None:
    raw = _frame(1, b"hello ") + _frame(2, b"oops") + _frame(1, b"world\n") + _frame(1, b"")
    demux = stream.Demux(limit=1024)

    for start in range(0, len(raw), size):
        demux.feed(raw[start : start + size])
    demux.close()

    assert demux.stdout.text(1024) == ("hello world\n", False)
    assert demux.stderr.text(1024) == ("oops", False)


def test_the_system_error_stream_counts_as_stderr_and_stdin_frames_are_ignored() -> None:
    demux = stream.Demux(limit=64)

    demux.feed(_frame(0, b"echo") + _frame(3, b"exec failed\n"))
    demux.close()

    assert demux.stdout.text(256) == ("", False)
    assert demux.stderr.text(256) == ("exec failed\n", False)


def test_an_unknown_stream_or_a_body_ending_mid_frame_is_refused() -> None:
    unknown = stream.Demux(limit=64)
    with pytest.raises(stream.StreamError, match="stream 7"):
        unknown.feed(_frame(7, b"x"))

    cut = stream.Demux(limit=64)
    cut.feed(_frame(1, b"whole")[:-2])
    with pytest.raises(stream.StreamError, match="mid-frame"):
        cut.close()

    header = stream.Demux(limit=64)
    header.feed(b"\x01\x00\x00")
    with pytest.raises(stream.StreamError, match="mid-frame"):
        header.close()


def test_long_output_keeps_its_head_and_tail_with_a_marked_elision_inside_the_bound() -> None:
    bounded = stream.Bounded(limit=300)
    data = b"".join(f"line {number:05d}\n".encode() for number in range(1000))

    bounded.feed(data)
    text, truncated = bounded.text(256)

    assert truncated is True
    assert len(text) <= 256
    assert text.startswith("line 00000\nline 00001\n")
    assert text.endswith("line 00998\nline 00999\n")
    assert "bytes left out" in text
    assert bounded.total == len(data)


def test_output_within_the_bound_is_kept_whole() -> None:
    bounded = stream.Bounded(limit=256)

    for piece in (b"a" * 100, b"b" * 100, b"c" * 56):
        bounded.feed(piece)

    assert bounded.text(256) == ("a" * 100 + "b" * 100 + "c" * 56, False)


def test_output_kept_in_bytes_but_over_the_character_bound_is_elided() -> None:
    bounded = stream.Bounded(limit=1024)
    bounded.feed(b"x" * 600)

    text, truncated = bounded.text(256)

    assert truncated is True
    assert len(text) <= 256
    assert text.startswith("x") and text.endswith("x")
    assert "bytes left out" in text


def test_undecodable_bytes_are_replaced_and_cut_characters_do_not_leak_replacement_marks() -> None:
    whole = stream.Bounded(limit=64)
    whole.feed(b"ok \xff\xfe end")
    assert whole.text(256) == ("ok �� end", False)

    cut = stream.Bounded(limit=301)
    cut.feed("é".encode() * 1000)
    text, truncated = cut.text(256)

    assert truncated is True
    assert "�" not in text
    assert text.startswith("é") and text.endswith("é")


def test_memory_stays_within_the_head_and_tail_however_long_the_stream() -> None:
    demux = stream.Demux(limit=256)
    chunk = _frame(1, b"y" * 4096)

    for _ in range(1000):
        demux.feed(chunk)
    demux.close()

    assert demux.stdout.total == 4096 * 1000
    assert demux.stdout.kept() <= 2 * 256
