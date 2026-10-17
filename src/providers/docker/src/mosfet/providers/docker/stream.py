"""Docker's multiplexed attach stream, split into stdout and stderr as it arrives and kept within a bound.

Without a TTY, ``POST /exec/{id}/start`` answers with frames: an 8-byte header (stream type, three zero bytes, the
payload length as a big-endian 32-bit integer) and then the payload. ``Demux`` reads them incrementally, so a
frame may be split anywhere across chunks, and hands each payload to the stream's ``Bounded``, which keeps only its
first and last ``limit`` bytes and counts the rest. Nothing here grows with the length of the output.

These are plain collaborators used inside one exchange activity of the host; they hold no machine state.
"""

import dataclasses
import typing

ELISION: typing.Final = "\n[... {count} bytes left out ...]\n"
"""The mark left where the middle of a long stream was elided; ``count`` is how many bytes it stands for."""

_HEADER_BYTES: typing.Final = 8
_STDIN: typing.Final = 0
_STDOUT: typing.Final = 1
_STDERR: typing.Final = 2
_SYSTEM_ERROR: typing.Final = 3
_CONTINUATION_MASK: typing.Final = 0xC0
_CONTINUATION: typing.Final = 0x80
_MAX_SEQUENCE: typing.Final = 4


class StreamError(ValueError):
    """The attach stream broke its framing: an unknown stream type, or a body that ended mid-frame."""


def _sequence_length(lead: int) -> int:
    if lead >= 0xF0:
        return 4
    if lead >= 0xE0:
        return 3
    if lead >= 0xC0:
        return 2
    return 1


def _cut_start(data: bytes) -> bytes:
    """``data`` without the continuation bytes of a character that began before the cut."""

    skip = 0
    while skip < min(len(data), _MAX_SEQUENCE - 1) and data[skip] & _CONTINUATION_MASK == _CONTINUATION:
        skip += 1
    return data[skip:]


def _cut_end(data: bytes) -> bytes:
    """``data`` without the leading bytes of a character that the cut left unfinished."""

    for back in range(1, min(len(data), _MAX_SEQUENCE) + 1):
        byte = data[-back]
        if byte & _CONTINUATION_MASK != _CONTINUATION:
            return data[:-back] if _sequence_length(byte) > back else data
    return data


@dataclasses.dataclass(slots=True)
class Bounded:
    """One output stream's first and last ``limit`` bytes, and how many bytes it carried in all."""

    limit: int
    head: bytearray = dataclasses.field(default_factory=bytearray)
    tail: bytearray = dataclasses.field(default_factory=bytearray)
    total: int = 0

    def feed(self, data: bytes) -> None:
        """Take the next ``data`` of the stream, keeping only the head and the latest tail."""

        self.total += len(data)
        room = self.limit - len(self.head)
        if room > 0:
            self.head += data[:room]
            data = data[room:]
        if data:
            self.tail += data
            excess = len(self.tail) - self.limit
            if excess > 0:
                del self.tail[:excess]

    def kept(self) -> int:
        """How many bytes are held now: never more than twice ``limit``."""

        return len(self.head) + len(self.tail)

    def text(self, max_chars: int) -> tuple[str, bool]:
        """The stream as UTF-8 text (undecodable bytes replaced) of at most ``max_chars``, and whether it was elided.

        A stream that does not fit keeps its head and its tail around an ``ELISION`` mark, all within the bound.
        """

        dropped = self.total - self.kept()
        if dropped == 0:
            whole = bytes(self.head + self.tail).decode("utf-8", "replace")
            if len(whole) <= max_chars:
                return whole, False
            head_text = tail_text = whole
        else:
            head_text = _cut_end(bytes(self.head)).decode("utf-8", "replace")
            tail_text = _cut_start(bytes(self.tail)).decode("utf-8", "replace")
        keep = max(0, max_chars - len(ELISION.format(count=self.total)))
        head_part = head_text[: keep // 2]
        tail_part = tail_text[len(tail_text) - (keep - keep // 2) :] if keep - keep // 2 else ""
        left_out = max(0, self.total - len(head_part.encode()) - len(tail_part.encode()))
        return head_part + ELISION.format(count=left_out) + tail_part, True


class Demux:
    """Splits Docker's multiplexed attach stream into ``stdout`` and ``stderr`` as its chunks arrive.

    Stdin frames are ignored and the system-error stream (type 3) counts as stderr. Each stream keeps its first
    and last ``limit`` bytes.
    """

    stdout: Bounded
    stderr: Bounded
    _header: bytearray
    _stream: Bounded | None
    _remaining: int

    def __init__(self, *, limit: int) -> None:
        self.stdout = Bounded(limit)
        self.stderr = Bounded(limit)
        self._header = bytearray()
        self._stream = None
        self._remaining = 0

    def feed(self, chunk: bytes) -> None:
        """Take the next ``chunk`` of the body, wherever it splits a frame.

        Raises:
            StreamError: When a frame names a stream type other than stdin, stdout, stderr or system error.
        """

        view = memoryview(chunk)
        while view:
            if self._remaining:
                taken = view[: self._remaining]
                if self._stream is not None:
                    self._stream.feed(bytes(taken))
                self._remaining -= len(taken)
                view = view[len(taken) :]
                continue
            needed = _HEADER_BYTES - len(self._header)
            self._header += view[:needed]
            view = view[needed:]
            if len(self._header) < _HEADER_BYTES:
                return
            kind = self._header[0]
            self._remaining = int.from_bytes(self._header[4:8], "big")
            self._header.clear()
            if kind == _STDOUT:
                self._stream = self.stdout
            elif kind in (_STDERR, _SYSTEM_ERROR):
                self._stream = self.stderr
            elif kind == _STDIN:
                self._stream = None
            else:
                raise StreamError(f"the attach stream named stream {kind}, which is not stdin, stdout or stderr")

    def close(self) -> None:
        """The body ended.

        Raises:
            StreamError: When it ended inside a frame header or payload.
        """

        if self._remaining or self._header:
            raise StreamError("the attach stream ended mid-frame")


__all__ = ["ELISION", "Bounded", "Demux", "StreamError"]
