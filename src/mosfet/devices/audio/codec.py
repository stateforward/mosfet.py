"""Binary PCM codec: one stream message is one chunk of signed 16-bit little-endian audio.

A message is a 6-byte big-endian header ``!IH`` (sample rate in hertz, channel count) followed by the
PCM bytes. It decodes to ``devices.audio.input`` and encodes ``devices.audio.output``. It is binary on
purpose: audio never rides as base64 inside JSON.
"""

from .events import InputData, InputEvent, OutputData, OutputEvent

import struct
import typing

import hsm

from mosfet.protocols import transform

_HEADER = struct.Struct("!IH")
_MEDIA_TYPE = "audio/pcm"
_MIN_SAMPLE_RATE_HZ = 8_000
_MAX_SAMPLE_RATE_HZ = 192_000
_MAX_CHANNELS = 8
_DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024


class PcmCodec:
    """Raw PCM between a stream and an audio port, with the rate and channel count in every message.

    The header bounds are mechanical: a sample rate from 8 kHz to 192 kHz and one to eight channels.
    One message carries at least one 16-bit sample and a whole number of them.

    Raises:
        ValueError: ``max_message_bytes`` too small to carry a header and one sample.
    """

    name: str
    max_message_bytes: int
    encodes_any: bool

    def __init__(self, *, max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES) -> None:
        if max_message_bytes < _HEADER.size + 2:
            raise ValueError(f"PcmCodec max_message_bytes must hold a header and one sample, got {max_message_bytes}.")
        self.name = "pcm"
        self.max_message_bytes = max_message_bytes
        self.encodes_any = True

    def decode(self, message: bytes) -> hsm.Event[typing.Any]:
        """Return ``devices.audio.input`` carrying the message's PCM, rate and channel count.

        Raises:
            DecodeError: ``oversize`` past ``max_message_bytes``; ``malformed`` for a short message, an
                odd PCM length, or a rate or channel count outside the header bounds.
        """

        if len(message) > self.max_message_bytes:
            raise transform.DecodeError("oversize", f"PCM message exceeds {self.max_message_bytes} bytes.")
        if len(message) < _HEADER.size + 2:
            raise transform.DecodeError("malformed", "PCM message needs a 6-byte header and one sample.")
        sample_rate_hz, channels = typing.cast(tuple[int, int], _HEADER.unpack_from(message))
        pcm = message[_HEADER.size :]
        if len(pcm) % 2:
            raise transform.DecodeError("malformed", "PCM s16le needs an even byte count.")
        if not _MIN_SAMPLE_RATE_HZ <= sample_rate_hz <= _MAX_SAMPLE_RATE_HZ or not 1 <= channels <= _MAX_CHANNELS:
            raise transform.DecodeError("malformed", "PCM header rate or channel count is out of range.")
        return InputEvent.with_data(
            InputData(audio=pcm, media_type=_MEDIA_TYPE, sample_rate_hz=sample_rate_hz, channels=channels)
        )

    def encodes(self, event: hsm.Event[typing.Any]) -> bool:
        """Whether ``event`` is named ``devices.audio.output``."""

        return event.name == OutputEvent.name

    def encode(self, event: hsm.Event[typing.Any]) -> bytes:
        """Return the header and PCM for one ``devices.audio.output`` chunk.

        Raises:
            EncodeError: ``schema_invalid`` for another event, audio that is not ``audio/pcm`` with a rate and
                channel count inside the header bounds, or an odd byte count; ``oversize`` past
                ``max_message_bytes``.
        """

        data = event.data
        if event.name != OutputEvent.name or not isinstance(data, OutputData):
            raise transform.EncodeError("schema_invalid", f"PcmCodec encodes only {OutputEvent.name!r}.")
        sample_rate_hz, channels = data.sample_rate_hz, data.channels
        if (
            data.media_type != _MEDIA_TYPE
            or sample_rate_hz is None
            or channels is None
            or not _MIN_SAMPLE_RATE_HZ <= sample_rate_hz <= _MAX_SAMPLE_RATE_HZ
            or not 1 <= channels <= _MAX_CHANNELS
            or len(data.audio) % 2
        ):
            raise transform.EncodeError("schema_invalid", "PcmCodec encodes audio/pcm s16le with a rate and channels.")
        message = _HEADER.pack(sample_rate_hz, channels) + data.audio
        if len(message) > self.max_message_bytes:
            raise transform.EncodeError("oversize", f"PCM message exceeds {self.max_message_bytes} bytes.")
        return message


__all__ = ["PcmCodec"]
