"""``willy_example.evaluate.sound``: the evaluator's audio, resampled to the room's rate, mixed and written as WAV."""

import array
import io
import wave

from willy_example.evaluate import sound, turns


def _samples(pcm: bytes) -> list[int]:
    values = array.array("h")
    values.frombytes(pcm)
    return values.tolist()


def _pcm(*values: int) -> bytes:
    return array.array("h", values).tobytes()


def test_audio_is_resampled_to_the_rooms_rate_by_linear_interpolation() -> None:
    # 24 kHz to 16 kHz: every second sample of three, interpolated between the samples around it.
    resampled = sound.resampled(_pcm(0, 300, 600, 900, 1200, 1500), sample_rate_hz=24_000)

    assert _samples(resampled) == [0, 450, 900, 1350]
    assert sound.resampled(_pcm(5, 6), sample_rate_hz=16_000) == _pcm(5, 6)
    assert sound.resampled(b"", sample_rate_hz=24_000) == b""
    # Two channels are mixed down to one.
    assert _samples(sound.resampled(_pcm(100, 300, -100, -300), sample_rate_hz=16_000, channels=2)) == [200, -200]


def test_willys_reply_is_its_chunks_one_after_another_at_the_rooms_rate() -> None:
    chunks = (
        turns.ChunkData(arrived=0.0, start=0.0, end=0.5, pcm=bytes(24_000), sample_rate_hz=24_000, channels=1),
        turns.ChunkData(arrived=0.1, start=0.5, end=1.0, pcm=bytes(24_000), sample_rate_hz=24_000, channels=1),
    )

    assert len(sound.reply(chunks)) == 2 * 16_000


def test_an_exchange_is_mixed_as_the_room_heard_it_and_written_as_a_16_khz_wav() -> None:
    exchange = turns.ExchangeData(
        exchange=0,
        said_from=10.0,
        said_until=10.5,
        cut_in_from=11.25,
        cut_in_until=11.5,
        chunks=(
            turns.ChunkData(
                arrived=11.0, start=11.0, end=11.5, pcm=_pcm(*[300] * 12_000), sample_rate_hz=24_000, channels=1
            ),
        ),
        outcome="replied",
    )

    mixed = sound.mixed(exchange, line=_pcm(*[1000] * 8_000), cut_in=_pcm(*[20_000] * 4_000))

    samples = _samples(mixed)
    assert len(samples) == round(1.5 * 16_000)
    assert samples[0] == 1000
    # A quiet half second between the person's line and Willy.
    assert samples[round(0.75 * 16_000)] == 0
    assert samples[round(1.1 * 16_000)] == 300
    # Where the person cut in over Willy, both are heard, clipped to what 16 bits hold.
    assert samples[round(1.3 * 16_000)] == 20_300
    with wave.open(io.BytesIO(sound.wav(mixed)), "rb") as read:
        assert (read.getnchannels(), read.getsampwidth(), read.getframerate()) == (1, 2, 16_000)
        assert read.readframes(read.getnframes()) == mixed
    loud = sound.mixed(
        exchange.model_copy(update={"cut_in_from": 11.0}),
        line=b"",
        cut_in=_pcm(*[32_700] * 4_000),
    )
    assert max(_samples(loud)) == 32_767
