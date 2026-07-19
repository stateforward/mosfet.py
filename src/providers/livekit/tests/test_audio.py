from __future__ import annotations

from bot import abilities
from bot.abilities import participating
from bot.abilities.hearing import speech
from bot.devices import audio as audio_device

import asyncio
import collections.abc
import dataclasses

import pytest
from livekit import rtc

from bot.providers.livekit.audio import (
    AudioBridge,
    AudioFrameDecoder,
    AudioFrameEncoder,
    AudioFrameError,
    AudioSourceWriter,
    PcmWavDecoder,
    VoiceDecoder,
    create_audio_bridge,
)

@dataclasses.dataclass(frozen=True)
class FakeAudioFrame:
    data: bytes
    sample_rate: int
    num_channels: int
    samples_per_channel: int

@dataclasses.dataclass
class FakeAudioSource:
    captured_frames: list[FakeAudioFrame] = dataclasses.field(default_factory=list)

    async def capture_frame(self, frame: FakeAudioFrame) -> None:
        self.captured_frames.append(frame)

@dataclasses.dataclass
class RecordingSpeechDecoder(speech.SpeechDecoder):
    inputs: list[bytes] = dataclasses.field(default_factory=list)

    async def decode(self, input: bytes) -> bytes:
        self.inputs.append(input)
        return b"hello caller"

async def await_value[T](value: collections.abc.Awaitable[T]) -> T:
    return await value

def fake_audio_frame_factory(
    *,
    data: bytes,
    sample_rate: int,
    num_channels: int,
    samples_per_channel: int,
) -> FakeAudioFrame:
    return FakeAudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )

def test_livekit_audio_frame_encoder_converts_bot_audio_output_to_livekit_frame() -> None:
    encoder = AudioFrameEncoder[FakeAudioFrame](audio_frame_factory=fake_audio_frame_factory)
    output = audio_device.AudioOutputData(
        audio=b"\x01\x00\x02\x00\x03\x00\x04\x00", media_type="audio/pcm", sample_rate_hz=48000, channels=2
    )

    frame = asyncio.run(await_value(encoder.encode(output)))

    assert frame == FakeAudioFrame(
        data=b"\x01\x00\x02\x00\x03\x00\x04\x00",
        sample_rate=48000,
        num_channels=2,
        samples_per_channel=2,
    )
    assert isinstance(encoder, abilities.Encoder)

def test_livekit_audio_frame_encoder_uses_livekit_sdk_audio_frame_by_default() -> None:
    encoder: AudioFrameEncoder[rtc.AudioFrame] = AudioFrameEncoder()

    frame = asyncio.run(
        await_value(
            encoder.encode(
                audio_device.AudioOutputData(audio=b"\x01\x00\x02\x00", media_type="audio/pcm", sample_rate_hz=48000, channels=1)
            )
        )
    )

    assert isinstance(frame, rtc.AudioFrame)
    assert frame.sample_rate == 48000
    assert frame.num_channels == 1
    assert frame.samples_per_channel == 2
    assert bytes(frame.data) == b"\x01\x00\x02\x00"

def test_livekit_audio_frame_encoder_uses_configured_pcm_defaults() -> None:
    encoder = AudioFrameEncoder[FakeAudioFrame](
        default_sample_rate_hz=16000,
        default_channels=1,
        audio_frame_factory=fake_audio_frame_factory,
    )

    frame = asyncio.run(await_value(encoder.encode(audio_device.AudioOutputData(audio=b"\x01\x00\x02\x00"))))

    assert frame.sample_rate == 16000
    assert frame.num_channels == 1
    assert frame.samples_per_channel == 2

@pytest.mark.parametrize("media_type", ["audio/opus", "audio/mpeg"])
def test_livekit_audio_frame_encoder_rejects_non_pcm_audio(media_type: str) -> None:
    encoder = AudioFrameEncoder[FakeAudioFrame](audio_frame_factory=fake_audio_frame_factory)

    with pytest.raises(AudioFrameError, match="PCM"):
        _ = asyncio.run(await_value(encoder.encode(audio_device.AudioOutputData(audio=b"\x01\x00", media_type=media_type))))

def test_livekit_audio_frame_encoder_rejects_misaligned_pcm() -> None:
    encoder = AudioFrameEncoder[FakeAudioFrame](audio_frame_factory=fake_audio_frame_factory)

    with pytest.raises(AudioFrameError, match="16-bit PCM"):
        _ = asyncio.run(
            await_value(
                encoder.encode(
                    audio_device.AudioOutputData(audio=b"\x01\x00\x02", media_type="audio/pcm", sample_rate_hz=48000, channels=1)
                )
            )
        )

def test_livekit_audio_frame_decoder_converts_livekit_frame_to_bot_audio_input() -> None:
    decoder = AudioFrameDecoder()
    frame = FakeAudioFrame(
        data=b"\x01\x00\x02\x00",
        sample_rate=24000,
        num_channels=1,
        samples_per_channel=2,
    )

    audio = asyncio.run(await_value(decoder.decode(frame)))

    assert audio == audio_device.AudioInputData(
        audio=b"\x01\x00\x02\x00",
        media_type="audio/pcm",
        sample_rate_hz=24000,
        channels=1,
    )
    assert isinstance(decoder, abilities.Decoder)

def test_livekit_audio_source_writer_captures_bot_audio_output() -> None:
    source = FakeAudioSource()
    encoder = AudioFrameEncoder[FakeAudioFrame](audio_frame_factory=fake_audio_frame_factory)
    writer = AudioSourceWriter[FakeAudioFrame](source=source, encoder=encoder)

    asyncio.run(
        writer.write(
            audio_device.AudioOutputData(audio=b"\x01\x00\x02\x00", media_type="audio/pcm", sample_rate_hz=48000, channels=1)
        )
    )

    assert source.captured_frames == [
        FakeAudioFrame(
            data=b"\x01\x00\x02\x00",
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2,
        )
    ]

def test_livekit_audio_bridge_receives_remote_frames_and_publishes_local_audio() -> None:
    source = FakeAudioSource()
    encoder = AudioFrameEncoder[FakeAudioFrame](audio_frame_factory=fake_audio_frame_factory)
    bridge: AudioBridge[FakeAudioFrame] = AudioBridge[FakeAudioFrame].from_audio_source(source=source, encoder=encoder)

    received = asyncio.run(
        await_value(
            bridge.receive_frame(
                FakeAudioFrame(
                    data=b"\x01\x00\x02\x00",
                    sample_rate=24000,
                    num_channels=1,
                    samples_per_channel=2,
                )
            )
        )
    )
    asyncio.run(
        bridge.publish_audio(
            audio_device.AudioOutputData(audio=b"\x03\x00\x04\x00", media_type="audio/pcm", sample_rate_hz=48000, channels=1)
        )
    )

    assert received == audio_device.AudioInputData(
        audio=b"\x01\x00\x02\x00",
        media_type="audio/pcm",
        sample_rate_hz=24000,
        channels=1,
    )
    assert source.captured_frames == [
        FakeAudioFrame(
            data=b"\x03\x00\x04\x00",
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2,
        )
    ]

def test_create_audio_bridge_stands_up_sdk_backed_bridge() -> None:
    async def exercise_bridge() -> audio_device.AudioInputData:
        bridge = create_audio_bridge(sample_rate_hz=48000, channels=1, queue_size_ms=10)
        frame = rtc.AudioFrame(
            data=b"\x01\x00\x02\x00",
            sample_rate=24000,
            num_channels=1,
            samples_per_channel=2,
        )

        received = await bridge.receive_frame(frame)
        await asyncio.wait_for(
            bridge.publish_audio(
                audio_device.AudioOutputData(audio=b"\x03\x00\x04\x00", media_type="audio/pcm", sample_rate_hz=48000, channels=1)
            ),
            timeout=1.0,
        )
        return received

    received = asyncio.run(exercise_bridge())

    assert received == audio_device.AudioInputData(
        audio=b"\x01\x00\x02\x00",
        media_type="audio/pcm",
        sample_rate_hz=24000,
        channels=1,
    )

def test_livekit_pcm_wav_decoder_wraps_pcm_in_wav_container_for_existing_speech_decoding_contract() -> None:
    decoder = PcmWavDecoder(sample_rate_hz=16000, channels=1)

    wav = asyncio.run(await_value(decoder.decode(b"\x01\x00\x02\x00")))

    assert wav.startswith(b"RIFF")
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert wav.endswith(b"\x01\x00\x02\x00")

def test_livekit_voice_decoder_wraps_pcm_audio_for_injected_speech_decoder() -> None:
    speech_decoder = RecordingSpeechDecoder()
    decoder = VoiceDecoder(
        pcm_decoder=PcmWavDecoder(sample_rate_hz=16000, channels=1),
        speech_decoder=speech_decoder,
    )

    transcript = asyncio.run(
        await_value(
            decoder.decode(
                participating.AudioStimulus(
                    source_participant_ref="caller",
                    content=b"\x01\x00\x02\x00",
                )
            )
        )
    )

    assert transcript == "hello caller"
    assert isinstance(decoder, abilities.VoiceDecoder)
    assert len(speech_decoder.inputs) == 1
    assert speech_decoder.inputs[0].startswith(b"RIFF")
    assert speech_decoder.inputs[0].endswith(b"\x01\x00\x02\x00")

def test_livekit_voice_decoder_is_not_package_root_constructable_export() -> None:
    import bot.providers.livekit as livekit

    assert "VoiceDecoder" not in livekit.__all__
    assert not hasattr(livekit, "VoiceDecoder")
    assert "PhoneService" in livekit.__all__
    assert "Phone" in livekit.__all__
