from __future__ import annotations

from mosfet.abilities.hearing import speech
from mosfet.abilities.communication.conversation import turn_detector

import asyncio
import collections.abc
import dataclasses
import pathlib
import sys
import threading
import types
import typing
import wave

import hsm
import pytest

from mosfet.providers.mlx_audio import SpeechDecoder, SpeechDecodingError, VoiceDecoder
from mosfet.providers.mlx_audio._mlx import SpeechDecodingModel, load_speech_decoding_model
from tests.hsm_instance_state import start_ability_tree


@dataclasses.dataclass(frozen=True)
class FakeTranscription:
    text: str


@dataclasses.dataclass
class FakeSpeechDecodingModel:
    result: object
    expected_audio: bytes = b"audio"
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def generate(self, audio: str, **kwargs: object) -> object:
        audio_path = pathlib.Path(audio)
        assert audio_path.read_bytes() == self.expected_audio
        self.calls.append({"audio": audio_path, **kwargs})
        return self.result


class FailingSpeechDecodingModel:
    def generate(self, audio: str, **kwargs: object) -> object:
        del audio, kwargs
        raise RuntimeError("mlx unavailable")


@dataclasses.dataclass
class StrictSpeechDecodingModel:
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def generate(self, audio: str, *, verbose: bool) -> object:
        audio_path = pathlib.Path(audio)
        assert audio_path.read_bytes() == b"audio"
        self.calls.append({"audio": audio_path, "verbose": verbose})
        return {"text": "strict transcript"}


def install_mlx_audio_loader(
    monkeypatch: pytest.MonkeyPatch,
    load_model: collections.abc.Callable[[str], object],
) -> None:
    mlx_audio_module = types.ModuleType("mlx_audio")
    stt_module = types.ModuleType("mlx_audio.stt")
    stt_utils_module = types.ModuleType("mlx_audio.stt.utils")
    setattr(stt_utils_module, "load_model", load_model)
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", stt_utils_module)


async def await_speech_decoding(output: collections.abc.Awaitable[bytes]) -> bytes:
    return await output


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


async def await_speech_decoding_ability(
    ability: speech.SpeechDecoding,
    input: bytes,
    completed: collections.abc.Callable[[], bool],
) -> None:
    await start_ability_tree(None, ability)
    _ = await ability.apply(input)
    for _ in range(100):
        if completed():
            return
        await asyncio.sleep(0)
    raise AssertionError("speech decoding ability did not run decoder")


def test_speech_decoder_uses_injected_model() -> None:
    model = FakeSpeechDecodingModel(result=FakeTranscription(text="hello there"))
    decoder = SpeechDecoder(
        model=model,
        model_id="local/whisper",
        language="en",
        verbose=True,
        generate_kwargs={"max_tokens": 32},
    )

    output = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert output == b"hello there"
    assert len(model.calls) == 1
    audio_path = model.calls[0]["audio"]
    assert isinstance(audio_path, pathlib.Path)
    assert not audio_path.exists()
    assert model.calls[0] == {
        "audio": model.calls[0]["audio"],
        "language": "en",
        "verbose": True,
        "max_tokens": 32,
    }
    assert isinstance(decoder, speech.SpeechDecoder)


def test_speech_decoder_uses_injected_loader() -> None:
    models: list[FakeSpeechDecodingModel] = []

    def load_model(model_id: str) -> FakeSpeechDecodingModel:
        assert model_id == "local/whisper"
        model = FakeSpeechDecodingModel(result={"text": "loaded transcript"})
        models.append(model)
        return model

    decoder = SpeechDecoder(model_id="local/whisper", load_model=load_model)

    output = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert output == b"loaded transcript"
    assert len(models) == 1


def test_speech_decoder_default_constructor_uses_mlx_audio_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_model_ids: list[str] = []
    model = FakeSpeechDecodingModel(result={"text": "default transcript"})

    def load_model(model_id: str) -> FakeSpeechDecodingModel:
        loaded_model_ids.append(model_id)
        return model

    install_mlx_audio_loader(monkeypatch, load_model)
    decoder = SpeechDecoder()

    output = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert output == b"default transcript"
    assert loaded_model_ids == ["mlx-community/whisper-large-v3-turbo"]
    assert len(model.calls) == 1


def test_speech_decoder_filters_unsupported_generate_kwargs() -> None:
    model = StrictSpeechDecodingModel()
    strict_model = typing.cast(SpeechDecodingModel, typing.cast(object, model))
    decoder = SpeechDecoder(
        model=strict_model,
        language="en",
        verbose=True,
        generate_kwargs={"max_tokens": 32},
    )

    output = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert output == b"strict transcript"
    assert len(model.calls) == 1
    audio_path = model.calls[0]["audio"]
    assert isinstance(audio_path, pathlib.Path)
    assert not audio_path.exists()
    assert model.calls[0] == {"audio": model.calls[0]["audio"], "verbose": True}


def test_speech_decoder_can_drive_speech_decoding_ability() -> None:
    model = FakeSpeechDecodingModel(result=FakeTranscription(text="ability transcript"))
    decoder = SpeechDecoder(model=model)
    ability = speech.SpeechDecoding(decoder=decoder)

    asyncio.run(await_speech_decoding_ability(ability, b"audio", lambda: bool(model.calls)))

    assert len(model.calls) == 1


def test_speech_decoder_is_awaitable() -> None:
    decoder = SpeechDecoder(model=FakeSpeechDecodingModel(result="plain transcript"))

    output = decoder.decode(b"audio")

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output) == b"plain transcript"


def test_speech_decoder_wraps_provider_errors() -> None:
    decoder = SpeechDecoder(model=FailingSpeechDecodingModel())

    with pytest.raises(SpeechDecodingError) as error:
        _ = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_speech_decoder_wraps_loader_errors() -> None:
    def load_model(model_id: str) -> FakeSpeechDecodingModel:
        del model_id
        raise RuntimeError("model unavailable")

    decoder = SpeechDecoder(load_model=load_model)

    with pytest.raises(SpeechDecodingError) as error:
        _ = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_speech_decoder_rejects_missing_transcription_text() -> None:
    decoder = SpeechDecoder(model=FakeSpeechDecodingModel(result={"segments": []}))

    with pytest.raises(SpeechDecodingError) as error:
        _ = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert isinstance(error.value.__cause__, TypeError)


def test_speech_decoder_owns_workers_per_decoder_and_shuts_down_queued_work() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingModel:
        threads: list[threading.Thread] = []

        def generate(self, audio: str, **kwargs: object) -> object:
            del audio, kwargs
            self.threads.append(threading.current_thread())
            started.set()
            assert release.wait(timeout=1.0)
            return {"text": "done"}

    model = BlockingModel()
    first_decoder = SpeechDecoder(model=model)
    second_decoder = SpeechDecoder(model=model)

    async def run() -> None:
        first = asyncio.create_task(first_decoder.decode(b"audio"))
        _ = await asyncio.to_thread(started.wait, 1.0)
        second = asyncio.create_task(first_decoder.decode(b"audio"))
        await asyncio.sleep(0)
        first_decoder.shutdown()
        _ = second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        assert await first == b"done"

        with pytest.raises(RuntimeError, match="closed"):
            _ = await first_decoder.decode(b"audio")

        assert await second_decoder.decode(b"audio") == b"done"
        assert len(set(model.threads)) == 2

    asyncio.run(run())
    second_decoder.shutdown()


def test_speech_decoder_rejects_work_beyond_one_active_and_one_queued_call() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingModel:
        def generate(self, audio: str, **kwargs: object) -> object:
            del audio, kwargs
            started.set()
            assert release.wait(timeout=1.0)
            return {"text": "done"}

    decoder = SpeechDecoder(model=BlockingModel())

    async def run() -> None:
        active = asyncio.create_task(decoder.decode(b"active"))
        assert await asyncio.to_thread(started.wait, 1.0)
        queued = asyncio.create_task(decoder.decode(b"queued"))
        await asyncio.sleep(0)
        with pytest.raises(SpeechDecodingError, match="capacity"):
            _ = await decoder.decode(b"rejected")
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        assert await active == b"done"
        await decoder.aclose()

    asyncio.run(run())


def test_speech_decoder_queued_cancellation_prevents_native_execution() -> None:
    first_started = threading.Event()
    release = threading.Event()
    calls: list[bytes] = []

    class BlockingModel:
        def generate(self, audio: str, **kwargs: object) -> object:
            del kwargs
            calls.append(pathlib.Path(audio).read_bytes())
            first_started.set()
            assert release.wait(timeout=1.0)
            return {"text": "done"}

    decoder = SpeechDecoder(model=BlockingModel())

    async def run() -> None:
        active = asyncio.create_task(decoder.decode(b"active"))
        assert await asyncio.to_thread(first_started.wait, 1.0)
        queued = asyncio.create_task(decoder.decode(b"queued"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        assert await active == b"done"
        await decoder.aclose()

    asyncio.run(run())
    assert calls == [b"active"]


def test_speech_decoder_active_cancellation_leaves_native_work_owned_until_completion() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = 0

    class BlockingModel:
        def generate(self, audio: str, **kwargs: object) -> object:
            nonlocal calls
            del audio, kwargs
            calls += 1
            started.set()
            assert release.wait(timeout=1.0)
            finished.set()
            return {"text": "done"}

    decoder = SpeechDecoder(model=BlockingModel())

    async def run() -> None:
        active = asyncio.create_task(decoder.decode(b"active"))
        assert await asyncio.to_thread(started.wait, 1.0)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert not finished.is_set()
        queued = asyncio.create_task(decoder.decode(b"queued"))
        await asyncio.sleep(0)
        with pytest.raises(SpeechDecodingError, match="capacity"):
            _ = await decoder.decode(b"rejected")
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        assert await asyncio.to_thread(finished.wait, 1.0)
        assert await decoder.decode(b"next") == b"done"
        await decoder.aclose()

    asyncio.run(run())
    assert calls == 2


def test_speech_decoder_close_is_bounded_when_native_work_never_returns() -> None:
    started = threading.Event()

    class PermanentlyBlockedModel:
        worker: threading.Thread | None = None

        def generate(self, audio: str, **kwargs: object) -> object:
            del audio, kwargs
            self.worker = threading.current_thread()
            started.set()
            threading.Event().wait()
            raise AssertionError("unreachable")

    model = PermanentlyBlockedModel()
    decoder = SpeechDecoder(model=model)

    async def start() -> asyncio.Task[bytes]:
        work = asyncio.create_task(decoder.decode(b"audio"))
        assert await asyncio.to_thread(started.wait, 1.0)
        return work

    work = asyncio.run(start())
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (decoder.close(), closed.set()))
    closer.start()
    assert closed.wait(timeout=0.5), "close must have a finite terminal bound"
    closer.join(timeout=0.1)
    assert model.worker is not None
    assert model.worker.daemon
    work.cancel()


def test_speech_decoder_aclose_is_bounded_when_native_work_never_returns() -> None:
    started = threading.Event()

    class PermanentlyBlockedModel:
        worker: threading.Thread | None = None

        def generate(self, audio: str, **kwargs: object) -> object:
            del audio, kwargs
            self.worker = threading.current_thread()
            started.set()
            threading.Event().wait()
            raise AssertionError("unreachable")

    model = PermanentlyBlockedModel()
    decoder = SpeechDecoder(model=model)

    async def run() -> None:
        work = asyncio.create_task(decoder.decode(b"audio"))
        assert await asyncio.to_thread(started.wait, 1.0)
        await asyncio.wait_for(decoder.aclose(), timeout=0.5)
        assert model.worker is not None
        assert model.worker.daemon
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work

    asyncio.run(run())


def test_speech_decoder_repeated_lifecycles_stop_cooperative_workers() -> None:
    workers: list[threading.Thread] = []

    class RecordingModel:
        def generate(self, audio: str, **kwargs: object) -> object:
            del audio, kwargs
            workers.append(threading.current_thread())
            return {"text": "done"}

    async def run() -> None:
        for _ in range(5):
            decoder = SpeechDecoder(model=RecordingModel())
            assert await decoder.decode(b"audio") == b"done"
            await decoder.aclose()

    asyncio.run(run())
    assert len(workers) == 5
    assert all(not worker.is_alive() for worker in workers)


def test_speech_decoder_aclose_waits_for_active_work_and_defers_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingModel:
        worker: threading.Thread | None = None

        def generate(self, audio: str, **kwargs: object) -> object:
            del audio, kwargs
            self.worker = threading.current_thread()
            started.set()
            assert release.wait(timeout=1.0)
            return {"text": "done"}

    model = BlockingModel()
    decoder = SpeechDecoder(model=model)

    async def run() -> None:
        work = asyncio.create_task(decoder.decode(b"audio"))
        assert await asyncio.to_thread(started.wait, 1.0)
        closing = asyncio.create_task(decoder.aclose())
        await asyncio.sleep(0)
        try:
            assert not closing.done()
            _ = closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert await work == b"done"

        worker = model.worker
        assert worker is not None
        assert not worker.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            _ = await decoder.decode(b"audio")

    asyncio.run(run())


def test_speech_decoder_default_models_are_isolated_between_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    first_started = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()

    class ConcurrentModel:
        def __init__(self) -> None:
            self._guard: threading.Lock = threading.Lock()
            self.active: int = 0
            self.max_active: int = 0

        def generate(self, audio: str, **kwargs: object) -> object:
            del audio, kwargs
            with self._guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                if self.active == 1:
                    first_started.set()
                else:
                    second_entered.set()
            assert release.wait(timeout=1.0)
            with self._guard:
                self.active -= 1
            return {"text": "done"}

    models: list[ConcurrentModel] = []
    loaded_model_ids: list[str] = []

    def load_model(model_id: str) -> ConcurrentModel:
        loaded_model_ids.append(model_id)
        model = ConcurrentModel()
        models.append(model)
        return model

    install_mlx_audio_loader(monkeypatch, load_model)
    first_decoder = SpeechDecoder(model_id="test/shared-cache")
    second_decoder = SpeechDecoder(model_id="test/shared-cache")

    async def run() -> None:
        first = asyncio.create_task(first_decoder.decode(b"audio"))
        assert await asyncio.to_thread(first_started.wait, 1.0)
        second = asyncio.create_task(second_decoder.decode(b"audio"))
        _ = await asyncio.to_thread(second_entered.wait, 0.1)
        release.set()
        assert await asyncio.gather(first, second) == [b"done", b"done"]
        _ = await asyncio.gather(first_decoder.aclose(), second_decoder.aclose())

    asyncio.run(run())

    assert loaded_model_ids == ["test/shared-cache", "test/shared-cache"]
    assert [model.max_active for model in models] == [1, 1]


def test_speech_decoder_reuses_its_owned_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_model_ids: list[str] = []

    def load_model(model_id: str) -> FakeSpeechDecodingModel:
        loaded_model_ids.append(model_id)
        return FakeSpeechDecodingModel(result={"text": model_id})

    install_mlx_audio_loader(monkeypatch, load_model)

    async def decode() -> None:
        decoder = SpeechDecoder(model_id="test/owned-cache")
        assert await decoder.decode(b"audio") == b"test/owned-cache"
        assert await decoder.decode(b"audio") == b"test/owned-cache"
        await decoder.aclose()

    asyncio.run(decode())
    assert loaded_model_ids == ["test/owned-cache"]


def test_voice_decoder_packages_raw_pcm_for_mlx_audio_file_decoder() -> None:
    """Listening's raw PCM product must reach MLX Audio as a valid WAV file."""

    class WavOnlyModel:
        def generate(self, audio: str, **kwargs: object) -> object:
            del kwargs
            with wave.open(audio, "rb") as stream:
                assert stream.getframerate() == 16_000
                assert stream.getnchannels() == 1
                assert stream.getsampwidth() == 2
                assert stream.readframes(1)
            return {"text": "hello from pcm"}

    source = pathlib.Path(__file__).parent / "assets" / "speech.wav"
    with wave.open(str(source), "rb") as stream:
        pcm = stream.readframes(stream.getnframes())
        sample_rate_hz = stream.getframerate()
        channels = stream.getnchannels()

    decoder = VoiceDecoder(speech_decoder=SpeechDecoder(model=WavOnlyModel()))
    output = asyncio.run(
        decoder.decode(
            turn_detector.AudioStimulus(
                source_participant_ref="caller",
                content=pcm,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
            )
        )
    )

    assert output == "hello from pcm"


@pytest.mark.live
def test_speech_decoder_serializes_concurrent_calls_to_shared_mlx_model() -> None:
    """The real MLX model must not receive concurrent GPU generations."""

    source = pathlib.Path(__file__).parent / "assets" / "speech.wav"
    audio = source.read_bytes()
    model: SpeechDecodingModel | None = None

    def load_model(model_id: str) -> SpeechDecodingModel:
        nonlocal model
        if model is None:
            model = load_speech_decoding_model(model_id)
        return model

    decoder = SpeechDecoder(load_model=load_model)

    async def run() -> tuple[bytes, ...]:
        first, second = await asyncio.gather(decoder.decode(audio), decoder.decode(audio))
        return first, second

    first, second = asyncio.run(run())

    assert first
    assert second == first
