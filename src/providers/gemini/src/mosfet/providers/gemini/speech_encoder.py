from __future__ import annotations

import asyncio
import base64
import collections.abc
import typing

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech

from .client import ContentClient, error_detail


_DEFAULT_TTS_MODEL = "gemini-3.1-flash-tts-preview"
_DEFAULT_VOICE = "Kore"
_DEFAULT_SAMPLE_RATE_HZ = 24_000


def _mapping(value: object) -> collections.abc.Mapping[str, object] | None:
    if not isinstance(value, collections.abc.Mapping):
        return None
    return typing.cast(collections.abc.Mapping[str, object], value)


def _bytes_from_payload(value: object) -> bytes | None:
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if isinstance(value, str) and value:
        try:
            return base64.b64decode(value, validate=False)
        except Exception:
            return None
    return None


def audio_bytes_from_interaction_response(response: collections.abc.Mapping[str, object]) -> bytes:
    """Extract PCM/audio bytes from an Interactions API TTS response.

    Prefer the convenience ``output_audio.data`` field documented for
    ``client.interactions.create`` speech generation.
    """

    output_audio = _mapping(response.get("output_audio"))
    if output_audio is not None:
        data = _bytes_from_payload(output_audio.get("data"))
        if data:
            return data

    # Fallback: scan model steps for audio content blocks.
    steps = response.get("steps")
    if isinstance(steps, collections.abc.Sequence) and not isinstance(steps, str | bytes | bytearray):
        for step in steps:
            step_mapping = _mapping(step)
            if step_mapping is None:
                continue
            for key in ("content", "output", "delta"):
                block = step_mapping.get(key)
                candidates: list[object]
                if isinstance(block, collections.abc.Sequence) and not isinstance(block, str | bytes | bytearray):
                    candidates = list(block)
                else:
                    candidates = [block]
                for candidate in candidates:
                    candidate_mapping = _mapping(candidate)
                    if candidate_mapping is None:
                        continue
                    if candidate_mapping.get("type") == "audio" or "data" in candidate_mapping:
                        data = _bytes_from_payload(candidate_mapping.get("data"))
                        if data:
                            return data

    raise encoding.Error("invalid_response", "Gemini TTS interaction did not include output_audio data.")


class SpeechEncoder(speech.SpeechEncoder):
    """Gemini text-to-speech as mono s16le PCM at ``sample_rate_hz`` (24 kHz by default).

    Uses the Interactions API recommended by current Gemini docs::

        client.interactions.create(
            model="gemini-3.1-flash-tts-preview",
            input=text,
            response_format={"type": "audio"},
            generation_config={"speech_config": [{"voice": "Kore"}]},
        )

    The model is not incremental and takes the whole text, and this encoder applies no expression
    (``expressiveness`` is the default, so all parts form one run): the reply is read to its end (joining the words
    of its pieces in order is the only buffering) and said as one chunk, aligned to all the words. A pause splits
    the text: each stretch between pauses is one chunk, with ``format.silence`` between them. ``voice_name`` is the
    Gemini voice the text is spoken in.
    """

    def __init__(
        self,
        *,
        client: ContentClient,
        model: str = _DEFAULT_TTS_MODEL,
        voice_name: str = _DEFAULT_VOICE,
        language_code: str | None = None,
        sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ,
        store: bool = False,
        generation_config: collections.abc.Mapping[str, object] | None = None,
    ) -> None:
        self.format: speech.Format = speech.Format(sample_rate_hz=sample_rate_hz, channels=1)
        self._client: ContentClient = client
        self._model: str = model
        self._voice_name: str = voice_name
        self._language_code: str | None = language_code
        self._store: bool = store
        self._generation_config: dict[str, object] = dict(generation_config or {})

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[speech.Piece]
    ) -> collections.abc.AsyncIterator[speech.Audio]:
        # The text between pauses, each stretch with the pause that opens it: a pause splits the text, so each
        # stretch is said whole and the silence between them in its place.
        stretches: list[tuple[speech.Pause | None, list[str]]] = [(None, [])]
        async for piece in input:
            if piece.expression is not None and piece.expression.pause is not None:
                stretches.append((piece.expression.pause, []))
            stretches[-1][1].append(piece.words)
        # The last chunk, held so blank words read after it ride it; blank words before any chunk ride the next.
        held: speech.Audio | None = None
        unsaid = ""
        for pause, words in stretches:
            if pause is not None:
                if held is not None:
                    yield held
                held, unsaid = speech.Audio(pcm=self.format.silence(pause), text=unsaid), ""
            said = "".join(words)
            if not said.strip():
                if held is None:
                    unsaid += said
                else:
                    held = held.model_copy(update={"text": held.text + said})
                continue
            pcm = await asyncio.to_thread(self._generate, said, self._voice_name)
            if held is not None:
                yield held
            held, unsaid = speech.Audio(pcm=pcm, text=unsaid + said), ""
        if held is None:
            raise encoding.Error("invalid_input", "Gemini speech encoding requires non-blank text.")
        yield held

    def _generate(self, text: str, voice_name: str) -> bytes:
        speech_entry: dict[str, object] = {"voice": voice_name}
        if self._language_code is not None:
            speech_entry["language_code"] = self._language_code
        generation_config: dict[str, object] = {**self._generation_config, "speech_config": [speech_entry]}
        try:
            response = self._client.create_interaction(
                model=self._model,
                input=text,
                response_format={"type": "audio"},
                generation_config=generation_config,
                store=self._store,
            )
        except Exception as error:
            raise encoding.Error("encode_failed", f"Gemini speech encoding failed: {error_detail(error)}") from error
        pcm = audio_bytes_from_interaction_response(response)
        output_audio = _mapping(response.get("output_audio"))
        reported = None if output_audio is None else output_audio.get("sample_rate")
        # Raw PCM carries no header, so the rate this encoder declares is the rate the audio gets played
        # at downstream — nothing can correct a disagreement, and a silent one is heard as a wrong-speed voice
        # rather than raised as an error. An absent report means the API did not say; the declared rate stands.
        if isinstance(reported, int) and reported > 0 and reported != self.format.sample_rate_hz:
            raise encoding.Error(
                "invalid_response",
                (
                    f"Gemini returned {reported} Hz PCM but this encoder declares {self.format.sample_rate_hz} Hz. "
                    + "Raw PCM has no header to carry the difference, so playing it would sound wrong rather than "
                    + f"fail. Configure sample_rate_hz to {reported}."
                ),
            )
        if len(pcm) % 2:
            raise encoding.Error("invalid_response", "Gemini returned PCM that ends mid-sample.")
        return pcm


__all__ = [
    "SpeechEncoder",
    "audio_bytes_from_interaction_response",
]
