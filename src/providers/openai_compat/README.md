# mosfet-provider-openai-compat

OpenAI-compatible provider package for stateforward.mosfet text generation, processing, and embedding abilities.

## Live Mercury tool-call eval

Evidence suite for Inception Mercury 2 dispatch tool calls (phone-bot intuition shape):

```sh
# from repo root; requires BOT_MERCURY_API_KEY (or MERCURY_API_KEY / INCEPTION_API_KEY)
uv run --package mosfet-provider-openai-compat --group dev \
  python -m pytest src/providers/openai_compat/tests/test_mercury_tool_call_live.py -m live -v -s
```

Cases: ring WAV + `kind=phone.ringing`, kind-only compact WAV, parseable offered events, ambient negative control.

The package owns the OpenAI SDK dependency for providers that expose the Chat Completions API shape. Pass a `model`,
and a `base_url` and optional `api_key` for any endpoint but the local one, to `ChatClient`, then use that client
through `TextGenerator` or `Processing`. `base_url` defaults to the loopback endpoint `http://127.0.0.1:8080/v1`, where
a local llama.cpp server listens, so `ChatClient(model="local-model")` reaches it; a cloud endpoint, OpenAI's
included, is always named.

```python
from bot.abilities.language.text import InputData, TextMessage, TextRole
from bot.providers.openai_compat import ChatClient, TextGenerator

client = ChatClient(
    base_url="https://llm.example.com/v1",
    api_key="provider-key",
    model="compatible-model",
)
generator = TextGenerator(client=client, provider="example-provider")

output = await generator.generate(
    InputData(messages=(TextMessage(role=TextRole.USER, content="Say hello."),))
)
```

Reasoning effort is a typed `ReasoningEffort` field on both generators. On Chat Completions, OpenAI reasoning models
accept function tools only with effort `none`: an unset effort defaults to `none` when tools are present, and a
requested effort other than `none` with tools raises `UnsupportedReasoningEffortError`. Use `ResponsesTextGenerator`
(the OpenAI Responses API, served by the same `ChatClient`) to combine function tools with reasoning:

```python
generator = ResponsesTextGenerator(client=client, reasoning_effort=ReasoningEffort.HIGH)
```

Both generators record token usage (including reasoning tokens) through `mosfet.telemetry.record_generator_usage`.

`TextGenerator.generate(input, deltas=...)` streams the completion (server-sent events through `ChatClient`'s
`async_client`, an `openai.AsyncOpenAI` built lazily from the same endpoint, key and headers) and calls `deltas` with
each `text.Delta` as it arrives: new content, or new arguments of one tool call. The returned output is assembled from
the whole stream and is the same, authoritative result as without `deltas`. `Processor.process(input, drafts=...)` uses
this to read the streamed `dispatch` arguments with `processing.DraftParser`, so a marked field (for example
the `words` of each of `communication.respond`'s `parts`) reaches its target as a draft while the model is still
writing it.
`ResponsesTextGenerator` does not stream; it ignores `deltas`.

Local models often send a nested array or object as a JSON string, for example `{"events": "[{...}]"}`. Tool call
arguments, and every `dispatch` value whose schema takes only an array or object (the `events` array, its items, and
each event's data down through nested fields), are decoded from such a string once; a string field stays as sent.
A response that still fails raises `ProcessingError` (or `TextGenerationError`) naming the cause's type and message,
never the values the model sent.

A user message may carry ordered content parts: `text.TextPart` and `text.ImagePart` (a typed `mosfet.media.Image`).
Chat Completions receives them as `text` and `image_url` parts and the Responses API as `input_text` and `input_image`
parts, each image as a base64 `data:` URL. Both generators default to a text-only `image_capacity` because the endpoint
may front any model; inject the capacity of a vision endpoint. A request over the capacity raises
`TextGenerationError`.

```python
from mosfet import media
from mosfet.abilities.language import text
from mosfet.providers.openai_compat import TextGenerator

generator = TextGenerator(
    client=client,
    image_capacity=media.ImageCapacity(
        max_images=4, max_total_bytes=8 * 1024 * 1024, media_types=frozenset({"image/png", "image/jpeg"})
    ),
)
photo = media.Image(content=png_bytes, media_type="image/png")
output = await generator.generate(
    text.InputData(
        messages=(
            text.TextMessage(
                role=text.TextRole.USER,
                content=(text.TextPart(text="What is in this photo?"), text.ImagePart(image=photo)),
            ),
        )
    )
)
```

## Vision stages

`vision.description.GenerativeDescriber` and `vision.detection.GenerativeDetector` serve
`seeing.Seeing`'s describer and object detector over any image-capable text generator, so a
vision-language model served behind an OpenAI-compatible endpoint (for example Qwen3.6 on vLLM)
describes and detects for the bot. Give the generator the capacity of that endpoint; both stages
refuse an image the capacity does not admit before calling it. Endpoint and model come from
configuration.

```python
import datetime

from mosfet import media
from mosfet.abilities import seeing, vision
from mosfet.providers import openai_compat

vlm = openai_compat.TextGenerator(
    client=openai_compat.ChatClient(base_url=config.vlm_base_url, model=config.vlm_model, api_key=config.vlm_api_key),
    image_capacity=media.ImageCapacity(
        max_images=1, max_total_bytes=8 * 1024 * 1024, media_types=frozenset({"image/png", "image/jpeg"})
    ),
)
eye = seeing.Seeing(
    change=change_measure,
    describer=vision.description.GenerativeDescriber(generator=vlm),
    object_detector=vision.detection.GenerativeDetector(generator=vlm),
    # Each stage serves one request at a time and a look interprets up to seeing.MAX_VIEWS (4) views
    # at once: cover four of the endpoint's slowest healthy calls.
    stage_timeout=datetime.timedelta(seconds=40),
)
```

Both stages send one user message (their instructions, then the image) with tools off. The
instructions (`DESCRIBE_INSTRUCTIONS`, `DETECT_INSTRUCTIONS`) are overridable defaults. The
detector asks for a JSON array of `{label, confidence, box}` with boxes normalized to the image;
items that are not valid objects are skipped and counted on `bot.vision.detection.invalid`.

Live check against a served model (prints what the model said, never pins it):

```sh
MOSFET_LIVE_VLM_BASE_URL=http://vlm.example:8000/v1 \
MOSFET_LIVE_VLM_MODEL=served-model-name \
  uv run --package mosfet-provider-openai-compat --group dev python -m pytest \
  src/providers/openai_compat/tests/test_vision_live.py -m live -v -s
```

`MOSFET_LIVE_VLM_API_KEY` is passed to the client when set.

`Processing` turns a stateforward.mosfet `InputData` frame into a typed output by asking the model to return JSON
that validates against the provided Pydantic/type schema.

## Speech

`SpeechEncoder` is a `vocal.speech.SpeechEncoder` over an OpenAI-compatible `/v1/audio/speech` endpoint (for example
Breeze TTS). The endpoint takes one whole `input`, so the reply is sent as one request per run of consecutive parts
with the same direction (the provider merge rule): a run closes when a part with another direction begins, or when
the reply ends, and only then is its request sent, in order; closing the stream before the first run closes sends
none. It asks for `response_format="pcm"` and streams the raw s16le body through the SDK's `with_streaming_response`,
yielding whole frames as they arrive with per-request alignment: every chunk carries `""` but a request's last, which
carries the run's words (so a stop before it plays records none of the run as said). Closing the iterator closes the
response, which is how such a server cancels. `sample_rate_hz` and `channels` declare what the server sends: an
`X-Sample-Rate` or `X-Sample-Format` header that disagrees is `encoding.Error("invalid_response")`. `voice` is the
provider voice the text is spoken in; without one the request names no voice.

Each request is built from the parts' typed expressions, never from their words. `expressiveness` declares what the
server applies and `markup` how it writes each declared sound and style inline (`ANGLE_TAGS` is a lab engine's
`<laugh>`/`<whisper>` convention): the run's style tag comes first, each part's sound tag goes before or after that
part's words, the run's natural-language cues become one `instructions` written after `voice_description` (a lab
engine's `instructions` replace the voice's stored description), and a declared pace range becomes `speed`. Words that
look like tags are sent unchanged. Undeclared cues and sounds change nothing.

`SpeechDecoder` is a `hearing.speech.SpeechDecoder` over `/v1/audio/transcriptions`: raw PCM at `sample_rate_hz` and
`channels` is wrapped as WAV (a WAV input is sent as is) and uploaded as multipart `file`; the stripped `text` comes
back as UTF-8 bytes, and blank text raises `SpeechDecodingError("decode_failed")`.

```python
from mosfet.abilities import vocal
from mosfet.providers import openai_compat

encoder = openai_compat.SpeechEncoder(model="tts", base_url=config.tts_base_url, voice="warm")
lab = openai_compat.SpeechEncoder(
    model="tts",
    base_url=config.tts_base_url,
    voice=voice_key,
    voice_description=voice_description,
    expressiveness=vocal.speech.Expressiveness(
        cues=frozenset({"instructions", "emotion", "intensity", "emphasis", "style", "sound"}),
        styles=("whisper",),
        sounds=frozenset({"laugh", "giggle", "chuckle", "gasp", "sigh"}),
    ),
    markup=openai_compat.ANGLE_TAGS,
)
decoder = openai_compat.SpeechDecoder(model="stt", base_url=config.stt_base_url, language="en")
```

Both map a 4xx answer to `invalid_input`, a 5xx answer or an unreachable server to `unavailable` and an SDK timeout to
`timeout`. `tests/test_speech_live.py` checks served endpoints when `BOT_OPENAI_COMPAT_SPEECH_BASE_URL` is set
(`-m live`).

## Embeddings

`Embedder` implements the core `embedding.Embedder` against a remote `/v1/embeddings` endpoint, so an
`embedding.Embedding` ability can embed through it. `EmbeddingClient` has its own `model`, `base_url`, and `api_key`
because an embedding model is served separately from a chat model; every request asks for float vectors. The operator
declares what the served model accepts in `capacity` (its `Space`, item limit, and image budget), and `instructions`
carries the model's task prefixes, applied once by the embedder.

```python
from mosfet import media
from mosfet.abilities import embedding
from mosfet.providers.openai_compat import Embedder, EmbeddingClient

embedder = Embedder(
    client=EmbeddingClient(base_url="http://spark.local:8080/v1", model="embeddinggemma-2"),
    capacity=embedding.Capacity(
        space=embedding.Space(model="google/embeddinggemma-2", dimensions=768),
        max_items=16,
        max_input_tokens=None,
        images=media.ImageCapacity(max_images=4, max_total_bytes=8 * 1024 * 1024, media_types=frozenset({"image/png"})),
        interleaved=True,
    ),
    instructions=embedding.EMBEDDING_GEMMA_2,
)
ability = embedding.Embedding(embedder=embedder)
```

A text-only item is sent as one prefixed string. An item with images is sent as an OAI content array
(`{"content": [{"type": "text", ...}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}`),
which llama-server accepts on `/v1/embeddings` from llama.cpp `4fbc76de`. Images are sent only within the declared
`capacity.images`; the default capacity is text-only. Vectors longer than `space.dimensions` are truncated
(Matryoshka) and every vector is L2-normalized client-side; set `send_dimensions=True` to ask the server for
`dimensions` instead. A response with the wrong row count, indices, length, or non-finite values fails with
`INVALID_RESPONSE`, and SDK failures map to closed kinds (`TIMEOUT`, `UNAVAILABLE`, `UNAUTHORIZED`, `OVERLOADED`,
`INVALID_INPUT`, `EMBED_FAILED`).

EmbeddingGemma-2 is non-causal, so llama-server must take the longest input in one batch: serve it with `-b` and
`-ub` at least that many tokens.

```sh
llama-server -m embeddinggemma-2-Q8_0.gguf --mmproj mmproj-embeddinggemma-2-Q8_0.gguf \
  --embeddings --pooling mean -ub 2048 -b 2048 --host 0.0.0.0 --port 8080
```

`tests/test_embeddings_live.py` checks a served model when `BOT_OPENAI_COMPAT_EMBEDDINGS_BASE_URL` and
`BOT_OPENAI_COMPAT_EMBEDDINGS_MODEL` are set (`-m live`).
