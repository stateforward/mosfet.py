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

The package owns the OpenAI SDK dependency for providers that expose the Chat Completions API shape. Pass a `base_url`,
`model`, and optional `api_key` to `ChatClient`, then use that client through `TextGenerator` or
`Processing`.

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
