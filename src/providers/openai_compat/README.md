# mosfet-provider-openai-compat

OpenAI-compatible provider package for stateforward.mosfet text generation and processing abilities.

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

`Processing` turns a stateforward.mosfet `InputData` frame into a typed output by asking the model to return JSON
that validates against the provided Pydantic/type schema.
