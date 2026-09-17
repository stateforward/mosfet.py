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

`Processing` turns a stateforward.mosfet `InputData` frame into a typed output by asking the model to return JSON
that validates against the provided Pydantic/type schema.
