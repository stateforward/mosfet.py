# bot-provider-openai-compat

OpenAI-compatible provider package for stateforward.bot text generation and processing abilities.

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

`Processing` turns a stateforward.bot `InputData` frame into a typed output by asking the model to return JSON
that validates against the provided Pydantic/type schema.
