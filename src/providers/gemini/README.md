# bot-provider-gemini

Google Gemini provider package for stateforward.bot text generation, processing, and speech.

This package owns the [`google-genai`](https://googleapis.github.io/python-genai/) SDK
dependency (2.11+). It follows current Gemini API guidance:

- **Interactions API** for speech (TTS / STT) — recommended path for new features
- **`models.generate_content`** for stateforward.bot text generation / processing (still fully
  supported; Interactions migration for text can follow)

## Text

```python
from bot.abilities.language.text import InputData, TextMessage, TextRole
from bot.providers.gemini import ChatClient, TextGenerator

client = ChatClient(
    model="gemini-3.5-flash",
    api_key="your-gemini-api-key",
)
generator = TextGenerator(client=client)

output = await generator.generate(
    InputData(messages=(TextMessage(role=TextRole.USER, content="Say hello."),))
)
```

`Processing` turns a stateforward.bot deliberative frame into typed JSON / operation tool calls.

## Speech (TTS / STT)

Current docs use `client.interactions.create` (not the legacy generateContent-only speech path):

```python
from bot.providers.gemini import ChatClient, SpeechDecoder, SpeechEncoder

client = ChatClient(api_key="your-gemini-api-key")

# TTS — gemini-3.1-flash-tts-preview
# interactions.create(model=..., input=text, response_format={"type":"audio"},
#                     generation_config={"speech_config":[{"voice":"Kore"}]})
audio = await SpeechEncoder(client=client, voice_name="Kore").encode(b"Hello.")

# STT — gemini-3.5-flash multimodal audio understanding
# interactions.create(model=..., input=[{"type":"text",...},{"type":"audio",...}])
transcript = await SpeechDecoder(client=client, mime_type="audio/wav").decode(audio)
```

| Component | Default model | API |
|---|---|---|
| `ChatClient` / text | `gemini-3.5-flash` | `models.generate_content` |
| `SpeechEncoder` (TTS) | `gemini-3.1-flash-tts-preview` | `interactions.create` |
| `SpeechDecoder` (STT) | `gemini-3.5-flash` | `interactions.create` |

`SpeechEncoder` returns raw PCM (24 kHz mono) by default; set `output_format="wav"` for a WAV container.
Interactions are created with `store=False` so encode/decode calls do not retain server-side state.

Credentials resolve from, in order: `api_key=`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY`.
Vertex AI: `ChatClient(vertexai=True, project=..., location=...)`.

```sh
uv run --package bot-provider-gemini --group dev python -m pytest src/providers/gemini/tests
```
