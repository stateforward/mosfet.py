# Willy

Willy is one bot for the whole house: the people who live there, and anyone who visits. It is not one person's
assistant. It learns who people are: a voice it has never heard is a fact it is shown, and whom that voice
belongs to, what to call them, and whether to ask at all are its own choices. A person may have several voices
(a cold, a phone, a whisper), and each voice is exactly one person's. Willy knows its own voice too.

Everything runs locally, on the lab model server (CT 108; see `.env.example`): the language model, its
embeddings, streaming speech to text, speech and voice design, voiceprints, diarization and tone of voice.

## The bot file

`willy_example/__main__.py` is the bot file. It reads the owner's settings and two secrets, then defines Willy
with one `bot.define` that constructs every part inline, and runs it with `bot.run`:

- **Rooms.** One `computer.Computer` per room the owner lists (`WILLY_ROOMS`), named for the room and set on
  something in it, an `environment.Object` such as a shelf or a counter, at its own place in the house. All are
  served by one `websocket.Server`. A far side in each room carries the room's microphone and speaker (and,
  optionally, camera) to it. A room's computer is shared, so it stamps nobody on what it hears: who is speaking is
  what Willy's ear recognizes.
- **Ear.** `listening.Listening` on the lab's streaming speech to text, with voiceprints into a voice
  `identification.Registry` on the same file database as memory, diarization (several voices in one turn) and
  tone of voice.
- **Mouth.** One `speaking.Speaking`, in the one voice Willy is given when it is born (`bot.voice`: the first
  time Willy starts, its model conceives a voice from its traits and the lab designs it). Every room's computer
  hears it and plays it on that room's speaker, like an intercom.
- **Mind.** `bot.cognition` on the lab's language model, with associations, consolidation, autonomy, routines
  and the speech seed that takes heard speech into a conversation. Its memory recalls by the lab's
  EmbeddingGemma 2 vectors.
- **Name.** `identity.Identity`: Willy is born nameless and may take a name when someone gives it one.
- **People.** `people.People`, the capacity to link a voice to a person, name a person, unlink a voice and
  forget a person, under the owner's household.
- **Eye** (`WILLY_CAMERA=1`). `seeing.Seeing` over the rooms' cameras, with the OpenCV Zoo's YuNet and SFace
  into a face registry that people also links and forgets.

Nothing in it tells Willy what to do: it is given no instruction text, no seeded reply and no rule about
whom to answer.

## Running it

```sh
uv sync --project examples/willy
cp examples/willy/.env.example examples/willy/.env          # the lab's addresses; edit the rooms
cp examples/willy/household.example.json examples/willy/household.json   # optional: who lives here
python -c "import secrets; print(secrets.token_urlsafe(32))" > examples/willy/secrets/computer_token_secret
set -a; . examples/willy/.env; set +a
examples/willy/.venv/bin/python -m willy_example --secrets-dir examples/willy/secrets --ready-file /tmp/willy-ready
```

Willy keeps its memory and every voiceprint in one SQLite file, `<data dir>/memory.sqlite3` (`--data-dir` or
`MOSFET_DATA_DIR`). The bot file opens nothing: the file is opened when Willy starts, inside the data dir
`bot.run` has created private (`0700`), and one that cannot be opened fails Willy's start there. The process options, exit codes and stop bound are `bot.run`'s
([`docs/runtime.md`](../../docs/runtime.md)). A missing or malformed setting or secret, or an unreadable or
malformed household file, ends the process with exit `1`, naming it, before anything starts.

## Settings

| Variable | Default | What it is |
|---|---|---|
| `WILLY_LLM_URL`, `WILLY_MODEL` | required | Willy's language model (the lab's `:8100/v1`, `qwen3.8-27b`) |
| `WILLY_EMBED_URL`, `WILLY_EMBED_MODEL` | required | memory's embeddings (`:8014/v1`, `embeddinggemma-2`, 768 dimensions) |
| `WILLY_STT_URL` | required | streaming speech to text, a `ws://` root (`:8001`) |
| `WILLY_TRANSCRIBE_URL` | required | batch transcription, which Willy hears its own name through (`:8001/v1`) |
| `WILLY_TTS_URL` | required | speech, its voices and its voice designer (`:8002`) |
| `WILLY_VOICEPRINT_URL`, `WILLY_DIAR_URL`, `WILLY_EMOTION_URL` | required | voiceprints (`:8003`), diarization (`:8005`), tone (`:8004`) |
| `WILLY_HOST`, `WILLY_PORT` | `127.0.0.1`, `8765` | where the rooms' computers are served |
| `WILLY_ROOMS` | `living-room:shelf` | each room and what its computer sits on, `room:object,...`; rooms are placed 10 m apart in the order given |
| `WILLY_CAMERA`, `WILLY_FACE_MODELS` | `0`, unset | `1` to see; the directory holding `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` |
| `WILLY_TRAITS` | `warm,patient,curious,honest` | the owner's traits for Willy, comma separated |
| `WILLY_HOUSEHOLD_FILE` | unset | a JSON list of who lives in the house; unset, nobody is declared and Willy learns everyone |

Every address is required: where the owner's model server is has no safe default. A room's name is the id of its
computer and the peer its far side connects as: 3 to 63 lower-case letters, digits or hyphens, starting with a
letter.

Secrets are files under `secrets/` (see [`secrets/README.md`](secrets/README.md)): `computer_token_secret`,
which every room's far side signs its token with, and `openai_api_key`, a placeholder for the lab's keyless
servers (`OPENAI_API_KEY=local` in `.env` does as well).

## The household, and where safety sits

The household file lists who lives in the house, each a `memory.Member`:

```json
[{"id": "gabe", "name": "Gabe", "role": "adult"}, {"id": "ada", "name": "Ada", "role": "child", "born": 2018}]
```

These are the owner's facts, and Willy cannot change them: no tool carries a role or a birth year, and memory
refuses to rename, unlink a declared voice of, or forget a declared person (`declared`). Willy sees the facts on
every turn that involves the person. Removing a member from the file undeclares them on the next start; Willy
may then forget them.

Kids are present, so it matters where the line is. The owner supplies facts (who lives here, adult or child, a
birth year) and capacities (what is attached here: the rooms' speakers and microphones, memory and people; there
is no web, purchasing or messaging). Code may refuse a capacity and may expose a fact. Code never decides a reply,
filters what Willy says, inserts a warning or treats a child's turn differently: what to say to anyone, and how,
is Willy's. Every link records whether the bot or the owner made it, and when, so the owner can audit and correct
it.

## Biometric privacy

Voice and face prints never leave the memory database: they never reach the model, a prompt, memory's episodes or
telemetry. The model sees only opaque refs (`voice:3f2a…`) and whether each is someone it knows. A print nobody
is known by is pruned 30 days after it was last heard (at most 1000 are kept); a print of someone named or declared is
kept, and one linked to a person nobody has named is still pruned.
Forgetting a person erases their prints with them.

## Security

The rooms' computers are served over plaintext `ws://` on `WILLY_HOST`. On anything but loopback, everything a
room's microphone hears crosses that network unencrypted: serve Willy on a network you trust. A room's far side
is admitted only with a token signed by `computer_token_secret`, issued for that room's name.

## Tests

```sh
uv sync --project examples/willy
examples/willy/.venv/bin/python -m pytest -m "not live" -q examples/willy/tests
```

The tests are hermetic. `test_build.py` covers the settings, the household file and the bot file run as a child
process (it becomes ready on closed endpoints and stops on SIGTERM; a bad setting exits `1` naming it).
`test_willy.py` writes its own small `bot.define` of Willy's shape, with a scripted model, the real ear, mouth,
memory and people, and fakes at the provider seams only: the speech server's wire behind
`speech_server.Transcription(connect=...)`, and the voiceprint and diarization servers behind
`httpx.MockTransport`. They pin what Willy perceives and what it can do (a stranger is `known=false` and linking
is offered; a link makes a declared child's voice hers; two voices in one turn are two sources; its own voice is
`you`; forgetting a person erases their print), never what it chooses.
