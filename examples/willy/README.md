# Willy

Willy is one bot for the whole house: the people who live there, and anyone who visits. It is not one person's
assistant. It learns who people are: a voice it has never heard is a fact it is shown, and whom that voice
belongs to, what to call them, and whether to ask at all are its own choices. A person may have several voices
(a cold, a phone, a whisper), and each voice is exactly one person's. Willy knows its own voice too.

Everything runs locally. The lab model server (CT 108; see `.env.example`) serves the language model, its
embeddings, streaming speech to text, voiceprints, diarization and tone of voice. Speech and voice design run on
the Breeze engine anygame's family shares on CT 107, where Willy has one voice of its own (see Mouth below).

## The bot file

`willy_example/__main__.py` is the bot file. It reads the owner's settings and two secrets, then defines Willy
with one `bot.define` that constructs every part inline, and runs it with `bot.run`:

- **Rooms.** One `computer.Computer` per room the owner lists (`WILLY_ROOMS`), named for the room and set on
  something in it, an `environment.Object` such as a shelf or a counter, at its own place in the house. All are
  served by one `websocket.Server`. A far side in each room, `willy_example.room`, carries the room's
  microphone and speaker (and, optionally, camera) to it. A room's computer is shared, so it stamps nobody on what it hears: who is speaking is
  what Willy's ear recognizes.
- **Ear.** `listening.Listening` on the lab's streaming speech to text, with voiceprints into a voice
  `identification.Registry` on the same file database as memory, diarization (several voices in one turn) and
  tone of voice.
- **Mouth.** One `speaking.Speaking`, in the one voice Willy is given when it is born (`bot.voice`: the first
  time Willy starts, its model conceives a voice from its traits and the lab designs it). Every room's computer
  hears it and plays it on that room's speaker, like an intercom. The lab's speech engine is the Breeze engine
  anygame's family shares on CT 107. Willy is allowed none of its voices (`bot.voice` with no `allowed=`), so
  it designs one of its own there and is never born as one of the family's or the sidekicks' voices. The engine
  keeps only a voice's id and its description, and speaks every line from that description, so nothing in it is
  marked as Willy's: Willy knows its voice by the id its memory keeps (the `self_voice` row in the data dir) and
  checks on every start that the engine still holds it.
- **Mind.** `bot.cognition` on the lab's language model, its intuition on a decision model (`decision.Processor`
  on the decision server `DECISION_BASE_URL` names, never a chat model), with associations, consolidation, autonomy, routines
  and the speech seed that takes heard speech into a conversation. Its memory recalls by the lab's
  EmbeddingGemma 2 vectors.
- **Name.** `identity.Identity`: Willy is born nameless and may take a name when someone gives it one.
- **People.** `people.People`, the capacity to link a voice to a person, name a person, unlink a voice and
  forget a person. Nobody is listed for Willy: it learns who everyone is itself.
- **Eye** (`WILLY_CAMERA=1`). `seeing.Seeing` over the rooms' cameras, with the OpenCV Zoo's YuNet and SFace
  into a face registry that people also links and forgets.

Nothing in it tells Willy what to do: it is given no instruction text, no seeded reply and no rule about
whom to answer.

## Running it

```sh
uv sync --project examples/willy
cp examples/willy/.env.example examples/willy/.env          # the lab's addresses; edit the rooms
python -c "import secrets; print(secrets.token_urlsafe(32))" > examples/willy/secrets/computer_token_secret
set -a; . examples/willy/.env; set +a
examples/willy/.venv/bin/python -m willy_example --secrets-dir examples/willy/secrets --ready-file /tmp/willy-ready
```

Willy keeps its memory and every voiceprint in one SQLite file, `<data dir>/memory.sqlite3` (`--data-dir` or
`MOSFET_DATA_DIR`). The bot file opens nothing: the file is opened when Willy starts, inside the data dir
`bot.run` has created private (`0700`), and one that cannot be opened fails Willy's start there. The process options, exit codes and stop bound are `bot.run`'s
([`docs/runtime.md`](../../docs/runtime.md)). A missing or malformed setting or secret ends the process with exit
`1`, naming it, before anything starts.

The data dir is also how Willy knows its voice. A new or wiped data dir forgets the id of the voice Willy
designed, so its next start designs one more voice on the shared engine and speaks in that one; the earlier
voice stays on the engine, unused, until someone deletes it there. Keep the data dir to keep Willy's voice.

## A room's far side

Each room needs a far side for its computer: a host in the room with a microphone and a speaker (and, with
`--camera`, a camera). `willy_example.room` is one:

```sh
examples/willy/.venv/bin/python -m willy_example.room --url ws://<willy-host>:8765 --peer kitchen \
  --secret-file examples/willy/secrets/computer_token_secret [--camera]
```

It connects as the room (`--peer` is the room's name in `WILLY_ROOMS`), signs its token with
`computer_token_secret`, and opens the computer's `microphone` and `speaker` channels (and `camera`). It decides
nothing: it captures 16 kHz mono from the host's default input and sends it, plays whatever the computer sends on
the host's default output (Willy's voice, at 24 kHz), and with `--camera` sends a JPEG still from the first
camera every second. Sound comes from `sounddevice` (PortAudio; on Linux install `libportaudio2`), the camera from
OpenCV.

Echo cancellation is the operating system's or the device's: capture through a device or a voice-processing
input that cancels what its speaker plays (a speakerphone, or the platform's echo-cancelled input). Without it
Willy hears itself through the room's microphone a moment later; it knows its own voice, so it never takes it for
someone else's.

SIGINT or SIGTERM leaves the room. Exit codes: `0` when the room is left or Willy's computer closes the session
normally, `1` when the connection fails, times out, is refused or ends abnormally (or the sound card cannot be
opened), `64` for bad flags, `78` for a missing or unreadable secret. It writes nothing to stdout. Its telemetry
(meter scope `willy_example.room`) counts connects by outcome, messages by channel, drops by channel and closes by
reason; the room's name, the url, the token and every sound and image never are attributes.

## Settings

| Variable | Default | What it is |
|---|---|---|
| `WILLY_LLM_URL`, `WILLY_MODEL` | required | Willy's language model (the lab's `:8100/v1`, `qwen3.8-27b`) |
| `WILLY_EMBED_URL`, `WILLY_EMBED_MODEL` | required | memory's embeddings (`:8014/v1`, `embeddinggemma-2`, 768 dimensions) |
| `WILLY_STT_URL` | required | streaming speech to text, a `ws://` root (`:8001`) |
| `WILLY_TRANSCRIBE_URL` | required | batch transcription, which Willy hears its own name through (`:8001/v1`) |
| `WILLY_TTS_URL` | required | speech, its voices and its voice designer: the Breeze engine anygame's family shares on CT 107 (`192.168.2.32:8002`) |
| `WILLY_VOICEPRINT_URL`, `WILLY_DIAR_URL`, `WILLY_EMOTION_URL` | required | voiceprints (`:8003`), diarization (`:8005`), tone (`:8004`) |
| `WILLY_HOST`, `WILLY_PORT` | `127.0.0.1`, `8765` | where the rooms' computers are served |
| `WILLY_ROOMS` | `living-room:shelf` | each room and what its computer sits on, `room:object,...`; rooms are placed 10 m apart in the order given |
| `WILLY_CAMERA`, `WILLY_FACE_MODELS` | `0`, unset | `1` to see; the directory holding `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` |
| `WILLY_TRAITS` | `warm,patient,curious,honest` | the owner's traits for Willy, comma separated |
| `DECISION_BASE_URL` | required | the decision server Willy's intuition decides on, which speaks the decisions wire format (anygame's decide, `http://127.0.0.1:8102` in `.env.example`) |
| `DECISION_MAX_CONTEXT_TOKENS` | unset | the decision model's context in tokens (`2048` for decide in `.env.example`); each decision request is fitted to it by cutting the turn's context, never its stimuli |

Every address is required: where the owner's model server is has no safe default. A room's name is the id of its
computer and the peer its far side connects as: 3 to 63 lower-case letters, digits or hyphens, starting with a
letter.

Secrets are files under `secrets/` (see [`secrets/README.md`](secrets/README.md)): `computer_token_secret`,
which every room's far side signs its token with, `openai_api_key`, a placeholder for the lab's keyless
servers (`OPENAI_API_KEY=local` in `.env` does as well), and, optionally, `decision_api_key`, the decision
server's key. The local decide takes no key, and without the secret Willy sends it none.

## Who lives here, and where safety sits

Nobody lists the household for Willy. Who lives in the house, who is visiting, what each of them is called and
whether someone is a child are things Willy learns from the people themselves and remembers, the same way it
learns everything else: a voice nobody has named is a fact it is shown, and what it makes of it is its own.

Kids are present, so it matters where the line is. The owner supplies capacities (what is attached here: the
rooms' speakers and microphones, memory and people; there is no web, purchasing or messaging) and Willy's traits.
Code may refuse a capacity and may expose a fact Willy has learned. Code never decides a reply, filters what Willy
says, inserts a warning or treats a child's turn differently: what to say to anyone, and how, is Willy's. Every
link records when Willy made it, so the owner can audit it.

## Biometric privacy

Voice and face prints never leave the memory database: they never reach the model, a prompt, memory's episodes or
telemetry. The model sees only opaque refs (`voice:3f2a…`) and whether each is someone it knows. Not even
`BOT_OTEL_CAPTURE=full`, the local debugging switch that logs every event's payload, writes a print or any
recording of anyone's voice or image: a print is logged as `<biometric:Print>`, and raw audio and image bytes as
their kind and length only (`<media:audio:32000 bytes>`). It does write text: transcripts of what anyone said,
prompts and model output reach `otel-logs.jsonl`, so leave it off unless you are debugging. Privacy wins over
debugging detail. A print nobody
is known by is pruned 30 days after it was last heard (at most 1000 are kept); a print of someone named is
kept, and one linked to a person nobody has named is still pruned.
Forgetting a person erases their prints with them. A voice heard over Willy's own (an overlap, while it is
speaking) is only matched to a voice it already knows: that audio is Willy's voice too, so no new voice is made
from it. Nor is one made from less than a second of a voice nobody knows; a known voice is named from any length.

## Security

The rooms' computers are served over plaintext `ws://` on `WILLY_HOST`. On anything but loopback, everything a
room's microphone hears crosses that network unencrypted: serve Willy on a network you trust. A room's far side
is admitted only with a token signed by `computer_token_secret`, issued for that room's name.

## Tests

```sh
uv sync --project examples/willy
examples/willy/.venv/bin/python -m pytest -m "not live" -q examples/willy/tests
```

The tests are hermetic. `test_build.py` covers the settings and the bot file run as a child
process (it becomes ready on closed endpoints and stops on SIGTERM; a bad setting exits `1` naming it).
`test_room.py` runs the room's far side against a real
websocket server and a real computer on a counter, with fake sound card and camera: what the microphone hears is
put into the house, what the computer hears comes out of the speaker, a still reaches the house, and a refused
token, nothing to connect to or a sound card that cannot open end it `1`. `test_willy.py` writes its own small
`bot.define` of Willy's shape, with a scripted model, the real ear, mouth,
memory and people, and fakes at the provider seams only: the speech server's wire behind
`speech_server.Transcription(connect=...)`, and the voiceprint and diarization servers behind
`httpx.MockTransport`. They pin what Willy perceives and what it can do (a stranger is `known=false` and linking
is offered; a voice Willy names is known by that name next turn; two voices in one turn are two sources; its own voice is
`you`; forgetting a person erases their print), never what it chooses.

## Live check

`tests/test_live.py` runs Willy against the lab itself, at the addresses in `.env.example`: CT 108, and the Breeze
speech engine on CT 107:

```sh
cd examples/willy && set -a && . ./.env.example && set +a && WILLY_LIVE=1 uv run pytest -m live tests/test_live.py -q
```

It has two gates, and both must hold: it is marked `live`, so `-m "not live"` (CI, and the command above under
Tests) deselects it, and it is skipped unless `WILLY_LIVE=1`. With `WILLY_LIVE=1` nothing is skipped: a missing
`WILLY_*` setting fails naming it, and a lab service that does not answer fails naming its unit and where it runs,
so a down server never passes. It assumes `mosfet-llm` and CT 108's speech and embedding units are up (the runbook,
`lab-ct108.md`, says how to stop and start them), and that anygame's `sidekick-tts-engine-breeze` answers on
CT 107. That engine is the family's: the check never restarts it.

The test writes its own `bot.define` of Willy's shape on the real parts and the lab's models, with one room served
on loopback and `willy_example.room` as its far side; the far side's sound card is a scripted microphone and
speaker, and the strangers' speech is made by the lab's speech engine at test time, in two catalog voices that are
not Willy's. Step by step, each bounded: the lab answers (L1, 10 s); Willy is born with a voice that memory keeps
(L2, 300 s); a stranger's voice reaches it unknown, with linking offered (L3); a link the test makes, standing in
for a selection, makes that voice known by the next utterance (L4); two voices in one utterance are two sources
(L5); its own voice, echoed through the room's microphone, is `you` (L6, 90 s); and forgetting the person erases
the print (L7); 60 s each unless given. It asserts perception and capacity only. Whatever Willy chose on each
turn (whether it asked who the stranger was, what it said, the tools it called) is recorded, never asserted:
printed as it happens, kept in `live-report.jsonl` beside the OTEL JSONL in the test's data directory, each
readable by its owner alone (0600). It takes about
fifteen minutes, most of it Willy's birth on the 27B model, which thinks by default.
