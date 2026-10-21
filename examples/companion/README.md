# Companion

The companion is one bot for the whole house: the people who live there, and anyone who visits. It is not one person's
assistant, and it has no given name: it learns its name from the people it lives with, as it learns who they
are. A voice it has never heard is a fact it is shown, and whom that voice
belongs to, what to call them, and whether to ask at all are its own choices. A person may have several voices
(a cold, a phone, a whisper), and each voice is exactly one person's. The companion knows its own voice too.

Everything runs locally. The companion talks to model servers on your own network, whose addresses go in `.env`: a
language model for intuition and one for reasoning (the same Gemma for now), its embeddings, streaming speech to text, voiceprints, diarization, tone of voice, a decision
model, and a speech engine that can design voices, where the companion has one voice of its own (see Mouth below).

## The bot file

`companion_example/__main__.py` is the bot file. It reads the owner's settings and two secrets, then defines the companion
with one `bot.define` that constructs every part inline, and runs it with `bot.run`:

- **Rooms.** One `computer.Computer` per room the owner lists (`COMPANION_ROOMS`), named for the room and set on
  something in it, an `environment.Object` such as a shelf or a counter, at its own place in the house. All are
  served by one `websocket.Server`. A far side in each room, `companion_example.room`, carries the room's
  microphone and speaker (and, optionally, camera) to it. A room's computer is shared, so it stamps nobody on what it hears: who is speaking is
  what the companion's ear recognizes.
- **Ear.** `listening.Listening` on the streaming speech to text server, with voiceprints into a voice
  `identification.Registry` on the same file database as memory, diarization (several voices in one turn) and
  tone of voice.
- **Mouth.** One `speaking.Speaking`, in the one voice the companion is given when it is born (`bot.voice`: the first
  time the companion starts, its model conceives a voice from its traits and the speech engine designs it). Every room's
  computer hears it and plays it on that room's speaker, like an intercom. The speech engine may be shared with
  other bots. The companion is allowed none of its voices (`bot.voice` with no `allowed=`), so it designs one of its own
  there and is never born as another bot's voice. The engine
  keeps only a voice's id and its description, and speaks every line from that description, so nothing in it is
  marked as the companion's: the companion knows its voice by the id its memory keeps (the `self_voice` row in the data dir) and
  checks on every start that the engine still holds it.
- **Mind.** `bot.cognition` in three tiers a turn takes up in order, each bounded by its own step timeout
  (`bot.timeouts(reflex=5 s, intuition=20 s, reasoning=180 s)`): its reflex on a decision model (`decision.Processor`
  on the decision server `DECISION_BASE_URL` names, never a chat model), its intuition on the intuition model with
  thinking off (fixed, not a setting), and its reasoning on the reasoning model, which thinks first within
  `COMPANION_THINKING_TOKENS` and also reflects and consolidates; with associations, consolidation, autonomy, routines
  and the speech seed that takes heard speech into a conversation. Its memory recalls by the
  embedding server's vectors.
- **Name.** `identity.Identity`: the companion is born nameless and may take a name when someone gives it one.
- **People.** Knowing who someone is, is remembering it: its memory offers the companion `memory.know` (a voice is the
  same person as someone, is called something, or is not the same person as someone) and `memory.forget` (a
  person, and their prints with them). The cognition is given the voice (and face) registries, so a person's prints
  are kept once the companion has named them. A new voice (or face) that is like someone the companion has named reaches it as
  `known=false` with whom it `resembles` and how closely; what that means is the companion's to work out. Nobody is
  listed for the companion: it learns who everyone is itself.
- **Eye** (`COMPANION_CAMERA=1`). `seeing.Seeing` over the rooms' cameras, with the OpenCV Zoo's YuNet and SFace
  into a face registry whose prints follow what the companion remembers, the same way.

Nothing in it tells the companion what to do: it is given no instruction text, no seeded reply and no rule about
whom to answer.

## Running it

```sh
uv sync --project examples/companion
cp examples/companion/.env.example examples/companion/.env          # your servers' addresses; edit the rooms
python -c "import secrets; print(secrets.token_urlsafe(32))" > examples/companion/secrets/computer_token_secret
set -a; . examples/companion/.env; set +a
examples/companion/.venv/bin/python -m companion_example --secrets-dir examples/companion/secrets --ready-file /tmp/companion-ready
```

The companion keeps its memory and every voiceprint in one SQLite file, `<data dir>/memory.sqlite3` (`--data-dir` or
`MOSFET_DATA_DIR`). The bot file opens nothing: the file is opened when the companion starts, inside the data dir
`bot.run` has created private (`0700`), and one that cannot be opened fails the companion's start there. The process options, exit codes and stop bound are `bot.run`'s
([`docs/runtime.md`](../../docs/runtime.md)). A missing or malformed setting or secret ends the process with exit
`1`, naming it, before anything starts.

The data dir is also how the companion knows its voice. A new or wiped data dir forgets the id of the voice the companion
designed, so its next start designs one more voice on the shared engine and speaks in that one; the earlier
voice stays on the engine, unused, until someone deletes it there. Keep the data dir to keep the companion's voice.

## A room's far side

Each room needs a far side for its computer: a host in the room with a microphone and a speaker (and, with
`--camera`, a camera). `companion_example.room` is one:

```sh
examples/companion/.venv/bin/python -m companion_example.room --url ws://<companion-host>:8765 --peer kitchen \
  --secret-file examples/companion/secrets/computer_token_secret [--camera]
```

It connects as the room (`--peer` is the room's name in `COMPANION_ROOMS`), signs its token with
`computer_token_secret`, and opens the computer's `microphone` and `speaker` channels (and `camera`). It decides
nothing: it captures 16 kHz mono from the host's default input and sends it, plays whatever the computer sends on
the host's default output (the companion's voice, at 24 kHz), and with `--camera` sends a JPEG still from the first
camera every second. Sound comes from `sounddevice` (PortAudio; on Linux install `libportaudio2`), the camera from
OpenCV.

Echo cancellation is the operating system's or the device's: capture through a device or a voice-processing
input that cancels what its speaker plays (a speakerphone, or the platform's echo-cancelled input). Without it
the companion hears itself through the room's microphone a moment later; it knows its own voice, so it never takes it for
someone else's.

SIGINT or SIGTERM leaves the room. Exit codes: `0` when the room is left or the companion's computer closes the session
normally, `1` when the connection fails, times out, is refused or ends abnormally (or the sound card cannot be
opened), `64` for bad flags, `78` for a missing or unreadable secret. It writes nothing to stdout. Its telemetry
(meter scope `companion_example.room`) counts connects by outcome, messages by channel, drops by channel and closes by
reason; the room's name, the url, the token and every sound and image never are attributes.

Given a keyboard, a room's far side also opens the computer's `keyboard` channel and types each line on it key by
key, then Enter. Nothing is focused on a room's computer, so the companion is told `computer.text.entered` with no field:
a typed line wakes a turn of its own, and answering aloud (`communication.respond`) is offered on it.

## Talk to the companion in a browser

`companion_example.ui` is a room's far side in a browser tab: your microphone, your speakers, a text box and
(optionally) your camera, all in one page. Run the bot file (see Running it), then, in a second terminal:

```sh
examples/companion/.venv/bin/python -m companion_example.ui --url ws://127.0.0.1:8765 --peer living-room \
  --secret-file examples/companion/secrets/computer_token_secret [--port 8780] [--camera]
```

and open `http://127.0.0.1:8780/`. The page shows whether it is linked to the companion's computer. Press **Start** to
turn on the microphone and sound (the browser asks for the microphone, and the camera with `--camera`); **Mute**
stops sending what you say, and **Push to talk** sends it only while you hold the button or the space bar. Type a
line and press Enter to type it on the room's computer. The companion's voice plays in the page, in order and without
gaps, and the log shows each line you typed and each time the companion spoke, with when.

`--peer` is the room you are in, one of `COMPANION_ROOMS`; a room's computer takes one far side at a time, so use a
room no other far side is connected as. The page is served on `127.0.0.1` only, and only a page served by the
bridge may open its socket. Opening the page in a second tab takes over from the first. The browser cancels the
echo of what the page plays from what its microphone hears; with speakers rather than headphones, the companion still
knows its own voice if some gets through. Like the room's far side, it decides nothing: it captures, carries and
plays. Exit codes are the room's (`0` left or closed normally, `1` failed or refused, or the port is taken, `64`
bad flags, `78` missing secret); the page's address goes to stderr and nothing to stdout. Its telemetry (meter
scope `companion_example.ui`) counts pages by outcome, messages by kind and drops by kind and why, never the room's
name, what was typed or any sound or image.

## Evaluate the companion by voice

`companion_example.evaluate` is a voice agent that evaluates a running companion: it joins one of the companion's rooms as that
room's far side (a `room.Room` whose microphone it speaks into and whose speaker it listens to), holds short
spoken conversations with the companion, and grades them. Give the companion a room the evaluator can have to itself, then run it
in a second terminal:

```sh
cd examples/companion
set -a; . ./.env.example; set +a
# Terminal 1: the companion, on loopback, with a room for the evaluator.
COMPANION_HOST=127.0.0.1 COMPANION_ROOMS=living-room:shelf,eval-room:table \
  .venv/bin/python -m companion_example --secrets-dir secrets --ready-file /tmp/companion-ready
# Terminal 2: the evaluator, in that room.
.venv/bin/python -m companion_example.evaluate --url ws://127.0.0.1:8765 --peer eval-room \
  --secret-file secrets/computer_token_secret --out /tmp/companion-eval [--scenario stranger-arrives ...]
```

Everything it uses is local, on the companion's own lab stack (`.env.example`): the people speak through the speech
engine (`COMPANION_TTS_URL`) in its catalog voices, never a designed voice (the companion's own is one); the companion's replies are
heard by batch speech to text (`COMPANION_TRANSCRIBE_URL`); and an evaluator model on the companion's reasoning model
(`COMPANION_REASONING_URL`/`COMPANION_REASONING_MODEL`) plays two parts. As the **persona** it plays each scenario's people: given the scenario's goal and what has been
said so far, it writes the next thing one of them says, or decides the conversation is done. Only the people's
lines are generated; nothing the companion should say is written anywhere. As the **judge** it grades each conversation
on a JSON schema: relevance, helpfulness, warmth (the companion's traits, `COMPANION_TRAITS`), honesty (no invented facts),
memory where a person comes back, what suits a child where a child speaks, and overall, 1 to 5, each with a short
rationale. Each schema is built for the scenario, so the server's structured output holds the answer to it: the
persona's names who may speak (the people's ids), bounds each line, and requires the words cut in on an interrupting
turn and none otherwise; the judge's offers the scenario's criteria, one grade each. An answer that still fails its
check is shown back once; failing twice, the persona ends the scenario and the judge's conversation is recorded
ungraded. The judge is the companion's own reasoning model by default, so it may favour answers like its own; the report says so.
`--judge-url` and `--judge-model` name a bigger one.

The person takes turns as people in a room do: they speak only once the companion's voice has died away (what the room's
speaker played, modelled as playing for as long as it lasts) and the echo window after it has passed
(`recognition.ECHO_SECONDS`), then wait for the companion's reply until it has been quiet for `--pause` seconds (1.5); no
sound within `--reply-timeout` (30) is no reply. Each exchange measures the response latency (from the end of the
person's line to the companion's first sound after it), how long the companion's reply played, no reply, talk-over (the companion's voice
playing while the person speaks) and, where the person cuts in, how long the companion kept talking. The microphone is
perfectly echo-cancelled: the evaluator never hears the companion through it. The companion's mouth plays in every room, so the
evaluator also hears what the companion says to anyone else in the house: run it when the other rooms are quiet.

The scenarios are data (`companion_example/evaluate/scenarios.json`), each a goal for the persona, the people (each
with a catalog voice), at most 3 to 6 of their turns, the breaks and cut-ins, and the criteria graded:

| Id | What happens |
|---|---|
| `stranger-arrives` | someone the companion has never met arrives and chats |
| `introduces-and-returns` | Maya introduces herself, chats, leaves for 20 s, then checks the companion remembers her (graded on memory: recognising her voice is the companion's; the judge grades whether the companion acted on knowing her) |
| `two-people` | Alex and Rose, in two voices, ask and answer each other with the companion in the room |
| `curious-child` | Lily, about seven, in a lighter voice, asks curious questions (graded on what suits a child) |
| `interrupts` | Sam asks for a long answer and cuts in while the companion is still giving it |

Everything goes in `--out` (created `0700`), each file readable by its owner alone (`0600`), since it is
household-style data: `transcript.jsonl` (every utterance: who, what, when, and a note on no reply, talk-over or a
cut-in), `audio/<scenario>-<turn>.wav` (each exchange as the room heard it, 16 kHz mono), `scores.json` (per
scenario and overall) and `report.html` (one page with nothing fetched: scores, latency median and p90, talk-overs,
no replies and the transcripts, with each exchange's audio). The last line on stdout is a one-line summary;
diagnostics go to stderr. Exit codes are the room's: `0` the evaluation finished, whatever it found, `1` it could
not (the room's session ended or was refused, a service failed, or `--out` cannot be made), `64` bad flags (an
unknown `--scenario` included), `78` a missing setting (`COMPANION_REASONING_URL`, `COMPANION_REASONING_MODEL`, `COMPANION_TTS_URL`,
`COMPANION_TRANSCRIBE_URL`) or secret. Its telemetry (meter scope `companion_example.evaluate`) counts exchanges by
outcome and scenarios by whether they were graded, and records each latency; nothing said, no scenario and no
person is an attribute.

## Settings

| Variable | Default | What it is |
|---|---|---|
| `COMPANION_INTUITION_URL`, `COMPANION_INTUITION_MODEL` | required | the model the companion's intuition answers on, with thinking off, an OpenAI-compatible `/v1` root, and its name |
| `COMPANION_REASONING_URL`, `COMPANION_REASONING_MODEL` | required | the model the companion's reasoning thinks on, and reflects and consolidates on, an OpenAI-compatible `/v1` root, and its name |
| `COMPANION_THINKING_TOKENS` | `1024` | how many tokens reasoning may think for before it chooses what to do (the chat template's thinking channel, capped by vLLM's `thinking_token_budget`); `0` turns its thinking off; intuition never thinks first |
| `COMPANION_MAX_CONTEXT_TOKENS` | unset | the intuition and reasoning models' context in tokens; set it to the chat server's own `--max-model-len` (the lab serves 32768). Each intuition and reasoning prompt is fitted to it, less the reply and thinking budget, by trimming only what the companion recalls (the recent turns keep the newest, the memories the strongest), never the turn itself; unset, prompts are sent whole |
| `COMPANION_EMBED_URL`, `COMPANION_EMBED_MODEL` | required | memory's embeddings, an OpenAI-compatible `/v1` root, and the model's name |
| `COMPANION_STT_URL` | required | streaming speech to text, a `ws://` root |
| `COMPANION_TRANSCRIBE_URL` | required | batch transcription, which the companion hears its own name through, a `/v1` root |
| `COMPANION_TTS_URL` | required | speech, its voices and its voice designer |
| `COMPANION_VOICEPRINT_URL`, `COMPANION_DIAR_URL`, `COMPANION_EMOTION_URL` | required | voiceprints, diarization (several voices in one utterance), tone of voice |
| `COMPANION_HOST`, `COMPANION_PORT` | `127.0.0.1`, `8765` | where the rooms' computers are served |
| `COMPANION_ROOMS` | `living-room:shelf` | each room and what its computer sits on, `room:object,...`; rooms are placed 10 m apart in the order given |
| `COMPANION_CAMERA`, `COMPANION_FACE_MODELS` | `0`, unset | `1` to see; the directory holding `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` |
| `COMPANION_TRAITS` | `warm,patient,curious,honest` | the owner's traits for the companion, comma separated |
| `DECISION_BASE_URL` | required | the decision server the companion's reflex decides on, which speaks the decisions wire format |
| `DECISION_MAX_CONTEXT_TOKENS` | unset | the decision model's context in tokens; each decision request is fitted to it by cutting the turn's context, never its stimuli |

Every address is required: where the owner's model server is has no safe default. A room's name is the id of its
computer and the peer its far side connects as: 3 to 63 lower-case letters, digits or hyphens, starting with a
letter.

Secrets are files under `secrets/` (see [`secrets/README.md`](secrets/README.md)): `computer_token_secret`,
which every room's far side signs its token with, `openai_api_key`, a placeholder for keyless local
servers (`OPENAI_API_KEY=local` in `.env` does as well), and, optionally, `decision_api_key`, the decision
server's key. Without the secret the companion sends the decision server no key.

## Who lives here, and where safety sits

Nobody lists the household for the companion. Who lives in the house, who is visiting, what each of them is called and
whether someone is a child are things the companion learns from the people themselves and remembers, the same way it
learns everything else: a voice nobody has named is a fact it is shown, and what it makes of it is its own.

Kids are present, so it matters where the line is. The owner supplies capacities (what is attached here: the
rooms' speakers and microphones, and memory, which knows people; there is no web, purchasing or messaging) and
The companion's traits.
Code may refuse a capacity and may expose a fact the companion has learned. Code never decides a reply, filters what the companion
says, inserts a warning or treats a child's turn differently: what to say to anyone, and how, is the companion's. Every
fact the companion remembers about a person records when it learned it, so the owner can audit it.

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
Forgetting a person erases their prints with them. A voice heard over the companion's own (an overlap, while it is
speaking) is only matched to a voice it already knows: that audio is the companion's voice too, so no new voice is made
from it. Nor is one made from less than a second of a voice nobody knows; a known voice is named from any length.

## Security

The rooms' computers are served over plaintext `ws://` on `COMPANION_HOST`. On anything but loopback, everything a
room's microphone hears crosses that network unencrypted: serve the companion on a network you trust. A room's far side
is admitted only with a token signed by `computer_token_secret`, issued for that room's name.

## Tests

```sh
uv sync --project examples/companion
examples/companion/.venv/bin/python -m pytest -m "not live" -q examples/companion/tests
```

The tests are hermetic. `test_build.py` covers the settings and the bot file run as a child
process (it becomes ready on closed endpoints and stops on SIGTERM; a bad setting exits `1` naming it).
`test_room.py` runs the room's far side against a real
websocket server and a real computer on a counter, with fake sound card and camera: what the microphone hears is
put into the house, what the computer hears comes out of the speaker, a still reaches the house, a typed line
reaches the computer's holder as text entered with no field, and a refused token, nothing to connect to or a
sound card that cannot open end it `1`. `test_ui.py` runs the browser bridge the same way, with a WebSocket client
standing in for the page: the page's sound reaches the house, what the computer plays reaches the page with its
rate, a typed line reaches the holder, what the bridge cannot take is dropped and the page told, the page is
served on loopback only and refuses another origin, and the exit codes hold. `test_companion.py` writes its own small
`bot.define` of the companion's shape, with a scripted model, the real ear, mouth,
and memory with its registries, and fakes at the provider seams only: the speech server's wire behind
`speech_server.Transcription(connect=...)`, and the voiceprint and diarization servers behind
`httpx.MockTransport`. They pin what the companion perceives and what it can do (a stranger is `known=false` and knowing who
they are is offered; a voice the companion names is known by that name next turn; two voices in one turn are two sources; its own voice is
`you`; forgetting a person erases their print; a line typed in a room wakes a turn with answering offered), never
what it chooses.

`tests/evaluate/` covers the evaluator, one file per module: the scenarios load and each is refused when its data
does not hold together, and a persona voice must be a catalog voice and never a designed one; the speaker's
playback model and the measures (latency, reply length, no reply, talk-over, a cut-in) on a manual clock, and the
turn-taking machine on the real clock with its windows scaled down (it speaks only after the companion's voice and the echo
window, waits out a break, hears a reply to its pause, marks talk-over, cuts in only while the companion is playing); the
audio resampled, mixed and written as WAV; the persona's and the judge's prompts and JSON against a scripted chat
model behind `httpx.MockTransport` (the schemas each sends for its scenario, the persona ending a scenario, a child
played as one, two people, a cut-in, one repair, and a judge answer that fails the schema twice recorded ungraded);
the report rendered from fixtures; a
whole evaluation against a real `websocket.Server` and `computer.Computer` with a scripted companion that answers each
line in the house, with the speech engine, speech to text and model scripted (the transcript, the WAVs, the
scores and the page, each `0600`); and the command line's exit codes. `tests/evaluate/test_live.py` runs one short
scenario with a running companion and the lab, under the same two gates as the live check below: `COMPANION_LIVE=1`, a
the companion whose `COMPANION_ROOMS` lists `COMPANION_EVALUATE_PEER` (default `eval-room`) at `COMPANION_EVALUATE_URL` (default
`ws://127.0.0.1:8765`), and the `computer_token_secret` it was started with.

## Live check

`tests/test_live.py` runs the companion against real servers, at the addresses in your `.env`:

```sh
cd examples/companion && set -a && . ./.env && set +a && COMPANION_LIVE=1 uv run pytest -m live tests/test_live.py -q
```

It has two gates, and both must hold: it is marked `live`, so `-m "not live"` (CI, and the command above under
Tests) deselects it, and it is skipped unless `COMPANION_LIVE=1`. With `COMPANION_LIVE=1` nothing is skipped: a missing
`COMPANION_*` setting fails naming it, and a server that does not answer fails naming it, so a down server never
passes. The check never restarts a server: a shared speech engine stays as it is.

The test writes its own `bot.define` of the companion's shape on the real parts and models, with one room served on
loopback and `companion_example.room` as its far side; the far side's sound card is a scripted microphone and
speaker, and the strangers' speech is made by the speech engine at test time, in two catalog voices that are not
the companion's. Step by step, each bounded: the servers answer (L1, 10 s); the companion is born with a voice that memory keeps
(L2, 300 s); a stranger's voice reaches it unknown, with knowing who they are offered (L3); a fact the test has
the companion remember, standing in for a selection, makes that voice known by the next utterance (L4); two voices in one
utterance are two sources (L5); its own voice, echoed through the room's microphone, is `you` (L6, 90 s); and
forgetting the person erases the print (L7); 60 s each unless given. It asserts perception and capacity only.
Whatever the companion chose on each turn (whether it asked who the stranger was, what it said, the tools it called) is
recorded, never asserted: printed as it happens, kept in `live-report.jsonl` beside the OTEL JSONL in the test's
data directory, each readable by its owner alone (0600). Most of its time is the companion's birth, while its model
conceives a voice.
