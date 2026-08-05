# RCA: room speech STTs; mid-call phone audio does not

**Status:** rechecked 2026-08-04 against Communication cutover + Gemini-intuition probe.  
**Original PASS artifact:** `examples/phone_bot/recordings/two_20260804T055930Z` (dial/answer/media; mid-call STT failed).  
**Latest probes:**  
- Mercury intuition: `two_20260804T192248Z` — admit OK, STT OK once, **no dial** (provider 400).  
- Gemini intuition: `two_20260804T195055Z` — admit OK, dial + ring OK, **no answer** (bad tool selection).  
**Scope:** dual-bot LiveKit phone demo — Alice `5550141`, Bob `5550142`.  
**Use with:** [rca.mdscript.md](rca.mdscript.md) for re-entry and re-measurement.

## Symptom (original, still the mid-call STT claim)

| Path | Acoustic energy | VAD / HearingSpeech | Conversation STT product |
|------|-----------------|---------------------|---------------------------|
| Harness → Alice room | Yes (16 kHz, ~3.6 s) | Yes | **Yes** when cognition reaches STT — historically transcript `"Call Bob at 555-0142"` on contribution `content` |
| LiveKit call (both sides) | Yes when call is up (mostly 48 kHz chunks) | Yes when answered | **Not re-proven** since Communication cutover — recent runs never reach answered media |

**Important:** the original mid-call STT failure and the current demo FAIL are not the same dominant cause. Recheck which layer dies first before applying the old fix order.

## Intended pipeline (current topology)

```text
environment.sound
  → Listening (VAD only; speech_decoder=None — no Listening STT)
  → cognition (SpeechEvent / speech.output)
  → SpeechHeard (Communication behavior seed)
  → communication.input → Communication routes to active Conversation
  → TurnDetector normalize → GeminiVoiceDecoder (AudioStimulus only)
  → contribution (text product on Response.content) → cognition
```

STT lives only on the Conversation path. If admit or normalize fails, there is no fallback STT.

## Root causes — recheck (2026-08-04)

### RC-1 (was P0) — Conversation single-flight + tool menu — **MITIGATED**

Code keeps `conversation.InputEvent` enabled while **active** (queue/redeliver); Communication owns admit and routes to `_active_conversation`.

| Metric | PASS baseline Alice / Bob | Mercury run Alice | Gemini run Alice / Bob |
|--------|--------------------------:|------------------:|-----------------------:|
| unavailable `conversation.input` | 70 / **294** | **0** | **0** / **0** |
| HearingSpeech | 74 / 140 | 2 | 2 / 0 |
| normalize.completed | 6 / 0 | 6 | 6 / 0 |
| communication.input path | n/a (pre-cutover) | yes | yes / no (no speech product) |

**Recheck rule:** RC-1 still dominates only if `unavailable event: bot.ability.conversation.input` is high with `HearingSpeech > 0`. On current builds that count is **zero** — do not treat RC-1 as the live primary.

### RC-2 (P1, latent) — TextStimulus into voice STT

```text
GeminiVoiceDecoder requires AudioStimulus, got TextStimulus
```

Seen on the original PASS run (4× each bot). **Not observed** on Mercury/Gemini probes (no mid-call STT volume). Remains valid packaging risk if `text/plain` admits reappear.

### RC-3 (P1, latent) — Call media ≠ harness audio

| | Harness (worked) | Call media |
|--|------------------|------------|
| Path | Room mouth → environment | LiveKit remote → ServiceAudio → earpiece → environment |
| Rate | **16 kHz**, longer continuous | **48 kHz**, ~1.2 s chunks (when answered) |

**Cannot re-validate mid-call** until a run answers and carries audible media again. Still a packaging hypothesis, not the current blocker.

### RC-4 (P1, structural) — Single STT chokepoint

Listening has `speech_decoder=None` by design. No second STT path if Conversation admit/normalize fails. Still true; less urgent while admit is healthy.

### RC-5 (P0 **now**, was out of scope) — Post-contribution cognition / provider

Not in the original mid-call STT ranking; **currently kills the demo before mid-call STT can be tested.**

| Run | Intuition | Call setup | Failure |
|-----|-----------|------------|---------|
| `two_20260804T192248Z` | Mercury → **HTTP 400** (`api.inceptionlabs.ai`) | dialed=0 | Provider request rejected (body not logged; client collapses to generic message). Body size ~228 B. Suspected OpenAI-compat fields (`tool_choice` / `reasoning_effort` / schema). |
| `two_20260804T195055Z` | Gemini flash | dialed + rang | Bob: `focus_device outside available device candidates` (no answer). Alice later: unavailable `speaking.input`. Peaks silent. |

Gemini probe proves the tool **menu can produce dial**; Mercury 400 is provider-side. Answer/focus/speaking availability is a separate selection/topology issue.

## Not root causes (unchanged + updates)

- LiveKit dial / identity-as-number (works when intuition selects dial)
- Speaking target-on-`bot` (fixed earlier)
- Complete absence of remote audio when call is up (original PASS)
- RC-1 unavailable `conversation.input` on current Communication topology
- “Bot chose not to listen” as the explanation for pre-call STT when Mercury 400s

## One-sentence RCA (updated)

**Original mid-call claim (still latent):** mid-call understanding can fail from STT packaging (RC-2/3) and a single Conversation STT chokepoint (RC-4) after RC-1 was fixed.  
**Current dual-bot FAIL:** pre-call admit/STT and Communication routing are healthy; the run dies on **intuition provider or post-admit tool selection** (RC-5)—Mercury 400 or bad focus/speaking picks—so mid-call STT is not yet re-exercised.

## Fix order (updated)

1. **RC-5a:** Make Mercury request shape valid *or* keep a known-good intuition provider (Gemini probe) for demos; surface provider error bodies on `RequestError`.
2. **RC-5b:** When ringing, ensure answer/focus tools match live topology (stop `focus_device outside candidates`; offer only enabled events).
3. **RC-2 / RC-3:** Only after answered media returns — TextStimulus guard + 48 kHz call-path packaging/assembly.
4. **RC-4:** Optional second STT path only if Conversation chokepoint reappears under load.
5. **RC-1:** Hold the line — re-open only if unavailable `conversation.input` returns.

## Code anchors (paths after rehomes)

| Area | Path |
|------|------|
| Conversation inactive/active + active InputEvent | `src/bot/abilities/communication/conversation/conversation.py` (`define_model`) |
| Communication ownership / route to active | `src/bot/abilities/communication/communication.py` |
| Admit seed | `src/bot/abilities/communication/behaviors.py` |
| Turn detector | `src/bot/abilities/communication/conversation/turn_detector/` |
| STT modality | `examples/phone_bot/.../GeminiVoiceDecoder`, turn_detector normalize |
| `_content_for_detector` | conversation package (audio/* → AudioStimulus) |
| Listening no STT | `phone_bot_example._listening` (`speech_decoder=None`) |
| Phone earpiece elevation | `phone.PhoneFirmware._receive_service_audio` → speaker → `environment.sound` |
| Tool menu | `processing.enabled_call_events` / `cognition.input.build_processing_input` |
| Intuition client (TEMP Gemini probe) | `examples/phone_bot/.../_phone_cognition` |
| OpenAI-compat error wrap | `src/providers/openai_compat/.../client.py` (`RequestError` message) |

## Demo artifacts

| Run | Role |
|-----|------|
| `examples/phone_bot/recordings/two_20260804T055930Z/` | Original PASS call + mid-call STT RCA evidence (high RC-1 counts) |
| `.../two_20260804T192248Z/` | RC-5 Mercury 400; RC-1 clear; pre-call STT path ok |
| `.../two_20260804T195055Z/` | RC-5 Gemini dial/ring; answer/focus fail; no mid-call STT |
