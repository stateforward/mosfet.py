---
name: rca
description: >
  Loads and re-applies the phone dual-bot root-cause analysis (mid-call STT
  history plus current pre-call/cognition blockers). Use when diagnosing why
  room speech STTs but call audio does not, when re-checking Communication
  admit / Conversation / Gemini STT after a demo run, when intuition/provider
  failures block dial or answer, or when an agent needs the durable RCA for
  two-bots LiveKit phone work.
  Invoke with /mdscript-exec docs/rca.mdscript.md or #load-findings / #recheck-run.
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Setup

* set `{{repo_root}}` to the stateforward.bot workspace root when empty
* set `{{rca_reference}}` to `docs/rca-phone-call-stt.md` under `{{repo_root}}`
* set `{{record_dir}}` from user input when provided; otherwise leave empty
* if the user asked only to recall findings
  * [Load Findings](#load-findings)
* if the user provided a recording directory or asked to recheck a demo run
  * [Resolve Record Dir](#resolve-record-dir)
* [Load Findings](#load-findings)

## Load Findings

* read [RCA reference](rca-phone-call-stt.md)
* set `{{primary_rc}}` to `RC-5 post-contribution cognition / provider (Mercury 400 or bad focus/speaking/answer selection) — blocks dial or answer before mid-call STT can be retested`
* set `{{secondary_rcs}}` to `RC-1 Conversation admit while active (MITIGATED — reopen only if unavailable conversation.input returns); RC-2 TextStimulus into GeminiVoiceDecoder (latent); RC-3 call media 48k chunked vs room 16k (latent until answered media); RC-4 STT only behind Conversation admit (structural)`
* if `{{record_dir}}` is not empty
  * [Recheck Run](#recheck-run)
* report the updated one-sentence RCA and the ranked RC list from the reference
* stop

## Resolve Record Dir

* if `{{record_dir}}` is empty
  * ask the user for `{{record_dir}}` as the absolute or repo-relative path to a two-bots recording directory (contains `*.log` and optionally `two_bot_summary.json`)
* if `{{record_dir}}` is relative
  * set `{{record_dir}}` to `{{repo_root}}/{{record_dir}}`
* if `{{record_dir}}` does not exist or has no `*.log`
  * report that the recording directory is missing or incomplete
  * stop
* [Recheck Run](#recheck-run)

## Recheck Run

* list log files under `{{record_dir}}` and set `{{caller_log}}` and `{{callee_log}}` to the two bot logs when present
* if either log is missing
  * report missing bot logs under `{{record_dir}}`
  * [Load Findings](#load-findings)
* count in each bot log: `HearingSpeech`, `communication.input`, `conversation.input`, `unavailable event: bot.ability.conversation.input`, `unavailable event`, `normalize.completed`, `normalize.failed`, `TextStimulus`, `environment.sound`, `remote_audio`, `phone.dial`, `/Phone/ringing`, `chat completion request failed`, `400 Bad Request`, `focus_device outside`, `speaking.input`
* set `{{rc1_evidence}}` to the unavailable conversation.input counts per bot
* set `{{rc2_evidence}}` to TextStimulus / GeminiVoiceDecoder failure counts per bot
* set `{{stt_evidence}}` to normalize.completed vs failed and any contribution transcript strings found
* set `{{rc5_evidence}}` to intuition/provider failures (400, chat completion failed) and bad selection reasons (focus_device outside, unavailable speaking.input, missing answer)
* if `two_bot_summary.json` exists under `{{record_dir}}`
  * read stages and speech peaks from that summary into `{{summary_stages}}`
* [Report Recheck](#report-recheck)

## Report Recheck

* compare `{{rc1_evidence}}`, `{{rc2_evidence}}`, `{{stt_evidence}}`, and `{{rc5_evidence}}` against the ranked causes in [RCA reference](rca-phone-call-stt.md)
* report whether RC-1 still dominates (high unavailable `conversation.input` with HearingSpeech > 0) — if near zero, state RC-1 remains mitigated
* report whether RC-5 dominates (provider 400 / intuition failed before dial, or dial/ring without answer due to focus/speaking selection)
* report whether any successful mid-call contribution transcript appeared
* report call setup health from `{{summary_stages}}` when set (dial, rang, answered, peaks)
* if RC-5 dominates
  * recommend fix order from the reference starting at provider request shape or answer/focus tool availability
* if unavailable conversation.input is near zero, call is answered with media, but STT still fails
  * recommend prioritizing RC-2 and RC-3 packaging checks
* if RC-1 still dominates
  * recommend reopening Conversation admit-while-active work
* stop

## Entry Shortcuts

* for findings only: `/mdscript-exec docs/rca.mdscript.md#load-findings`
* for a new recording: `/mdscript-exec docs/rca.mdscript.md#resolve-record-dir`
* for recheck with path already known: set `{{record_dir}}` then `/mdscript-exec docs/rca.mdscript.md#recheck-run`
