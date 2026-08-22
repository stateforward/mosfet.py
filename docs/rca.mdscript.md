---
name: rca
description: >
  Loads and re-applies the phone dual-bot root-cause analysis (mid-call STT
  history plus current pre-call/cognition blockers). Use when diagnosing why
  room speech STTs but call audio does not, when re-checking Communication
  admission / the `communication.respond` → Communication → Speaking route /
  Gemini STT after a demo run, when intuition/provider
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
* set `{{primary_rc}}` to `RC-5 post-contribution cognition / provider (Mercury 400, bad answer/focus selection, or communication.respond route failure) — blocks dial, answer, or response before mid-call STT can be retested`
* set `{{rc1_token}}` to `unavailable event: bot.ability.communication.input`
* set `{{secondary_rcs}}` to `RC-1 Communication admission while active (MITIGATED — reopen only if {{rc1_token}} returns); RC-2 TextStimulus into GeminiVoiceDecoder (latent); RC-3 call media 48k chunked vs room 16k (latent until answered media); RC-4 STT only behind Conversation admission (structural)`
* treat event names and counts in dated historical RCA tables as non-executable context; use the current Communication topology below for rechecks
* if `{{record_dir}}` is not empty
  * [Recheck Run](#recheck-run)
* report the updated one-sentence RCA and the ranked RC list from the reference
* stop

## Resolve Record Dir

* set `{{repo_root}}` to the stateforward.bot workspace root when empty
* set `{{rca_reference}}` to `docs/rca-phone-call-stt.md` under `{{repo_root}}` when empty
* if `{{record_dir}}` is empty
  * set `{{return_stamp}}` to the current UTC timestamp in `YYYYMMDDTHHMMSSZ` form
  * set `{{return_dir}}` to `~/.agents/projects/bot.py/returns`
  * create `{{return_dir}}` when missing
  * set `{{return_script}}` to `{{return_dir}}/rca-record-dir-{{return_stamp}}.mdscript.md`
  * write an executable MDScript to `{{return_script}}` that records this source workflow, resumes at `Resume Record Dir`, binds the pending answer to `{{record_dir}}`, preserves the current `{{repo_root}}`, `{{rca_reference}}`, and `{{record_dir}}` values, applies the user's latest answer to `{{record_dir}}`, and continues at [Resume Record Dir](#resume-record-dir)
  * ask the user for `{{record_dir}}` as the absolute or repo-relative path to a two-bots recording directory, resume at [Resume Record Dir](#resume-record-dir), and end the prompt with the exact final line `mdscript-exec {{return_script}}`
  * stop
* [Validate Record Dir](#validate-record-dir)

## Resume Record Dir

* set `{{repo_root}}` to the stateforward.bot workspace root when empty
* set `{{rca_reference}}` to `docs/rca-phone-call-stt.md` under `{{repo_root}}` when empty
* if `{{record_dir}}` is empty
  * report that the returned recording directory is empty
  * stop
* [Validate Record Dir](#validate-record-dir)

## Validate Record Dir

* if `{{record_dir}}` is relative
  * set `{{record_dir}}` to `{{repo_root}}/{{record_dir}}`
* if `{{record_dir}}` does not exist or has no `*.log`
  * report that the recording directory is missing or incomplete
  * stop
* [Recheck Run](#recheck-run)

## Recheck Run

* list all `*.log` files under `{{record_dir}}` into `{{candidate_logs}}`
* classify each candidate from explicit bot identity recorded in the log, then set `{{caller_candidates}}` and `{{callee_candidates}}`
* if either candidate set is empty
  * report which bot identity is missing and list `{{candidate_logs}}`
  * stop
* if either candidate set contains more than one log
  * report the ambiguous identity, its matching paths, and the identity evidence used for classification
  * stop
* set `{{caller_log}}` to the sole `{{caller_candidates}}` path and `{{callee_log}}` to the sole `{{callee_candidates}}` path
* set `{{response_route}}` to `bot.ability.communication.respond → Communication/responding → injected Speaking`
* set `{{rc1_token}}` to `unavailable event: bot.ability.communication.input`
* count in each bot log: `HearingSpeech`, `communication.input`, `{{rc1_token}}`, `bot.ability.communication.respond`, `bot.ability.communication.respond.completed`, `bot.ability.communication.respond.failed`, `Communication/responding`, `normalize.completed`, `normalize.failed`, `TextStimulus`, `environment.sound`, `remote_audio`, `phone.dial`, `/Phone/ringing`, `chat completion request failed`, `400 Bad Request`, `focus_device outside`, `missing answer`
* set `{{rc1_evidence}}` to `{{rc1_token}}` counts per bot
* set `{{rc2_evidence}}` to TextStimulus / GeminiVoiceDecoder failure counts per bot
* set `{{stt_evidence}}` to normalize.completed vs failed and any contribution transcript strings found
* set `{{rc5_evidence}}` to intuition/provider failures (400, chat completion failed), `{{response_route}}` outcomes, and bad selection reasons (focus_device outside, missing answer)
* if `two_bot_summary.json` exists under `{{record_dir}}`
  * read stages and speech peaks from that summary into `{{summary_stages}}`
* [Report Recheck](#report-recheck)

## Report Recheck

* compare `{{rc1_evidence}}`, `{{rc2_evidence}}`, `{{stt_evidence}}`, and `{{rc5_evidence}}` against the ranked causes in [RCA reference](rca-phone-call-stt.md)
* report whether RC-1 still dominates (`{{rc1_token}}` high with HearingSpeech > 0) — if near zero, state RC-1 remains mitigated
* report whether RC-5 dominates (provider 400 / intuition failed before dial, dial/ring without answer due to answer/focus selection, or a failed `{{response_route}}` after `communication.respond`)
* report whether any successful mid-call contribution transcript appeared
* report call setup health from `{{summary_stages}}` when set (dial, rang, answered, peaks)
* if RC-5 dominates
  * recommend fix order from the reference starting at provider request shape or answer/focus tool availability
* if `{{rc1_token}}` is near zero, call is answered with media, but STT still fails
  * recommend prioritizing RC-2 and RC-3 packaging checks
* if RC-1 still dominates
  * recommend reopening Communication admit-while-active work for `{{rc1_token}}`
* stop

## Entry Shortcuts

* for findings only: `/mdscript-exec docs/rca.mdscript.md#load-findings`
* for a new recording: `/mdscript-exec docs/rca.mdscript.md#resolve-record-dir`
* for recheck with path already known: set `{{record_dir}}` then `/mdscript-exec docs/rca.mdscript.md#recheck-run`
