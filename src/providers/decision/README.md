# mosfet-provider-decision

Decision-model provider for stateforward.mosfet: the bot's reflex decides on any server that
speaks the decisions wire format, such as anygame's decide, or a hosted server that keeps to it.

```python
from mosfet import bot
from mosfet.providers import decision

bot.cognition(reasoning=processor, reflex=decision.Processor(base_url="http://127.0.0.1:8102"))
```

- `Processor` is a decision processor (`processing.DecisionProcessor`), the kind reflex's decision
  layer runs on. It maps a cognition turn to one request. Its state is the turn as the model may see it: core's
  model-facing projection of the stimulus, the same text reasoning reads, with typed causal parents
  left out and media as a placeholder, never as bytes or base64; what someone said or typed is kept
  whole. It asks a choice question over the offered event menu
  (each event described by the first line of its schema's description) plus a reserved pass
  criterion ("none of these") for each later tier the turn may go to (R-23), and one choice question per required payload key whose criteria
  are values taken mechanically from the turn's own evidence (the stimulus data, media left out, then the
  schema's enum options). The answer label *is* the value, so a payload can never leave the menu,
  and an event with a required key that has no grounded value is left unoffered, before any
  request. Only the option the question chose is returned, with its probability as the
  confidence. A pass leaves the turn unhandled and names the tier it hands the turn to (`hand_to`):
  `__intuition__`, "answer it in words now, with intuition", offered only when the frame names intuition
  (`processing.InputData.hand_ons`), or `__reasoning__`, "it needs thought, so think it through first, with
  reasoning", always offered. `pass_wording=` rewords them, one wording per tier.
- `Client` sends one request and returns the answer. It opens an `httpx.AsyncClient` per request,
  so cancelling a turn closes its request. Every failure is a `decision.Error` that names the HTTP
  status or the error type and never a body.

## Servers and keys

`Processor(base_url=..., api_key=None, model=None, timeout=None, max_context_tokens=None)` and
`Client(base_url=..., api_key=None, timeout=None)`:

- `base_url` is required: the server to ask, for example `http://127.0.0.1:8102`. There is no
  default server.
- `api_key`, when given, is sent as `Authorization: Bearer <key>`. Without it, no authorization
  header is sent at all.
- `model` names the model a server that serves several should answer with.
- `timeout` bounds each request in seconds; `None` keeps httpx's default.
- `max_context_tokens` is the decision model's context (2048 for anygame's decide). See below.

## Fitting the model's context

With `max_context_tokens`, each request is fitted to the decision model's context before it is
sent. Its size is estimated from its JSON body at the fewest bytes per token any answer's `usage.input_tokens`
has shown (the server's own tokenizer and prompt template), and at 3 until a server has said. A request over the
limit has the selection question's instructions cut: those are the turn's world, self and acted
blocks, which are context assembly. They keep as many leading whole lines as fit (the world block
comes first), and when not even one fits, the question's own default instructions are sent
instead. The state (the turn's stimuli) and the menu are never cut, ranked or dropped. A request
that still does not fit is sent as it is, and the server's refusal is the decision layer's failure, which
cognition hands on to the next tier in the same turn. `None` sends every request whole.

## Wire format

`POST {base_url}/v1/systemone` with

```json
{"state": {"turn": "<environment:sound kind=\"phone.ringing\" stimulus:event=\"environment.sound\" />"},
 "questions": {"selection": {"type": "choice", "instructions": "...",
                             "criteria": {"phone.answer_call": "Answer the ringing call.",
                                          "__intuition__": "None of these fits: answer it in words now, with intuition.",
                                          "__reasoning__": "None of these fits: it needs thought, so think it through first, with reasoning."}}}}
```

answered by

```json
{"model": "pplx-decider-flash",
 "answers": {"selection": {"type": "choice", "choice": "phone.answer_call", "confidence": 0.7,
                           "probabilities": {"phone.answer_call": 0.8, "__intuition__": 0.15, "__reasoning__": 0.05}}},
 "usage": {"input_tokens": 10, "output_tokens": 0}}
```

`model` is optional in the answer, and `usage` and any other keys are ignored. mosfet asks only
choice questions.

## Telemetry

Requests and answers are logged through `mosfet.telemetry` with `provider="decision"` and the
model that answered (`model`), the answer as JSON when payload capture is on. The server's address
never reaches telemetry.

`bot.provider.decision.fit.count{outcome}` counts the requests that were over
`max_context_tokens`: `cut` when cutting the instructions made them fit, `overflow` when they were
still over and were sent anyway.
