"""Print each bot turn recorded in otel-logs.jsonl / otel-spans.jsonl as a span tree.

A turn is one trace: the runtime propagates trace context through every hop a stimulus causes
(``mosfet.telemetry.Traced`` / ``EventContextBinding``), so records are grouped strictly by
``trace_id``. A trace is a turn when it carries a captured ``bot.input`` event; ``--all`` also
prints the other traces (bring-up, background work). Within a trace, spans are printed as a tree
under their parents, each followed by the records logged in it, in time order. Record lines show
one step: an HSM event and the transition that consumed it, a generator/processor request or
response (selected tool calls with arguments, text, reasoning, confidence, latency), a dispatch
outcome, or a behavior inventory status write. ``bot.hsm.observe`` spans and quiet construction
spans are transparent: their records and children print at their parent's depth.

Full detail needs the runtime to have run with ``BOT_OTEL_CAPTURE=full`` (see
``mosfet.telemetry.capture``); without it only spans and generator counts exist.

Usage::

    uv run python scripts/otel_turns.py [--dir DIR] [--last N] [--width N] [--all]
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import pathlib
import typing

Record = dict[str, typing.Any]

# Envelope keys that repeat the whole turn context on every hop; elided unless --all.
_BULKY_KEYS = frozenset({"instructions", "processing_input", "turn", "request", "cognition_input", "prior_episodes"})
_NOISE_EVENTS = ("attachment.", "hsm/initial", "bot.ability.attachment.terminal")
_QUIET_OUTCOMES = frozenset({"ok", "observed", ""})
# Construction spans of per-turn child machines: noise unless they failed.
_QUIET_SPANS = frozenset({"bot.define", "bot.started"})


def _read(path: pathlib.Path) -> list[Record]:
    if not path.exists():
        return []
    records: list[Record] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _time(value: str | None) -> datetime.datetime:
    if not value:
        return datetime.datetime.min.replace(tzinfo=datetime.UTC)
    text = value.rstrip("Z")
    head, _, fraction = text.partition(".")
    return datetime.datetime.fromisoformat(f"{head}.{(fraction or '0')[:6]:0<6}").replace(tzinfo=datetime.UTC)


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if width <= 0 or len(text) <= width else text[: width - 1] + "…"


def _elide(value: object, *, keep_all: bool) -> object:
    if keep_all:
        return value
    if isinstance(value, dict):
        mapping = typing.cast(dict[str, object], value)
        return {
            key: (f"<{key}>" if key in _BULKY_KEYS and item not in (None, "", [], {}) else _elide(item, keep_all=False))
            for key, item in mapping.items()
        }
    if isinstance(value, list):
        return [_elide(item, keep_all=False) for item in typing.cast(list[object], value)]
    return value


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


def _input_text(body: Record) -> str:
    """Pull the stimulus the turn started from out of a bot.input event."""

    observation = (body.get("event", {}).get("data") or {}).get("observation") or {}
    data = observation.get("data") or {}
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    text = inner.get("text") if isinstance(inner, dict) else None
    sender = inner.get("sender") if isinstance(inner, dict) else None
    name = data.get("name") or observation.get("event") or "?"
    if text is not None:
        return f"{name} from {sender or '?'}: {text!r}"
    return f"{name} {_json(data)}"


def _request_summary(body: Record, width: int) -> str:
    messages = body.get("messages")
    tools = body.get("tools")
    if messages is None:
        return f"counts messages={body.get('message_count')} tools={body.get('tool_count')} (payload capture off)"
    last = messages[-1] if isinstance(messages, list) and messages else messages
    if isinstance(last, dict) and "content" in last:
        prompt = f"{last.get('role', '?')}: {last.get('content')}"
    else:
        prompt = _json(last)
    names: list[str] = []
    if isinstance(tools, list):
        for tool in typing.cast(list[Record], tools):
            names.append(str(tool.get("name") or (tool.get("function") or {}).get("name") or "?"))
    elif isinstance(tools, dict):
        for key, question in typing.cast(Record, tools).items():
            criteria = question.get("criteria") if isinstance(question, dict) else None
            names.append(f"{key}{sorted(criteria) if isinstance(criteria, dict) else ''}")
    count = len(messages) if isinstance(messages, list) else 1
    return f"{count} msgs, tools={names} | last {_clip(prompt, width)}"


def _response_summary(body: Record, width: int) -> str:
    response = body.get("response")
    parts: list[str] = [f"{body.get('latency_ms')}ms"]
    if body.get("error"):
        parts.append(f"ERROR {body['error']}")
    if isinstance(response, dict):
        answer = typing.cast(Record, response)
        if "status" in answer:
            parts.append(f"status={answer.get('status')}")
            if answer.get("incomplete_details"):
                parts.append(f"incomplete={_json(answer['incomplete_details'])}")
        for item in answer.get("output") or ():
            kind = item.get("type")
            if kind == "function_call":
                parts.append(f"CALL {item.get('name')}({item.get('arguments')})")
            elif kind == "reasoning":
                summary = " ".join(str(part.get("text", "")) for part in item.get("summary") or ())
                if summary:
                    parts.append(f"reasoning: {_clip(summary, width)}")
            elif kind == "message":
                text = "".join(str(part.get("text", "")) for part in item.get("content") or ())
                parts.append(f"text: {text!r}")
        for choice in answer.get("choices") or ():
            message = choice.get("message") or {}
            parts.append(f"finish={choice.get('finish_reason')}")
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                parts.append(f"CALL {function.get('name')}({function.get('arguments')})")
            if message.get("content"):
                parts.append(f"text: {message['content']!r}")
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                parts.append(f"reasoning: {_clip(str(reasoning), width)}")
        for name, question in (answer.get("answers") or answer.get("choices_by_name") or {}).items():
            if isinstance(question, dict):
                parts.append(
                    f"{name}={question.get('choice')!r} conf={question.get('confidence')} "
                    + f"p={_json(question.get('probabilities'))}"
                )
        for call in answer.get("function_calls") or ():
            parts.append(f"CALL {call.get('name')}({_json(call.get('arguments'))})")
        if "confidence" in answer and "answers" not in answer:
            parts.append(f"confidence={answer.get('confidence')}")
        if answer.get("validation"):
            parts.append(f"validation={_json(answer['validation'])}")
    elif response is not None:
        parts.append(_clip(_json(response), width))
    return " | ".join(parts)


def _log_line(record: Record, *, width: int, keep_all: bool) -> str | None:
    attributes = record.get("attributes") or {}
    body = record.get("body") or {}
    component = attributes.get("component", "")
    stage = attributes.get("stage", "")
    if component == "text.generator":
        provider = attributes.get("provider", "?")
        if stage == "request":
            return f"GEN  {provider} request   {_request_summary(body, width)}"
        if stage == "response":
            return f"GEN  {provider} response  {_response_summary(body, width)}"
        if stage == "usage":
            return f"GEN  {provider} usage     {_json(body)}"
    if stage == "behavior_status" and isinstance(body, dict):
        return (
            f"BHV  {body.get('behavior')} {body.get('from')} -> {body.get('to')} cause={body.get('cause')} "
            + f"revision={body.get('revision')} reason={body.get('reason')!r}"
        )
    if attributes.get("stage") == "dispatch" and isinstance(body, dict) and "accepted" in body:
        return (
            f"DISP {component} {body.get('event')} {body.get('source')}->{body.get('target')} "
            + f"accepted={body.get('accepted')} [{body.get('id')}]"
        )
    if isinstance(body, dict) and body.get("occurrence") == "event":
        event = body.get("event") or {}
        name = str(event.get("name", ""))
        if not keep_all and name.startswith(_NOISE_EVENTS):
            return None
        transition = body.get("transition") or {}
        if transition:
            move = f" {_short(transition.get('from'))} -> {_short(transition.get('to'))}"
        else:
            move = f" @{_short(body.get('state'))} ({str(body.get('element', '')).rsplit('/', 1)[-1]})"
        data = _elide(event.get("data"), keep_all=keep_all)
        detail = "" if data is None else " " + _clip(_json(data), width)
        return f"EVT  {body.get('component')}{move}  {name} [{event.get('id')}]{detail}"
    if isinstance(body, str):
        return f"LOG  {record.get('severity')} {_clip(body, width)}"
    return None


def _short(state: object) -> str:
    text = str(state or "")
    parts = [part for part in text.split("/") if part]
    return "/".join(parts[-2:]) if parts else text


def _span_line(span: Record, width: int) -> str | None:
    attributes = span.get("attributes") or {}
    outcome = str(attributes.get("bot.outcome", ""))
    interesting = {
        key: value
        for key, value in attributes.items()
        if key.startswith("bot.") and key not in ("bot.component.name", "bot.stage", "bot.outcome")
    }
    if span.get("name") in _QUIET_SPANS and outcome in _QUIET_OUTCOMES:
        return None
    if span.get("name") == "bot.hsm.observe":
        return None
    if outcome in _QUIET_OUTCOMES and span.get("status") != "ERROR" and not interesting:
        return None
    duration = span.get("duration_ns")
    millis = f"{duration / 1e6:.1f}ms" if isinstance(duration, int) else "?"
    return _clip(
        f"SPAN {span.get('name')} stage={attributes.get('bot.stage')} outcome={outcome or '?'} "
        + f"status={span.get('status')} {millis} {_json(interesting) if interesting else ''}",
        width * 2,
    )


def _is_turn_start(record: Record) -> bool:
    body = record.get("body")
    if not isinstance(body, dict) or body.get("occurrence") != "event":
        return False
    return (body.get("event") or {}).get("name") == "bot.input"


def _transparent(span: Record) -> bool:
    attributes = span.get("attributes") or {}
    if span.get("name") == "bot.hsm.observe":
        return True
    return span.get("name") in _QUIET_SPANS and str(attributes.get("bot.outcome", "")) in _QUIET_OUTCOMES


def _print_trace(
    trace_id: str,
    spans: list[Record],
    logs: list[Record],
    *,
    width: int,
    keep_all: bool,
) -> None:
    by_id = {str(span["span_id"]): span for span in spans}
    children: dict[str | None, list[Record]] = collections.defaultdict(list)
    for span in spans:
        parent = span.get("parent_span_id")
        children[str(parent) if parent in by_id else None].append(span)
    records: dict[str | None, list[Record]] = collections.defaultdict(list)
    for record in logs:
        span_id = record.get("span_id")
        records[str(span_id) if span_id in by_id else None].append(record)
    moments = [_time(span.get("start_time")) for span in spans] + [_time(record.get("timestamp")) for record in logs]
    began = min(moments)
    start = next(
        (record for record in sorted(logs, key=lambda r: _time(r.get("timestamp"))) if _is_turn_start(record)), None
    )
    roots = sorted(children[None], key=lambda span: _time(span.get("start_time")))
    root_names = ", ".join(str(span.get("name")) for span in roots) or "-"
    print(
        f"=== trace {trace_id}  {began.isoformat(timespec='milliseconds')}  spans={len(spans)} records={len(logs)} "
        + f"root={root_names}"
    )
    if start is not None:
        print(f"    input: {_input_text(start['body'])}")

    def emit(span_id: str | None, depth: int) -> None:
        items: list[tuple[datetime.datetime, str, Record]] = [
            (_time(record.get("timestamp")), "log", record) for record in records[span_id]
        ]
        items += [(_time(child.get("start_time")), "span", child) for child in children[span_id]]
        items.sort(key=lambda item: item[0])
        for moment, kind, record in items:
            offset = f"+{(moment - began).total_seconds():7.3f}s"
            if kind == "log":
                line = _log_line(record, width=width, keep_all=keep_all)
                if line is not None:
                    print(f"  {offset} {'  ' * depth}{line}")
                continue
            if _transparent(record):
                emit(str(record["span_id"]), depth)
                continue
            line = _span_line(record, width) or _span_label(record)
            print(f"  {offset} {'  ' * depth}{line}")
            emit(str(record["span_id"]), depth + 1)

    emit(None, 0)
    print()


def _span_label(span: Record) -> str:
    attributes = span.get("attributes") or {}
    duration = span.get("duration_ns")
    millis = f"{duration / 1e6:.1f}ms" if isinstance(duration, int) else "?"
    return (
        f"SPAN {span.get('name')} stage={attributes.get('bot.stage')} "
        + f"outcome={attributes.get('bot.outcome', '?')} {millis}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("--dir", default=".", help="directory holding otel-logs.jsonl / otel-spans.jsonl")
    _ = parser.add_argument("--logs", default=None, help="log file (default DIR/otel-logs.jsonl)")
    _ = parser.add_argument("--spans", default=None, help="span file (default DIR/otel-spans.jsonl)")
    _ = parser.add_argument("--last", type=int, default=0, help="only the last N traces")
    _ = parser.add_argument("--width", type=int, default=240, help="clip long values (0 = never)")
    _ = parser.add_argument(
        "--all", action="store_true", help="include non-turn traces, lifecycle noise, and bulky payload keys"
    )
    arguments = parser.parse_args()
    root = pathlib.Path(arguments.dir)
    logs = _read(pathlib.Path(arguments.logs) if arguments.logs else root / "otel-logs.jsonl")
    spans = _read(pathlib.Path(arguments.spans) if arguments.spans else root / "otel-spans.jsonl")

    trace_spans: dict[str, list[Record]] = collections.defaultdict(list)
    trace_logs: dict[str, list[Record]] = collections.defaultdict(list)
    for span in spans:
        if span.get("trace_id"):
            trace_spans[str(span["trace_id"])].append(span)
    for record in logs:
        if record.get("trace_id"):
            trace_logs[str(record["trace_id"])].append(record)
    traces: list[tuple[datetime.datetime, str]] = []
    for trace_id in {*trace_spans, *trace_logs}:
        if not arguments.all and not any(_is_turn_start(record) for record in trace_logs[trace_id]):
            continue
        moments = [_time(span.get("start_time")) for span in trace_spans[trace_id]]
        moments += [_time(record.get("timestamp")) for record in trace_logs[trace_id]]
        traces.append((min(moments), trace_id))
    traces.sort()
    if arguments.last > 0:
        traces = traces[-arguments.last :]
    if not traces:
        print("no turns found (a turn is a trace with a captured bot.input event; run with BOT_OTEL_CAPTURE=full)")
        return
    for _, trace_id in traces:
        _print_trace(
            trace_id,
            trace_spans[trace_id],
            trace_logs[trace_id],
            width=arguments.width,
            keep_all=arguments.all,
        )


if __name__ == "__main__":
    main()
