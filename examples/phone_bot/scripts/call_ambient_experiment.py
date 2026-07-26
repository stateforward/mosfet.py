#!/usr/bin/env python3
"""Random call vs ambient stimulus experiment against phone_bot cognition.

Starts a local phone bot (no LiveKit room) and injects either:

- **call** — firmware ``IncomingCall`` → elevated ``PhoneSoundData`` (``kind=phone.ringing``)
- **ambient** — plain ``environment.sound`` with ``kind=ambient`` (no call_id)

Records cognition outputs (answer / ignore / focus / other). Uses real Mercury intuition +
Terra keys from repo ``.env`` and ``examples/phone_bot/.env``.

Run from ``examples/phone_bot``::

    uv run python scripts/call_ambient_experiment.py --trials 8 --seed 1
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import pathlib
import random
import sys
import time
import typing
import uuid

import hsm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from phone_bot_example import AppConfig, PhoneBot  # noqa: E402

from bot.abilities import memory  # noqa: E402
from bot.abilities.cognition import types as cognition_types  # noqa: E402
from bot.devices.phone import events as phone_events  # noqa: E402
from bot.devices.phone.phone import PhoneFirmware, RING_SOUND_WAV  # noqa: E402
from bot.devices.phone.events import PhoneSoundData  # noqa: E402
from bot.behavior import storage as behavior_storage  # noqa: E402
from bot.environment import SoundData, SoundEvent, Environment  # noqa: E402


def _load_env_files(*paths: pathlib.Path) -> None:
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _minimal_wav() -> bytes:
    return (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )


def _firmware(phone: object) -> PhoneFirmware:
    return typing.cast(PhoneFirmware, object.__getattribute__(phone, "_firmware_instance"))


async def _wait_until(pred: typing.Callable[[], bool], *, timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(label)


def _selection_names(output: object) -> list[str]:
    if not isinstance(output, tuple):
        return []
    names: list[str] = []
    for item in output:
        name = getattr(item, "event", None)
        if isinstance(name, str):
            names.append(name)
    return names


def _list_behaviors(store: memory.Memory) -> list[dict[str, object]]:
    """Return inventory rows: name, status, triggers (any status)."""

    try:
        recalled = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses()))
        )
        results = getattr(recalled, "results", ()) or ()
        behavior_rows = tuple(row.as_mapping() for row in (results[0].rows if len(results) > 0 else ()))
        trigger_rows = tuple(row.as_mapping() for row in (results[1].rows if len(results) > 1 else ()))
        instances = behavior_storage.instances_from_behavior_results(behavior_rows, trigger_rows)
    except Exception as error:
        return [{"error": str(error)}]
    return [
        {
            "name": item.name,
            "status": item.status,
            "triggers": list(item.triggers),
            "description": item.description[:120] if item.description else "",
        }
        for item in instances
    ]


async def _hang_up_if_needed(phone: object, call_id: str | None) -> None:
    if not call_id:
        return
    firmware = _firmware(phone)
    state = firmware.state() or ""
    if "/hung_up" in state:
        return
    try:
        await firmware.event_recorder().receive(
            phone.context(),  # type: ignore[attr-defined]
            phone_events.RemoteHangUpEvent.with_data(phone_events.RemoteHangUpData(call_id=call_id)),
        )
    except Exception as error:
        print(f"  hang_up note: {error}", flush=True)
    await asyncio.sleep(0.3)


async def _inject_call(environment: Environment, body: PhoneBot, *, call_id: str) -> None:
    phone = body.phone()
    firmware = _firmware(phone)
    await firmware.event_recorder().receive(
        phone.context(),
        phone_events.IncomingCallEvent.with_data(
            phone_events.IncomingCallData(call_id=call_id, display_hint="experiment-caller")
        ),
    )
    # Ensure body sees elevated ring even if environment broadcast timing races.
    await asyncio.sleep(0.15)
    await hsm.dispatch(
        environment,
        body,
        dataclasses.replace(
            SoundEvent.with_data(
                PhoneSoundData(
                    audio=RING_SOUND_WAV,
                    media_type="audio/wav",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="phone.ringing",
                    call_id=call_id,
                )
            ),
            id=call_id,
            source=hsm.id(phone),
            target="",
        ),
    )


async def _inject_ambient(environment: Environment, body: PhoneBot) -> None:
    await hsm.dispatch(
        environment,
        body,
        dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=_minimal_wav(),
                    media_type="audio/wav",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="ambient",
                )
            ),
            id=f"ambient-{uuid.uuid4().hex[:8]}",
            source="experiment-ambient",
            target="",
        ),
    )


async def run_experiment(*, trials: int, seed: int | None, settle_seconds: float) -> int:
    repo = pathlib.Path(__file__).resolve().parents[3]
    example = pathlib.Path(__file__).resolve().parents[1]
    _load_env_files(repo / ".env", example / ".env")

    rng = random.Random(seed)
    config = AppConfig.from_env_file(example / ".env")
    if not config.cognition.can_process():
        print("missing cognition keys (Mercury/OpenAI); abort", file=sys.stderr)
        return 2

    environment = Environment()
    body = PhoneBot("experiment", cognition_config=config.cognition, speech_config=config.speech)
    _ = await body.attach(environment)
    await _wait_until(
        lambda: (body.state() or "").endswith("/active/unfocused"),
        timeout=60.0,
        label="bot activate",
    )

    print(
        f"experiment start trials={trials} seed={seed} "
        f"mercury={config.cognition.intuition_model} terra={config.cognition.model}",
        flush=True,
    )

    results: list[dict[str, object]] = []
    active_call_id: str | None = None
    behaviors_before = _list_behaviors(body.memory())
    print(f"behaviors_before={behaviors_before}", flush=True)

    for index in range(1, trials + 1):
        kind = rng.choice(("call", "ambient"))
        before = len(body.outputs())
        before_fail = len(body.failures())
        behaviors_pre = _list_behaviors(body.memory())
        call_id = f"livekit:exp-{index}-{uuid.uuid4().hex[:6]}"
        print(f"\n=== trial {index}/{trials} type={kind} ===", flush=True)
        # Clear previous call so firmware can ring again.
        await _hang_up_if_needed(body.phone(), active_call_id)
        active_call_id = None
        t0 = time.monotonic()
        try:
            if kind == "call":
                active_call_id = call_id
                await _inject_call(environment, body, call_id=call_id)
            else:
                await _inject_ambient(environment, body)
            await _wait_until(
                lambda: len(body.outputs()) > before or len(body.failures()) > before_fail,
                timeout=settle_seconds,
                label=f"trial {index} cognition terminal",
            )
        except Exception as error:
            print(f"trial {index} ERROR: {error}", flush=True)
            results.append({"trial": index, "type": kind, "error": str(error), "call_id": call_id})
            continue
        elapsed = time.monotonic() - t0
        new_outputs = body.outputs()[before:]
        new_failures = body.failures()[before_fail:]
        # Give reflection time to select create + revise + store after a handled turn.
        await asyncio.sleep(4.0)
        behaviors_post = _list_behaviors(body.memory())
        names: list[str] = []
        for out in new_outputs:
            names.extend(_selection_names(out))
        fail_msgs = []
        for fail in new_failures:
            if hasattr(fail, "message"):
                fail_msgs.append(str(fail.message))
            else:
                fail_msgs.append(repr(fail))
        pre_names = {str(h.get("name")) for h in behaviors_pre if "name" in h}
        post_names = {str(h.get("name")) for h in behaviors_post if "name" in h}
        created = sorted(post_names - pre_names)
        row: dict[str, object] = {
            "trial": index,
            "type": kind,
            "call_id": call_id if kind == "call" else None,
            "seconds": round(elapsed, 2),
            "selections": names,
            "answered": any(n == phone_events.AnswerCallEvent.name for n in names),
            "ignored": any(n == cognition_types.IgnoreEvent.name for n in names),
            "focused": any(n == "bot.focus_device" for n in names),
            "failures": len(new_failures),
            "failure_messages": fail_msgs,
            "behaviors_created": created,
            "behavior_count": len([h for h in behaviors_post if "name" in h]),
            "bot_state": body.state(),
        }
        results.append(row)
        print(
            f"trial {index} done selections={names} "
            f"answered={row['answered']} ignored={row['ignored']} "
            f"focused={row['focused']} failures={fail_msgs or 0} "
            f"behaviors_created={created} behavior_count={row['behavior_count']} t={elapsed:.1f}s",
            flush=True,
        )

    await _hang_up_if_needed(body.phone(), active_call_id)
    behaviors_after = _list_behaviors(body.memory())

    print("\n=== summary ===", flush=True)
    calls = [r for r in results if r.get("type") == "call" and "error" not in r]
    ambients = [r for r in results if r.get("type") == "ambient" and "error" not in r]
    errors = [r for r in results if "error" in r]
    created_any = [r for r in results if r.get("behaviors_created")]
    print(
        f"calls: n={len(calls)} answered={sum(1 for r in calls if r.get('answered'))} "
        f"ignored={sum(1 for r in calls if r.get('ignored'))}",
        flush=True,
    )
    print(
        f"ambient: n={len(ambients)} answered={sum(1 for r in ambients if r.get('answered'))} "
        f"ignored={sum(1 for r in ambients if r.get('ignored'))}",
        flush=True,
    )
    print(f"errors: {len(errors)}", flush=True)
    print(f"behavior_creates: {len(created_any)} trials", flush=True)
    print(f"behaviors_before={behaviors_before}", flush=True)
    print(f"behaviors_after={behaviors_after}", flush=True)
    for r in results:
        print(r, flush=True)
    return 0 if not errors else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--trials", type=int, default=24)
    _ = parser.add_argument("--seed", type=int, default=None)
    _ = parser.add_argument("--settle-seconds", type=float, default=120.0)
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run_experiment(
                trials=int(args.trials),
                seed=args.seed,
                settle_seconds=float(args.settle_seconds),
            )
        )
    )


if __name__ == "__main__":
    main()
