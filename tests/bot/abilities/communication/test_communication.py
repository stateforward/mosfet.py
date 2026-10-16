"""Communication is event-driven: activate via typed events; Memory owns durable recall."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import inspect
import itertools
import json
import re
import pathlib
import typing

import hsm
import mosfet
import hsm.muid as muid
import pytest

from mosfet.abilities import communication
from mosfet.abilities.communication import conversation
from mosfet.abilities import decoding
from mosfet.abilities import processing
from mosfet.abilities import speaking
from mosfet.abilities import vocal
from mosfet.abilities.communication.conversation import turn_detector
from mosfet import event
from mosfet import StimulusData
from mosfet.abilities import ability
from mosfet.devices import audio
from mosfet.protocols import attachment
from mosfet.environment import Environment
from tests import metric_reader
from tests.hsm_instance_state import start_ability_tree
from tests.hsm_model import transition_map

RATE = 8_000
_ENCODE_IDLE_TIMEOUT = datetime.timedelta(seconds=5)


class Script(vocal.speech.SpeechEncoder):
    """Reads the text to its end, then streams 20 ms of PCM per word, each chunk aligned to its word, optionally
    pausing before each word or stalling forever (at a word)."""

    def __init__(self, *, pause: float = 0.0, stall: bool = False, stall_on: str | None = None) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=RATE, channels=1)
        self.pause: float = pause
        self.stall: bool = stall
        self.stall_on: str | None = stall_on
        self.calls: list[str] = []

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece.words async for piece in input])
        self.calls.append(said)
        if self.stall:
            _ = await asyncio.Event().wait()
        for word in re.findall(r"\s*\S+\s*", said):
            if self.pause:
                await asyncio.sleep(self.pause)
            if self.stall_on is not None and word.strip().startswith(self.stall_on):
                _ = await asyncio.Event().wait()
            yield vocal.speech.Audio(pcm=b"\x01\x00" * (RATE // 50), text=word)


def _instant() -> collections.abc.Callable[[], float]:
    """A speaker clock always past its schedule, so the mouth reports each chunk played at once."""

    ticks = itertools.count(0.0, 1_000.0)
    return lambda: next(ticks)


def _speaking(
    encoder: Script | None = None,
    *,
    speaker: audio.Speaker | None = None,
    conversation: collections.abc.Sequence[ability.Ability[typing.Any, typing.Any]] = (),
    encode_idle_timeout: datetime.timedelta = _ENCODE_IDLE_TIMEOUT,
) -> speaking.Speaking:
    return speaking.Speaking(
        encoder=encoder or Script(),
        speaker=speaker or audio.Speaker(frame=datetime.timedelta(milliseconds=10), clock=_instant()),
        conversation=conversation,
        encode_idle_timeout=encode_idle_timeout,
    )


class RecordingConversation(conversation.Conversation):
    def __init__(self) -> None:
        super().__init__(turn_detector=turn_detector.TurnDetector(decoder=IdentityTextDecoder()))
        self.outputs: list[conversation.Messages] = []

    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> typing.Awaitable[bool]:
        if event.name == ability.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
            terminal = event.data
            if terminal.name == conversation.OutputEvent.name and isinstance(terminal.data, conversation.Messages):
                self.outputs.append(terminal.data)
        return super().dispatch(ctx, event)


class TerminalCollector(hsm.Instance):
    events: list[hsm.Event]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "TerminalCollector", event: hsm.Event) -> None:
        del ctx
        instance.events.append(event)

    model = mosfet.define(
        "TerminalCollector",
        hsm.initial(hsm.target("/TerminalCollector/recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )


class IdentityTextDecoder(decoding.Decoder[typing.Any, str]):
    @typing.override
    async def decode(self, input: typing.Any) -> str:
        if isinstance(input, turn_detector.TextStimulus):
            return input.content
        raise AssertionError(f"unexpected stimulus {input!r}")


def _conversation() -> conversation.Conversation:
    return conversation.Conversation(
        turn_detector=turn_detector.TurnDetector(decoder=IdentityTextDecoder()),
    )


async def _wait_until(condition: typing.Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not met")


async def _owned(ctx: hsm.Context, speaker: speaking.Speaking, comm: communication.Communication) -> TerminalCollector:
    """Start Speaking and Communication, with a collector as Communication's owner."""

    collector = TerminalCollector()
    await start_ability_tree(ctx, speaker)
    _ = await mosfet.started(ctx, collector, collector.model)
    comm_model = comm.model
    assert comm_model is not None
    _ = await mosfet.started(ctx, comm, comm_model)
    await comm.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=collector)))
    await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
    await _wait_until(lambda: (speaker.state() or "").endswith("/wired/voiced/idle"))
    return collector


def _respond(text: str, response_id: str) -> hsm.Event[communication.RespondData]:
    return communication.RespondEvent.with_data_and_id(communication.RespondData(text=text), response_id)


def _failures(collector: TerminalCollector) -> list[ability.FailureData]:
    return [
        item.data
        for item in collector.events
        if item.name == communication.FailedEvent.name and isinstance(item.data, ability.FailureData)
    ]


def _spoken(collector: TerminalCollector) -> list[hsm.Event[typing.Any]]:
    return [item for item in collector.events if item.name == speaking.OutputEvent.name]


def test_communication_construction_uses_active_conversation_keyword() -> None:
    conv = _conversation()
    comm = communication.Communication(active_conversation=conv, speaking=_speaking())
    assert comm.model is not None
    with pytest.raises(ValueError, match="response_idle_timeout"):
        _ = communication.Communication(
            active_conversation=conv, speaking=_speaking(), response_idle_timeout=datetime.timedelta(0)
        )


def test_communication_responding_follows_speaking_by_events_without_private_waiters() -> None:
    comm = communication.Communication(active_conversation=_conversation(), speaking=_speaking())
    model = comm.model
    assert model is not None

    transitions = transition_map(model)
    responding = transitions["/CommunicationLifecycle/attached/behavior/responding"]
    waiting = transitions["/CommunicationLifecycle/attached/behavior/responding/waiting"]
    assert speaking.OutputEvent.name in responding
    assert ability.FailedEvent.name in responding
    assert speaking.ProgressEvent.name in waiting
    assert not any(state.endswith(("preparing_response", "dispatching_response")) for state in transitions)


def test_communication_respond_routes_one_typed_request_to_speaking_and_records_what_was_said() -> None:
    async def run() -> tuple[list[str], list[conversation.Messages], list[hsm.Event[typing.Any]]]:
        ctx = Environment()
        conv = RecordingConversation()
        script = Script()
        speaker = _speaking(script, conversation=(conv,))
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        collector = await _owned(ctx, speaker, comm)

        _ = await hsm.dispatch(ctx, comm, _respond("Hello from Communication.", "response-1"))
        await _wait_until(lambda: bool(_spoken(collector)))
        await _wait_until(lambda: len(conv.outputs) == 1)
        return script.calls, conv.outputs, _spoken(collector)

    calls, outputs, spoken = asyncio.run(run())
    assert calls == ["Hello from Communication."]
    assert [message.direction for message in outputs[0].messages] == ["outbound"]
    assert outputs[0].messages[0].content == "Hello from Communication."
    assert spoken[0].id == "response-1"
    assert isinstance(spoken[0].data, speaking.OutputData)
    assert spoken[0].data.outcome == "completed"


def test_communication_ignores_malformed_respond_payload() -> None:
    async def run() -> tuple[str, list[str]]:
        ctx = Environment()
        script = Script()
        speaker = _speaking(script)
        comm = communication.Communication(active_conversation=_conversation(), speaking=speaker)
        _ = await _owned(ctx, speaker, comm)
        _ = await hsm.dispatch(
            ctx,
            comm,
            communication.RespondEvent.with_data_and_id(typing.cast(communication.RespondData, object()), "bad"),
        )
        return comm.state() or "", script.calls

    state, calls = asyncio.run(run())
    assert "/behavior/active" in state
    assert calls == []


def test_communication_respond_surfaces_speaking_failure() -> None:
    class Deaf(audio.Speaker):
        """A mouth that never answers an attach, so Speaking's own attach fails and it answers requests failed."""

        @typing.override
        async def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> bool:
            del ctx, event
            return True

    async def run() -> tuple[list[ability.FailureData], str]:
        ctx = Environment()
        speaker = _speaking(speaker=Deaf())
        comm = communication.Communication(active_conversation=_conversation(), speaking=speaker)
        collector, body = TerminalCollector(), TerminalCollector()
        _ = await mosfet.started(ctx, body, body.model)
        _ = await speaker.attach(
            ctx,
            attachment.AttachEvent.with_data(
                attachment.AttachData(actor=body, timeout=datetime.timedelta(milliseconds=50))
            ),
        )
        await _wait_until(lambda: any(event.name == attachment.AttachFailedEvent.name for event in body.events))
        _ = await mosfet.started(ctx, collector, collector.model)
        _ = await mosfet.started(ctx, comm, typing.cast(hsm.Model, comm.model))
        await comm.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        _ = await hsm.dispatch(ctx, comm, _respond("This fails.", "response-failure-1"))
        await _wait_until(lambda: bool(_failures(collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        return _failures(collector), comm.state() or ""

    failures, state = asyncio.run(run())
    assert len(failures) == 1
    assert failures[0].message == "Speaking is not attached."
    assert state.endswith("/active")


def test_communication_respond_without_id_mints_unique_operation_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[str, str, str, str]:
        ctx = Environment()
        observed: list[hsm.Event[typing.Any]] = []

        class RecordingSpeaking(speaking.Speaking):
            def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> typing.Awaitable[bool]:
                if event.name == speaking.InputEvent.name:
                    observed.append(event)
                return super().dispatch(ctx, event)

        speaker = RecordingSpeaking(
            encoder=Script(),
            speaker=audio.Speaker(frame=datetime.timedelta(milliseconds=10), clock=_instant()),
        )
        comm = communication.Communication(active_conversation=_conversation(), speaking=speaker)
        collector = await _owned(ctx, speaker, comm)
        monkeypatch.setattr(muid, "make", lambda: "")
        _ = await hsm.dispatch(
            ctx,
            comm,
            dataclasses.replace(communication.RespondEvent.with_data(communication.RespondData(text="no id")), id=""),
        )
        await _wait_until(lambda: bool(observed))
        await _wait_until(lambda: bool(_spoken(collector)))
        request = observed[0]
        return request.id, request.source, request.target, hsm.id(comm)

    operation_id, source, target, producer_id = asyncio.run(run())
    assert operation_id
    assert operation_id != producer_id
    assert source == producer_id
    assert target


def test_communication_rejects_speaking_terminals_for_another_response_or_recipient() -> None:
    async def run() -> tuple[str, list[ability.FailureData], list[hsm.Event[typing.Any]]]:
        ctx = Environment()
        speaker = _speaking(Script(stall=True))
        comm = communication.Communication(active_conversation=_conversation(), speaking=speaker)
        collector = await _owned(ctx, speaker, comm)
        _ = await hsm.dispatch(ctx, comm, _respond("current response", "current-response"))
        await _wait_until(lambda: "/behavior/responding" in (comm.state() or ""))
        output = speaking.OutputData(text="stale", outcome="completed", sample_rate_hz=RATE, channels=1)
        for id, target in (("stale-response", hsm.id(comm)), ("current-response", "wrong-recipient")):
            _ = await hsm.dispatch(
                ctx,
                comm,
                dataclasses.replace(
                    speaking.OutputEvent.with_data(output), id=id, source=hsm.id(speaker), target=target
                ),
            )
            _ = await hsm.dispatch(
                ctx,
                comm,
                dataclasses.replace(
                    ability.FailedEvent.with_data(ability.FailureData(message="spoofed failure")),
                    id=id,
                    source=hsm.id(speaker),
                    target=target,
                ),
            )
        return comm.state() or "", _failures(collector), _spoken(collector)

    state, failures, spoken = asyncio.run(run())
    assert "/behavior/responding" in state
    assert failures == []
    assert spoken == []


def test_communication_gives_up_on_a_response_after_its_idle_timeout_of_silence_and_stops_it() -> None:
    async def run() -> tuple[str, list[tuple[str, str]], list[hsm.Event[typing.Any]], list[str], str]:
        ctx = Environment()
        stops: list[str] = []

        class RecordingSpeaking(speaking.Speaking):
            @typing.override
            def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
                if isinstance(event.data, speaking.StopData):
                    stops.append(event.id)
                return super().dispatch(ctx, event)

        # Speaking itself would wait far longer than the test for an encoding that never starts.
        speaker = RecordingSpeaking(
            encoder=Script(stall=True),
            speaker=audio.Speaker(frame=datetime.timedelta(milliseconds=10), clock=_instant()),
            encode_idle_timeout=datetime.timedelta(seconds=60),
        )
        comm = communication.Communication(
            active_conversation=_conversation(),
            speaking=speaker,
            response_idle_timeout=datetime.timedelta(milliseconds=30),
        )
        collector = await _owned(ctx, speaker, comm)
        _ = await hsm.dispatch(ctx, comm, _respond("This is never spoken.", "response-timeout"))
        await _wait_until(lambda: bool(_failures(collector)))
        state = comm.state() or ""
        # The response it gave up on is stopped, so Speaking is free again; its late terminal is ignored.
        await _wait_until(lambda: (speaker.state() or "").endswith("/wired/voiced/idle"))
        await asyncio.sleep(0.01)
        failed = [
            (event.id, event.data.message)
            for event in collector.events
            if event.name == communication.FailedEvent.name and isinstance(event.data, ability.FailureData)
        ]
        return state, failed, _spoken(collector), stops, speaker.state() or ""

    state, failures, spoken, stops, speaking_state = asyncio.run(run())
    assert state.endswith("/active")
    assert failures == [("response-timeout", "Speaking response timed out.")]
    assert stops == ["response-timeout"]
    assert spoken == []
    assert speaking_state.endswith("/wired/voiced/idle")


def test_speaking_progress_keeps_a_long_response_alive_past_the_idle_timeout() -> None:
    async def run() -> tuple[list[ability.FailureData], list[hsm.Event[typing.Any]]]:
        ctx = Environment()
        # Six words, 40 ms apart: a quarter second of speech against a 100 ms silence deadline.
        speaker = _speaking(Script(pause=0.04))
        comm = communication.Communication(
            active_conversation=_conversation(),
            speaking=speaker,
            response_idle_timeout=datetime.timedelta(milliseconds=100),
        )
        collector = await _owned(ctx, speaker, comm)
        _ = await hsm.dispatch(ctx, comm, _respond("one two three four five six", "long-response"))
        await _wait_until(lambda: bool(_spoken(collector) or _failures(collector)))
        return _failures(collector), _spoken(collector)

    failures, spoken = asyncio.run(run())
    assert failures == []
    assert [item.id for item in spoken] == ["long-response"]


def _draft(
    text: str, response_id: str, *, offset: int = 0, item: int = 0, seq: int = 0
) -> hsm.Event[processing.DraftData]:
    return communication.DraftEvent.with_data_and_id(
        processing.DraftData(text=text, offset=offset, event=communication.RespondEvent.name, item=item, seq=seq),
        response_id,
    )


class ForwardRecordingSpeaking(speaking.Speaking):
    """A real Speaking that also notes every draft, final text and stop Communication forwards to it."""

    def __init__(self, encoder: Script) -> None:
        super().__init__(
            encoder=encoder,
            speaker=audio.Speaker(frame=datetime.timedelta(milliseconds=10), clock=_instant()),
        )
        self.forwarded: list[hsm.Event[typing.Any]] = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if isinstance(event.data, (speaking.InputData, speaking.DraftData, speaking.StopData)):
            self.forwarded.append(event)
        return super().dispatch(ctx, event)


def test_respond_is_marked_for_drafting_through_communication_draft() -> None:
    marker = communication.RespondData.__draft__
    assert marker == processing.Draft(field="text", event=communication.DraftEvent)
    assert communication.DraftEvent.name == "bot.ability.communication.draft"
    assert communication.DraftEvent.kind != processing.EventKind
    assert "__draft__" not in json.dumps(communication.RespondData.model_json_schema())


def test_drafts_open_the_response_and_reach_speaking_under_its_id_until_the_final_text() -> None:
    async def run() -> tuple[list[tuple[str, str, str, int | None]], list[hsm.Event[typing.Any]], str]:
        ctx = Environment()
        voice = ForwardRecordingSpeaking(Script())
        comm = communication.Communication(active_conversation=_conversation(), speaking=voice)
        collector = await _owned(ctx, voice, comm)
        _ = await hsm.dispatch(ctx, comm, _draft("Sure. I can", "op-1", seq=0))
        await _wait_until(lambda: "/behavior/responding" in (comm.state() or ""))
        _ = await hsm.dispatch(ctx, comm, _draft(" help.", "op-1", offset=11, seq=1))
        # A draft of some other selection is not this response's text.
        _ = await hsm.dispatch(
            ctx,
            comm,
            communication.DraftEvent.with_data_and_id(
                processing.DraftData(text="not a response", offset=0, event="bot.ability.other", item=1, seq=2),
                "op-1",
            ),
        )
        _ = await hsm.dispatch(ctx, comm, _respond("Sure. I can help.", "op-1"))
        await _wait_until(lambda: bool(_spoken(collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        forwarded = [
            (
                event.name,
                event.id or "",
                event.data.words
                if isinstance(event.data, speaking.DraftData)
                else vocal.speech.said(typing.cast(speaking.InputData, event.data).parts),
                getattr(event.data, "offset", None),
            )
            for event in voice.forwarded
        ]
        return forwarded, _spoken(collector), hsm.id(comm)

    forwarded, spoken, comm_id = asyncio.run(run())
    assert forwarded == [
        # Each delta goes on unchanged, at its offset; then the final text.
        (speaking.DraftEvent.name, "op-1", "Sure. I can", 0),
        (speaking.DraftEvent.name, "op-1", " help.", 11),
        (speaking.InputEvent.name, "op-1", "Sure. I can help.", None),
    ]
    assert [item.id for item in spoken] == ["op-1"]
    output = spoken[0].data
    assert isinstance(output, speaking.OutputData)
    assert (output.text, output.spoken, output.outcome) == ("Sure. I can help.", "Sure. I can help.", "completed")
    del comm_id


def test_another_response_waits_for_the_one_being_spoken_even_while_this_one_is_still_drafted() -> None:
    async def run() -> list[tuple[str | None, str]]:
        ctx = Environment()
        voice = ForwardRecordingSpeaking(Script())
        comm = communication.Communication(active_conversation=_conversation(), speaking=voice)
        collector = await _owned(ctx, voice, comm)
        _ = await hsm.dispatch(ctx, comm, _draft("First", "op-1"))
        _ = await hsm.dispatch(ctx, comm, _draft("Second", "op-2"))
        _ = await hsm.dispatch(ctx, comm, _respond("Second.", "op-2"))
        _ = await hsm.dispatch(ctx, comm, _draft(" one", "op-1", offset=5, seq=1))
        _ = await hsm.dispatch(ctx, comm, _respond("First one.", "op-1"))
        await _wait_until(lambda: len(_spoken(collector)) == 2)
        return [(item.id, typing.cast(speaking.OutputData, item.data).spoken) for item in _spoken(collector)]

    assert asyncio.run(run()) == [("op-1", "First one."), ("op-2", "Second.")]


def test_a_draft_for_a_finished_response_is_ignored_and_counted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = sum(point.value for point in metrics.points("bot.communication.response.late.count"))

        async def run() -> tuple[list[str], str]:
            ctx = Environment()
            script = Script()
            voice = _speaking(script)
            comm = communication.Communication(active_conversation=_conversation(), speaking=voice)
            collector = await _owned(ctx, voice, comm)
            _ = await hsm.dispatch(ctx, comm, _respond("Said once.", "once"))
            await _wait_until(lambda: bool(_spoken(collector)))
            await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
            _ = await hsm.dispatch(ctx, comm, _draft("Said once. And again", "once"))
            await asyncio.sleep(0.05)
            return script.calls, comm.state() or ""

        calls, state = asyncio.run(run())
        late = sum(point.value for point in metrics.points("bot.communication.response.late.count")) - before

    assert calls == ["Said once."]
    assert state.endswith("/active")
    assert late == 1


def test_the_response_deadline_outlasts_how_long_speaking_waits_for_more_drafted_text() -> None:
    respond_deadline = typing.cast(
        datetime.timedelta,
        inspect.signature(communication.Communication).parameters["response_idle_timeout"].default,
    )
    draft_deadline = typing.cast(
        datetime.timedelta, inspect.signature(speaking.Speaking).parameters["draft_timeout"].default
    )
    assert respond_deadline == datetime.timedelta(seconds=15)
    assert draft_deadline == datetime.timedelta(seconds=10)
    assert respond_deadline > draft_deadline


def test_a_finished_response_cannot_be_reopened_by_a_late_respond(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        # Points are cumulative for the process: measure the difference this test makes.
        before = sum(point.value for point in metrics.points("bot.communication.response.late.count"))

        async def run() -> tuple[list[str], list[hsm.Event[typing.Any]]]:
            ctx = Environment()
            script = Script()
            speaker = _speaking(script)
            comm = communication.Communication(active_conversation=_conversation(), speaking=speaker)
            collector = await _owned(ctx, speaker, comm)
            _ = await hsm.dispatch(ctx, comm, _respond("Said once.", "once"))
            await _wait_until(lambda: bool(_spoken(collector)))
            await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
            _ = await hsm.dispatch(ctx, comm, _respond("Said once.", "once"))
            await asyncio.sleep(0.05)
            return script.calls, _spoken(collector)

        calls, spoken = asyncio.run(run())
        late = sum(point.value for point in metrics.points("bot.communication.response.late.count")) - before

    assert calls == ["Said once."]
    assert len(spoken) == 1
    assert late == 1


def test_stop_is_offered_only_while_a_response_is_being_spoken() -> None:
    async def run() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        ctx = Environment()
        speaker = _speaking(Script(stall=True))
        comm = communication.Communication(active_conversation=_conversation(), speaking=speaker)
        _ = await _owned(ctx, speaker, comm)
        before = tuple(item.name for item in processing.enabled_call_events(comm))
        _ = await hsm.dispatch(ctx, comm, _respond("This is never spoken.", "response-1"))
        await _wait_until(lambda: "/behavior/responding" in (comm.state() or ""))
        during = tuple(item.name for item in processing.enabled_call_events(comm))
        _ = await hsm.dispatch(ctx, comm, communication.StopEvent.with_data(communication.StopData()))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        after = tuple(item.name for item in processing.enabled_call_events(comm))
        return before, during, after

    before, during, after = asyncio.run(run())
    assert communication.StopEvent.kind == processing.EventKind
    assert communication.StopEvent.name not in before
    assert communication.StopEvent.name in during
    assert communication.StopEvent.name not in after
    assert speaking.StopEvent.name not in during


def test_stop_reaches_speaking_and_the_response_ends_stopped_with_what_was_said() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[conversation.Messages], list[str], str]:
        ctx = Environment()
        conv = RecordingConversation()
        script = Script(stall_on="stuck")
        speaker = _speaking(script, conversation=(conv,))
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        collector = await _owned(ctx, speaker, comm)
        _ = await hsm.dispatch(ctx, comm, _respond("Said first. Then stuck here.", "response-1"))
        await _wait_until(lambda: len(script.calls) == 1)
        # The instant mouth reports the first words played within a few loop turns.
        await asyncio.sleep(0.05)
        _ = await hsm.dispatch(
            ctx, comm, communication.StopEvent.with_data_and_id(communication.StopData(), "operation-7")
        )
        await _wait_until(lambda: bool(_spoken(collector)))
        await _wait_until(lambda: len(conv.outputs) == 1)
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        return (
            _spoken(collector),
            conv.outputs,
            [failure.message for failure in _failures(collector)],
            comm.state() or "",
        )

    spoken, outputs, failures, state = asyncio.run(run())
    assert [item.id for item in spoken] == ["response-1"]
    output = spoken[0].data
    assert isinstance(output, speaking.OutputData)
    assert output.outcome == "stopped"
    assert (output.spoken, output.unspoken) == ("Said first. Then ", "stuck here.")
    assert outputs[0].messages[0].content == "Said first. Then "
    assert failures == []
    assert state.endswith("/active")


def test_an_overlap_fact_alone_never_stops_the_response_being_spoken() -> None:
    """Hearing another voice while speaking is a fact for the bot to judge; nothing seeded stops speech on it."""

    from mosfet.abilities import cognition, listening
    from mosfet.abilities.communication import behaviors

    async def run() -> tuple[str, str, list[str]]:
        ctx = Environment()
        script = Script(stall_on="stuck")
        speaker = _speaking(script)
        comm = communication.Communication(active_conversation=_conversation(), speaking=speaker)
        collector = await _owned(ctx, speaker, comm)
        _ = await hsm.dispatch(ctx, comm, _respond("Said first. Then stuck here.", "response-1"))
        await _wait_until(lambda: len(script.calls) == 1)
        fact = listening.OverlapEvent.with_data(listening.OverlapData(voiced_seconds=0.8, text="wait"))
        handoff = cognition.InputEvent.with_data(cognition.InputData(stimulus=fact))
        for target in (comm, speaker):
            _ = await hsm.dispatch(ctx, target, fact)
            _ = await hsm.dispatch(ctx, target, handoff)
        await asyncio.sleep(0.1)
        return comm.state() or "", speaker.state() or "", [item.name for item in collector.events]

    comm_state, speaker_state, owner_events = asyncio.run(run())
    assert listening.OverlapEvent.name not in behaviors.SPEECH_HEARD_TRIGGERS
    assert "/behavior/responding" in comm_state
    assert "/speaking/" in speaker_state and "/stopping" not in speaker_state
    assert speaking.OutputEvent.name not in owner_events


def test_communication_catalog_via_conversations_keyword() -> None:
    first = _conversation()
    second = _conversation()
    speaker = _speaking()
    ability = communication.Communication(
        conversations=(first, second),
        active_conversation=first,
        speaking=speaker,
    )
    assert ability.model is not None


def test_communication_activate_event_while_inactive() -> None:
    """ActivateEvent is accepted while Communication is inactive (not engaged)."""

    async def run() -> None:
        ctx = hsm.Context()
        first = _conversation()
        second = _conversation()
        speaker = _speaking()
        ability = communication.Communication(
            active_conversation=first,
            conversations=(first, second),
            speaking=speaker,
        )
        model = ability.submodel
        assert model is not None
        _ = await mosfet.started(ctx, ability, model)
        await _wait_until(lambda: "inactive" in (ability.state() or ""))
        _ = await hsm.dispatch(
            ctx,
            ability,
            communication.ActivateEvent.with_data(
                communication.ActivateData(conversation=second),
            ),
        )
        assert "inactive" in (ability.state() or "")

    asyncio.run(run())


def test_communication_activate_ignored_while_engaged() -> None:
    """While behavior/active, Activate has no transition (event-driven refuse)."""

    async def run() -> str:
        ctx = hsm.Context()
        first = _conversation()
        second = _conversation()
        speaker = _speaking()
        ability = communication.Communication(active_conversation=first, speaking=speaker)
        assert ability.model is not None
        _ = await mosfet.started(ctx, ability, ability.model)

        class Owner(hsm.Instance):
            model = mosfet.define("Owner", hsm.initial(hsm.target("/Owner/a")), hsm.state("a"))

        owner = Owner()
        _ = await mosfet.started(ctx, owner, owner.model)
        _ = await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await _wait_until(lambda: "/behavior/active" in (ability.state() or ""))
        _ = await hsm.dispatch(
            ctx,
            ability,
            communication.ActivateEvent.with_data(
                communication.ActivateData(conversation=second),
            ),
        )
        return ability.state() or ""

    state = asyncio.run(run())
    assert "/behavior/active" in state


def test_communication_offers_respond_but_not_input_events() -> None:
    async def run() -> tuple[str, ...]:
        ctx = Environment()
        conv = _conversation()
        speaker = _speaking()
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        await start_ability_tree(ctx, speaker)
        await start_ability_tree(ctx, comm)
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        return tuple(item.name for item in processing.enabled_call_events(comm))

    offered = asyncio.run(run())
    assert communication.RespondEvent.name in offered
    assert communication.InputEvent.name not in offered
    assert speaking.InputEvent.name not in offered


def test_conversation_output_does_not_automatically_speak() -> None:
    async def run() -> list[str]:
        ctx = Environment()
        conv = _conversation()
        encoder = Script()
        speaker = _speaking(encoder)
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        await start_ability_tree(ctx, speaker)
        await start_ability_tree(ctx, comm)
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        _ = await hsm.dispatch(
            ctx,
            comm,
            conversation.OutputEvent.with_data(
                conversation.Messages(
                    messages=(
                        conversation.Message(
                            sequence=0,
                            direction="inbound",
                            source_ids=frozenset({"caller"}),
                            target_ids=frozenset({"bot"}),
                            content="inbound only",
                            content_type="text/plain",
                            provenance=conversation.MessageProvenance(event="test.inbound"),
                        ),
                    )
                )
            ),
        )
        return encoder.calls

    assert asyncio.run(run()) == []


def test_outbound_latest_conversation_output_is_not_forwarded() -> None:
    async def run() -> tuple[list[hsm.Event], list[str]]:
        ctx = Environment()
        conv = _conversation()
        encoder = Script()
        speaker = _speaking(encoder)
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        collector = TerminalCollector()

        await start_ability_tree(ctx, speaker)
        collector_model = collector.model
        assert collector_model is not None
        comm_model = comm.model
        assert comm_model is not None
        _ = await mosfet.started(ctx, collector, collector_model)
        _ = await mosfet.started(ctx, comm, comm_model)
        await comm.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        _ = await hsm.dispatch(
            ctx,
            comm,
            conversation.OutputEvent.with_data(
                conversation.Messages(
                    messages=(
                        conversation.Message(
                            sequence=0,
                            direction="inbound",
                            source_ids=frozenset({"caller"}),
                            target_ids=frozenset({"bot"}),
                            content="inbound",
                            content_type="text/plain",
                            provenance=conversation.MessageProvenance(event="test.inbound"),
                        ),
                        conversation.Message(
                            sequence=1,
                            direction="outbound",
                            source_ids=frozenset({"bot"}),
                            target_ids=frozenset({"caller"}),
                            content="outbound",
                            content_type="text/plain",
                            provenance=conversation.MessageProvenance(event="test.outbound"),
                        ),
                    )
                )
            ),
        )
        forwarded = [
            item
            for item in collector.events
            if item.name == conversation.OutputEvent.name and isinstance(item.data, conversation.Messages)
        ]
        return forwarded, encoder.calls

    forwarded, calls = asyncio.run(run())
    assert forwarded == []
    assert calls == []


def test_conversation_input_enabled_while_active() -> None:
    """RC-1: conversation.input remains tool-offerable while a turn is active."""

    async def run() -> tuple[str, ...]:
        ctx = hsm.Context()
        hang = asyncio.Event()

        class HangingDecoder(decoding.Decoder[typing.Any, str]):
            @typing.override
            async def decode(self, input: typing.Any) -> str:
                del input
                await hang.wait()
                return "done"

        hung = conversation.Conversation(
            turn_detector=turn_detector.TurnDetector(decoder=HangingDecoder()),
        )
        assert hung.model is not None
        _ = await mosfet.started(ctx, hung, hung.model)

        class Owner(hsm.Instance):
            model = mosfet.define("Owner", hsm.initial(hsm.target("/Owner/a")), hsm.state("a"))

        owner = Owner()
        _ = await mosfet.started(ctx, owner, owner.model)
        _ = await hung.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await _wait_until(lambda: (hung.state() or "").endswith("/behavior/inactive"))

        _ = await hsm.dispatch(
            ctx,
            hung,
            hung.input_event.with_data_and_id(
                conversation.TurnData(
                    source_ids=frozenset({"caller"}),
                    target_ids=frozenset({"bot"}),
                    content="in flight",
                    content_type="text/plain",
                ),
                "op-active",
            ),
        )
        await _wait_until(lambda: "/active/running" in (hung.state() or ""))
        offered = tuple(event.name for event in processing.enabled_call_events(hung))
        hang.set()
        await _wait_until(lambda: (hung.state() or "").endswith("/behavior/active/waiting"))
        return offered

    offered = asyncio.run(run())
    assert conversation.InputEvent.name in offered


def test_communication_seed_is_native_and_fresh_per_candidate_run() -> None:
    """The immutable native descriptor ships a factory that returns fresh HSMs."""

    from mosfet.abilities import listening
    from mosfet.abilities.communication import behaviors

    seeded = behaviors.speech_heard_seed()
    first = seeded.factory()
    second = seeded.factory()

    assert seeded.name == behaviors.SPEECH_HEARD_NAME
    assert seeded.triggers == (listening.SpeechEvent.name,)
    assert tuple(field.name for field in dataclasses.fields(seeded)) == (
        "name",
        "triggers",
        "factory",
        "input_adapter",
    )
    assert isinstance(first, behaviors.SpeechHeard)
    assert isinstance(second, behaviors.SpeechHeard)
    assert first is not second


def test_communication_seed_does_not_force_observation() -> None:
    """Reusable seed models leave telemetry observation to their owning runtime."""

    model = communication.behaviors.SpeechHeard.submodel
    assert model is not None
    assert not any(name.endswith("/observer") for name in model.members)


def test_speech_heard_dispatches_one_terminal_output_without_self_output_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SpeechHeard emits its typed terminal wrapper without re-dispatching its output to itself."""

    from mosfet.abilities import cognition, listening
    from mosfet.abilities.hearing import voice
    from mosfet.abilities.communication import behaviors

    async def run() -> tuple[hsm.Event[typing.Any], ...]:
        instance = behaviors.SpeechHeard()
        ctx = hsm.Context()
        await start_ability_tree(ctx, instance)
        dispatched: list[hsm.Event[typing.Any]] = []
        dispatch = hsm.dispatch

        def record_dispatch(
            dispatch_ctx: hsm.Context | None,
            target: hsm.Dispatchable | None,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[bool]:
            if target is instance:
                dispatched.append(event)
            return dispatch(dispatch_ctx, target, event)

        monkeypatch.setattr(hsm, "dispatch", record_dispatch)
        speech = listening.SpeechData(
            content=b"speech",
            voice_detection=voice.detection.ApplyData(segments=()),
            sample_rate_hz=16_000,
            channels=1,
            media_type="audio/pcm",
            source_ids=frozenset({"voice:caller"}),
        )
        await instance.apply(cognition.InputData(stimulus=listening.SpeechEvent.with_data(speech)), ctx=ctx)
        return tuple(dispatched)

    dispatched = asyncio.run(run())
    assert [event.name for event in dispatched] == [
        communication.behaviors.SpeechHeard.input_event.name,
        ability.TerminalOutputEvent.name,
    ]
    terminal = dispatched[1].data
    assert isinstance(terminal, hsm.Event)
    assert terminal.name == communication.behaviors.SpeechHeard.output_event.name


def test_speech_heard_emits_unhandled_without_source_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty source_ids is not identified speech; SpeechHeard must complete unhandled immediately."""

    from mosfet.abilities import cognition, listening
    from mosfet.abilities.hearing import voice
    from mosfet.abilities.communication import behaviors

    async def run() -> tuple[hsm.Event[typing.Any], ...]:
        instance = behaviors.SpeechHeard()
        ctx = hsm.Context()
        await start_ability_tree(ctx, instance)
        dispatched: list[hsm.Event[typing.Any]] = []
        dispatch = hsm.dispatch

        def record_dispatch(
            dispatch_ctx: hsm.Context | None,
            target: hsm.Dispatchable | None,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[bool]:
            if target is instance:
                dispatched.append(event)
            return dispatch(dispatch_ctx, target, event)

        monkeypatch.setattr(hsm, "dispatch", record_dispatch)
        speech = listening.SpeechData(
            content=b"playback-audio",
            voice_detection=voice.detection.ApplyData(segments=()),
            sample_rate_hz=48_000,
            channels=1,
            media_type="audio/pcm",
        )
        await instance.apply(cognition.InputData(stimulus=listening.SpeechEvent.with_data(speech)), ctx=ctx)
        return tuple(dispatched)

    dispatched = asyncio.run(run())
    assert [event.name for event in dispatched] == [
        communication.behaviors.SpeechHeard.input_event.name,
        ability.TerminalOutputEvent.name,
    ]
    terminal = dispatched[1].data
    assert isinstance(terminal, hsm.Event)
    assert terminal.name == communication.behaviors.SpeechHeard.output_event.name
    assert isinstance(terminal.data, processing.OutputData)
    assert terminal.data.handled is False
    assert terminal.data.events == ()


def test_routed_hsm_payload_preserves_nested_stimulus_event_chain() -> None:
    """Communication's typed routed event keeps env → Listening → admit ancestry across JSON hops."""

    from mosfet.abilities import listening
    from mosfet.abilities.hearing import voice
    from mosfet.environment import SoundData, SoundEvent

    sound = SoundData(audio=b"sound", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
    speech = listening.SpeechData(
        content=b"speech",
        voice_detection=voice.detection.ApplyData(segments=()),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({"voice:caller"}),
        parent=StimulusData.from_event(SoundEvent.with_data_and_id(sound, "sound-1")),
    )
    input_data = conversation.TurnData(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset(),
        content=b"speech",
        content_type="audio/pcm",
        sample_rate_hz=16_000,
        channels=1,
        parent=StimulusData.from_event(listening.SpeechEvent.with_data_and_id(speech, "speech-1")),
    )
    communication_event = communication.InputEvent.with_data_and_id(input_data, "communication-1")
    routed = conversation.RoutedInputData(parent=StimulusData.from_event(communication_event))
    routed_event = conversation.RoutedInputEvent.with_data_and_id(routed, "route-1")
    canonical = typing.cast(dict[str, object], event.event_json_value(routed_event))
    assert canonical["name"] == conversation.RoutedInputEvent.name
    assert canonical["id"] == "route-1"
    routed_data = typing.cast(dict[str, object], canonical["data"])
    communication_parent = typing.cast(dict[str, object], routed_data["parent"])
    assert communication_parent["event"] == communication.InputEvent.name
    assert communication_parent["id"] == "communication-1"
    communication_data = typing.cast(dict[str, object], communication_parent["data"])
    assert "content" not in communication_data
    speech_parent = typing.cast(dict[str, object], communication_data["parent"])
    assert speech_parent["event"] == listening.SpeechEvent.name
    assert speech_parent["id"] == "speech-1"
    speech_data = typing.cast(dict[str, object], speech_parent["data"])
    assert "content" not in speech_data
    sound_parent = typing.cast(dict[str, object], speech_data["parent"])
    assert sound_parent["event"] == "environment.sound"
    assert sound_parent["id"] == "sound-1"
    sound_data = typing.cast(dict[str, object], sound_parent["data"])
    assert "audio" not in sound_data
