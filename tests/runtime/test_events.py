import typing

import hsm
import pydantic
import pytest
from mosfet import event, runtime

_EVENTS: tuple[hsm.Event[typing.Any], ...] = (
    runtime.StopEvent,
    runtime.ActorsStartedEvent,
    runtime.ActorsFailedEvent,
    runtime.ActorsStoppedEvent,
    runtime.SupervisorsStartedEvent,
    runtime.SupervisorsFailedEvent,
    runtime.SupervisorRunningEvent,
    runtime.SupervisorExhaustedEvent,
    runtime.SupervisorStopEvent,
    runtime.SupervisorStoppedEvent,
)


def test_no_runtime_event_is_offered_to_a_model_as_a_tool() -> None:
    assert [runtime_event.name for runtime_event in _EVENTS if runtime_event.kind == event.EventKind] == []
    assert all(runtime_event.name.startswith("runtime.") for runtime_event in _EVENTS)


def test_stop_names_its_signal_exactly_when_a_signal_asked_for_it() -> None:
    assert runtime.StopData(reason="signal", signal="SIGINT").signal == "SIGINT"
    assert runtime.StopData(reason="request").signal is None
    with pytest.raises(pydantic.ValidationError):
        _ = runtime.StopData(reason="signal")
    with pytest.raises(pydantic.ValidationError):
        _ = runtime.StopData(reason="request", signal="SIGTERM")


@pytest.mark.parametrize(("reason", "code"), [("stopped", 0), ("forced", 1), ("exhausted", 70), ("boot_failed", 78)])
def test_exit_code_follows_its_reason(
    reason: typing.Literal["stopped", "forced", "exhausted", "boot_failed"], code: int
) -> None:
    assert runtime.ExitedData.for_reason(reason).code == code
    with pytest.raises(pydantic.ValidationError):
        _ = runtime.ExitedData.model_validate({"code": 0 if code else 1, "reason": reason})


def test_supervisor_reports_carry_bounded_bot_names() -> None:
    assert runtime.SupervisorData(name="builder", attempt=1).name == "builder"
    with pytest.raises(pydantic.ValidationError):
        _ = runtime.SupervisorData(name="Builder", attempt=1)
    with pytest.raises(pydantic.ValidationError):
        _ = runtime.SupervisorData(name="builder", attempt=0)
    with pytest.raises(pydantic.ValidationError):
        _ = runtime.SupervisorStoppedData(name="builder", outcome=typing.cast(typing.Any, "lost"))
