import datetime

import pydantic
import pytest
from mosfet import runtime

_S = datetime.timedelta(seconds=1)


def test_defaults_keep_every_bot_stop_inside_the_grace() -> None:
    policy = runtime.Policy()

    assert policy.stop_grace == 8 * _S
    assert policy.bot_stop_timeout == 4 * _S
    assert policy.bot_release_timeout == 1 * _S
    assert policy.start_timeout == 120 * _S
    assert policy.max_restarts == 5
    assert policy.backoff_initial == 1 * _S
    assert policy.backoff_max == 60 * _S
    assert policy.stable_after == 60 * _S


@pytest.mark.parametrize(
    "field",
    [
        "stop_grace",
        "bot_stop_timeout",
        "bot_release_timeout",
        "start_timeout",
        "backoff_initial",
        "backoff_max",
        "stable_after",
    ],
)
def test_durations_must_be_positive(field: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = runtime.Policy.model_validate({field: datetime.timedelta()})


@pytest.mark.parametrize("max_restarts", [0, runtime.MAX_RESTARTS + 1])
def test_restart_budget_is_bounded(max_restarts: int) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = runtime.Policy(max_restarts=max_restarts)


def test_backoff_cap_is_at_least_the_initial_delay() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = runtime.Policy(backoff_initial=10 * _S, backoff_max=5 * _S)


def test_a_bot_stop_must_fit_strictly_inside_the_grace() -> None:
    _ = runtime.Policy(stop_grace=8 * _S, bot_stop_timeout=4 * _S, bot_release_timeout=3 * _S)
    with pytest.raises(pydantic.ValidationError):
        _ = runtime.Policy(stop_grace=8 * _S, bot_stop_timeout=4 * _S, bot_release_timeout=4 * _S)


def test_policy_is_frozen() -> None:
    with pytest.raises(pydantic.ValidationError):
        setattr(runtime.Policy(), "max_restarts", 2)
