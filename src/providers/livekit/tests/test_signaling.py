from __future__ import annotations

import json

import pytest
from livekit import rtc

from bot.devices import phone as phone_device
from bot.providers.livekit import MappingDirectory
from bot.providers.livekit import signaling


def test_the_four_methods_are_the_whole_vocabulary() -> None:
    """Every method a phone answers is registered, and each one is distinct on the wire."""

    assert signaling.Methods == (
        signaling.SetupMethod,
        signaling.AcceptMethod,
        signaling.DeclineMethod,
        signaling.ByeMethod,
    )
    assert len(set(signaling.Methods)) == 4


def test_a_directory_resolves_a_dial_target_to_the_identity_that_answers_it() -> None:
    directory = MappingDirectory({"reception": "agent-b"})

    assert directory.resolve(phone_device.TransferTarget(kind="device", value="reception")) == "agent-b"
    assert directory.resolve(phone_device.TransferTarget(kind="device", value="nobody")) is None


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (rtc.RpcError.ErrorCode.RECIPIENT_NOT_FOUND, "remote_unavailable"),
        (rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD, "remote_unavailable"),
        (rtc.RpcError.ErrorCode.RECIPIENT_DISCONNECTED, "remote_unavailable"),
        (rtc.RpcError.ErrorCode.CONNECTION_TIMEOUT, "signaling_failed"),
        (rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT, "signaling_failed"),
        (rtc.RpcError.ErrorCode.SEND_FAILED, "signaling_failed"),
    ],
)
def test_rpc_failures_normalize_to_the_phone_failure_vocabulary(
    code: rtc.RpcError.ErrorCode,
    expected: str,
) -> None:
    """Nobody there is a fact about the far end; a lost message is not."""

    assert signaling.failure_kind(rtc.RpcError(code, "boom")) == expected


def test_an_invocation_carries_the_identity_the_sfu_authenticated() -> None:
    """Caller identity comes from the transport. A payload could only ever claim one."""

    caller_identity, message = signaling.invocation(
        rtc.RpcInvocationData(
            request_id="RQ_1",
            caller_identity="agent-a",
            payload=json.dumps({"call_id": "livekit:1"}),
            response_timeout=5.0,
        )
    )

    assert caller_identity == "agent-a"
    assert message == signaling.MessageData(call_id="livekit:1")


@pytest.mark.parametrize(
    "payload",
    ["not json", json.dumps({}), json.dumps({"call_id": ""})],
)
def test_a_message_that_is_not_call_setup_is_refused_toward_the_sender(payload: str) -> None:
    """Rejecting on the wire tells the sender; failing quietly would leave them waiting."""

    with pytest.raises(rtc.RpcError):
        _ = signaling.invocation(
            rtc.RpcInvocationData(
                request_id="RQ_1",
                caller_identity="agent-a",
                payload=payload,
                response_timeout=5.0,
            )
        )


def test_an_unidentified_caller_is_refused() -> None:
    with pytest.raises(rtc.RpcError):
        _ = signaling.invocation(
            rtc.RpcInvocationData(
                request_id="RQ_1",
                caller_identity="",
                payload=json.dumps({"call_id": "livekit:1"}),
                response_timeout=5.0,
            )
        )
