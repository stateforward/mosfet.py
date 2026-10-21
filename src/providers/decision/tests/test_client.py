"""Client contract tests over httpx.MockTransport: the decisions wire format, no network."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from mosfet.providers import decision


def _request() -> decision.Request:
    return decision.Request(
        state={"turn": {"envelope": {"name": "environment.sound"}, "data": {"kind": "phone.ringing"}}},
        questions={
            "selection": decision.Choice(
                instructions="Choose from the menu.",
                criteria={"phone.answer_call": "Answer the ringing call.", "__reasoning__": None},
            )
        },
    )


def _answer(choice: str = "phone.answer_call") -> dict[str, object]:
    return {
        "model": "pplx-decider-flash",
        "answers": {
            "selection": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.7,
                "probabilities": {"phone.answer_call": 0.8, "__reasoning__": 0.2},
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }


class _Server:
    """A decision server stand-in: records every request and answers what it was told to."""

    requests: list[httpx.Request]
    _response: httpx.Response

    def __init__(self, response: httpx.Response) -> None:
        self.requests = []
        self._response = response

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._response


def _decide(client: decision.Client) -> decision.Response:
    return asyncio.run(client.decide(_request()))


def test_a_decision_is_one_post_of_the_wire_format_to_v1_systemone() -> None:
    server = _Server(httpx.Response(200, json=_answer()))
    client = decision.Client(base_url="http://decide.test:8102", transport=httpx.MockTransport(server))

    response = _decide(client)

    (sent,) = server.requests
    assert (sent.method, str(sent.url)) == ("POST", "http://decide.test:8102/v1/systemone")
    assert json.loads(sent.content) == {
        "state": {"turn": {"envelope": {"name": "environment.sound"}, "data": {"kind": "phone.ringing"}}},
        "questions": {
            "selection": {
                "type": "choice",
                "instructions": "Choose from the menu.",
                "criteria": {"phone.answer_call": "Answer the ringing call.", "__reasoning__": None},
            }
        },
    }
    assert response.model == "pplx-decider-flash"
    answer = response.answers["selection"]
    assert (answer.choice, answer.confidence, answer.probabilities) == (
        "phone.answer_call",
        0.7,
        {"phone.answer_call": 0.8, "__reasoning__": 0.2},
    )


def test_the_requested_model_rides_the_request_only_when_named() -> None:
    server = _Server(httpx.Response(200, json=_answer()))
    client = decision.Client(base_url="http://decide.test", transport=httpx.MockTransport(server))

    _ = asyncio.run(client.decide(_request().model_copy(update={"model": "decider-v2"})))

    (sent,) = server.requests
    assert json.loads(sent.content)["model"] == "decider-v2"


def test_no_key_sends_no_authorization_and_a_key_is_a_bearer() -> None:
    """D-R-32: an optional api_key. A local server is sent no placeholder bearer at all."""

    server = _Server(httpx.Response(200, json=_answer()))
    transport = httpx.MockTransport(server)

    _ = _decide(decision.Client(base_url="http://decide.test", transport=transport))
    _ = _decide(decision.Client(base_url="http://decide.test", api_key="hosted-key", transport=transport))

    unkeyed, keyed = server.requests
    assert "authorization" not in unkeyed.headers
    assert keyed.headers["authorization"] == "Bearer hosted-key"


def test_an_answer_without_a_model_is_still_a_decision() -> None:
    answer = _answer()
    del answer["model"]
    server = _Server(httpx.Response(200, json=answer))

    response = _decide(decision.Client(base_url="http://decide.test", transport=httpx.MockTransport(server)))

    assert response.model is None


@pytest.mark.parametrize("status", [400, 422, 503])
def test_a_refusal_is_an_error_naming_its_status_and_never_its_body(status: int) -> None:
    """decide answers 400 for a malformed request and 422 for an engine error (a context overflow, run 7)."""

    server = _Server(httpx.Response(status, json={"detail": "ValueError: secret prompt text overflowed"}))
    client = decision.Client(base_url="http://decide.test", transport=httpx.MockTransport(server))

    with pytest.raises(decision.Error) as raised:
        _ = _decide(client)

    assert str(raised.value) == f"Decision server answered HTTP {status}."
    assert "secret" not in str(raised.value)


def test_an_unreachable_server_is_an_error_naming_the_transport_failure() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused to 10.0.0.7", request=request)

    client = decision.Client(base_url="http://decide.test", transport=httpx.MockTransport(refuse))

    with pytest.raises(decision.Error) as raised:
        _ = _decide(client)

    assert str(raised.value) == "Decision server unreachable: ConnectError."


@pytest.mark.parametrize(
    "body",
    [b"not json at all", json.dumps({"answers": {"selection": {"type": "choice"}}}).encode()],
    ids=["not-json", "not-a-decision"],
)
def test_a_body_that_is_not_a_decision_is_an_error_naming_only_its_type(body: bytes) -> None:
    server = _Server(httpx.Response(200, content=body))
    client = decision.Client(base_url="http://decide.test", transport=httpx.MockTransport(server))

    with pytest.raises(decision.Error) as raised:
        _ = _decide(client)

    assert str(raised.value) == "Decision server answered something that is not a decision: ValidationError."


def test_a_client_needs_a_server_and_a_key_that_is_not_empty() -> None:
    with pytest.raises(ValueError, match="base_url"):
        _ = decision.Client(base_url=" ")
    with pytest.raises(ValueError, match="api_key"):
        _ = decision.Client(base_url="http://decide.test", api_key="")
