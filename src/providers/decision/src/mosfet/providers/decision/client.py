"""The decisions wire format and its HTTP client: the transport boundary to a decision server.

A decision server answers named questions about one state document with calibrated
probabilities: ``POST {base_url}/v1/systemone`` with ``{"state": ..., "questions": {name: question}}``
answers ``{"model": ..., "answers": {name: answer}, "usage": {...}}``. anygame's decide serves it
locally, and a hosted server that keeps to the same format works the same way. mosfet asks only
``choice`` questions (one option of a named set), so the models here carry that kind only.

:class:`Client` is a plain collaborator, not a state machine. Each :meth:`Client.decide` opens its
own ``httpx.AsyncClient`` and closes it before returning, so cancelling the turn that awaits it
closes its request too. Every failure is one :class:`Error` that names the HTTP status or the
error type, never a body, a header or a value.
"""

from __future__ import annotations

import typing

import httpx
import pydantic

_PATH: typing.Final = "/v1/systemone"


class Error(RuntimeError):
    """A decision server could not be reached, refused the request, or answered something that is not a decision."""


class Choice(pydantic.BaseModel):
    """One ``choice`` question: pick exactly one of the named options, judged against the state."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    type: typing.Literal["choice"] = pydantic.Field(
        default="choice",
        description="The question kind; mosfet asks choice questions only.",
        examples=["choice"],
    )
    instructions: str | None = pydantic.Field(
        default=None,
        description="What the question asks of the state, in plain words; None leaves it to the options.",
        examples=["Choose the single criterion that answers this turn."],
    )
    criteria: dict[str, str | None] = pydantic.Field(
        min_length=1,
        description="The options by label, each with its description (None: the label says it all).",
        examples=[{"phone.answer_call": "Answer the ringing call.", "__unhandled__": "None of these."}],
    )


class Request(pydantic.BaseModel):
    """One decision request: a state document and the named questions asked about it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    state: pydantic.JsonValue = pydantic.Field(
        description="The state the questions are judged against: a string, an object or an array.",
        examples=[{"turn": {"envelope": {"name": "environment.sound"}, "data": {"kind": "phone.ringing"}}}],
    )
    questions: dict[str, Choice] = pydantic.Field(
        min_length=1,
        description="The questions by name; each answer comes back under the same name.",
    )
    model: str | None = pydantic.Field(
        default=None,
        description="The model a server that serves several should answer with; None lets the server choose.",
        examples=["pplx-decider-flash"],
    )


class ChoiceAnswer(pydantic.BaseModel):
    """The answer to one choice question: the chosen label, its confidence and every option's probability."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="ignore")

    type: typing.Literal["choice"] = pydantic.Field(
        default="choice",
        description="The question kind this answers.",
        examples=["choice"],
    )
    choice: str = pydantic.Field(
        description="The label of the chosen option.",
        examples=["phone.answer_call"],
    )
    confidence: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="How confident the server is in the choice, from 0 to 1.",
        examples=[0.7],
    )
    probabilities: dict[str, float] = pydantic.Field(
        default_factory=dict[str, float],
        description="Each option's calibrated probability, by label.",
        examples=[{"phone.answer_call": 0.8, "__unhandled__": 0.2}],
    )


class Response(pydantic.BaseModel):
    """A decision server's answer. Usage and any other keys are ignored."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="ignore")

    model: str | None = pydantic.Field(
        default=None,
        description="The model that answered, when the server names it.",
        examples=["pplx-decider-flash"],
    )
    answers: dict[str, ChoiceAnswer] = pydantic.Field(
        description="The answers by question name.",
    )


class Client:
    """Asks the decision server at ``base_url``: one ``POST /v1/systemone`` per :meth:`decide`.

    ``api_key`` is sent as ``Authorization: Bearer <key>`` when given, and nothing is sent without
    it. ``timeout`` (seconds) bounds the HTTP call; None keeps httpx's default. ``transport``
    replaces the network (tests use ``httpx.MockTransport``).
    """

    _base_url: str
    _api_key: str | None
    _timeout: float | None
    _transport: httpx.AsyncBaseTransport | None

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Raises ``ValueError`` for a blank ``base_url``, an empty ``api_key`` or a ``timeout`` that is not positive."""

        if not base_url.strip():
            raise ValueError("base_url must name the decision server, for example http://127.0.0.1:8102.")
        if api_key is not None and not api_key:
            raise ValueError("api_key must not be empty; pass None to send no key.")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport

    async def decide(self, request: Request) -> Response:
        """Send ``request`` and return the server's answer.

        Raises :class:`Error` when the server cannot be reached, answers anything but a 2xx, or
        answers a body that is not a decision.
        """

        body = request.model_dump(mode="json")
        if request.model is None:
            del body["model"]
        headers = {} if self._api_key is None else {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with (
                httpx.AsyncClient(transport=self._transport)
                if self._timeout is None
                else httpx.AsyncClient(transport=self._transport, timeout=self._timeout)
            ) as http:
                answered = await http.post(f"{self._base_url}{_PATH}", json=body, headers=headers)
        except httpx.HTTPError as error:
            raise Error(f"Decision server unreachable: {type(error).__name__}.") from error
        if not answered.is_success:
            raise Error(f"Decision server answered HTTP {answered.status_code}.")
        try:
            return Response.model_validate_json(answered.content)
        except pydantic.ValidationError as error:
            raise Error(
                f"Decision server answered something that is not a decision: {type(error).__name__}."
            ) from error


__all__ = [
    "Choice",
    "ChoiceAnswer",
    "Client",
    "Error",
    "Request",
    "Response",
]
