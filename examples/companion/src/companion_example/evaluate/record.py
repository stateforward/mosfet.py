"""The record of an evaluation: what was said, what each exchange measured, and how the judge graded it.

Times are seconds since the evaluation began. The companion's lines are what the lab's speech to text heard it say; the
people's are what the persona wrote for them.
"""

import typing

import pydantic

from companion_example.evaluate import scenarios, turns

type Ended = typing.Literal["done", "turns_used", "persona_unanswered"]
"""How a scenario's conversation ended: the persona said it was done, the people used every turn, or the persona
could not be got to answer."""

COMPANION: typing.Final = "companion"
"""The speaker id of everything the companion says."""


class Utterance(pydantic.BaseModel):
    """One thing said in the room, by the companion or by one of the people."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    scenario: str = pydantic.Field(description="The scenario it was said in.", examples=["stranger-arrives"])
    exchange: int = pydantic.Field(ge=0, description="The exchange it belongs to, from 0.")
    speaker: str = pydantic.Field(
        description="`companion`, or the id of the person who said it.", examples=["companion"]
    )
    name: str = pydantic.Field(description="Who said it, as people call them.", examples=["Jasper", "Maya"])
    text: str = pydantic.Field(description="What was said; empty when the companion said nothing.")
    note: str | None = pydantic.Field(
        default=None,
        description="What a reader needs to know about it: no reply, talked over, a cut-in.",
        examples=["Jasper said nothing within 30 s"],
    )
    started_s: float | None = pydantic.Field(default=None, description="When it began, if anything was heard.")
    ended_s: float | None = pydantic.Field(default=None, description="When it ended.")


class Exchange(pydantic.BaseModel):
    """One exchange's measures, and where its audio is kept."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    scenario: str
    exchange: int = pydantic.Field(ge=0)
    speaker: str = pydantic.Field(description="The person whose line began the exchange.")
    measures: turns.Measures
    audio: str = pydantic.Field(description="The exchange as the room heard it: a WAV file in the output directory.")


class Grade(pydantic.BaseModel):
    """The judge's grade on one criterion."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"criterion": "warmth", "score": 4, "rationale": "Friendly and patient throughout."}]
        },
    )

    criterion: scenarios.Criterion = pydantic.Field(description="The criterion graded.")
    score: int = pydantic.Field(ge=1, le=5, description="1 poor, 2 weak, 3 adequate, 4 good, 5 excellent.")
    rationale: str = pydantic.Field(
        min_length=1, max_length=600, description="One or two sentences on why, citing what the companion said."
    )


class Verdict(pydantic.BaseModel):
    """The judge's grades on one conversation: each criterion, and overall."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    grades: tuple[Grade, ...] = pydantic.Field(min_length=1, description="One grade per criterion, each once.")
    overall: int = pydantic.Field(ge=1, le=5, description="The conversation as a whole, 1 to 5.")
    rationale: str = pydantic.Field(min_length=1, max_length=800, description="A short reason for the overall grade.")


class Ungraded(pydantic.BaseModel):
    """A conversation the judge could not grade, and why."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    reason: str = pydantic.Field(min_length=1, examples=["the judge's answer failed the schema twice"])


class ScenarioResult(pydantic.BaseModel):
    """One scenario's conversation with the companion, measured and graded."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    criteria: tuple[scenarios.Criterion, ...]
    people: tuple[scenarios.Person, ...]
    ended: Ended
    utterances: tuple[Utterance, ...]
    exchanges: tuple[Exchange, ...]
    graded: Verdict | Ungraded


class Judge(pydantic.BaseModel):
    """Which model judged, and whether it is the companion's own (a bias every report states)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    model: str = pydantic.Field(examples=["gemma-4-26b-a4b"])
    shares_companions_model: bool = pydantic.Field(
        description="The judge is the model the companion reasons on, so it may favour answers like its own."
    )


class Results(pydantic.BaseModel):
    """Everything one evaluation found."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    judge: Judge
    scenarios: tuple[ScenarioResult, ...]


__all__ = [
    "COMPANION",
    "Ended",
    "Exchange",
    "Grade",
    "Judge",
    "Results",
    "ScenarioResult",
    "Ungraded",
    "Utterance",
    "Verdict",
]
