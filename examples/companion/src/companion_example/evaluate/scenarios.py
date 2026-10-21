"""What the evaluator's people try to do with the companion: scenarios, as data.

A scenario is what one conversation with the companion is for, never what is said in it. It names the people in the room
(each with a catalog voice of the lab's speech engine), the goal the persona plays them toward, how many times at
most they speak, the pauses the room falls quiet for (a person stepping out and coming back) and the turns on which
a person cuts in over the companion, and which criteria the judge grades it on. Every line a person says is written by
the persona at run time; nothing the companion says is ever written anywhere.

The built-in scenarios are package data (``scenarios.json``), read with :func:`builtin`. Their people call the companion
:data:`COMPANION_NAME`.
"""

import collections.abc
import importlib.resources
import typing

import pydantic
from mosfet.abilities.vocal import voice

type Criterion = typing.Literal["relevance", "helpfulness", "warmth", "honesty", "memory", "child_appropriate"]
"""A criterion the judge grades a conversation on.

``relevance``: the companion answered what was said. ``helpfulness``: it moved the person's goal along. ``warmth``: it
sounded like the companion (warm, patient, curious, honest). ``honesty``: it invented no facts. ``memory``: it acted on
knowing someone it had met. ``child_appropriate``: what it said suits a young child.
"""

COMPANION_NAME: typing.Final = "Jasper"
"""What the evaluator's household calls its companion. It is the people's, never the bot's: the companion is born
nameless and is never told it, so it can only learn it from them. Lab speech to text hears it reliably.
"""

CRITERIA: typing.Final[tuple[Criterion, ...]] = (
    "relevance",
    "helpfulness",
    "warmth",
    "honesty",
    "memory",
    "child_appropriate",
)
"""Every criterion, in the order reports list them."""

Slug = typing.Annotated[
    str,
    pydantic.StringConstraints(pattern=r"^[a-z][a-z0-9-]{1,47}$"),
    pydantic.Field(
        description="An id: 2 to 48 lower-case letters, digits or hyphens, starting with a letter.",
        examples=["stranger-arrives", "maya"],
    ),
]


class Person(pydantic.BaseModel):
    """Someone in the room the persona plays."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "id": "maya",
                    "name": "Maya",
                    "voice": "warm",
                    "about": "A woman in her thirties who has just moved into the house; relaxed and chatty.",
                }
            ]
        },
    )

    id: Slug
    name: str = pydantic.Field(
        min_length=1, max_length=40, description="What the person is called; theirs to say or not.", examples=["Maya"]
    )
    voice: str = pydantic.Field(
        min_length=1,
        max_length=256,
        description="The catalog voice the person speaks in, by its key on the lab's speech engine.",
        examples=["warm", "canvas"],
    )
    child: bool = pydantic.Field(
        default=False, description="Whether the person is a child: the persona speaks like one.", examples=[True]
    )
    about: str = pydantic.Field(
        min_length=1,
        max_length=400,
        description="Who the person is and how they talk, for the persona.",
        examples=["A curious seven-year-old who asks why about everything."],
    )


class Break(pydantic.BaseModel):
    """A while the room is quiet between two of the people's turns: someone stepped out and comes back."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"after_turn": 2, "seconds": 20, "note": "You went to the kitchen and are coming back."}]
        },
    )

    after_turn: int = pydantic.Field(
        ge=1, description="How many of the people's turns come before the break.", examples=[2]
    )
    seconds: float = pydantic.Field(gt=0, le=600, description="How long the room is quiet.", examples=[20])
    note: str = pydantic.Field(
        min_length=1,
        max_length=300,
        description="What happened in the break, as the persona is told it.",
        examples=["You stepped out for a while and are coming back into the room."],
    )


class Scenario(pydantic.BaseModel):
    """One conversation with the companion: who is in the room, what they want, and what it is graded on.

    Raises:
        pydantic.ValidationError: two people share an id or a voice, a criterion is listed twice,
            ``child_appropriate`` is listed without a child or a child is in the room without it, or a break or an
            interrupting turn falls outside the scenario's turns.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    id: Slug
    title: str = pydantic.Field(min_length=1, max_length=80, examples=["A stranger arrives"])
    goal: str = pydantic.Field(
        min_length=1,
        max_length=1000,
        description="What the people are trying to do, for the persona; never anything the companion should say.",
        examples=["You have just come home for the first time. Say hello and chat a little."],
    )
    people: tuple[Person, ...] = pydantic.Field(min_length=1, max_length=3)
    max_turns: int = pydantic.Field(ge=1, le=8, description="The most times the people speak.", examples=[4])
    criteria: tuple[Criterion, ...] = pydantic.Field(
        min_length=1, description="What the judge grades the conversation on.", examples=[["relevance", "warmth"]]
    )
    breaks: tuple[Break, ...] = pydantic.Field(default=(), description="Quiet spells between turns.")
    interrupt_turns: tuple[int, ...] = pydantic.Field(
        default=(),
        description="The people's turns (from 0) on which a person also cuts in while the companion answers.",
        examples=[[1]],
    )

    @pydantic.model_validator(mode="after")
    def _holds_together(self) -> typing.Self:
        if len({person.id for person in self.people}) != len(self.people):
            raise ValueError("two people share an id")
        if len({person.voice for person in self.people}) != len(self.people):
            raise ValueError("two people share a voice")
        if len(set(self.criteria)) != len(self.criteria):
            raise ValueError("a criterion is listed twice")
        if ("child_appropriate" in self.criteria) != any(person.child for person in self.people):
            raise ValueError("child_appropriate is graded exactly when a child is in the room")
        if any(not 0 <= turn < self.max_turns for turn in self.interrupt_turns):
            raise ValueError("an interrupting turn is not one of the scenario's turns")
        if any(not 1 <= pause.after_turn < self.max_turns for pause in self.breaks):
            raise ValueError("a break does not fall between two of the scenario's turns")
        if len({pause.after_turn for pause in self.breaks}) != len(self.breaks):
            raise ValueError("two breaks fall at the same turn")
        return self


class Scenarios(pydantic.BaseModel):
    """Every scenario the evaluator holds, in the order it runs them.

    Raises:
        pydantic.ValidationError: two scenarios share an id.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    scenarios: tuple[Scenario, ...] = pydantic.Field(min_length=1)

    @pydantic.model_validator(mode="after")
    def _unique(self) -> typing.Self:
        if len({scenario.id for scenario in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario ids must be unique")
        return self


def builtin() -> Scenarios:
    """The evaluator's built-in scenarios, read from this package's ``scenarios.json`` (file I/O)."""

    text = (
        importlib.resources.files("companion_example.evaluate").joinpath("scenarios.json").read_text(encoding="utf-8")
    )
    return Scenarios.model_validate_json(text)


def chosen(held: Scenarios, ids: collections.abc.Iterable[str]) -> tuple[Scenario, ...]:
    """The scenarios ``ids`` names, in the order ``held`` runs them; every scenario when it names none.

    Raises:
        ValueError: an id names no scenario ``held`` has.
    """

    wanted = set(ids)
    unknown = sorted(wanted - {scenario.id for scenario in held.scenarios})
    if unknown:
        raise ValueError(f"no scenario {', '.join(unknown)}; there are {', '.join(s.id for s in held.scenarios)}")
    return tuple(scenario for scenario in held.scenarios if not wanted or scenario.id in wanted)


def check_voices(run: collections.abc.Iterable[Scenario], catalog: collections.abc.Iterable[voice.Entry]) -> None:
    """Check every person's voice is one the speech engine lists, and not a designed voice.

    The companion designs its own voice on the same engine, so a designed (``custom``) voice may be the companion's: no
    person ever speaks in one.

    Raises:
        ValueError: naming the voice the engine does not list, or that it lists as designed.
    """

    listed = {entry.ref.key: entry for entry in catalog}
    for scenario in run:
        for person in scenario.people:
            entry = listed.get(person.voice)
            if entry is None:
                raise ValueError(f"the speech engine lists no voice {person.voice!r} ({scenario.id}, {person.id})")
            if entry.labels.get("custom") == "true":
                raise ValueError(
                    f"voice {person.voice!r} is a designed voice, which may be the companion's own ({scenario.id}, {person.id})"
                )


__all__ = [
    "COMPANION_NAME",
    "CRITERIA",
    "Break",
    "Criterion",
    "Person",
    "Scenario",
    "Scenarios",
    "Slug",
    "builtin",
    "check_voices",
    "chosen",
]
