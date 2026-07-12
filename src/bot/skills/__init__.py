"""Bot skills primitives for stateforward.bot.

A skill is a learned proficiency built on one or more abilities. Markdown
skills are stored instructional source, while Starlark skills are stored
behavior source that can train into HSM-visible execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import typing

import pydantic

from bot import abilities

SkillSourceKind = typing.Literal["markdown", "starlark"]

class SkillSource(pydantic.BaseModel):
    """Stored source for a skill."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Stored skill source. Markdown is instructional source; Starlark is behavior source.",
            "examples": [
                {
                    "kind": "markdown",
                    "source": "# Summarize support calls\nPrefer concise summaries with open questions.",
                },
                {
                    "kind": "starlark",
                    "source": 'kind = "skill_behavior"\nname = "summarize_support_call"',
                },
            ],
        },
    )

    kind: SkillSourceKind = pydantic.Field(
        description="SourceData representation for the skill.",
        examples=["markdown", "starlark"],
    )
    source: str = pydantic.Field(
        min_length=1,
        description=(
            "Plain-text skill source stored as the canonical editable representation. Starlark source is "
            "compiled by stateforward.bot and does not imply a public host ABI."
        ),
        examples=["# Summarize support calls\nPrefer concise summaries with open questions."],
    )

@dataclass(frozen=True)
class Skill:
    """Learned proficiency built on one or more abilities.

    Answers: What has the system learned to do well?
    """

    name: str
    abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...] = ()
    description: str = ""
    examples: tuple[str, ...] = ()
    source: SkillSource | None = None

__all__ = ["Skill", "SkillSource", "SkillSourceKind"]
