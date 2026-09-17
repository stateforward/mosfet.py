import pydantic
import pytest

from mosfet import abilities
from mosfet.skills import Skill, SkillSource


def test_skill_defines_learned_proficiency_built_on_abilities() -> None:
    language_generation = abilities.Ability[object, object]()
    speech_production = abilities.Ability[object, object]()
    source = SkillSource(kind="markdown", source="# Speaking English\nProduce fluent English speech.")
    skill = Skill(
        name="Speaking English",
        abilities=(language_generation, speech_production),
        description="Produce fluent English speech.",
        examples=("answer a phone call in English",),
        source=source,
    )

    assert skill.name == "Speaking English"
    assert skill.abilities == (language_generation, speech_production)
    assert skill.description == "Produce fluent English speech."
    assert skill.examples == ("answer a phone call in English",)
    assert skill.source == source
    assert Skill.__doc__ is not None
    assert "What has the system learned to do well?" in Skill.__doc__


def test_skill_source_supports_markdown_and_starlark_source_tiers() -> None:
    markdown_source = SkillSource(kind="markdown", source="# Escalate politely\nAsk before interrupting.")
    starlark_source = SkillSource(
        kind="starlark",
        source='kind = "skill_behavior"\nname = "escalate_politely"',
    )

    assert markdown_source.kind == "markdown"
    assert markdown_source.source.startswith("# Escalate politely")
    assert starlark_source.kind == "starlark"
    assert "skill_behavior" in starlark_source.source


def test_skill_source_rejects_empty_or_unknown_source_tiers() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = SkillSource(kind="markdown", source="")

    with pytest.raises(pydantic.ValidationError):
        _ = SkillSource.model_validate({"kind": "python", "source": "print('not a skill tier')"})
