import hsm

from mosfet import abilities
from mosfet.abilities import embedding, identification


def test_ability_defines_operation_contract() -> None:
    ability = abilities.Ability[object, object]()

    assert isinstance(ability, hsm.Instance)
    assert abilities.Ability.input_event.name == "bot.ability.input"
    assert abilities.Ability.output_event.name == "bot.ability.output"
    assert abilities.Ability.__doc__ is not None
    assert "What operations can the system perform?" in abilities.Ability.__doc__


def test_abilities_export_the_embedding_package_without_an_events_module() -> None:
    assert abilities.embedding is embedding
    assert "embedding" in abilities.__all__
    assert "events" not in embedding.__all__
    assert {"Embedding", "Embedder", "InputData", "OutputData", "FailureData", "InputEvent"} <= set(embedding.__all__)


def test_abilities_export_the_identification_package() -> None:
    assert abilities.identification is identification
    assert "identification" in abilities.__all__
    assert "events" not in identification.__all__
    assert {"Registry", "Identifier", "Matcher", "Store", "InProcessStore", "Print", "decide", "ref"} <= set(
        identification.__all__
    )
