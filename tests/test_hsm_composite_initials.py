"""HSM-INIT-001: named composites declare explicit nested initials."""

from __future__ import annotations

import hsm

from bot.abilities.conversation.text import TextConversation
from bot.abilities.listening.listening import Listening
from bot.abilities.participating.participating import Participating
from bot.abilities.reading.reading import Reading


def _behavior_model(machine_cls: type[hsm.Instance]) -> hsm.Model:
    """Return the behavior model that owns the domain composites (not lifecycle wrapper)."""

    submodel = getattr(machine_cls, "submodel", None)
    if isinstance(submodel, hsm.Model):
        return submodel
    return machine_cls.model


def _composite_state(model: hsm.Model, *, name: str) -> tuple[str, object]:
    """Return path + composite state element whose path ends with ``/{name}``."""

    matches = [
        (path, element)
        for path, element in model.members.items()
        if path.endswith(f"/{name}") and not path.endswith(f"/{name}/.initial")
    ]
    assert matches, f"composite state {name!r} not found in model members"
    path, element = sorted(matches, key=lambda item: item[0].count("/"), reverse=True)[0]
    assert getattr(element, "initial", None), f"{path} missing hsm.initial"
    return path, element


def test_hsm_init_required_composites_declare_nested_initials() -> None:
    """The five composites called out for HSM-INIT-001 each have an unguarded nested initial."""

    cases: list[tuple[type[hsm.Instance], str, str]] = [
        (Reading, "Focused", "Classifying/Applying"),
        (Participating, "perceiving", "routing"),
        (Listening, "DecodingSpeech", "Detected"),
        # Revision.authoring is a leaf: attempt index is free-running on the write payload,
        # not nested attempt_0/1/2 states (retries continue until same diagnostics twice).
        (TextConversation, "active", "decoding"),
    ]
    for machine_cls, composite_name, expected_target_suffix in cases:
        model = _behavior_model(machine_cls)
        _path, state = _composite_state(model, name=composite_name)
        initial_path = state.initial
        assert isinstance(initial_path, str) and initial_path.endswith("/.initial")
        transition_paths = [
            path
            for path in model.members
            if path.startswith(f"{initial_path}/")
            and "/transition_" in path
            and "/observer/" not in path
            and path.count("/transition_") == 1
        ]
        assert transition_paths, f"{composite_name} initial has no transition under {initial_path}"
        transition = model.members[sorted(transition_paths)[0]]
        target = getattr(transition, "target", None)
        assert isinstance(target, str) and target.endswith(expected_target_suffix), (
            f"{composite_name} initial target {target!r} does not end with {expected_target_suffix!r}"
        )
        assert getattr(transition, "guard", None) is None
