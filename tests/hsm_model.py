from typing import Protocol, runtime_checkable


@runtime_checkable
class TransitionLike(Protocol):
    source: str | None
    guard: str | None
    effect: list[str]
    target: str | None


TransitionMap = dict[str, dict[str, list[TransitionLike]]]


@runtime_checkable
class ModelWithTransitionMap(Protocol):
    transition_map: TransitionMap
    members: dict[str, object]


def transition_map(model: object) -> TransitionMap:
    assert isinstance(model, ModelWithTransitionMap)
    return model.transition_map


def choice_transitions(model: object, source: str) -> list[TransitionLike]:
    assert isinstance(model, ModelWithTransitionMap)
    transitions: list[TransitionLike] = []
    for member in model.members.values():
        if isinstance(member, TransitionLike) and getattr(member, "source", None) == source:
            transitions.append(member)
    return transitions
