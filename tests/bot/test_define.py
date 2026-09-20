from __future__ import annotations

import asyncio
import importlib
import pkgutil

import hsm
import pytest

import mosfet
from mosfet import telemetry
from mosfet.define import topology

_DEFINE = importlib.import_module("mosfet.define")


def _demo() -> hsm.Model:
    return mosfet.define(
        "Demo",
        mosfet.initial(mosfet.target("idle")),
        mosfet.state("idle", mosfet.transition(mosfet.on("go"), mosfet.target("../run"))),
        mosfet.state("run"),
    )


def test_define_returns_working_model() -> None:
    model = _demo()
    go = hsm.Event[None](name="go")

    class Demo(hsm.Instance):
        pass

    async def run() -> str:
        demo = Demo()
        _ = await mosfet.started(None, demo, model)
        await hsm.dispatch(None, demo, go)
        return demo.state() or ""

    assert asyncio.run(run()) == "/Demo/run"


def test_topology_includes_idle_run_and_go() -> None:
    payload = topology(_demo())
    assert payload["name"] == "/Demo"
    assert payload["initial"] == "/Demo/.initial"
    states = {state["qualified_name"]: state for state in payload["states"]}
    assert "/Demo/idle" in states
    assert "/Demo/run" in states
    assert states["/Demo/idle"]["parent"] == "/Demo"
    assert states["/Demo/run"]["parent"] == "/Demo"
    assert any(
        transition["source"] == "/Demo/idle" and transition["target"] == "/Demo/run" and "go" in transition["events"]
        for transition in payload["transitions"]
    )


def test_topology_includes_runtime_owner_when_supplied() -> None:
    owner_model = mosfet.define("Owner", mosfet.initial(mosfet.target("ready")), mosfet.state("ready"))
    child_model = mosfet.define("Child", mosfet.initial(mosfet.target("ready")), mosfet.state("ready"))

    async def run() -> str:
        owner = await mosfet.started(None, hsm.Instance(), owner_model)
        payload = topology(child_model, owner=owner)
        owner_name = payload.get("owner")
        assert owner_name is not None
        return owner_name

    assert asyncio.run(run()) == "/Owner"


def test_topology_can_explicitly_clear_runtime_owner() -> None:
    payload = topology(_demo(), clear_owner=True)

    assert "owner" in payload
    assert payload["owner"] is None


def test_define_without_otlp_does_not_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_MODEL_PUBLISH", raising=False)
    calls: list[object] = []

    def _fail_post(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _fail_post)
    _ = _demo()
    assert calls == []


def test_define_with_endpoint_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_MODEL_PUBLISH", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[object, str]] = []

    def _record(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)
    model = _demo()
    assert len(calls) == 1
    payload, url = calls[0]
    published = topology(model)
    assert url == "http://127.0.0.1:5173/v1/models"
    assert payload == published
    assert any(
        transition["source"] == "/Demo/idle" and "go" in transition["events"] for transition in published["transitions"]
    )


def test_define_with_endpoint_but_without_opt_in_does_not_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOT_MODEL_PUBLISH", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[object, str]] = []

    def _record(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)
    _ = _demo()
    assert calls == []


def test_define_publish_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_MODEL_PUBLISH", "1")
    monkeypatch.setenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    def _boom(payload: object, url: str) -> None:
        del payload, url
        raise TimeoutError("collector down")

    monkeypatch.setattr(_DEFINE, "post_model", _boom)
    model = _demo()
    assert model.qualified_name == "/Demo"


def _consumed_here(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[object]) -> None:
    """An explicit no-op effect: the event is consumed and deliberately does nothing."""

    del ctx, instance, event


def _guard_true(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[object]) -> bool:
    del ctx, instance, event
    return True


def _consuming_model(*extra: hsm.Element) -> hsm.Model:
    """A model whose only ``go`` transition has a guard, no target, and no effect."""

    return mosfet.define(
        "Consuming",
        hsm.initial(hsm.target("/Consuming/idle")),
        hsm.state("idle", hsm.transition(hsm.on("go"), hsm.guard(_guard_true))),
        *extra,
    )


def test_define_rejects_a_transition_with_no_target_and_no_effect() -> None:
    """``hsm`` requires a target or an effect, and ``mosfet.define`` does not soften that.

    A transition that consumes an event and does nothing has to say so with a named no-op
    effect; "no target, no effect" is indistinguishable from an unfinished transition.
    """

    with pytest.raises(hsm.ErrorValidatingModel):
        _ = _consuming_model()


def test_define_rejects_that_transition_even_when_the_model_is_observed() -> None:
    """Observation must not be what makes a model valid.

    ``hsm`` inserts an observation effect into every matching transition when it *finalizes* a
    model, which is after it validates it. Binding the event context around behaviors must not
    reorder those two: a model that only passes once observation has filled in an effect is a
    model whose validity depends on telemetry being wired.
    """

    with pytest.raises(hsm.ErrorValidatingModel):
        _ = _consuming_model(hsm.observe(telemetry.observer))


def test_an_explicit_no_op_effect_is_how_a_consuming_transition_is_written() -> None:
    """The form every such transition in this library uses: a named effect that does nothing."""

    model = mosfet.define(
        "Consuming",
        hsm.initial(hsm.target("/Consuming/idle")),
        hsm.state(
            "idle",
            hsm.transition(hsm.on("go"), hsm.guard(_guard_true), hsm.effect(_consumed_here)),
        ),
        hsm.observe(telemetry.observer),
    )
    assert model.qualified_name == "/Consuming"


def _transitions_only_observation_makes_valid(model: hsm.Model) -> list[str]:
    """Transitions with no target whose every effect was inserted by ``hsm.observe``.

    ``hsm.DefaultModelFinalizer`` inserts each observation's effect under that observation's own
    namespace, so what the model's author wrote is what is left once those are removed.
    """

    namespaces = [
        member.qualified_name for member in model.members.values() if isinstance(member, hsm.ObservationElement)
    ]
    offenders: list[str] = []
    for member in model.members.values():
        if not isinstance(member, hsm.TransitionElement) or member.target != "":
            continue
        authored = [effect for effect in member.effect if not any(effect.startswith(f"{ns}/") for ns in namespaces)]
        if not authored:
            offenders.append(member.qualified_name)
    return offenders


def test_library_models_are_valid_without_their_observation_effects() -> None:
    """No model in the library leans on observation for its validity.

    Walking every model the package defines at import time is what makes this a statement about
    the library rather than about the models one test happens to build. ``hsm`` validates before
    it finalizes, so a model that needs its observation effect to pass is a model that would
    stop defining the moment its owning boundary stopped observing it.
    """

    models: list[hsm.Model] = []
    package = importlib.import_module("mosfet")
    for info in pkgutil.walk_packages(package.__path__, "mosfet."):
        module = importlib.import_module(info.name)
        for value in vars(module).values():
            if not isinstance(value, type):
                continue
            member = getattr(value, "model", None)
            if isinstance(member, hsm.Model) and member not in models:
                models.append(member)
    assert models, "no models were discovered under mosfet"
    offenders = {
        model.qualified_name: names for model in models if (names := _transitions_only_observation_makes_valid(model))
    }
    assert offenders == {}
