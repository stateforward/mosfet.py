from bot import abilities
from bot.abilities import cognition

import asyncio
import typing
import uuid
import weakref

import hsm

from tests.hsm_instance_state import ability_terminal_owner, remember_ability_terminal_owner

_TAbilityOutput = typing.TypeVar("_TAbilityOutput")

def shared_hsm_context(ctx: hsm.Context | None = None) -> hsm.Context:
    base = hsm.Context() if ctx is None else ctx
    instances = base.value(hsm.Keys.Instances)
    if isinstance(instances, weakref.WeakValueDictionary):
        return base
    return base.with_value(hsm.Keys.Instances, weakref.WeakValueDictionary[object, hsm.Instance]())

def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model

async def start_abilities_for_test(
    ctx: hsm.Context,
    *abilities: abilities.Ability[typing.Any, typing.Any],
) -> None:
    for ability in abilities:
        recorder = _AbilityTerminalRecorder(
            output_event=ability.output_event,
            failed_event=ability.failed_event,
            ability=ability,
        )
        _ = await hsm.started(ctx, recorder, require_model(recorder.model))
        _ = await ability.attach(owner=recorder, ctx=ctx)
        remember_ability_terminal_owner(ability, recorder)

class _AbilityTerminalRecorder(hsm.Instance):
    model: typing.ClassVar[hsm.Model | None] = None
    output_event: hsm.Event[typing.Any]
    failed_event: hsm.Event[typing.Any]
    ability: abilities.Ability[typing.Any, typing.Any] | None
    results: dict[str, asyncio.Future[object]]

    def __init__(
        self,
        *,
        output_event: hsm.Event[typing.Any],
        failed_event: hsm.Event[typing.Any],
        ability: abilities.Ability[typing.Any, typing.Any] | None = None,
    ) -> None:
        super().__init__()
        self.output_event = output_event
        self.failed_event = failed_event
        self.ability = ability
        self.results = {}

    def result_for(self, operation_id: str) -> asyncio.Future[object]:
        result = self.results.get(operation_id)
        if result is None:
            result = asyncio.get_running_loop().create_future()
            self.results[operation_id] = result
        return result

def _record_terminal_event(
    ctx: hsm.Context,
    instance: _AbilityTerminalRecorder,
    event: hsm.Event[typing.Any],
) -> None:
    del ctx
    operation_id = event.id if event.id else None
    result = instance.results.get(operation_id) if operation_id is not None else None
    product_event = event
    product_data = event.data
    if event.name == cognition.InputEvent.name and isinstance(event.data, cognition.InputData):
        if instance.output_event.name == cognition.InputEvent.name:
            _mirror_terminal_event(instance.ability, "outputs", event.data)
            _mirror_terminal_event(instance.ability, "handoffs", event.data)
            if result is not None and not result.done():
                result.set_result(event.data)
            return
        product_event = event.data.stimulus
        product_data = product_event.data if isinstance(product_event, hsm.Event) else product_event
    if isinstance(product_event, hsm.Event) and product_event.name == instance.output_event.name:
        _mirror_terminal_event(instance.ability, "outputs", product_data)
        if result is not None and not result.done():
            result.set_result(product_data)
        return
    if event.name != instance.failed_event.name:
        return
    failure = event.data
    if isinstance(failure, abilities.FailureData):
        _mirror_terminal_event(instance.ability, "failures", failure)
        if result is not None and not result.done():
            result.set_exception(RuntimeError(failure.message))
        return
    message = getattr(failure, "message", str(failure))
    _mirror_terminal_event(instance.ability, "failures", failure)
    if result is not None and not result.done():
        result.set_exception(RuntimeError(message))

def _mirror_terminal_event(ability: abilities.Ability[typing.Any, typing.Any] | None, attribute: str, data: object) -> None:
    if ability is None:
        return
    values = getattr(ability, attribute, None)
    if isinstance(values, list):
        values = typing.cast(list[object], values)
        values.append(data)

_AbilityTerminalRecorder.model = hsm.define(
    "AbilityTerminalRecorder",
    hsm.initial(hsm.target("/AbilityTerminalRecorder/recording")),
    hsm.state(
        "recording",
        hsm.transition(
            hsm.on(hsm.AnyEvent),
            hsm.effect(_record_terminal_event),
        ),
    ),
)

async def dispatch_ability_for_test(
    ability: abilities.Ability[typing.Any, _TAbilityOutput],
    ctx: hsm.Context | None,
    input: object,
    *,
    timeout: float = 1.0,
) -> _TAbilityOutput:
    shared_ctx = shared_hsm_context(ctx)
    _register_ability_graph(shared_ctx, ability)
    owner = ability_terminal_owner(ability)
    recorder = owner if isinstance(owner, _AbilityTerminalRecorder) else None
    if owner is not None and recorder is None and callable(getattr(owner, "result_for", None)):
        recorder = typing.cast(_AbilityTerminalRecorder, owner)
    if recorder is None:
        recorder = _AbilityTerminalRecorder(
            output_event=ability.output_event,
            failed_event=ability.failed_event,
            ability=ability,
        )
        _ = await hsm.started(shared_ctx, recorder, require_model(recorder.model))
        _ = await ability.attach(owner=recorder, ctx=shared_ctx)
        remember_ability_terminal_owner(ability, recorder)
    elif not ability.state():
        _ = await ability.attach(owner=recorder, ctx=shared_ctx)
    operation_id = uuid.uuid4().hex
    result = recorder.result_for(operation_id)
    _ = await hsm.dispatch(shared_ctx, ability, ability.input_event.with_data_and_id(input, operation_id))
    return typing.cast(_TAbilityOutput, await asyncio.wait_for(result, timeout=timeout))

def _register_ability_graph(ctx: hsm.Context, ability: abilities.Ability[typing.Any, typing.Any]) -> None:
    instances = ctx.value(hsm.Keys.Instances)
    if not isinstance(instances, weakref.WeakValueDictionary):
        return
    snapshot = ability.take_snapshot()
    if snapshot.ID:
        instances[snapshot.ID] = ability
