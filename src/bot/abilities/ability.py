import asyncio
import collections.abc
import dataclasses
import typing
import uuid
import weakref

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from bot.telemetry import observer


TInput = typing.TypeVar("TInput")
TOutput = typing.TypeVar("TOutput")
_DataType = type[object] | tuple[type[object], ...] | None


class AttachEventData(pydantic.BaseModel):
    """Request for an owned ability to attach its runtime behavior."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        json_schema_extra={
            "description": "Request for an owned ability to attach its runtime behavior.",
            "examples": [{}],
        },
    )

    owner: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]] = pydantic.Field(
        exclude=True,
        repr=False,
        description="Runtime-only structural owner receiving this ability's terminal events.",
    )


class DetachEventData(pydantic.BaseModel):
    """Request for an owned ability to detach its runtime behavior."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Request for an owned ability to detach its runtime behavior.",
            "examples": [{}],
        },
    )


class FailureData(pydantic.BaseModel):
    """FailureData signal produced when an ability operation cannot complete."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "FailureData signal produced when an ability operation cannot complete.",
            "examples": [{"message": "Provider request failed."}],
        },
    )

    message: str = pydantic.Field(
        description="Human-readable failure message for the ability operation.",
        examples=["Provider request failed."],
    )


def _schema_for_data_type(
    data_type: type[object],
    *,
    description: str | None,
    examples: collections.abc.Sequence[object] | None,
) -> object:
    if description is None and examples is None:
        return data_type
    field = _schema_metadata_field(description=description, examples=examples)
    schema_type: object = typing.Annotated[data_type, field]
    return typing.cast(pydantic.TypeAdapter[object], pydantic.TypeAdapter(schema_type))


def _schema_metadata_field(
    *,
    description: str | None,
    examples: collections.abc.Sequence[object] | None,
) -> object:
    if description is None:
        return typing.cast(object, pydantic.Field(examples=list(examples or ())))
    if examples is None:
        return typing.cast(object, pydantic.Field(description=description))
    return typing.cast(object, pydantic.Field(description=description, examples=list(examples)))


InputEvent = hsm.Event[typing.Any](
    name="bot.ability.input",
    schema=_schema_for_data_type(
        object,
        description=(
            "InputData event data accepted by a generic ability. Concrete abilities should replace this with "
            "a specific input event schema."
        ),
        examples=["Summarize this note."],
    ),
)
OutputEvent = hsm.Event[typing.Any](
    name="bot.ability.output",
    schema=_schema_for_data_type(
        object,
        description=(
            "OutputData event data dispatched by a generic ability. Concrete abilities should replace this with "
            "a specific output event schema."
        ),
        examples=["Summary text."],
    ),
)
FailedEvent = hsm.Event[FailureData](
    name="bot.ability.failed",
    schema=FailureData,
)
AttachEvent = hsm.Event[AttachEventData](
    name="bot.ability.attach",
    schema=AttachEventData,
)
DetachEvent = hsm.Event[DetachEventData](
    name="bot.ability.detach",
    schema=DetachEventData,
)
_AttachCompletedEvent = hsm.Event[object](
    name="bot.ability.attach.completed",
    kind=hsm.CompletionEventKind,
    schema=object,
)
TerminalOutputEvent = hsm.Event[hsm.Event[typing.Any]](
    name="bot.ability.terminal.output",
    kind=hsm.CompletionEventKind,
    schema=hsm.Event[typing.Any],
)
TerminalErrorEvent = hsm.Event[hsm.Event[typing.Any]](
    name="bot.ability.terminal.error",
    kind=hsm.ErrorEventKind,
    schema=hsm.Event[typing.Any],
)


def ability_input_event(
    name: str,
    data_type: type[TInput],
    *,
    description: str | None = None,
    examples: collections.abc.Sequence[object] | None = None,
) -> hsm.Event[TInput]:
    """Build an ability input event with a concrete Pydantic payload schema."""

    schema = _schema_for_data_type(
        data_type,
        description=description,
        examples=examples,
    )
    return hsm.Event[TInput](
        name=name,
        schema=schema,
    )


def ability_output_event(
    name: str,
    data_type: type[TOutput],
    *,
    description: str | None = None,
    examples: collections.abc.Sequence[object] | None = None,
) -> hsm.Event[TOutput]:
    """Build an ability output event with a concrete Pydantic payload schema."""

    schema = _schema_for_data_type(
        data_type,
        description=description,
        examples=examples,
    )
    return hsm.Event[TOutput](
        name=name,
        schema=schema,
    )


class Ability(hsm.Instance, typing.Generic[TInput, TOutput]):
    """Specific event-driven operational capacity.

    Answers: What operations can the system perform?
    """

    input_event: typing.ClassVar[hsm.Event[typing.Any]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[typing.Any]] = OutputEvent
    failed_event: typing.ClassVar[hsm.Event[typing.Any]] = FailedEvent
    input_data_type: typing.ClassVar[_DataType] = None
    output_data_type: typing.ClassVar[_DataType] = None
    submodel: typing.ClassVar[hsm.Model | None] = None

    @staticmethod
    def _can_claim_owner(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, AttachEventData):
            return False
        owner = data.owner
        if owner is instance:
            return False
        owner_ref = instance._owner_ref
        if owner_ref is None:
            return True
        return owner_ref() is owner

    @staticmethod
    def _claim_owner(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        data = typing.cast(AttachEventData, event.data)
        instance._owner_ref = weakref.ref(data.owner)

    @staticmethod
    def _clear_owner(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, event
        instance._owner_ref = None

    @staticmethod
    def current_owner(instance: "Ability[typing.Any, typing.Any]") -> hsm.Instance | None:
        """Return the structural owner claimed through the attach lifecycle."""

        owner_ref = instance._owner_ref
        if owner_ref is None:
            return None
        return owner_ref()

    @staticmethod
    def _has_terminal_event(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, hsm.Event)

    @staticmethod
    def _dispatch_attach_completed(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        _ = hsm.dispatch(ctx, instance, _AttachCompletedEvent.with_data(None))

    @staticmethod
    def _forward_terminal_event(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        owner_ref = instance._owner_ref
        if owner_ref is None:
            return
        owner = owner_ref()
        if owner is None:
            return
        terminal = event.data
        assert isinstance(terminal, hsm.Event)
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(terminal, source=hsm.id(instance), target=hsm.id(owner)),
        )

    model: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Ability",
        hsm.initial(hsm.target("/Ability/detached")),
        hsm.state(
            "detached",
            hsm.transition(
                hsm.on(AttachEvent),
                hsm.guard(_can_claim_owner),
                hsm.effect(_claim_owner),
                hsm.target("/Ability/attaching"),
            ),
        ),
        hsm.state(
            "attaching",
            hsm.entry(_dispatch_attach_completed),
            hsm.transition(
                hsm.on(_AttachCompletedEvent),
                hsm.target("/Ability/attached"),
            ),
        ),
        hsm.state(
            "attached",
            hsm.transition(
                hsm.on(DetachEvent),
                hsm.effect(_clear_owner),
                hsm.target("/Ability/detached"),
            ),
            hsm.transition(
                hsm.on(TerminalOutputEvent),
                hsm.guard(_has_terminal_event),
                hsm.effect(_forward_terminal_event),
            ),
            hsm.transition(
                hsm.on(TerminalErrorEvent),
                hsm.guard(_has_terminal_event),
                hsm.effect(_forward_terminal_event),
            ),
        ),
        hsm.observe(observer),
    )
    _owner_ref: weakref.ReferenceType[hsm.Instance] | None

    @staticmethod
    def _define_model(name: str, submodel: hsm.Model) -> hsm.Model:
        root = f"/{name}Lifecycle"
        attached = f"{root}/attached"
        return hsm.define(
            f"{name}Lifecycle",
            hsm.initial(hsm.target(f"{root}/detached")),
            hsm.state(
                "detached",
                hsm.transition(
                    hsm.on(AttachEvent),
                    hsm.guard(Ability._can_claim_owner),
                    hsm.effect(Ability._claim_owner),
                    hsm.target(f"{root}/attaching"),
                ),
            ),
            hsm.state(
                "attaching",
                hsm.entry(Ability._dispatch_attach_completed),
                hsm.transition(
                    hsm.on(_AttachCompletedEvent),
                    hsm.target(attached),
                ),
            ),
            hsm.state(
                "attached",
                hsm.initial(hsm.target(f"{attached}/behavior")),
                hsm.submachine_state("behavior", submodel),
                hsm.transition(
                    hsm.on(DetachEvent),
                    hsm.effect(Ability._clear_owner),
                    hsm.target(f"{root}/detached"),
                ),
                hsm.transition(
                    hsm.on(TerminalOutputEvent),
                    hsm.guard(Ability._has_terminal_event),
                    hsm.effect(Ability._forward_terminal_event),
                ),
                hsm.transition(
                    hsm.on(TerminalErrorEvent),
                    hsm.guard(Ability._has_terminal_event),
                    hsm.effect(Ability._forward_terminal_event),
                ),
            ),
            hsm.observe(observer),
        )

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        declared_submodel = cls.__dict__.get("submodel")
        if declared_submodel is None:
            declared_submodel = cls.submodel
        assert declared_submodel is not None, f"{cls.__name__} must define submodel."
        submodel = typing.cast(hsm.Model, declared_submodel)
        cls.submodel = submodel
        cls.model = Ability._define_model(cls.__name__, submodel)

    def __init__(self) -> None:
        super().__init__()
        self._owner_ref = None

    def attach(self, *, owner: hsm.Instance, ctx: hsm.Context | None = None) -> collections.abc.Awaitable[None]:
        """Start this ability lifecycle and dispatch its attach event.

        Prefer the owner's lifetime context when available so this ability outlives
        transient activity contexts at the call site (HSM-CONTEXT-001). Callers that need
        durable parenting should pass the owner's context explicitly.
        """

        if ctx is not None:
            context = ctx
        else:
            try:
                context = owner.context()
            except Exception:
                context = self.context()
        model = self.model

        async def start_and_dispatch() -> None:
            if model is None:
                return
            try:
                _ = await hsm.started(context, self, model)
            except hsm.ErrorValidatingModel as error:
                if "already has a running HSM" not in str(error):
                    raise
            _ = await hsm.dispatch(context, self, AttachEvent.with_data(AttachEventData(owner=owner)))

        task = asyncio.Task(
            start_and_dispatch(),
            loop=asyncio.get_running_loop(),
            eager_start=True,
        )
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return task

    def detach(self, *, ctx: hsm.Context | None = None) -> collections.abc.Awaitable[None]:
        """Dispatch this ability's detach event."""

        return hsm.dispatch(
            self.context() if ctx is None else ctx,
            self,
            DetachEvent.with_data(DetachEventData()),
        )

    def apply(self, input: TInput, *, ctx: hsm.Context | None = None) -> collections.abc.Awaitable[None]:
        """Dispatch this ability's input event."""

        return hsm.dispatch(
            self.context() if ctx is None else ctx,
            self,
            self.input_event.with_data_and_id(input, uuid.uuid4().hex),
        )


__all__ = [
    "AttachEvent",
    "DetachEvent",
    "FailedEvent",
    "InputEvent",
    "OutputEvent",
    "TerminalErrorEvent",
    "TerminalOutputEvent",
    "Ability",
    "AttachEventData",
    "DetachEventData",
    "FailureData",
    "TInput",
    "TOutput",
    "ability_input_event",
    "ability_output_event",
]
