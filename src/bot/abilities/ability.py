import asyncio
import collections.abc
import dataclasses
import datetime
import typing
import uuid

import hsm
import pydantic

from bot.protocols import attachment
from bot.telemetry import observer


TInput = typing.TypeVar("TInput")
TOutput = typing.TypeVar("TOutput")
_DataType = type[object] | tuple[type[object], ...] | None


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


def _is_base_model_type(data_type: object) -> typing.TypeGuard[type[pydantic.BaseModel]]:
    if not isinstance(data_type, type):
        return False
    try:
        return issubclass(data_type, pydantic.BaseModel)
    except TypeError:
        return False


def _model_with_json_schema_extra(
    data_type: type[pydantic.BaseModel],
    *,
    description: str | None,
    examples: collections.abc.Sequence[object] | None,
) -> type[pydantic.BaseModel]:
    """Return a BaseModel subclass with merged root json_schema_extra (still a model, not TypeAdapter)."""

    existing_extra = data_type.model_config.get("json_schema_extra")
    extra: dict[str, object] = {}
    if isinstance(existing_extra, dict):
        extra = dict(typing.cast(dict[str, object], existing_extra))
    if description is not None:
        extra["description"] = description
    if examples is not None:
        extra["examples"] = list(examples)
    # Preserve relevant base config; create_model needs an explicit ConfigDict for extra merge.
    config = pydantic.ConfigDict(
        frozen=bool(data_type.model_config.get("frozen", False)),
        extra=data_type.model_config.get("extra", "ignore"),  # type: ignore[arg-type]
        arbitrary_types_allowed=bool(data_type.model_config.get("arbitrary_types_allowed", False)),
        json_schema_extra=extra,
    )
    return typing.cast(
        type[pydantic.BaseModel],
        pydantic.create_model(
            f"{data_type.__name__}EventSchema",
            __base__=data_type,
            __config__=config,
        ),
    )


def _schema_for_data_type(
    data_type: type[object],
    *,
    description: str | None,
    examples: collections.abc.Sequence[object] | None,
) -> object:
    """Resolve an event payload schema.

    Prefer concrete ``BaseModel`` types (required fields / Field descriptions stay on the model).
    Do not wrap models in ``TypeAdapter`` — that breaks create_model patching and hides
    domain required lists. Non-model types may still use TypeAdapter when metadata is needed.
    """

    if _is_base_model_type(data_type):
        if description is None and examples is None:
            return data_type
        return _model_with_json_schema_extra(
            data_type,
            description=description,
            examples=examples,
        )
    if description is None and examples is None:
        return data_type
    # Non-BaseModel payloads (e.g. bytes) still need TypeAdapter to attach description/examples.
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


class Ability(hsm.Instance, attachment.Attachment, typing.Generic[TInput, TOutput]):
    """Specific event-driven operational capacity.

    Answers: What operations can the system perform?
    """

    _attachment_limit: typing.ClassVar[int | None] = 1
    input_event: typing.ClassVar[hsm.Event[typing.Any]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[typing.Any]] = OutputEvent
    failed_event: typing.ClassVar[hsm.Event[typing.Any]] = FailedEvent
    input_data_type: typing.ClassVar[_DataType] = None
    output_data_type: typing.ClassVar[_DataType] = None
    submodel: typing.ClassVar[hsm.Model | None] = None

    @staticmethod
    def _has_terminal_event(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, hsm.Event)

    @staticmethod
    def _forward_terminal_event(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        if not instance._attachments:
            return
        owner = instance._attachments[0]
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
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._can_attach),
                hsm.effect(
                    attachment.Attachment._attach,
                    attachment.Attachment._remember_attachment_timeout,
                    attachment.Attachment._queue_attach_complete,
                ),
                hsm.target("/Ability/attaching"),
            ),
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._is_attach_request),
                hsm.effect(attachment.Attachment._dispatch_attach_failed),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._is_detach_request),
                hsm.effect(attachment.Attachment._dispatch_detach_complete_absent),
            ),
        ),
        hsm.state(
            "attaching",
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.effect(attachment.Attachment._deliver_attach_complete),
                hsm.target("/Ability/attached"),
            ),
            hsm.transition(
                hsm.after(attachment.Attachment._attachment_timeout_delay),
                hsm.effect(attachment.Attachment._timeout_attachment),
                hsm.target("/Ability/detached"),
            ),
        ),
        hsm.state(
            "attached",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._is_attached),
                hsm.effect(attachment.Attachment._dispatch_attach_complete_existing),
            ),
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._is_attach_request),
                hsm.effect(attachment.Attachment._dispatch_attach_failed),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._detach_last_attachment),
                hsm.effect(attachment.Attachment._detach, attachment.Attachment._dispatch_detach_complete_removed),
                hsm.target("/Ability/detached"),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._is_detach_request),
                hsm.effect(attachment.Attachment._dispatch_detach_failed),
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
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._can_attach),
                    hsm.effect(
                        attachment.Attachment._attach,
                        attachment.Attachment._remember_attachment_timeout,
                        attachment.Attachment._queue_attach_complete,
                    ),
                    hsm.target(f"{root}/attaching"),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._is_attach_request),
                    hsm.effect(attachment.Attachment._dispatch_attach_failed),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._is_detach_request),
                    hsm.effect(attachment.Attachment._dispatch_detach_complete_absent),
                ),
            ),
            hsm.state(
                "attaching",
                hsm.transition(
                    hsm.on(attachment.AttachCompleteEvent),
                    hsm.effect(attachment.Attachment._deliver_attach_complete),
                    hsm.target(attached),
                ),
                hsm.transition(
                    hsm.after(attachment.Attachment._attachment_timeout_delay),
                    hsm.effect(attachment.Attachment._timeout_attachment),
                    hsm.target(f"{root}/detached"),
                ),
            ),
            hsm.state(
                "attached",
                hsm.initial(hsm.target(f"{attached}/behavior")),
                hsm.submachine_state("behavior", submodel),
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._is_attached),
                    hsm.effect(attachment.Attachment._dispatch_attach_complete_existing),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._is_attach_request),
                    hsm.effect(attachment.Attachment._dispatch_attach_failed),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._detach_last_attachment),
                    hsm.effect(
                        attachment.Attachment._detach,
                        attachment.Attachment._dispatch_detach_complete_removed,
                    ),
                    hsm.target(f"{root}/detached"),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._is_detach_request),
                    hsm.effect(attachment.Attachment._dispatch_detach_failed),
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
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)

    def attach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[attachment.AttachData],
    ) -> collections.abc.Awaitable[None]:
        """Start this ability lifecycle and dispatch its attach request."""

        model = self.model

        async def start_and_dispatch() -> None:
            if model is None:
                return
            try:
                _ = await hsm.started(ctx, self, model)
            except hsm.ErrorValidatingModel as error:
                if "already has a running HSM" not in str(error):
                    raise
            _ = await hsm.dispatch(ctx, self, event)

        task = asyncio.Task(
            start_and_dispatch(),
            loop=asyncio.get_running_loop(),
            eager_start=True,
        )
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return task

    def detach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[attachment.DetachData],
    ) -> collections.abc.Awaitable[None]:
        """Dispatch this ability's detach request."""

        return hsm.dispatch(ctx, self, event)

    def apply(self, input: TInput, *, ctx: hsm.Context | None = None) -> collections.abc.Awaitable[None]:
        """Dispatch this ability's input event."""

        return hsm.dispatch(
            self.context() if ctx is None else ctx,
            self,
            self.input_event.with_data_and_id(input, uuid.uuid4().hex),
        )


__all__ = [
    "FailedEvent",
    "InputEvent",
    "OutputEvent",
    "TerminalErrorEvent",
    "TerminalOutputEvent",
    "Ability",
    "FailureData",
    "TInput",
    "TOutput",
    "ability_input_event",
    "ability_output_event",
]
