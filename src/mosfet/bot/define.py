"""``bot.define``: one bot, as data, with every mistake in its file reported at once.

``bot.define(name, *elements)`` mirrors ``hsm.define`` line for line: a :class:`RedefinableBot` walks
the elements into a :class:`~mosfet.bot.elements.Model`, a :class:`Validator` (an
``hsm.ModelValidator``) checks each member by kind, and a :class:`Finalizer` puts what they found in
source order. Where hsm raises at the first mistake, the walk and the validator *collect*
:class:`~mosfet.bot.diagnostics.Diagnostic`\\ s, so one :class:`DefinitionError` carries every error
in the file, rendered the way rustc renders them. Each member is keyed like an hsm state:

>>> from mosfet import bot
>>> from mosfet.bot import testing
>>> from mosfet.devices import computer
>>> model = bot.define(
...     "greeter",
...     bot.traits("curious"),
...     bot.cognition(testing.Scripted()),
...     bot.attach(computer.Computer(id="desk")),
... )
>>> list(model.members)
['/greeter', '/greeter/.traits', '/greeter/.cognition', '/greeter/desk']

The hsm partials among the elements (``bot.state``, ``bot.transition``, ...) are the bot's one
firmware: they compile after the walk, together, into a machine of its own
(``mosfet.bot.firmware``), and the validator checks the commands it offers cognition.

``bot.redefine(model, *elements)`` is a new model from the old one's elements and more, as
``hsm.redefine`` is; the old model is unchanged:

>>> from mosfet.devices import phone
>>> more = bot.redefine(model, bot.attach(phone.Phone()))
>>> [[part.id for part in outline.parts] for outline in bot.describe(model, more).bots]
[['desk'], ['desk', 'phone']]

Defining starts, opens and reads nothing: it holds the instances the bot file constructed.
"""

import collections.abc
import dataclasses
import datetime
import difflib
import posixpath
import typing
import warnings

import hsm
import pydantic

import mosfet.device

from mosfet import event, lifecycle, runtime
from mosfet.abilities import memory, persona, processing
from mosfet.abilities.communication.conversation.conversation import snake_case

from . import diagnostics, elements, firmware, lints

_NAME: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(runtime.BotName)


class Error(hsm.ErrorValidatingModel):
    """The bot DSL's errors: a bot file that does not define, or a bot that does not start.

    Each is hsm-shaped: ``str()`` starts with the ``file:line:`` of the bot file's own code.
    """


class DefinitionError(Error):
    """``bot.define`` found errors in a bot file.

    ``diagnostics`` holds every error it found, in source order. ``str()`` is hsm's ``file:line:``
    for the first one, then every diagnostic rendered and the summary line.
    """

    def __init__(self, name: str, found: collections.abc.Sequence[diagnostics.Diagnostic]) -> None:
        first = next(diagnostic for diagnostic in found if diagnostic.level == "error").primary
        super().__init__(
            hsm.Location(filename=first.filename, lineno=first.lineno),
            diagnostics.report(found, action="define", name=name),
        )
        self.diagnostics: tuple[diagnostics.Diagnostic, ...] = tuple(found)


class Validator(hsm.ModelValidator):
    """hsm's ``_validate_<kind>`` dispatch over a bot's members; each check appends, never raises."""

    @typing.override
    def validate(self, model: hsm.Model | elements.Model) -> None:
        # hsm's dispatch reads only ``members``, which a bot model has as the namespace it is.
        super().validate(typing.cast(hsm.Model, model))

    def _validate_model(self, model: elements.Model, member: elements.Model) -> None:
        del member
        call = diagnostics.Span.of(model.location)
        name = model.qualified_name[1:]
        if not name:
            first = diagnostics.argument(call, 0)
            insertion = (
                None
                if first is None
                else diagnostics.Suggestion(
                    span=first[0].model_copy(update={"end_lineno": first[0].lineno, "end_col": first[0].col}),
                    replacement='"bot", ',
                    applicability="maybe-incorrect",
                )
            )
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0001",
                    level="error",
                    message="`bot.define` takes the bot's name first",
                    primary=diagnostics.head(call, "no name"),
                    notes=("a bot's name is lower snake case, at most 64 characters",),
                    help=diagnostics.Help(message="name the bot", suggestion=insertion),
                )
            )
        else:
            try:
                _ = _NAME.validate_python(name)
            except pydantic.ValidationError:
                found = diagnostics.argument(call, 0)
                fixed = _snake(name)
                model.findings.append(
                    diagnostics.Diagnostic(
                        code="B0001",
                        level="error",
                        message=f"bot name `{name}` is not lower snake case of at most 64 characters",
                        primary=(diagnostics.head(call, "") if found is None else found[0]).model_copy(
                            update={"label": "the bot's name"}
                        ),
                        notes=("a bot's name is a telemetry attribute and a topology name, so it is bounded",),
                        help=diagnostics.Help(
                            message="name it in lower snake case",
                            suggestion=None
                            if found is None
                            else diagnostics.Suggestion(
                                span=found[0], replacement=f'"{fixed}"', applicability="machine-applicable"
                            ),
                        ),
                    )
                )
        if not any(isinstance(item, elements.CognitionElement) for item in model.members.values()):
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0002",
                    level="error",
                    message=f"bot `{name or 'bot'}` has no `bot.cognition`",
                    primary=diagnostics.head(call, "no cognition"),
                    notes=("cognition is what perceives and decides; a bot without one is a body with no mind",),
                    help=diagnostics.Help(
                        message="give it a cognition that thinks with a `processing.Processor`",
                        suggestion=diagnostics.append(call, "bot.cognition(processor)"),
                    ),
                )
            )

    def _validate_allow(self, model: elements.Model, member: elements.AllowElement) -> None:
        call = diagnostics.Span.of(member.location)
        for index, name in enumerate(member.lints):
            if isinstance(name, str) and name in lints.LINTS:
                continue
            found = diagnostics.argument(call, index)
            primary = call if found is None else found[0]
            nearest = difflib.get_close_matches(name, list(lints.LINTS), n=1) if isinstance(name, str) else []
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0018",
                    level="error",
                    message=f"`bot.allow` names no lint `{name}`"
                    if isinstance(name, str)
                    else "`bot.allow` takes lint names",
                    primary=primary.model_copy(update={"label": "not a lint"}),
                    notes=(f"the lints are {', '.join(f'`{known}`' for known in lints.LINTS)}",),
                    help=diagnostics.Help(
                        message=f"did you mean `{nearest[0]}`?" if nearest else "name a lint, or remove it",
                        suggestion=None
                        if not nearest or found is None
                        else diagnostics.Suggestion(
                            span=primary, replacement=f'"{nearest[0]}"', applicability="machine-applicable"
                        ),
                    ),
                )
            )

    def _validate_attachment(self, model: elements.Model, member: elements.AttachmentElement) -> None:
        if lifecycle.is_started(member.instance):
            what = "device" if isinstance(member.instance, mosfet.device.Device) else "ability"
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0005",
                    level="error",
                    message=f"{what} `{member.name()}` is already started",
                    primary=diagnostics.Span.of(member.location, "started before it was given"),
                    notes=("the bot starts the parts it is given, once, as one body",),
                    help=diagnostics.Help(
                        message=f"give the bot a new, unstarted `{type(member.instance).__name__}(...)`"
                    ),
                )
            )

    def _validate_traits(self, model: elements.Model, member: elements.TraitsElement) -> None:
        try:
            _ = persona.Traits.model_validate({"traits": member.traits})
        except pydantic.ValidationError as error:
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0010",
                    level="error",
                    message="these traits are not one to 24 words or short phrases",
                    primary=diagnostics.Span.of(member.location, "the bot's traits"),
                    notes=(
                        diagnostics.cause(error),
                        "traits are the owner's fixed seed: one to 24, each at most 200 characters",
                    ),
                    help=diagnostics.Help(message='give one to 24 short traits: `bot.traits("curious", "patient")`'),
                )
            )

    def _validate_cognition(self, model: elements.Model, member: elements.CognitionElement) -> None:
        call = diagnostics.Span.of(member.location)
        checks: tuple[tuple[int | str, object, bool, str], ...] = (
            (0, member.processor, isinstance(member.processor, processing.Processor), "a `processing.Processor`"),
            (
                "memory",
                member.memory,
                member.memory is None or isinstance(member.memory, memory.Memory),
                "a `memory.Memory`",
            ),
            ("routines", member.routines, member.routines is None or callable(member.routines), "a wall clock"),
        )
        for key, value, valid, takes in checks:
            if valid:
                continue
            found = diagnostics.argument(call, key)
            primary = call if found is None else found[0]
            written = "..." if found is None else diagnostics.text(primary)
            where = "bot.cognition" if key == 0 else f"bot.cognition({key}=)"
            misplacement = elements.misplaced(value, primary, written, where)
            model.findings.append(
                misplacement.model_copy(
                    update={
                        "notes": (f"`{where}` takes {takes}",),
                        # The rewrite is of a whole element; this is one argument of bot.cognition.
                        "help": diagnostics.Help(message=misplacement.help.message if misplacement.help else ""),
                    }
                )
                if misplacement is not None
                else diagnostics.Diagnostic(
                    code="B0006",
                    level="error",
                    message=f"`{where}` takes {takes}, not `{type(value).__name__}`",
                    primary=primary.model_copy(update={"label": f"this is a `{type(value).__name__}`"}),
                    help=diagnostics.Help(message=f"pass {takes}"),
                )
            )

    def _validate_firmware(self, model: elements.Model, member: firmware.FirmwareElement) -> None:
        offered: dict[str, elements.AttachmentElement] = {}
        for attached in model.members.values():
            if not isinstance(attached, elements.AttachmentElement):
                continue
            for name in ("model", "firmware_model"):
                actor_model = typing.cast(object, getattr(attached.instance, name, None))
                if isinstance(actor_model, hsm.Model):
                    for offering in actor_model.events.values():
                        if offering.kind == event.EventKind:
                            _ = offered.setdefault(offering.name, attached)
        bot = model.name() or "bot"
        for command in member.commands():
            written = member.named.get(command.name)
            use = diagnostics.Span.of(member.location if written is None else written, "offered to cognition here")
            attached = offered.get(command.name)
            if attached is not None:
                model.findings.append(
                    diagnostics.Diagnostic(
                        code="B0013",
                        level="error",
                        message=f"firmware command `{command.name}` is also an event `{attached.name()}` offers",
                        primary=use.model_copy(update={"label": "the firmware's command"}),
                        secondary=(diagnostics.Span.of(attached.location, f"`{attached.name()}` offers it too"),),
                        notes=(
                            "cognition selects a tool by its name, so one name offered by two actors would be ambiguous",
                        ),
                        help=diagnostics.Help(
                            message="give the firmware's command a name of its own, such as "
                            + f"`{bot}.{command.name.rsplit('.', 1)[-1]}`"
                        ),
                    )
                )
            schema = command.schema
            if not isinstance(schema, type) or not issubclass(schema, pydantic.BaseModel):
                continue
            for name, field in schema.model_fields.items():
                if field.description:
                    continue
                declared = diagnostics.definition(schema, name, model.location.filename, "no description")
                model.findings.append(
                    diagnostics.Diagnostic(
                        code="B0012",
                        level="error",
                        message=f"field `{name}` of firmware command `{command.name}` has no description",
                        primary=use.model_copy(update={"label": f"`{schema.__name__}.{name}` has no description"})
                        if declared is None
                        else declared,
                        secondary=() if declared is None else (use,),
                        notes=(
                            "cognition is offered a firmware command as a tool, and the model choosing a tool "
                            + "reads what each of its fields means",
                        ),
                        help=diagnostics.Help(message=f'describe it: `{name}: ... = pydantic.Field(description="…")`'),
                    )
                )

    def _validate_timeouts(self, model: elements.Model, member: elements.TimeoutsElement) -> None:
        call = diagnostics.Span.of(member.location)
        for key, value in (
            ("processing", member.processing),
            ("deactivation", member.deactivation),
            ("attachment", member.attachment),
        ):
            # The annotation is the bot file's promise, not a fact: check what was actually given.
            given = typing.cast(object, value)
            if given is None or (isinstance(given, datetime.timedelta) and given > datetime.timedelta()):
                continue
            found = diagnostics.argument(call, key)
            primary = call if found is None else found[0]
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0011",
                    level="error",
                    message=f"the {key} timeout is not a positive `datetime.timedelta`",
                    primary=primary.model_copy(update={"label": "not positive"}),
                    notes=("a timeout bounds how long the body waits; zero or less would never wait",),
                    help=diagnostics.Help(
                        message="give a positive duration",
                        suggestion=None
                        if found is None
                        else diagnostics.Suggestion(
                            span=found[0],
                            replacement="datetime.timedelta(seconds=30)",
                            applicability="maybe-incorrect",
                        ),
                    ),
                )
            )


def _snake(name: str) -> str:
    """``name`` in lower snake case, starting with a letter and at most 64 characters: a valid bot name."""

    words = "".join(character if character.isalnum() else " " for character in snake_case(name)).split()
    snake = "_".join(words).lower() or "bot"
    if not snake[0].isalpha():
        snake = f"bot_{snake}"
    return snake[:64]


class Finalizer:
    """Puts what the walk, the validator and the lints found in source order, the order rustc reports in."""

    def finalize(self, model: elements.Model) -> elements.Model:
        """``model``, its findings sorted by file (the ``bot.define`` call's first), line and column."""

        model.findings[:] = diagnostics.ordered(model.findings, model.location.filename)
        return model


@dataclasses.dataclass(kw_only=True)
class RedefinableBot(hsm.RedefinableElement[elements.Model]):
    """The root of a bot's definition, as ``hsm.RedefinableModel`` is a state machine's."""

    @typing.override
    def redefine(
        self, model: hsm.Model | elements.Model, stack: list[hsm.Element], element: elements.Model | None = None
    ) -> elements.Model:
        """Walk every element into a new model, validate and lint it, and put what was found in order.

        One pass finds every error and every warning, as rustc's does. A model with no error emits
        each lint it does not allow as a :class:`~mosfet.bot.lints.LintWarning` at its own
        ``file:line`` and keeps it in ``findings``.

        Raises:
            DefinitionError: If any error was found, with every error and warning found.
        """

        del model, element
        result = elements.Model(
            qualified_name=self.qualified_name, owned_elements=list(self.owned_elements), location=self.location
        )
        partials: list[firmware.Partial] = []
        for owned in self.owned_elements:
            if isinstance(owned, firmware.Partial):
                partials.append(owned)
                continue
            _ = typing.cast(elements.Element[hsm.Element], owned).redefine(result, [*stack, result])
        # The firmware compiles after the walk, from every partial at once, as one machine.
        _ = firmware.compile(result, partials)
        # A part the walk or the compile refused is missing from the model, so a lint about what the
        # bot lacks would only be noise until it defines.
        missing = any(found.level == "error" for found in result.findings)
        Validator().validate(result)
        result.findings.extend(lints.lint(result, partial=missing))
        _ = Finalizer().finalize(result)
        if any(found.level == "error" for found in result.findings):
            raise DefinitionError(result.qualified_name[1:] or "bot", result.findings)
        for found in result.findings:
            warnings.warn_explicit(
                lints.LintWarning(found), lints.LintWarning, found.primary.filename, found.primary.lineno
            )
        return result


def _elements(given: collections.abc.Sequence[object], location: hsm.Location, first: int) -> list[hsm.Element]:
    """``given`` as bot elements: an hsm partial kept as a :class:`~mosfet.bot.firmware.Partial` of the
    firmware, and any other argument that is not an element as a :class:`~mosfet.bot.elements.Stray`."""

    return [
        item
        if isinstance(item, elements.Element)
        else firmware.Partial(partial=typing.cast(hsm.Element, item), index=first + index, location=location)
        if firmware.partial(item)
        else elements.Stray(value=item, index=first + index, location=location)
        for index, item in enumerate(given)
    ]


def define(name: str, *elements_: hsm.Element) -> elements.Model:
    """One bot, as data: ``name`` (lower snake case) and its elements.

    The hsm partials among them (``bot.state``, ``bot.transition``, ...) are the bot's one firmware.

    Raises:
        DefinitionError: If the bot file has any error; it carries every diagnostic found, rendered.
    """

    location = diagnostics.Location.capture()
    given: tuple[object, ...] = elements_
    written = typing.cast(object, name)
    if isinstance(written, str):
        qualified_name, first = posixpath.join("/", name), 1
    else:
        qualified_name, first, given = "/", 0, (name, *elements_)
    return RedefinableBot(
        qualified_name=qualified_name,
        owned_elements=_elements(given, location, first),
        location=location,
    ).redefine(elements.Model(qualified_name=""), [])


def redefine(model: elements.Model, *elements_: hsm.Element) -> elements.Model:
    """A new model with ``model``'s elements and ``elements_``, as ``hsm.redefine``; ``model`` is unchanged.

    Raises:
        DefinitionError: If the result has any error; an id given again is a duplicate.
    """

    location = diagnostics.Location.capture()
    return RedefinableBot(
        qualified_name=model.qualified_name,
        owned_elements=[*model.owned_elements, *_elements(elements_, location, 1)],
        location=model.location,
    ).redefine(elements.Model(qualified_name=""), [])


__all__ = ["DefinitionError", "Error", "Finalizer", "RedefinableBot", "Validator", "define", "redefine"]
