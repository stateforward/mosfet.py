"""Clippy-style lints: warnings about a bot file that defines, but probably not as meant.

A lint reads only a defined model's structure (its members and their classes), never a value the
bot file passed in, and never runs anything. Each lint is a warning named in snake case and on by
default; ``bot.allow("<lint>")`` allows it for one bot when what it warns about is deliberate.
``bot.define`` emits each one as a :class:`LintWarning` (pytest's ``-W error::mosfet.bot.LintWarning`` makes
them fail), and :func:`check` returns them as diagnostics.

A new lint is a function here, added to :data:`LINTS` with its golden test and its explanation.
"""

import collections.abc
import difflib
import re

from mosfet import abilities, event
from mosfet.abilities import memory

import mosfet.device

from . import diagnostics, elements

_OPENINGS = ("always", "never", "you must", "reply", "respond", "do not", "don't", "dont")
_SECOND_PERSON = frozenset({"you", "your", "yours", "yourself", "you're", "youre"})
_WORDS = re.compile(r"[a-z']+")
_MAX_TRAIT_WORDS = 12


class LintWarning(UserWarning):
    """A lint's warning: the rendered diagnostic, with the diagnostic itself as ``diagnostic``."""

    def __init__(self, diagnostic: diagnostics.Diagnostic) -> None:
        super().__init__(diagnostics.render(diagnostic))
        self.diagnostic: diagnostics.Diagnostic = diagnostic


def _warning(
    lint: str,
    message: str,
    primary: diagnostics.Span,
    *,
    notes: tuple[str, ...] = (),
    help: diagnostics.Help | None = None,
) -> diagnostics.Diagnostic:
    return diagnostics.Diagnostic(
        code=lint,
        level="warning",
        message=message,
        primary=primary,
        notes=(*notes, f'allow it with `bot.allow("{lint}")` if this is deliberate'),
        help=help,
    )


def _members[TMember: elements.Member](model: elements.Model, kind: type[TMember]) -> list[TMember]:
    return [member for member in model.members.values() if isinstance(member, kind)]


def _senses(model: elements.Model) -> list[elements.AttachmentElement]:
    """The devices the bot holds and its input abilities: what it can perceive."""

    return [
        member
        for member in _members(model, elements.AttachmentElement)
        if isinstance(member.instance, mosfet.device.Device) or "input" in member.roles
    ]


def _removal(
    model: elements.Model, member: elements.Member, applicability: diagnostics.Applicability
) -> diagnostics.Suggestion | None:
    """Deleting ``member``'s element from the ``bot.define`` call, when it was written there."""

    span = diagnostics.removal(diagnostics.Span.of(model.location), diagnostics.Span.of(member.location))
    if span is None:
        return None
    return diagnostics.Suggestion(span=span, replacement="", applicability=applicability)


def no_devices(model: elements.Model) -> list[diagnostics.Diagnostic]:
    """A bot with a cognition but nothing it can perceive or act through, so nothing can wake it."""

    if not _members(model, elements.CognitionElement) or _senses(model):
        return []
    call = diagnostics.Span.of(model.location)
    return [
        _warning(
            "no_devices",
            f"bot `{model.qualified_name[1:]}` has a cognition but nothing it can perceive or act through",
            diagnostics.head(call, "no device or input ability"),
            notes=("a bot that perceives nothing is never woken; this is allowed, and it is what the file says",),
            help=diagnostics.Help(
                message="attach something it can perceive, for example a computer a person can type into",
                suggestion=diagnostics.append(call, 'bot.attach(computer.Computer(id="desk"))'),
            ),
        )
    ]


def ability_without_tools(model: elements.Model) -> list[diagnostics.Diagnostic]:
    """An acquired ability whose model offers cognition no event, so nothing can use or address it."""

    found: list[diagnostics.Diagnostic] = []
    for member in _members(model, elements.AttachmentElement):
        instance = member.instance
        if not isinstance(instance, abilities.Ability) or member.roles != ["attach"]:
            continue
        model_of = type(instance).model
        if model_of is not None and any(offered.kind == event.EventKind for offered in model_of.events.values()):
            continue
        written = diagnostics.argument(diagnostics.Span.of(member.location), 0)
        text = "..." if written is None else diagnostics.text(written[0])
        if isinstance(instance, memory.StmMemory):
            advice = f"a sensory register is the body's: `bot.register({text})`"
        else:
            advice = f"`bot.input({text})` if it is a sense, `bot.output({text})` if it is an effector"
        found.append(
            _warning(
                "ability_without_tools",
                f"ability `{member.name()}` offers cognition nothing to select",
                diagnostics.Span.of(member.location, f"`{type(instance).__name__}` has no tool events"),
                notes=("an acquired ability is used through the events it offers cognition; this one offers none",),
                help=diagnostics.Help(message=advice),
            )
        )
    return found


def instruction_like_traits(model: elements.Model) -> list[diagnostics.Diagnostic]:
    """A trait that reads as an instruction to the bot rather than as who it is."""

    found: list[diagnostics.Diagnostic] = []
    for member in _members(model, elements.TraitsElement):
        call = diagnostics.Span.of(member.location)
        for index, trait in enumerate(member.traits):
            if not isinstance(trait, str):
                continue
            lowered = trait.strip().lower()
            words = _WORDS.findall(lowered)
            reason = (
                "it opens like an instruction"
                if any(lowered == opening or lowered.startswith(f"{opening} ") for opening in _OPENINGS)
                else "it speaks to the bot as `you`"
                if _SECOND_PERSON.intersection(words)
                else f"it is more than {_MAX_TRAIT_WORDS} words"
                if len(words) > _MAX_TRAIT_WORDS
                else None
            )
            if reason is None:
                continue
            argument = diagnostics.argument(call, index)
            primary = call if argument is None else argument[0]
            found.append(
                _warning(
                    "instruction_like_traits",
                    f"trait {index + 1} reads as an instruction: {reason}",
                    primary.model_copy(update={"label": "an instruction, not a trait"}),
                    notes=("traits are who the bot is; how it acts is the bot's own to develop, never scripted",),
                    help=diagnostics.Help(message='give a trait in a word or a short phrase: "patient", "curious"'),
                )
            )
    return found


def empty_timeouts(model: elements.Model) -> list[diagnostics.Diagnostic]:
    """``bot.timeouts()`` with no timeout in it, which changes nothing."""

    return [
        _warning(
            "empty_timeouts",
            "`bot.timeouts()` sets no timeout",
            diagnostics.Span.of(member.location, "changes nothing"),
            notes=("with no values, the body keeps every default timeout",),
            help=diagnostics.Help(
                message="remove it, or set a timeout", suggestion=_removal(model, member, "machine-applicable")
            ),
        )
        for member in _members(model, elements.TimeoutsElement)
        if member.processing is None and member.deactivation is None and member.attachment is None
    ]


def unused_register(model: elements.Model) -> list[diagnostics.Diagnostic]:
    """A sensory register on a bot with no device and no input ability, so nothing is ever recorded."""

    if _senses(model):
        return []
    return [
        _warning(
            "unused_register",
            "the sensory register records nothing on a bot that perceives nothing",
            diagnostics.Span.of(member.location, "records nothing"),
            notes=("the register records stimuli the body admits, and only devices and input abilities bring them",),
            help=diagnostics.Help(
                message="remove it, or attach a sense", suggestion=_removal(model, member, "maybe-incorrect")
            ),
        )
        for member in _members(model, elements.RegisterElement)
    ]


LINTS: dict[str, collections.abc.Callable[[elements.Model], list[diagnostics.Diagnostic]]] = {
    "no_devices": no_devices,
    "ability_without_tools": ability_without_tools,
    "instruction_like_traits": instruction_like_traits,
    "empty_timeouts": empty_timeouts,
    "unused_register": unused_register,
}
"""Every lint by name, each on by default."""


def allowed(model: elements.Model) -> frozenset[str]:
    """The lint names ``model``'s ``bot.allow`` allows.

    A name that is no lint (``B0018``) allows the lint its help names, the nearest one: that is the
    lint the bot file means, so one pass never warns about what it was written to allow.
    """

    return frozenset(
        lint if lint in LINTS else next(iter(difflib.get_close_matches(lint, list(LINTS), n=1)), lint)
        for member in _members(model, elements.AllowElement)
        for lint in member.lints
        if isinstance(lint, str)
    )


ABSENCE: frozenset[str] = frozenset({"no_devices", "unused_register"})
"""The lints that judge what a bot lacks, which a part that failed to define would make it seem to lack."""


def lint(model: elements.Model, *, partial: bool = False) -> list[diagnostics.Diagnostic]:
    """Every lint ``model`` does not allow, run on it, in source order.

    ``partial`` is a model missing a part that failed to define: the :data:`ABSENCE` lints wait for it.
    """

    allowing = allowed(model) | (ABSENCE if partial else frozenset[str]())
    found = [found for name, run in LINTS.items() if name not in allowing for found in run(model)]
    return diagnostics.ordered(found, model.location.filename)


def check(*models: elements.Model) -> tuple[diagnostics.Diagnostic, ...]:
    """The lint warnings of defined ``models``, as ``bot.define`` found them, each bot's ``bot.allow`` kept."""

    return tuple(found for model in models for found in lint(model))


__all__ = [
    "ABSENCE",
    "LINTS",
    "LintWarning",
    "ability_without_tools",
    "allowed",
    "check",
    "empty_timeouts",
    "instruction_like_traits",
    "lint",
    "no_devices",
    "unused_register",
]
