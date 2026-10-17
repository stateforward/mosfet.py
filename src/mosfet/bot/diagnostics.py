"""Diagnostics for bot files: rustc-style errors, clippy-style warnings, and what each one means.

A bot file is written in a DSL that its writer, a person or a model, may never have seen. So every
mistake comes back as plain text that says *what* is wrong, *where* (the span of the writer's own
call, with its source line and a caret under it), *why*, and *exactly how to rewrite it*::

    error[B0003]: device id `desk` is used twice
      --> greeter.py:15:5
       |
    14 |     bot.attach(computer.Computer(id="desk")),
       |     ---------------------------------------- first used here
    15 |     bot.attach(phone.Phone(id="desk")),
       |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ used again here
       |
       = note: a bot keys its devices by `id`
    help: give the second device its own id
       |
    15 |     bot.attach(phone.Phone(id="phone")),
       |                               ~~~~~~~
       = explain: bot.explain("B0003")

The pieces are hsm's own: :class:`Location` is ``hsm.Location`` with columns, and the validator that
finds the mistakes is an ``hsm.ModelValidator``. A :class:`Diagnostic` is also typed data, so a reader
that prefers structure takes ``Diagnostic.model_dump_json()``.

**No values.** A diagnostic names ids, classes, elements and fields. The source lines it echoes keep
the diagnosed span as written, and every string and number literal outside a span is masked as
``"…"`` or ``…``, so a credential written on a nearby line or argument never prints. A cause raised by
the writer's code renders as ``<Type>: <message>``, and a ``pydantic.ValidationError`` as ``loc: msg``
without its input.

Every code and lint has an :class:`Explanation`: :func:`explain` returns it, like ``rustc --explain``,
and ``docs/bot_diagnostics.md`` is generated from the same table by :func:`document`::

    python -c 'from mosfet.bot import diagnostics; print(diagnostics.document(), end="")' > docs/bot_diagnostics.md
"""

import ast
import collections.abc
import dataclasses
import difflib
import inspect
import io
import linecache
import os
import pathlib
import tokenize
import typing

import hsm
import pydantic

# Frames inside this package and inside hsm are the DSL's machinery, never the writer's call.
_SKIPPED_DIRECTORIES = (
    os.path.dirname(os.path.abspath(__file__)) + os.sep,
    os.path.dirname(os.path.abspath(hsm.__file__)) + os.sep,
)

Level = typing.Literal["error", "warning"]
"""How bad a diagnostic is: an ``error`` stops ``bot.define``; a ``warning`` is a lint that lets it finish."""

Applicability = typing.Literal["machine-applicable", "maybe-incorrect"]
"""Whether applying a suggestion as written is known to clear its diagnostic (rustc's applicability)."""


def _characters(line: str, offset: int) -> int:
    """The character column of UTF-8 byte ``offset`` in ``line``: frame positions count bytes."""

    return len(line.encode("utf-8")[:offset].decode("utf-8", errors="ignore"))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Location(hsm.Location):
    """Where a bot element was written: hsm's file and line, plus the columns of the call expression.

    ``col`` and ``end_col`` are 0-based UTF-8 byte offsets, ``end_col`` exclusive, on ``lineno`` and
    ``end_lineno``: Python's own positions, so capturing one reads no source. :meth:`Span.of` turns
    them into characters. A location with ``end_lineno == 0`` has no columns.
    """

    col: int = 0
    end_lineno: int = 0
    end_col: int = 0

    @classmethod
    @typing.override
    def capture(cls) -> "Location":
        """The call expression in the first frame outside ``mosfet.bot`` and ``hsm``: the writer's own call."""

        frame = inspect.currentframe()
        try:
            caller = frame
            while frame is not None:
                caller = frame
                if not os.path.abspath(frame.f_code.co_filename).startswith(_SKIPPED_DIRECTORIES):
                    break
                frame = frame.f_back
            if caller is None:
                return cls(filename="<unknown>", lineno=0)
            filename = caller.f_code.co_filename
            positions = inspect.getframeinfo(caller, context=0).positions
            lineno = caller.f_lineno
            if positions is None or None in (
                positions.lineno,
                positions.end_lineno,
                positions.col_offset,
                positions.end_col_offset,
            ):
                return cls(filename=filename, lineno=lineno)
            start = positions.lineno or lineno
            end = positions.end_lineno or start
            return cls(
                filename=filename,
                lineno=start,
                col=positions.col_offset or 0,
                end_lineno=end,
                end_col=positions.end_col_offset or 0,
            )
        finally:
            del frame


class Span(pydantic.BaseModel):
    """A stretch of a bot file's source, with what a diagnostic says about it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    filename: str = pydantic.Field(description="The source file, as Python names it.", examples=["greeter.py"])
    lineno: int = pydantic.Field(ge=0, description="The 1-based line the span starts on.", examples=[15])
    col: int = pydantic.Field(default=0, ge=0, description="The 0-based character column it starts at.", examples=[4])
    end_lineno: int = pydantic.Field(
        default=0,
        ge=0,
        description="The 1-based line it ends on; 0 when the span has no columns and covers its whole line.",
        examples=[15],
    )
    end_col: int = pydantic.Field(
        default=0, ge=0, description="The 0-based character column it ends before (exclusive).", examples=[38]
    )
    label: str = pydantic.Field(
        default="", description="What the diagnostic says about this span.", examples=["used again here"]
    )

    @classmethod
    def of(cls, location: hsm.Location, label: str = "") -> "Span":
        """The span of ``location``, labelled ``label``: its columns in characters when it has them.

        This reads the location's source line, so it belongs where a diagnostic is made, never on a
        path that defines a bot cleanly.
        """

        if isinstance(location, Location) and location.end_lineno >= location.lineno > 0:
            lines = _source(location.filename)
            first = lines[location.lineno - 1] if location.lineno <= len(lines) else ""
            last = lines[location.end_lineno - 1] if location.end_lineno <= len(lines) else ""
            return cls(
                filename=location.filename,
                lineno=location.lineno,
                col=_characters(first, location.col),
                end_lineno=location.end_lineno,
                end_col=_characters(last, location.end_col),
                label=label,
            )
        return cls(filename=location.filename, lineno=location.lineno, label=label)


class Suggestion(pydantic.BaseModel):
    """A rewrite: replace the source under ``span`` with ``replacement``."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    span: Span = pydantic.Field(description="The source to replace; an empty span inserts at its start.")
    replacement: str = pydantic.Field(
        description="The text to put there, in the DSL's real names.", examples=['phone.Phone(id="phone")']
    )
    applicability: Applicability = pydantic.Field(
        description=(
            "`machine-applicable` when applying the replacement as written yields a bot that defines "
            "without this diagnostic; `maybe-incorrect` when it needs the writer's judgment."
        ),
        examples=["machine-applicable"],
    )


class Help(pydantic.BaseModel):
    """How to fix a diagnostic: what to do, and when it can be written down, the rewrite itself."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    message: str = pydantic.Field(description="What to do.", examples=["give the second device its own id"])
    suggestion: Suggestion | None = pydantic.Field(default=None, description="The rewrite, when there is one.")


class Diagnostic(pydantic.BaseModel):
    """One thing wrong with a bot file: what, where, why, and how to rewrite it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    code: str = pydantic.Field(
        pattern=r"^(B[0-9]{4}|[a-z][a-z0-9_]*)$",
        description=(
            "A stable code: an error is `B` and four digits (`B00xx` from `bot.define`, `B01xx` from "
            "`bot.start` and `bot.run`); a lint is its snake_case name. A code's meaning never changes, "
            'and `bot.explain("<code>")` says what it means.'
        ),
        examples=["B0003", "no_devices"],
    )
    level: Level = pydantic.Field(description="`error` or `warning`.", examples=["error"])
    message: str = pydantic.Field(
        description="What is wrong, in one line.", examples=["device id `desk` is used twice"]
    )
    primary: Span = pydantic.Field(description="Where it is wrong: the writer's own call.")
    secondary: tuple[Span, ...] = pydantic.Field(
        default=(), description="Other places that explain it, such as where an id was first used."
    )
    notes: tuple[str, ...] = pydantic.Field(
        default=(), description="Why it is wrong.", examples=[["a bot keys its devices by `id`"]]
    )
    help: Help | None = pydantic.Field(default=None, description="How to fix it.")


class Explanation(pydantic.BaseModel):
    """What one code or lint means, why the DSL has it, and a wrong bot file beside a corrected one."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    code: str = pydantic.Field(pattern=r"^(B[0-9]{4}|[a-z][a-z0-9_]*)$", examples=["B0003"])
    level: Level = pydantic.Field(examples=["error"])
    title: str = pydantic.Field(
        description="The mistake, in one line.", examples=["a device or ability id is used twice"]
    )
    meaning: str = pydantic.Field(description="What the code means.")
    why: str = pydantic.Field(description="Why the DSL requires it.")
    wrong: str = pydantic.Field(description="Bot file code that has the mistake.")
    right: str = pydantic.Field(description="The same code, corrected.")


EXPLANATIONS: tuple[Explanation, ...] = ()
"""Every code and lint, in code order. A retired code keeps its entry and is never reused."""


def explain(code: str) -> str:
    """What ``code`` means, why, and a wrong bot file beside a corrected one, like ``rustc --explain``.

    Raises:
        ValueError: If ``code`` is no diagnostic's code; the message names the nearest ones.
    """

    for explanation in EXPLANATIONS:
        if explanation.code == code:
            return "\n".join(
                (
                    f"{explanation.level}[{explanation.code}]: {explanation.title}",
                    "",
                    explanation.meaning,
                    "",
                    f"Why: {explanation.why}",
                    "",
                    "Wrong:",
                    "",
                    _indent(explanation.wrong),
                    "",
                    "Right:",
                    "",
                    _indent(explanation.right),
                )
            )
    nearest = difflib.get_close_matches(code, [explanation.code for explanation in EXPLANATIONS], n=3)
    hint = f"; did you mean {', '.join(f'`{near}`' for near in nearest)}?" if nearest else ""
    raise ValueError(f"unknown diagnostic code `{code}`{hint}")


def document() -> str:
    """``docs/bot_diagnostics.md``: the index and every explanation, generated from :data:`EXPLANATIONS`."""

    lines = [
        "# Bot diagnostics",
        "",
        "<!-- Generated from mosfet.bot.diagnostics.EXPLANATIONS. Do not edit by hand; regenerate with",
        "python -c 'from mosfet.bot import diagnostics; print(diagnostics.document(), end=\"\")' > docs/bot_diagnostics.md -->",
        "",
        "`bot.define` reports every mistake in a bot file at once, the way rustc does: each with a stable",
        "code, the source span, notes on why, and a `help:` that shows the rewrite. Errors (`B00xx` from",
        "`bot.define`, `B01xx` from `bot.start` and `bot.run`) stop the bot; lints (snake_case names) are",
        'warnings. `bot.explain("<code>")` prints any entry below.',
        "",
    ]
    lines.extend(
        f"- [`{explanation.code}`](#{explanation.code.lower()}) ({explanation.level}): {explanation.title}"
        for explanation in EXPLANATIONS
    )
    for explanation in EXPLANATIONS:
        lines.extend(
            (
                "",
                f"## {explanation.code}",
                "",
                f"**{explanation.level}:** {explanation.title}",
                "",
                explanation.meaning,
                "",
                f"Why: {explanation.why}",
                "",
                "Wrong:",
                "",
                "```python",
                explanation.wrong,
                "```",
                "",
                "Right:",
                "",
                "```python",
                explanation.right,
                "```",
            )
        )
    return "\n".join(lines) + "\n"


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" if line else "" for line in text.splitlines())


def cause(error: BaseException) -> str:
    """``error`` as ``<Type>: <message>``; a ``pydantic.ValidationError`` as ``loc: msg``, never its input."""

    if isinstance(error, pydantic.ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in detail['loc']) or '<value>'}: {detail['msg']}"
            for detail in error.errors(include_url=False, include_context=False, include_input=False)
        )
        return f"ValidationError: {details}"
    return f"{type(error).__name__}: {error}"


def call(location: Span) -> ast.Call | None:
    """The call expression under ``location``, parsed, with character positions in the file's coordinates.

    ``None`` when the source is not available or the span is not a call (a span without columns).
    """

    if location.end_lineno < location.lineno or location.lineno < 1:
        return None
    lines = _source(location.filename)
    if len(lines) < location.end_lineno:
        return None
    segment = [line.rstrip("\r\n") for line in lines[location.lineno - 1 : location.end_lineno]]
    segment[-1] = segment[-1][: location.end_col]
    segment[0] = segment[0][location.col :]
    try:
        expression = ast.parse("(" + "\n".join(segment) + ")", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call):
        return None
    for node in ast.walk(expression):
        if not isinstance(node, ast.expr | ast.keyword):
            continue
        for attribute in ("lineno", "end_lineno"):
            line = typing.cast(int | None, getattr(node, attribute, None))
            if line is not None:
                setattr(node, attribute, line + location.lineno - 1)
        for attribute, line_attribute in (("col_offset", "lineno"), ("end_col_offset", "end_lineno")):
            column = typing.cast(int | None, getattr(node, attribute, None))
            line = typing.cast(int | None, getattr(node, line_attribute, None))
            if column is None or line is None:
                continue
            text = segment[line - location.lineno] if line - location.lineno < len(segment) else ""
            # ast columns count UTF-8 bytes; the first line also carries the "(" added above.
            prefix = "(" + text if line == location.lineno else text
            characters = _characters(prefix, column)
            setattr(node, attribute, characters - 1 + location.col if line == location.lineno else characters)
    return expression


def node_span(location: Span, node: ast.expr | ast.keyword, label: str = "") -> Span:
    """The span of ``node``, an expression from :func:`call` under ``location``, labelled ``label``."""

    return Span(
        filename=location.filename,
        lineno=node.lineno,
        col=node.col_offset,
        end_lineno=node.end_lineno or node.lineno,
        end_col=node.end_col_offset or node.col_offset,
        label=label,
    )


def text(span: Span) -> str:
    """The source under ``span`` as written (empty when the source is not available)."""

    lines = _source(span.filename)
    if span.end_lineno < span.lineno or len(lines) < span.end_lineno or span.lineno < 1:
        return ""
    chunk = [line.rstrip("\r\n") for line in lines[span.lineno - 1 : span.end_lineno]]
    chunk[-1] = chunk[-1][: span.end_col]
    chunk[0] = chunk[0][span.col :]
    return "\n".join(chunk)


def _source(filename: str) -> list[str]:
    linecache.checkcache(filename)
    return linecache.getlines(filename)


# A literal's extent in its file: (start line, start column, end line, end column), columns in characters.
_Literal = tuple[int, int, int, int]


def _literals(filename: str) -> tuple[_Literal, ...] | None:
    """Every string and number literal in ``filename``; ``None`` when it does not tokenize."""

    literals: list[_Literal] = []
    fstring: tuple[int, int] | None = None
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO("".join(_source(filename))).readline):
            if token.type == tokenize.FSTRING_START:
                if depth == 0:
                    fstring = token.start
                depth += 1
            elif token.type == tokenize.FSTRING_END:
                depth -= 1
                if depth == 0 and fstring is not None:
                    literals.append((*fstring, *token.end))
                    fstring = None
            elif depth == 0 and token.type in (tokenize.STRING, tokenize.NUMBER):
                literals.append((*token.start, *token.end))
    except (tokenize.TokenError, SyntaxError):
        return None
    return tuple(literals)


def _mask(
    line: str, lineno: int, literals: tuple[_Literal, ...] | None, revealed: list[tuple[int, int]]
) -> tuple[str, typing.Callable[[int], int]]:
    """``line`` with every literal outside ``revealed`` masked, and the map from old columns to new.

    Without ``literals`` (the file does not tokenize) everything outside ``revealed`` is masked.
    """

    pieces: list[tuple[int, int, str]] = []
    if literals is None:
        cursor = len(line) - len(line.lstrip())
        for low, high in [*sorted(revealed), (len(line), len(line))]:
            if line[cursor:low].strip():
                pieces.append((cursor, low, "…"))
            cursor = max(cursor, high)
    for start_line, start_col, end_line, end_col in literals or ():
        if not start_line <= lineno <= end_line:
            continue
        start = start_col if start_line == lineno else 0
        end = end_col if end_line == lineno else len(line)
        if any(low <= start and end <= high for low, high in revealed):
            continue
        is_string = line[start:end][-1:] in ("'", '"') or start_line != end_line
        pieces.append((start, end, '"…"' if is_string and start_line == lineno else "…"))
    pieces.sort()
    masked: list[str] = []
    cursor = 0
    for start, end, replacement in pieces:
        masked.append(line[cursor:start])
        masked.append(replacement)
        cursor = end
    masked.append(line[cursor:])

    def column(old: int) -> int:
        shift = 0
        for start, end, replacement in pieces:
            if end <= old:
                shift += len(replacement) - (end - start)
        return old + shift

    return "".join(masked), column


def _display(filename: str, relative_to: pathlib.Path | None) -> str:
    root = pathlib.Path.cwd() if relative_to is None else relative_to
    path = pathlib.Path(filename)
    if not path.is_absolute():
        return filename
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return filename


def _extent(span: Span, line: str) -> tuple[int, int]:
    """The columns ``span`` covers on its first line: to its end, or to the end of the line it runs past."""

    if span.end_lineno == 0:
        stripped = line.rstrip()
        return len(stripped) - len(stripped.lstrip()), len(stripped)
    end = span.end_col if span.end_lineno == span.lineno else len(line.rstrip())
    return span.col, max(end, span.col + 1)


def render(diagnostic: Diagnostic, *, relative_to: pathlib.Path | None = None) -> str:
    """``diagnostic`` as rustc prints one: header, ``-->``, source with carets, notes, help, explain.

    File names under ``relative_to`` (the current directory when ``None``) print relative to it.
    """

    marked = [(diagnostic.primary, "^"), *((span, "-") for span in diagnostic.secondary)]
    files = [diagnostic.primary.filename]
    files.extend(span.filename for span, _ in marked if span.filename not in files)
    suggestion = diagnostic.help.suggestion if diagnostic.help is not None else None
    numbers = [span.lineno for span, _ in marked]
    if suggestion is not None:
        numbers.append(suggestion.span.lineno)
    width = len(str(max(numbers)))
    gutter = " " * width
    out = [f"{diagnostic.level}[{diagnostic.code}]: {diagnostic.message}"]
    for filename in files:
        spans = sorted((item for item in marked if item[0].filename == filename), key=lambda item: item[0].lineno)
        first = spans[0][0] if filename != diagnostic.primary.filename else diagnostic.primary
        arrow = "-->" if filename == diagnostic.primary.filename else ":::"
        out.append(f"{gutter}{arrow} {_display(filename, relative_to)}:{first.lineno}:{first.col + 1}")
        lines = _source(filename)
        literals = _literals(filename)
        shown = sorted({span.lineno for span, _ in spans if 0 < span.lineno <= len(lines)})
        if not shown:
            continue
        out.append(f"{gutter} |")
        previous = 0
        for lineno in shown:
            if previous and lineno > previous + 1:
                out.append("...")
            previous = lineno
            line = lines[lineno - 1].rstrip("\r\n")
            here = [(span, mark) for span, mark in spans if span.lineno == lineno]
            extents = [_extent(span, line) for span, _ in here]
            masked, column = _mask(line, lineno, literals, extents)
            out.append(f"{lineno:>{width}} | {masked}".rstrip())
            for (span, mark), (start, end) in sorted(zip(here, extents, strict=True), key=lambda item: item[1]):
                new_start, new_end = column(start), column(end)
                out.append(f"{gutter} | {' ' * new_start}{mark * (new_end - new_start)} {span.label}".rstrip())
    out.append(f"{gutter} |")
    out.extend(f"{gutter} = note: {note}" for note in diagnostic.notes)
    if diagnostic.help is not None:
        out.append(f"help: {diagnostic.help.message}")
        if suggestion is not None:
            out.extend(_suggested(suggestion, diagnostic.primary, width))
    out.append(f'{gutter} = explain: bot.explain("{diagnostic.code}")')
    return "\n".join(out)


def _suggested(suggestion: Suggestion, primary: Span, width: int) -> list[str]:
    """The suggestion's line with the replacement applied and ``~`` under it, literals masked as elsewhere."""

    span = suggestion.span
    lines = _source(span.filename)
    if span.end_lineno != span.lineno or "\n" in suggestion.replacement or not 0 < span.lineno <= len(lines):
        return []
    line = lines[span.lineno - 1].rstrip("\r\n")
    start, end = span.col, max(span.end_col, span.col)
    delta = len(suggestion.replacement) - (end - start)
    edited = line[:start] + suggestion.replacement + line[end:]

    def moved(column: int) -> int:
        return column if column <= start else (start if column < end else column + delta)

    found = _literals(span.filename)
    literals = (
        None
        if found is None
        else tuple(
            (lineno, moved(col), end_line, moved(end_col) if end_line == span.lineno else end_col)
            for lineno, col, end_line, end_col in found
            if not (lineno == end_line == span.lineno and start <= col and end_col <= end)
        )
    )
    revealed = [(start, start + len(suggestion.replacement))]
    if primary.filename == span.filename and primary.lineno == span.lineno:
        low, high = _extent(primary, line)
        revealed.append((moved(low), high + delta if high >= end else moved(high)))
    masked, column = _mask(edited, span.lineno, literals, revealed)
    gutter = " " * width
    marker_start, marker_end = column(start), column(start + len(suggestion.replacement))
    return [
        f"{gutter} |",
        f"{span.lineno:>{width}} | {masked}".rstrip(),
        f"{gutter} | {' ' * marker_start}{'~' * (marker_end - marker_start)}".rstrip(),
    ]


def summary(diagnostics: collections.abc.Iterable[Diagnostic], *, action: str, name: str) -> str:
    """The closing line: ``error: could not <action> bot `<name>`: 2 errors, 1 warning``."""

    levels = [diagnostic.level for diagnostic in diagnostics]
    errors, warnings = levels.count("error"), levels.count("warning")
    counts = [
        f"{count} {noun}{'' if count == 1 else 's'}"
        for count, noun in ((errors, "error"), (warnings, "warning"))
        if count
    ]
    if errors:
        return f"error: could not {action} bot `{name}`: {', '.join(counts)}"
    return f"warning: bot `{name}` has {', '.join(counts) or 'no warnings'}"


def report(
    diagnostics: collections.abc.Sequence[Diagnostic],
    *,
    action: str,
    name: str,
    relative_to: pathlib.Path | None = None,
) -> str:
    """Every diagnostic rendered, in order, then the :func:`summary` line."""

    rendered = [render(diagnostic, relative_to=relative_to) for diagnostic in diagnostics]
    return "\n\n".join([*rendered, summary(diagnostics, action=action, name=name)])


__all__ = [
    "EXPLANATIONS",
    "Applicability",
    "Diagnostic",
    "Explanation",
    "Help",
    "Level",
    "Location",
    "Span",
    "Suggestion",
    "call",
    "cause",
    "document",
    "explain",
    "node_span",
    "render",
    "report",
    "summary",
    "text",
]
