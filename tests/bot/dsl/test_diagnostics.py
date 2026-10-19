"""rustc-style diagnostics: columns, the rendered text, masking, causes, and explanations (DSL-27, DSL-34)."""

import ast
import linecache
import pathlib
import typing

import hsm
import pydantic
import pytest

from mosfet import bot
from mosfet.bot import diagnostics

GOLDEN = pathlib.Path(__file__).parent / "golden"
DOCS = pathlib.Path(__file__).parents[3] / "docs" / "bot_diagnostics.md"

GREETER = '''\
"""A greeter bot with two devices that share one id."""

from mosfet import bot
from mosfet.devices import computer, phone

API_KEY = "sk-SENTINEL-NEIGHBOUR"

processor, decider, server = object(), object(), object()

greeter = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.attach(computer.Computer(id="desk", transport=server)),
    bot.attach(phone.Phone(id="desk")), bot.traits("sk-SENTINEL-SAME-LINE", 4242),
)
'''


@pytest.fixture
def greeter(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    path = tmp_path / "greeter.py"
    _ = path.write_text(GREETER, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return path


def _span(path: pathlib.Path, needle: str, label: str = "", *, occurrence: int = 0) -> bot.Span:
    """The span of the ``occurrence``-th ``needle`` in ``path``."""

    lines = path.read_text(encoding="utf-8").splitlines()
    found = [(number, line.index(needle)) for number, line in enumerate(lines, start=1) if needle in line]
    number, column = found[occurrence]
    return bot.Span(
        filename=str(path), lineno=number, col=column, end_lineno=number, end_col=column + len(needle), label=label
    )


def _duplicate(path: pathlib.Path) -> bot.Diagnostic:
    second = _span(path, 'bot.attach(phone.Phone(id="desk"))', "used again here")
    value = _span(path, '"desk"', occurrence=1)
    return bot.Diagnostic(
        code="B0003",
        level="error",
        message="device id `desk` is used twice",
        primary=second,
        secondary=(_span(path, 'bot.attach(computer.Computer(id="desk", transport=server))', "first used here"),),
        notes=("a bot keys its devices by `id`",),
        help=bot.Help(
            message="give the second device its own id",
            suggestion=bot.Suggestion(span=value, replacement='"phone"', applicability="machine-applicable"),
        ),
    )


def _golden(name: str) -> str:
    return (GOLDEN / name).read_text(encoding="utf-8")


def test_location_captures_the_callers_call_expression_with_its_columns() -> None:
    location = bot.Location.capture()

    line = linecache.getline(__file__, location.lineno)
    assert location.filename == __file__
    assert location.end_lineno == location.lineno
    assert line[location.col : location.end_col] == "bot.Location.capture()"


def test_location_columns_are_python_byte_offsets_and_a_span_counts_characters() -> None:
    pair = ("é—ü", bot.Location.capture())

    location = pair[1]
    line = linecache.getline(__file__, location.lineno)
    assert line.encode()[location.col : location.end_col] == b"bot.Location.capture()"
    span = bot.Span.of(location, "here")
    assert line[span.col : span.end_col] == "bot.Location.capture()"
    assert diagnostics.text(span) == "bot.Location.capture()"
    assert span.label == "here"


def test_a_span_of_a_location_without_columns_covers_its_line() -> None:
    span = bot.Span.of(hsm.Location(filename="x.py", lineno=3))

    assert (span.lineno, span.end_lineno) == (3, 0)


def test_location_is_hsms_location_with_columns() -> None:
    location = bot.Location.capture()

    assert isinstance(location, hsm.Location)
    assert location.col > 0


def test_a_diagnostic_renders_like_rustc(greeter: pathlib.Path) -> None:
    assert diagnostics.render(_duplicate(greeter)) == _golden("render_duplicate.txt").rstrip("\n")


def test_help_without_a_suggestion_prints_only_its_message(greeter: pathlib.Path) -> None:
    diagnostic = bot.Diagnostic(
        code="B0004",
        level="error",
        message="`bot.attach` takes a constructed device or ability, not a class",
        primary=_span(greeter, "computer.Computer", "this is the class `computer.Computer`"),
        help=bot.Help(message="construct it"),
    )

    assert diagnostics.render(diagnostic) == _golden("render_help_only.txt").rstrip("\n")


def test_literals_outside_the_span_never_print(greeter: pathlib.Path) -> None:
    rendered = diagnostics.render(_duplicate(greeter))

    assert "SENTINEL" not in rendered
    assert "4242" not in rendered
    assert 'bot.traits("…", …)' in rendered
    assert 'phone.Phone(id="desk")' in rendered


def test_masking_keeps_the_carets_under_the_span(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "masked.py"
    _ = path.write_text('pair = ("a long secret literal", 123456, bot.attach(x))\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    diagnostic = bot.Diagnostic(code="B0006", level="error", message="m", primary=_span(path, "bot.attach(x)", "here"))

    lines = diagnostics.render(diagnostic).splitlines()

    source = next(line for line in lines if line.startswith("1 |"))
    carets = lines[lines.index(source) + 1]
    assert source == '1 | pair = ("…", …, bot.attach(x))'
    assert carets.index("^") == source.index("bot.attach")
    assert carets.endswith("^^^^^^^^^^^^^ here")


def test_a_multi_line_string_outside_the_span_is_masked_on_every_line(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "multi.py"
    _ = path.write_text('x = """SENTINEL-A\nSENTINEL-B""", bot.attach(y)\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    diagnostic = bot.Diagnostic(
        code="B0006",
        level="error",
        message="m",
        primary=_span(path, "bot.attach(y)", "here"),
        secondary=(bot.Span(filename=str(path), lineno=1, col=0, end_lineno=1, end_col=1, label="x"),),
    )

    rendered = diagnostics.render(diagnostic)

    assert "SENTINEL" not in rendered
    assert '1 | x = "…"' in rendered
    assert "2 | …, bot.attach(y)" in rendered


def test_an_f_string_outside_the_span_is_masked(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "fstring.py"
    _ = path.write_text('x = (f"SENTINEL-{1}", bot.attach(y))\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    diagnostic = bot.Diagnostic(code="B0006", level="error", message="m", primary=_span(path, "bot.attach(y)"))

    rendered = diagnostics.render(diagnostic)

    assert "SENTINEL" not in rendered
    assert '1 | x = ("…", bot.attach(y))' in rendered


def test_a_file_that_does_not_tokenize_masks_everything_outside_the_span(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "broken.py"
    _ = path.write_text('x = ("SENTINEL", bot.attach(y))\n"""unterminated\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    diagnostic = bot.Diagnostic(code="B0006", level="error", message="m", primary=_span(path, "bot.attach(y)", "here"))

    lines = diagnostics.render(diagnostic).splitlines()

    assert "1 | …bot.attach(y)…" in lines
    assert "  |  ^^^^^^^^^^^^^ here" in lines


def test_a_span_in_another_file_gets_its_own_header(greeter: pathlib.Path, tmp_path: pathlib.Path) -> None:
    other = tmp_path / "parts.py"
    _ = other.write_text('DESK = computer.Computer(id="desk")\nPORT = 8080\n', encoding="utf-8")
    diagnostic = _duplicate(greeter).model_copy(
        update={"secondary": (_span(other, 'computer.Computer(id="desk")', "first used here"),)}
    )

    assert diagnostics.render(diagnostic) == _golden("render_other_file.txt").rstrip("\n")


def test_a_span_without_columns_underlines_its_whole_line(greeter: pathlib.Path) -> None:
    diagnostic = bot.Diagnostic(
        code="B0006",
        level="error",
        message="m",
        primary=bot.Span(filename=str(greeter), lineno=12, label="hsm says so"),
    )

    lines = diagnostics.render(diagnostic).splitlines()

    assert "12 |     bot.cognition(processor, intuition=decider)," in lines
    assert "   |     " + "^" * len("bot.cognition(processor, intuition=decider),") + " hsm says so" in lines


def test_without_source_only_the_header_and_location_print() -> None:
    diagnostic = bot.Diagnostic(
        code="B0006", level="error", message="m", primary=bot.Span(filename="<string>", lineno=3, col=2, end_lineno=3)
    )

    assert diagnostics.render(diagnostic).splitlines() == [
        "error[B0006]: m",
        " --> <string>:3:3",
        "  |",
        '  = explain: bot.explain("B0006")',
    ]


def test_every_rendered_diagnostic_ends_with_its_explain_line(greeter: pathlib.Path) -> None:
    assert diagnostics.render(_duplicate(greeter)).splitlines()[-1] == '   = explain: bot.explain("B0003")'


def test_a_file_outside_the_directory_prints_its_whole_path(greeter: pathlib.Path) -> None:
    rendered = diagnostics.render(_duplicate(greeter), relative_to=pathlib.Path("/nonexistent"))

    assert f"  --> {greeter}:14:5" in rendered


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        (("error",), "error: could not define bot `greeter`: 1 error"),
        (("error", "error", "warning"), "error: could not define bot `greeter`: 2 errors, 1 warning"),
        (("warning", "warning"), "warning: bot `greeter` has 2 warnings"),
    ],
)
def test_the_summary_counts_errors_and_warnings(levels: tuple[diagnostics.Level, ...], expected: str) -> None:
    found = [
        bot.Diagnostic(code="B0006", level=level, message="m", primary=bot.Span(filename="x", lineno=1))
        for level in levels
    ]

    assert diagnostics.summary(found, action="define", name="greeter") == expected


def test_a_report_is_every_diagnostic_then_the_summary(greeter: pathlib.Path) -> None:
    duplicate = _duplicate(greeter)

    text = diagnostics.report([duplicate, duplicate], action="define", name="greeter")

    assert text == "\n\n".join(
        [
            diagnostics.render(duplicate),
            diagnostics.render(duplicate),
            "error: could not define bot `greeter`: 2 errors",
        ]
    )


def test_a_pydantic_cause_renders_without_its_input() -> None:
    class Secret(pydantic.BaseModel):
        port: int

    with pytest.raises(pydantic.ValidationError) as caught:
        _ = Secret.model_validate({"port": "sk-SENTINEL"})

    rendered = diagnostics.cause(caught.value)

    assert rendered.startswith("ValidationError: port: ")
    assert "SENTINEL" not in rendered


def test_another_cause_renders_as_its_type_and_message() -> None:
    assert diagnostics.cause(ValueError("no transport")) == "ValueError: no transport"


def test_a_diagnostic_is_json_with_described_fields(greeter: pathlib.Path) -> None:
    duplicate = _duplicate(greeter)

    assert bot.Diagnostic.model_validate_json(duplicate.model_dump_json()) == duplicate
    properties = typing.cast(dict[str, dict[str, object]], bot.Diagnostic.model_json_schema()["properties"])
    assert all("description" in field for field in properties.values())


def test_call_parses_the_written_call_with_file_columns(greeter: pathlib.Path) -> None:
    span = _span(greeter, 'bot.attach(phone.Phone(id="desk"))')

    node = diagnostics.call(span)

    assert isinstance(node, ast.Call)
    argument = diagnostics.node_span(span, node.args[0])
    assert diagnostics.text(argument) == 'phone.Phone(id="desk")'


def test_call_spans_lines(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "lines.py"
    _ = path.write_text('model = bot.define(\n    "é",\n    bot.attach(x),\n)\n', encoding="utf-8")
    location = bot.Span(filename=str(path), lineno=1, col=8, end_lineno=4, end_col=1)

    node = diagnostics.call(location)

    assert node is not None
    assert [diagnostics.text(diagnostics.node_span(location, argument)) for argument in node.args] == [
        '"é"',
        "bot.attach(x)",
    ]


def test_call_is_none_without_source_or_columns() -> None:
    assert diagnostics.call(bot.Span(filename="<string>", lineno=1, col=0, end_lineno=1, end_col=4)) is None
    assert diagnostics.call(bot.Span(filename=__file__, lineno=1)) is None


def test_the_explanation_of_a_missing_cognition_names_where_a_processor_comes_from() -> None:
    explained = " ".join(bot.explain("B0002").split())  # the prose is wrapped

    assert "`from mosfet.abilities import processing`" in explained
    assert "`testing.Scripted()`" in explained
    assert "`from mosfet.bot import testing`" in explained


def test_the_explanation_of_a_stray_names_the_firmware_elements_too() -> None:
    explained = " ".join(bot.explain("B0017").split())  # the prose is wrapped

    for name in ("allow", "state", "initial", "transition", "attribute", "on", "after", "target"):
        assert f"`bot.{name}`" in explained


def test_explain_rejects_an_unknown_code() -> None:
    with pytest.raises(ValueError, match="unknown diagnostic code `B9999`"):
        _ = bot.explain("B9999")


def test_every_explanation_has_a_unique_code() -> None:
    codes = [explanation.code for explanation in diagnostics.EXPLANATIONS]

    assert len(codes) == len(set(codes))


def test_docs_in_sync() -> None:
    assert DOCS.read_text(encoding="utf-8") == diagnostics.document()
