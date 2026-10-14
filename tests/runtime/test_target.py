import pytest
from mosfet import runtime

from tests.runtime.fixtures import factories


def test_parse_splits_module_and_attribute() -> None:
    target = runtime.Target.parse("tests.runtime.fixtures.factories:quiet")

    assert target == runtime.Target(module="tests.runtime.fixtures.factories", attribute="quiet")
    assert str(target) == "tests.runtime.fixtures.factories:quiet"


@pytest.mark.parametrize(
    "text",
    [
        "not-a-target",
        "module",
        "module:",
        ":build",
        "pkg..mod:build",
        "pkg.mod:build:extra",
        "pkg.mod:bu-ild",
        "1pkg:build",
    ],
)
def test_parse_rejects_anything_but_module_path_and_attribute(text: str) -> None:
    with pytest.raises(runtime.UsageError):
        _ = runtime.Target.parse(text)


def test_resolve_returns_the_factory() -> None:
    assert runtime.Target.parse("tests.runtime.fixtures.factories:quiet").resolve() is factories.quiet


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("tests.runtime.fixtures.absent:build", "import_failed"),
        ("tests.runtime.fixtures.factories:absent", "factory_missing"),
        ("tests.runtime.fixtures.factories:not_callable", "factory_missing"),
    ],
)
def test_resolve_reports_why_the_factory_is_unavailable(text: str, kind: str) -> None:
    with pytest.raises(runtime.BootError) as raised:
        _ = runtime.Target.parse(text).resolve()

    assert raised.value.kind == kind
