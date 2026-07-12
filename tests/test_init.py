import importlib.metadata
import pathlib
import tomllib
import typing

from bot import __version__
from tests.type_helpers import object_dict


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_package_version_is_dynamic() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    project = object_dict(typing.cast(object, metadata["project"]))

    assert "version" not in project
    assert project["dynamic"] == ["version"]
    assert metadata["tool"]["hatch"]["version"]["path"] == "src/bot/__init__.py"


def test_version() -> None:
    assert __version__ == importlib.metadata.version("stateforward.bot")
