from __future__ import annotations

import importlib
import importlib.metadata
import collections.abc
import pathlib
import tomllib
import typing


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOT_PROJECT = PACKAGE_ROOT.parents[2] / "pyproject.toml"


def object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, collections.abc.Mapping)
    return dict(typing.cast(collections.abc.Mapping[str, object], value))


def object_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return list(typing.cast(list[object], value))


def toml_dict(path: pathlib.Path) -> dict[str, object]:
    data: object = tomllib.loads(path.read_text())
    return object_dict(data)


def test_provider_has_package_owned_dependencies() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"

    provider_metadata = toml_dict(provider_project)
    root_metadata = toml_dict(ROOT_PROJECT)
    provider_project_metadata = object_dict(provider_metadata["project"])
    provider_dependencies = object_list(provider_project_metadata["dependencies"])
    root_project = object_dict(root_metadata["project"])
    optional_dependencies = object_dict(root_project.get("optional-dependencies", {}))

    assert provider_project_metadata["name"] == "bot-provider-elevenlabs"
    assert "elevenlabs>=2.49.0,<3.0.0" in provider_dependencies
    assert "stateforward.bot" in provider_dependencies
    assert "elevenlabs" not in optional_dependencies


def test_provider_version_is_dynamic() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"
    provider_metadata = toml_dict(provider_project)
    provider_project_metadata = object_dict(provider_metadata["project"])

    assert provider_project_metadata["readme"] == "README.md"
    assert "version" not in provider_project_metadata
    assert provider_project_metadata["dynamic"] == ["version"]
    provider_tool_metadata = object_dict(provider_metadata["tool"])
    provider_hatch_metadata = object_dict(provider_tool_metadata["hatch"])
    provider_version_metadata = object_dict(provider_hatch_metadata["version"])

    assert provider_version_metadata["path"] == "src/bot/providers/elevenlabs/__init__.py"


def test_provider_version_matches_installed_metadata() -> None:
    provider_module = importlib.import_module("bot.providers.elevenlabs")
    provider_version = getattr(provider_module, "__version__", None)

    assert isinstance(provider_version, str)
    assert provider_version == importlib.metadata.version("bot-provider-elevenlabs")
