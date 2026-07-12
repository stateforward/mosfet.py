from __future__ import annotations

import collections.abc
import importlib
import importlib.metadata
import pathlib
import tomllib
import typing


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOT_PROJECT = PACKAGE_ROOT.parents[2] / "pyproject.toml"


def object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return typing.cast(dict[str, object], value)


def string_list(value: object) -> list[str]:
    assert isinstance(value, list)
    sequence = typing.cast(collections.abc.Sequence[object], value)
    assert all(isinstance(item, str) for item in sequence)
    return typing.cast(list[str], sequence)


def test_provider_has_package_owned_dependencies() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"

    provider_metadata = object_dict(tomllib.loads(provider_project.read_text()))
    root_metadata = object_dict(tomllib.loads(ROOT_PROJECT.read_text()))
    root_project = object_dict(root_metadata["project"])
    optional_dependencies = object_dict(root_project.get("optional-dependencies", {}))
    provider_project_metadata = object_dict(provider_metadata["project"])
    dependencies = string_list(provider_project_metadata["dependencies"])

    assert provider_project_metadata["name"] == "bot-provider-gemini"
    assert "google-genai>=2.11.0,<3.0.0" in dependencies
    assert "stateforward.bot>=0.1.0,<0.2.0" in dependencies
    assert "gemini" not in optional_dependencies


def test_provider_version_is_dynamic() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"
    provider_metadata = object_dict(tomllib.loads(provider_project.read_text()))
    provider_project_metadata = object_dict(provider_metadata["project"])
    tool_metadata = object_dict(provider_metadata["tool"])
    hatch_metadata = object_dict(tool_metadata["hatch"])
    version_metadata = object_dict(hatch_metadata["version"])

    assert provider_project_metadata["readme"] == "README.md"
    assert "version" not in provider_project_metadata
    assert provider_project_metadata["dynamic"] == ["version"]
    assert version_metadata["path"] == "src/bot/providers/gemini/__init__.py"


def test_provider_version_matches_installed_metadata() -> None:
    provider_module = importlib.import_module("bot.providers.gemini")
    provider_version = getattr(provider_module, "__version__", None)

    assert isinstance(provider_version, str)
    assert provider_version == importlib.metadata.version("bot-provider-gemini")
