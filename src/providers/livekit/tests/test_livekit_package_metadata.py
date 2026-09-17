from __future__ import annotations

import importlib
import importlib.metadata
import pathlib
import typing
import tomllib


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOT_PROJECT = PACKAGE_ROOT.parents[2] / "pyproject.toml"


def object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return typing.cast(dict[str, object], value)


def toml_dict(path: pathlib.Path) -> dict[str, object]:
    return typing.cast(dict[str, object], tomllib.loads(path.read_text()))


def string_list(value: object) -> list[str]:
    assert isinstance(value, list)
    values = typing.cast(list[object], value)
    strings: list[str] = []
    for item in values:
        assert isinstance(item, str)
        strings.append(item)
    return strings


def test_livekit_provider_has_package_owned_dependencies() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"

    provider_metadata = toml_dict(provider_project)
    root_metadata = toml_dict(ROOT_PROJECT)
    root_project = object_dict(root_metadata["project"])
    optional_dependencies = object_dict(root_project.get("optional-dependencies", {}))
    provider_project_metadata = object_dict(provider_metadata["project"])
    provider_dependencies = string_list(provider_project_metadata["dependencies"])

    assert provider_project_metadata["name"] == "mosfet-provider-livekit"
    assert "livekit>=1.1.10,<2.0.0" in provider_dependencies
    assert "stateforward.mosfet" in provider_dependencies
    assert "livekit" not in optional_dependencies


def test_livekit_provider_version_is_dynamic() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"
    provider_metadata = toml_dict(provider_project)
    provider_project_metadata = object_dict(provider_metadata["project"])
    provider_tool_metadata = object_dict(provider_metadata["tool"])
    hatch_metadata = object_dict(provider_tool_metadata["hatch"])
    version_metadata = object_dict(hatch_metadata["version"])

    assert provider_project_metadata["readme"] == "README.md"
    assert "version" not in provider_project_metadata
    assert provider_project_metadata["dynamic"] == ["version"]
    assert version_metadata["path"] == "src/mosfet/providers/livekit/__init__.py"


def test_livekit_provider_version_matches_installed_metadata() -> None:
    provider_module = importlib.import_module("mosfet.providers.livekit")
    provider_version = getattr(provider_module, "__version__", None)

    assert isinstance(provider_version, str)
    assert provider_version == importlib.metadata.version("mosfet-provider-livekit")
