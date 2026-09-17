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


def object_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return typing.cast(list[object], value)


def toml_document(path: pathlib.Path) -> dict[str, object]:
    return typing.cast(dict[str, object], tomllib.loads(path.read_text()))


def test_provider_has_package_owned_dependencies() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"

    provider_metadata = toml_document(provider_project)
    root_metadata = toml_document(ROOT_PROJECT)
    provider_project_metadata = object_dict(provider_metadata["project"])
    provider_dependencies = object_list(provider_project_metadata["dependencies"])
    root_project = object_dict(root_metadata["project"])
    optional_dependencies = object_dict(root_project.get("optional-dependencies", {}))

    assert provider_project_metadata["name"] == "mosfet-provider-moonshine"
    assert "moonshine-voice>=0.0.70,<0.1.0" in provider_dependencies
    assert "stateforward.mosfet" in provider_dependencies
    assert "moonshine-voice" not in optional_dependencies


def test_provider_version_is_dynamic() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"
    provider_metadata = toml_document(provider_project)
    provider_project_metadata = object_dict(provider_metadata["project"])
    tool_metadata = object_dict(provider_metadata["tool"])
    hatch_metadata = object_dict(tool_metadata["hatch"])
    version_metadata = object_dict(hatch_metadata["version"])

    assert provider_project_metadata["readme"] == "README.md"
    assert "version" not in provider_project_metadata
    assert provider_project_metadata["dynamic"] == ["version"]
    assert version_metadata["path"] == "src/mosfet/providers/moonshine/__init__.py"


def test_provider_version_matches_installed_metadata() -> None:
    provider_module = importlib.import_module("mosfet.providers.moonshine")
    provider_version = getattr(provider_module, "__version__", None)

    assert isinstance(provider_version, str)
    assert provider_version == importlib.metadata.version("mosfet-provider-moonshine")
