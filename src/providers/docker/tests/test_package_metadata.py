from __future__ import annotations

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


def _project(path: pathlib.Path) -> dict[str, object]:
    return object_dict(object_dict(tomllib.loads(path.read_text()))["project"])


def test_provider_owns_its_httpx_dependency() -> None:
    provider = _project(PACKAGE_ROOT / "pyproject.toml")

    assert provider["name"] == "mosfet-provider-docker"
    assert provider["dependencies"] == ["httpx>=0.28,<1", "stateforward.mosfet"]
    assert "httpx" not in str(_project(ROOT_PROJECT)["dependencies"])


def test_provider_version_is_dynamic() -> None:
    metadata = object_dict(tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text()))
    provider = object_dict(metadata["project"])
    hatch = object_dict(object_dict(metadata["tool"])["hatch"])

    assert provider["readme"] == "README.md"
    assert "version" not in provider
    assert provider["dynamic"] == ["version"]
    assert object_dict(hatch["version"])["path"] == "src/mosfet/providers/docker/__init__.py"


def test_provider_version_matches_installed_metadata() -> None:
    provider_module = importlib.import_module("mosfet.providers.docker")
    provider_version = getattr(provider_module, "__version__", None)

    assert isinstance(provider_version, str)
    assert provider_version == importlib.metadata.version("mosfet-provider-docker")
