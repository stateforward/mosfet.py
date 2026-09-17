from __future__ import annotations

import importlib
import importlib.metadata
import pathlib
import tomllib
import typing


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOT_PROJECT = PACKAGE_ROOT.parents[2] / "pyproject.toml"


def test_provider_has_package_owned_dependencies() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"

    provider_metadata = tomllib.loads(provider_project.read_text())
    root_metadata = tomllib.loads(ROOT_PROJECT.read_text())
    root_project = typing.cast("dict[str, object]", root_metadata["project"])
    optional_dependencies = typing.cast("dict[str, object]", root_project.get("optional-dependencies", {}))
    dependencies = typing.cast("list[str]", provider_metadata["project"]["dependencies"])

    assert provider_metadata["project"]["name"] == "mosfet-provider-postgres-memory"
    assert "stateforward.mosfet" in dependencies
    assert "psycopg" not in optional_dependencies
    assert "boto3" not in optional_dependencies
    assert "pglite" not in optional_dependencies


def test_provider_version_is_dynamic() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"
    provider_metadata = tomllib.loads(provider_project.read_text())
    provider_project_metadata = typing.cast("dict[str, object]", provider_metadata["project"])

    assert provider_project_metadata["readme"] == "README.md"
    assert "version" not in provider_project_metadata
    assert provider_project_metadata["dynamic"] == ["version"]
    assert provider_metadata["tool"]["hatch"]["version"]["path"] == "src/mosfet/providers/postgres_memory/__init__.py"


def test_provider_version_matches_installed_metadata() -> None:
    provider_module = importlib.import_module("mosfet.providers.postgres_memory")
    provider_version = typing.cast("object", getattr(provider_module, "__version__", None))

    assert isinstance(provider_version, str)
    assert provider_version == importlib.metadata.version("mosfet-provider-postgres-memory")
