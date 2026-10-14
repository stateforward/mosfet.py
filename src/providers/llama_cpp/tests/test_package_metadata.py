from __future__ import annotations

import collections.abc
import importlib
import importlib.metadata
import pathlib
import tomllib
import typing

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]
PINNED_REQUIREMENT = "llama-cpp-python==0.3.36+mosfet.4fbc76de; sys_platform == 'darwin' or sys_platform == 'linux'"


def object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return typing.cast(dict[str, object], value)


def string_list(value: object) -> list[str]:
    assert isinstance(value, list)
    sequence = typing.cast(collections.abc.Sequence[object], value)
    assert all(isinstance(item, str) for item in sequence)
    return typing.cast(list[str], value)


def provider_metadata() -> dict[str, object]:
    return object_dict(tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text()))


def test_provider_owns_the_pinned_llama_cpp_python_build() -> None:
    metadata = provider_metadata()
    project = object_dict(metadata["project"])
    root_project = object_dict(object_dict(tomllib.loads((REPO_ROOT / "pyproject.toml").read_text()))["project"])
    sources = object_dict(object_dict(object_dict(metadata["tool"])["uv"])["sources"])

    assert project["name"] == "mosfet-provider-llama-cpp"
    assert string_list(project["dependencies"]) == [PINNED_REQUIREMENT, "stateforward.mosfet"]
    assert sources["llama-cpp-python"] == {"path": "vendor/llama-cpp-python", "editable": False}
    assert sources["stateforward-mosfet"] == {"workspace": True}
    assert not any("llama" in dependency for dependency in string_list(root_project["dependencies"]))


def test_provider_version_is_dynamic() -> None:
    metadata = provider_metadata()
    project = object_dict(metadata["project"])
    hatch = object_dict(object_dict(metadata["tool"])["hatch"])

    assert project["readme"] == "README.md"
    assert "version" not in project
    assert project["dynamic"] == ["version"]
    assert object_dict(hatch["version"])["path"] == "src/mosfet/providers/llama_cpp/__init__.py"


def test_wheel_ships_only_the_import_package_and_never_the_build_recipe() -> None:
    hatch = object_dict(object_dict(provider_metadata()["tool"])["hatch"])
    wheel = object_dict(object_dict(hatch["build"])["targets"])["wheel"]

    assert object_dict(wheel)["packages"] == ["src/mosfet"]
    assert (PACKAGE_ROOT / "vendor" / "llama-cpp-python" / "pyproject.toml").is_file()
    assert not (PACKAGE_ROOT / "src" / "mosfet" / "vendor").exists()


def test_build_recipe_is_not_a_workspace_member() -> None:
    root = object_dict(tomllib.loads((REPO_ROOT / "pyproject.toml").read_text()))
    members = string_list(object_dict(object_dict(object_dict(root["tool"])["uv"])["workspace"])["members"])
    recipe = PACKAGE_ROOT / "vendor" / "llama-cpp-python"

    assert not any(recipe.match(str(REPO_ROOT / member)) for member in members)


def test_public_exports_are_intentional() -> None:
    module = importlib.import_module("mosfet.providers.llama_cpp")

    assert getattr(module, "__all__") == ["Accelerator", "Library", "LibraryError", "Weights", "__version__"]


def test_provider_version_matches_installed_metadata() -> None:
    module = importlib.import_module("mosfet.providers.llama_cpp")
    version = getattr(module, "__version__", None)

    assert isinstance(version, str)
    assert version == importlib.metadata.version("mosfet-provider-llama-cpp")
