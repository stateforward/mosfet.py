from __future__ import annotations

import importlib
import importlib.metadata
import pathlib
import tomllib


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOT_PROJECT = PACKAGE_ROOT.parents[2] / "pyproject.toml"


def object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def test_provider_has_package_owned_dependencies() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"

    provider_metadata = tomllib.loads(provider_project.read_text())
    root_metadata = tomllib.loads(ROOT_PROJECT.read_text())
    root_project = object_dict(root_metadata["project"])
    optional_dependencies = object_dict(root_project.get("optional-dependencies", {}))

    assert provider_metadata["project"]["name"] == "mosfet-provider-opencv"
    # W3-6: the full (not headless) wheel, pinned; headless is a Docker deployment note only.
    assert provider_metadata["project"]["dependencies"] == [
        "numpy>=2.4,<3",
        "opencv-python==4.13.0.92",
        "stateforward.mosfet",
    ]
    assert "opencv-python" not in optional_dependencies
    assert "opencv" not in str(root_metadata["project"]["dependencies"])


def test_provider_version_is_dynamic() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"
    provider_metadata = tomllib.loads(provider_project.read_text())
    provider_project_metadata = object_dict(provider_metadata["project"])

    assert provider_project_metadata["readme"] == "README.md"
    assert "version" not in provider_project_metadata
    assert provider_project_metadata["dynamic"] == ["version"]
    assert provider_metadata["tool"]["hatch"]["version"]["path"] == "src/mosfet/providers/opencv/__init__.py"


def test_provider_version_matches_installed_metadata() -> None:
    provider_module = importlib.import_module("mosfet.providers.opencv")
    provider_version = getattr(provider_module, "__version__", None)

    assert isinstance(provider_version, str)
    assert provider_version == importlib.metadata.version("mosfet-provider-opencv")
