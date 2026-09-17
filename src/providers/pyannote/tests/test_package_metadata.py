from __future__ import annotations

import pathlib
import typing
import tomllib


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _project_metadata() -> dict[str, object]:
    raw_document: object = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert isinstance(raw_document, dict)
    document = typing.cast(dict[str, object], raw_document)
    project = document["project"]
    assert isinstance(project, dict)
    return typing.cast(dict[str, object], project)


def test_pyannote_dependency_is_declared() -> None:
    dependencies = _project_metadata()["dependencies"]
    assert isinstance(dependencies, list)
    assert dependencies == ["pyannote.audio>=4.0.7,<5.0.0", "stateforward.mosfet"]


def test_package_metadata_points_to_local_version_and_readme() -> None:
    project = _project_metadata()
    assert project["name"] == "mosfet-provider-pyannote"
    assert project["dynamic"] == ["version"]
    assert project["readme"] == "README.md"
    raw_document: object = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert isinstance(raw_document, dict)
    document = typing.cast(dict[str, object], raw_document)
    tool = document["tool"]
    assert isinstance(tool, dict)
    hatch = typing.cast(dict[str, object], tool)["hatch"]
    assert isinstance(hatch, dict)
    version = typing.cast(dict[str, object], hatch)["version"]
    assert isinstance(version, dict)
    assert typing.cast(dict[str, object], version)["path"] == "src/mosfet/providers/pyannote/__init__.py"


def test_readme_records_exact_dependency() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "pyannote.audio" in readme
    assert "pyannote.audio>=4.0.7,<5.0.0" in readme
