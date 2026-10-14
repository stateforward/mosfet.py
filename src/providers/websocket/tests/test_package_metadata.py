from __future__ import annotations

import importlib
import importlib.metadata
import pathlib
import tomllib
import typing

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]


def project() -> dict[str, typing.Any]:
    metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
    return typing.cast(dict[str, typing.Any], metadata["project"])


def test_websocket_provider_owns_its_websockets_dependency() -> None:
    metadata = project()

    assert metadata["name"] == "mosfet-provider-websocket"
    assert metadata["dependencies"] == ["websockets>=16.0,<17", "stateforward.mosfet"]


def test_websocket_provider_version_is_dynamic_and_matches_installed_metadata() -> None:
    metadata = project()
    module = importlib.import_module("mosfet.providers.websocket")

    assert metadata["dynamic"] == ["version"]
    assert "version" not in metadata
    assert getattr(module, "__version__", None) == importlib.metadata.version("mosfet-provider-websocket")
