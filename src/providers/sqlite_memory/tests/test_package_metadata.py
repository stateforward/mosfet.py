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
    dev_dependencies = typing.cast("list[str]", provider_metadata["dependency-groups"]["dev"])

    assert provider_metadata["project"]["name"] == "bot-provider-sqlite-memory"
    assert "stateforward.bot" in dependencies
    assert "sqlite-vec>=0.1.10a4,<0.2.0" not in dependencies
    assert "sqlite-vec>=0.1.10a4,<0.2.0" in dev_dependencies
    assert "sqlite-vec" not in optional_dependencies
    assert "sqlite-objstore" not in optional_dependencies


def test_provider_bundles_objstore_extension_source() -> None:
    vendor_root = PACKAGE_ROOT / "vendor/sqlite-objstore"
    build_hook = PACKAGE_ROOT / "hatch_build.py"

    assert build_hook.exists()
    assert (vendor_root / "include/objstore/objstore.h").exists()
    assert (vendor_root / "src/objstore.c").exists()
    assert (vendor_root / "third_party/blake3/blake3.c").exists()
    assert (PACKAGE_ROOT / "vendor/sqlite/sqlite3ext.h").exists()


def test_provider_builds_both_native_extensions_into_wheel() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"
    provider_metadata = tomllib.loads(provider_project.read_text())
    hook_metadata = provider_metadata["tool"]["hatch"]["build"]["hooks"]["custom"]

    assert (
        "src/bot/providers/sqlite_memory/_native/*"
        in provider_metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["artifacts"]
    )
    assert "sqlite-vec>=0.1.10a4,<0.2.0" in hook_metadata["dependencies"]


def test_provider_bundles_redistribution_licenses() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"
    provider_metadata = tomllib.loads(provider_project.read_text())
    force_include = provider_metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert (PACKAGE_ROOT / "vendor/sqlite-objstore/LICENSE").exists()
    assert (PACKAGE_ROOT / "vendor/sqlite-objstore/third_party/blake3/LICENSE_A2").exists()
    assert (PACKAGE_ROOT / "vendor/sqlite-objstore/third_party/blake3/LICENSE_A2LLVM").exists()
    assert (PACKAGE_ROOT / "vendor/sqlite-objstore/third_party/blake3/LICENSE_CC0").exists()
    assert (PACKAGE_ROOT / "vendor/sqlite-vec/LICENSE-MIT").exists()
    assert (PACKAGE_ROOT / "vendor/sqlite-vec/LICENSE-APACHE").exists()
    assert "vendor/sqlite-objstore/third_party/blake3/LICENSE_A2" in force_include
    assert "vendor/sqlite-objstore/third_party/blake3/LICENSE_A2LLVM" in force_include
    assert "vendor/sqlite-objstore/third_party/blake3/LICENSE_CC0" in force_include
    assert "vendor/sqlite-vec/LICENSE-MIT" in force_include
    assert "vendor/sqlite-vec/LICENSE-APACHE" in force_include


def test_provider_version_is_dynamic() -> None:
    provider_project = PACKAGE_ROOT / "pyproject.toml"
    provider_metadata = tomllib.loads(provider_project.read_text())
    provider_project_metadata = typing.cast("dict[str, object]", provider_metadata["project"])

    assert provider_project_metadata["readme"] == "README.md"
    assert "version" not in provider_project_metadata
    assert provider_project_metadata["dynamic"] == ["version"]
    assert provider_metadata["tool"]["hatch"]["version"]["path"] == "src/bot/providers/sqlite_memory/__init__.py"


def test_provider_version_matches_installed_metadata() -> None:
    provider_module = importlib.import_module("bot.providers.sqlite_memory")
    provider_version = typing.cast("object", getattr(provider_module, "__version__", None))

    assert isinstance(provider_version, str)
    assert provider_version == importlib.metadata.version("bot-provider-sqlite-memory")
