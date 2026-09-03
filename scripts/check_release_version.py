from __future__ import annotations

import argparse
import ast
import collections.abc
import os
import pathlib
import sys
import tomllib
import typing
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PackageSpec:
    name: str
    pyproject_path: pathlib.Path
    version_path: str


def _discover_provider_specs() -> tuple[PackageSpec, ...]:
    """Derive provider package specs from the workspace layout (no hardcoded list)."""

    specs: list[PackageSpec] = []
    for pyproject_path in sorted(REPO_ROOT.glob("src/providers/*/pyproject.toml")):
        metadata = tomllib.loads(pyproject_path.read_text())
        project = metadata.get("project")
        if not isinstance(project, dict):
            continue
        name = project.get("name")
        if not isinstance(name, str) or not name.startswith("bot-provider-"):
            continue
        tool = metadata.get("tool")
        version_path: object = None
        if isinstance(tool, dict):
            hatch = tool.get("hatch")
            if isinstance(hatch, dict):
                hatch_version = hatch.get("version")
                if isinstance(hatch_version, dict):
                    version_path = hatch_version.get("path")
        if not isinstance(version_path, str):
            continue
        specs.append(
            PackageSpec(
                name=name,
                pyproject_path=pyproject_path,
                version_path=version_path,
            )
        )
    return tuple(specs)


PACKAGE_SPECS = (
    PackageSpec(
        name="stateforward.bot",
        pyproject_path=REPO_ROOT / "pyproject.toml",
        version_path="src/bot/__init__.py",
    ),
    *_discover_provider_specs(),
)


class ReleaseVersionError(Exception):
    pass


def read_version(version_file: pathlib.Path) -> str:
    module = ast.parse(version_file.read_text(), filename=str(version_file))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue

        has_version_target = any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
        if has_version_target and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value

    raise ReleaseVersionError(f"{version_file} does not assign a string __version__")


def require_table(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReleaseVersionError(f"{label} must be a TOML table")
    return typing.cast("dict[str, object]", value)


def verify_dynamic_metadata(spec: PackageSpec) -> str:
    metadata = tomllib.loads(spec.pyproject_path.read_text())
    project = require_table(metadata.get("project"), f"{spec.pyproject_path}: [project]")

    if "version" in project:
        raise ReleaseVersionError(f"{spec.name} must not set [project].version statically")
    if project.get("dynamic") != ["version"]:
        raise ReleaseVersionError(f'{spec.name} must set [project].dynamic = ["version"]')

    tool = require_table(metadata.get("tool"), f"{spec.pyproject_path}: [tool]")
    hatch = require_table(tool.get("hatch"), f"{spec.pyproject_path}: [tool.hatch]")
    hatch_version = require_table(hatch.get("version"), f"{spec.pyproject_path}: [tool.hatch.version]")
    if hatch_version.get("path") != spec.version_path:
        raise ReleaseVersionError(f"{spec.name} must read its version from {spec.version_path}")

    return read_version(spec.pyproject_path.parent / spec.version_path)


def tag_to_version(tag: str) -> str:
    if tag.startswith("refs/tags/"):
        tag = tag.removeprefix("refs/tags/")
    if not tag.startswith("v"):
        raise ReleaseVersionError(f"release tag must start with 'v': {tag}")

    version = tag.removeprefix("v")
    if not version:
        raise ReleaseVersionError("release tag must include a version after 'v'")
    return version


def run(tag: str | None) -> int:
    versions = {spec.name: verify_dynamic_metadata(spec) for spec in PACKAGE_SPECS}
    for package_name, version in versions.items():
        print(f"{package_name} {version}")

    if tag is None:
        return 0

    expected_version = tag_to_version(tag)
    mismatches = {
        package_name: package_version
        for package_name, package_version in versions.items()
        if package_version != expected_version
    }
    if mismatches:
        mismatch_text = ", ".join(
            f"{package_name}={package_version}" for package_name, package_version in mismatches.items()
        )
        raise ReleaseVersionError(f"tag v{expected_version} does not match package versions: {mismatch_text}")

    return 0


def default_release_tag() -> str | None:
    if os.environ.get("GITHUB_REF_TYPE") != "tag":
        return None
    return os.environ.get("GITHUB_REF_NAME")


def main(argv: collections.abc.Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate package version metadata before release.")
    _ = parser.add_argument(
        "--tag",
        default=default_release_tag(),
        help="Release tag to compare against, usually GITHUB_REF_NAME.",
    )
    args = parser.parse_args(argv)

    try:
        tag_value = typing.cast(object, getattr(args, "tag"))
        tag = tag_value if isinstance(tag_value, str) else None
        return run(tag)
    except ReleaseVersionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
