from __future__ import annotations

import pathlib
import re
import subprocess
import tomllib
import typing

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "smoke_wheels.sh"
CORE_WHEEL = "stateforward_mosfet-0.0.0-py3-none-any.whl"


def wheel_filename(project_name: str, tags: str = "py3-none-any") -> str:
    """Return a wheel filename for ``project_name`` with the distribution name normalized as the wheel spec requires."""
    return f"{re.sub(r'[-_.]+', '_', project_name).lower()}-0.0.0-{tags}.whl"


def map_only(dist: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([SCRIPT, "--map-only", dist], capture_output=True, text=True, check=False)


def mapping(dist: pathlib.Path) -> dict[str, str]:
    result = map_only(dist)
    assert result.returncode == 0, result.stderr
    return dict(line.split("\t") for line in result.stdout.splitlines())


def test_core_wheel_name_matches_root_project() -> None:
    project_name = typing.cast(object, tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["name"])

    assert isinstance(project_name, str)
    assert wheel_filename(project_name) == CORE_WHEEL


def test_map_only_maps_core_and_every_workspace_provider(tmp_path: pathlib.Path) -> None:
    expected = {CORE_WHEEL: "mosfet"}
    for provider_dir in sorted((REPO_ROOT / "src" / "providers").iterdir()):
        pyproject = provider_dir / "pyproject.toml"
        if not pyproject.is_file():
            continue
        import_package = provider_dir / "src" / "mosfet" / "providers" / provider_dir.name
        assert (import_package / "__init__.py").is_file(), import_package
        project_name = typing.cast(object, tomllib.loads(pyproject.read_text())["project"]["name"])
        assert isinstance(project_name, str), pyproject
        expected[wheel_filename(project_name)] = f"mosfet.providers.{provider_dir.name}"
    for filename in expected:
        (tmp_path / filename).touch()

    assert mapping(tmp_path) == expected


def test_map_only_maps_a_new_platform_specific_provider_wheel(tmp_path: pathlib.Path) -> None:
    provider_wheel = wheel_filename("mosfet-provider-llama-cpp", "cp313-cp313-macosx_14_0_arm64")
    (tmp_path / CORE_WHEEL).touch()
    (tmp_path / provider_wheel).touch()

    assert mapping(tmp_path) == {CORE_WHEEL: "mosfet", provider_wheel: "mosfet.providers.llama_cpp"}


def test_map_only_fails_on_a_wheel_outside_the_naming_contract(tmp_path: pathlib.Path) -> None:
    (tmp_path / CORE_WHEEL).touch()
    (tmp_path / "bot_provider_gemini-0.0.0-py3-none-any.whl").touch()

    result = map_only(tmp_path)

    assert result.returncode == 1
    assert "No import smoke mapping for built wheel: bot_provider_gemini-0.0.0-py3-none-any.whl" in result.stderr


@pytest.mark.parametrize(
    ("filenames", "message"),
    [
        ((), "No wheels were built"),
        (
            ("mosfet_provider_gemini-0.0.0-py3-none-any.whl",),
            "Expected exactly one stateforward.mosfet wheel, found 0.",
        ),
    ],
)
def test_map_only_requires_exactly_one_core_wheel(
    tmp_path: pathlib.Path, filenames: tuple[str, ...], message: str
) -> None:
    for filename in filenames:
        (tmp_path / filename).touch()

    result = map_only(tmp_path)

    assert result.returncode == 1
    assert message in result.stderr


def fake_uv(bin_dir: pathlib.Path, log: pathlib.Path) -> None:
    """Install a ``uv`` that records its arguments and makes venvs whose python always imports successfully."""
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    _ = uv.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{log}"\n'
        'if [[ "$1" == venv ]]; then\n'
        '  environment="${@: -1}"\n'
        '  mkdir -p "$environment/bin"\n'
        "  printf '#!/bin/sh\\nexit 0\\n' > \"$environment/bin/python\"\n"
        '  chmod +x "$environment/bin/python"\n'
        "fi\n"
    )
    uv.chmod(0o755)


def smoke(arguments: list[str], tmp_path: pathlib.Path) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    log = tmp_path / "uv.log"
    fake_uv(tmp_path / "bin", log)
    environment = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run([SCRIPT, *arguments], capture_output=True, text=True, check=False, env=environment)
    return result, log.read_text().splitlines() if log.exists() else []


def test_find_links_reaches_every_provider_install(tmp_path: pathlib.Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    provider_wheel = wheel_filename("mosfet-provider-llama-cpp", "py3-none-any")
    (dist / CORE_WHEEL).touch()
    (dist / provider_wheel).touch()
    native = tmp_path / "dist-native"

    result, calls = smoke(["--find-links", str(native), str(dist)], tmp_path)

    assert result.returncode == 0, result.stderr
    installs = [call for call in calls if call.startswith("pip install")]
    assert len(installs) == 2
    provider_install = next(call for call in installs if provider_wheel in call)
    assert f"--find-links {native}" in provider_install


def test_installs_use_only_the_index_without_find_links(tmp_path: pathlib.Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / CORE_WHEEL).touch()

    result, calls = smoke([str(dist)], tmp_path)

    assert result.returncode == 0, result.stderr
    assert not any("--find-links" in call for call in calls)


def test_find_links_requires_a_directory(tmp_path: pathlib.Path) -> None:
    result, _ = smoke(["--find-links"], tmp_path)

    assert result.returncode == 2
    assert "--find-links needs a directory" in result.stderr
