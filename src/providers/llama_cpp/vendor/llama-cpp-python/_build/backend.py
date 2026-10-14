"""PEP 517 backend that rebuilds llama-cpp-python 0.3.36 against the pinned llama.cpp commit.

The upstream 0.3.36 sdist vendors an older llama.cpp. This backend fetches that sdist and the pinned llama.cpp
source archive (both sha256-pinned in ``recipe.toml``), swaps the vendored tree, verifies the ABI headers the ctypes
bindings were proven against, stamps the local version, and hands the staged tree to scikit-build-core. Every step
fails closed with ``RecipeError``.

``BOT_LLAMA_CPP_SOURCES=<dir>`` serves the archives from a local directory (air-gapped and Docker builds), named as
the last segment of their pinned URL; a local file is used only when its sha256 matches the pin.

The backend runs on the build interpreter (3.11 or newer, for ``tomllib``). scikit-build-core reads the project from
its working directory, so it runs in a child process inside the staged tree; this process never changes directory.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import typing
import urllib.parse
import urllib.request

PROJECT = pathlib.Path(__file__).resolve().parents[1]
"""The recipe project: ``pyproject.toml`` (static metadata) and ``recipe.toml`` (pins)."""
SOURCES_ENV = "BOT_LLAMA_CPP_SOURCES"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
"""The largest source archive fetched or read (the pinned ones are about 76 MiB and 31 MiB)."""
MAX_MEMBERS = 20_000
"""The most members one archive may hold (the pinned ones hold about 3,900 and 4,100)."""
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
"""The most bytes one archive may expand to (the pinned ones expand to about 216 MB and 176 MB)."""
DOWNLOAD_TIMEOUT_SECONDS = 120.0
VENDORED = pathlib.PurePosixPath("vendor/llama.cpp")
"""Where the sdist vendors llama.cpp; its own copy is never extracted and the pinned tree takes its place."""

Download = collections.abc.Callable[[str, pathlib.Path], None]
"""Writes the resource at an HTTPS URL to a local file."""

_CHILD = """\
import json, pathlib, sys
from scikit_build_core import build
hook, arguments, result = sys.argv[1], json.loads(sys.argv[2]), pathlib.Path(sys.argv[3])
result.write_text(json.dumps(getattr(build, hook)(*arguments)))
"""


class RecipeError(RuntimeError):
    """The recipe, a source archive, or the staged tree does not match the pins; nothing is built."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class Archive:
    """A source archive pinned by URL and sha256."""

    url: str
    sha256: str

    def filename(self) -> str:
        """The archive's name in ``BOT_LLAMA_CPP_SOURCES``: the last segment of its URL path."""
        return pathlib.PurePosixPath(urllib.parse.urlsplit(self.url).path).name


@dataclasses.dataclass(frozen=True, kw_only=True)
class Recipe:
    """Everything the build is pinned to, read from ``recipe.toml`` and the static ``[project]`` metadata."""

    version: str
    """The upstream llama-cpp-python release the sdist is."""
    local_version: str
    """The PEP 440 local label stamped on the build, so ``__version__`` names the llama.cpp commit."""
    sdist: Archive
    llama_cpp: Archive
    commit: str
    """The llama.cpp commit; the archive's pax ``comment`` header must equal it."""
    abi: collections.abc.Mapping[str, str]
    """sha256 of each header the ctypes bindings depend on, by path inside the llama.cpp tree."""
    dependencies: tuple[str, ...]
    """The static runtime requirements; the staged upstream metadata must declare exactly these."""
    requires_python: str

    @classmethod
    def load(cls, project: pathlib.Path) -> Recipe:
        """Read the recipe in ``project`` and refuse one whose pins disagree with its static metadata."""
        pins = _toml(project / "recipe.toml")
        metadata = _table(_toml(project / "pyproject.toml"), "project")
        upstream = _table(pins, "llama_cpp_python")
        llama_cpp = _table(pins, "llama_cpp")
        abi = _table(pins, "abi")
        recipe = cls(
            version=_text(upstream, "version"),
            local_version=_text(pins, "local_version"),
            sdist=Archive(url=_text(upstream, "sdist_url"), sha256=_text(upstream, "sdist_sha256")),
            llama_cpp=Archive(url=_text(llama_cpp, "archive_url"), sha256=_text(llama_cpp, "archive_sha256")),
            commit=_text(llama_cpp, "commit"),
            abi={path: _text(abi, path) for path in abi},
            dependencies=_texts(metadata, "dependencies"),
            requires_python=_text(metadata, "requires-python"),
        )
        for archive in (recipe.sdist, recipe.llama_cpp):
            if urllib.parse.urlsplit(archive.url).scheme != "https":
                raise RecipeError(f"{archive.url} is not an https URL")
        version = _text(metadata, "version")
        if version != f"{recipe.version}+{recipe.local_version}":
            pinned = f"{recipe.version}+{recipe.local_version}"
            raise RecipeError(f"pyproject.toml version {version!r} is not {pinned} from recipe.toml")
        build = _text(_table(pins, "build"), "scikit_build_core")
        requires = _texts(_table(_toml(project / "pyproject.toml"), "build-system"), "requires")
        if requires != (f"scikit-build-core=={build}",):
            raise RecipeError(f"build-system requires {list(requires)} is not scikit-build-core=={build}")
        return recipe


def stage(
    destination: pathlib.Path,
    recipe: Recipe,
    *,
    sources: pathlib.Path | None,
    download: Download,
    llama_cpp: bool,
) -> pathlib.Path:
    """Stage the pinned source tree under ``destination`` and return its root.

    With ``llama_cpp`` false only the sdist is staged, which is all that metadata and build requirements need.
    """
    archive = _fetch(recipe.sdist, destination, sources=sources, download=download)
    tree = _extract(archive, destination / "sdist", skip=VENDORED)
    _check_upstream_metadata(tree / "pyproject.toml", recipe)
    init = tree / "llama_cpp" / "__init__.py"
    rewritten, count = re.subn(
        rf'^__version__ = "{re.escape(recipe.version)}"$',
        f'__version__ = "{recipe.version}+{recipe.local_version}"',
        init.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RecipeError(f'{init} does not declare __version__ = "{recipe.version}" exactly once')
    _ = init.write_text(rewritten, encoding="utf-8")
    if not llama_cpp:
        return tree
    archive = _fetch(recipe.llama_cpp, destination, sources=sources, download=download)
    with tarfile.open(archive, mode="r:gz") as opened:
        _ = opened.next()
        commit = opened.pax_headers.get("comment")
    if commit != recipe.commit:
        raise RecipeError(f"{recipe.llama_cpp.url} is commit {commit!r}, not the pinned commit {recipe.commit}")
    llama_tree = _extract(archive, destination / "llama.cpp", skip=None)
    for path, pinned in recipe.abi.items():
        header = llama_tree / path
        if not header.is_file():
            raise RecipeError(f"llama.cpp {recipe.commit} has no ABI header {path}")
        if _sha256(header) != pinned:
            raise RecipeError(f"llama.cpp ABI header {path} does not match its pinned sha256 {pinned}")
    vendored = tree / VENDORED
    vendored.parent.mkdir(parents=True, exist_ok=True)
    _ = llama_tree.rename(vendored)
    return tree


def get_requires_for_build_wheel(config_settings: dict[str, str | list[str]] | None = None) -> list[str]:
    """PEP 517: scikit-build-core's requirements (cmake and ninja when the host lacks them) for the staged tree."""
    with tempfile.TemporaryDirectory(prefix="mosfet-llama-cpp-") as directory:
        tree = _stage_from_environment(pathlib.Path(directory), llama_cpp=False)
        requires = _scikit_build(tree, "get_requires_for_build_wheel", [config_settings])
    if not isinstance(requires, list):
        raise RecipeError(f"scikit-build-core returned {requires!r} as build requirements")
    return [str(requirement) for requirement in typing.cast(list[object], requires)]


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: dict[str, str | list[str]] | None = None
) -> str:
    """PEP 517: write the staged tree's wheel metadata, whose version carries the pinned local label."""
    with tempfile.TemporaryDirectory(prefix="mosfet-llama-cpp-") as directory:
        tree = _stage_from_environment(pathlib.Path(directory), llama_cpp=False)
        arguments = [os.path.abspath(metadata_directory), config_settings]
        return str(_scikit_build(tree, "prepare_metadata_for_build_wheel", arguments))


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """PEP 517: compile the staged tree with scikit-build-core; ``CMAKE_ARGS`` selects CUDA or CPU on Linux."""
    with tempfile.TemporaryDirectory(prefix="mosfet-llama-cpp-") as directory:
        tree = _stage_from_environment(pathlib.Path(directory), llama_cpp=True)
        metadata = None if metadata_directory is None else os.path.abspath(metadata_directory)
        arguments = [os.path.abspath(wheel_directory), config_settings, metadata]
        return str(_scikit_build(tree, "build_wheel", arguments))


def build_sdist(sdist_directory: str, config_settings: dict[str, str | list[str]] | None = None) -> str:
    """PEP 517: unsupported; the recipe is consumed only as a path source that builds a wheel."""
    del sdist_directory, config_settings
    raise NotImplementedError("the llama-cpp-python build recipe builds a wheel only; it has no sdist")


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """PEP 660: unsupported; the recipe is consumed with ``editable = false``."""
    del wheel_directory, config_settings, metadata_directory
    raise NotImplementedError("the llama-cpp-python build recipe builds a regular wheel only; it is never editable")


def _stage_from_environment(destination: pathlib.Path, *, llama_cpp: bool) -> pathlib.Path:
    sources = os.environ.get(SOURCES_ENV)
    return stage(
        destination,
        Recipe.load(PROJECT),
        sources=pathlib.Path(sources) if sources else None,
        download=_download,
        llama_cpp=llama_cpp,
    )


def _scikit_build(tree: pathlib.Path, hook: str, arguments: collections.abc.Sequence[object]) -> object:
    result = tree.parent / f"{hook}.json"
    command = [sys.executable, "-P", "-c", _CHILD, hook, json.dumps(list(arguments)), str(result)]
    try:
        _ = subprocess.run(command, cwd=tree, check=True)
    except subprocess.CalledProcessError as error:
        raise RecipeError(f"scikit-build-core {hook} failed with exit status {error.returncode}") from error
    return typing.cast(object, json.loads(result.read_text(encoding="utf-8")))


def _fetch(
    archive: Archive, destination: pathlib.Path, *, sources: pathlib.Path | None, download: Download
) -> pathlib.Path:
    local = None if sources is None else sources / archive.filename()
    if local is not None and local.is_file():
        path = local
    else:
        path = destination / archive.filename()
        download(archive.url, path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise RecipeError(f"{path} is larger than {MAX_ARCHIVE_BYTES} bytes")
    if _sha256(path) != archive.sha256:
        raise RecipeError(f"{path} does not match the pinned sha256 {archive.sha256} of {archive.url}")
    return path


def _download(url: str, target: pathlib.Path) -> None:
    if urllib.parse.urlsplit(url).scheme != "https":
        raise RecipeError(f"{url} is not an https URL")
    written = 0
    response = typing.cast(typing.BinaryIO, urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS))
    with response, target.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_ARCHIVE_BYTES:
                raise RecipeError(f"{url} is larger than {MAX_ARCHIVE_BYTES} bytes")
            _ = file.write(chunk)


def _extract(archive: pathlib.Path, target: pathlib.Path, *, skip: pathlib.PurePosixPath | None) -> pathlib.Path:
    """Extract a verified archive with one top-level directory, minus ``skip`` under it, and return that directory."""
    members: list[tarfile.TarInfo] = []
    roots: set[str] = set()
    expanded = 0
    with tarfile.open(archive, mode="r:gz") as opened:
        for count, member in enumerate(opened, start=1):
            path = pathlib.PurePosixPath(member.name)
            if count > MAX_MEMBERS:
                raise RecipeError(f"archive {archive.name} holds more than {MAX_MEMBERS} members")
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise RecipeError(f"archive {archive.name} member {member.name!r} escapes the extraction directory")
            if not (member.isfile() or member.isdir()):
                raise RecipeError(f"archive {archive.name} member {member.name!r} is not a file or directory")
            expanded += member.size
            if expanded > MAX_EXTRACTED_BYTES:
                raise RecipeError(f"archive {archive.name} expands past {MAX_EXTRACTED_BYTES} bytes")
            roots.add(path.parts[0])
            inner = pathlib.PurePosixPath(*path.parts[1:])
            if skip is not None and len(path.parts) > 1 and inner.is_relative_to(skip):
                continue
            members.append(member)
        if len(roots) != 1:
            raise RecipeError(f"archive {archive.name} has top-level entries {sorted(roots)}, not one directory")
        opened.extractall(target, members=members, filter="data")
    return target / roots.pop()


def _check_upstream_metadata(pyproject: pathlib.Path, recipe: Recipe) -> None:
    metadata = _table(_toml(pyproject), "project")
    name = re.sub(r"[-_.]+", "-", _text(metadata, "name")).lower()
    if name != "llama-cpp-python":
        raise RecipeError(f"the staged sdist is {name}, not llama-cpp-python")
    dependencies = _texts(metadata, "dependencies")
    if dependencies != recipe.dependencies:
        static = list(recipe.dependencies)
        raise RecipeError(f"upstream dependencies {list(dependencies)} differ from the static dependencies {static}")
    requires_python = _text(metadata, "requires-python")
    if requires_python != recipe.requires_python:
        static = recipe.requires_python
        raise RecipeError(f"upstream requires-python {requires_python!r} differs from the static {static!r}")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _toml(path: pathlib.Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _table(table: collections.abc.Mapping[str, object], key: str) -> dict[str, object]:
    value = table.get(key)
    if not isinstance(value, dict):
        raise RecipeError(f"[{key}] is missing or is not a table")
    return typing.cast(dict[str, object], value)


def _text(table: collections.abc.Mapping[str, object], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise RecipeError(f"{key} is missing or is not a non-empty string")
    return value


def _texts(table: collections.abc.Mapping[str, object], key: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list):
        raise RecipeError(f"{key} is missing or is not a list")
    items = typing.cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise RecipeError(f"{key} holds a value that is not a string")
    return tuple(typing.cast(list[str], items))
