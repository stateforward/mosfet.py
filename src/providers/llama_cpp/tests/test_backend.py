from __future__ import annotations

import collections.abc
import hashlib
import io
import pathlib
import tarfile
import tomllib
import typing

import pytest

import backend

RECIPE_PROJECT = pathlib.Path(__file__).resolve().parents[1] / "vendor" / "llama-cpp-python"
COMMIT = "4fbc76dec51d0add466f0210855c0596589b60d4"
UPSTREAM_DEPENDENCIES = ("typing-extensions>=4.5.0", "numpy>=1.20.0", "diskcache>=5.6.1", "jinja2>=2.11.3")
HEADERS = {
    "include/llama.h": b"struct llama_context;\n",
    "tools/mtmd/mtmd.h": b"struct mtmd_context;\n",
    "tools/mtmd/mtmd-helper.h": b"struct mtmd_bitmap;\n",
}
UPSTREAM_PYPROJECT = """\
[build-system]
requires = ["scikit-build-core[pyproject]>=0.9.2"]
build-backend = "scikit_build_core.build"

[project]
name = "llama_cpp_python"
dynamic = ["version"]
dependencies = [
    "typing-extensions>=4.5.0",
    "numpy>=1.20.0",
    "diskcache>=5.6.1",
    "jinja2>=2.11.3",
]
requires-python = ">=3.8"
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tarball(
    files: collections.abc.Mapping[str, bytes],
    *,
    comment: str | None = None,
    links: collections.abc.Mapping[str, str] | None = None,
) -> bytes:
    buffer = io.BytesIO()
    pax_headers = {} if comment is None else {"comment": comment}
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT, pax_headers=pax_headers) as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        for name, target in (links or {}).items():
            member = tarfile.TarInfo(name)
            member.type = tarfile.SYMTYPE
            member.linkname = target
            archive.addfile(member)
    return buffer.getvalue()


def sdist(
    *, pyproject: str = UPSTREAM_PYPROJECT, init: bytes = b'"""llama_cpp"""\n\n__version__ = "0.3.36"\n'
) -> bytes:
    return tarball(
        {
            "llama_cpp_python-0.3.36/pyproject.toml": pyproject.encode(),
            "llama_cpp_python-0.3.36/llama_cpp/__init__.py": init,
            "llama_cpp_python-0.3.36/vendor/llama.cpp/include/llama.h": b"struct stale_vendored_header;\n",
        }
    )


def llama_tree(*, comment: str | None = COMMIT, headers: collections.abc.Mapping[str, bytes] = HEADERS) -> bytes:
    return tarball({f"llama.cpp-{COMMIT}/{path}": data for path, data in headers.items()}, comment=comment)


def recipe(sdist_data: bytes, llama_data: bytes) -> backend.Recipe:
    return backend.Recipe(
        version="0.3.36",
        local_version="mosfet.test",
        sdist=backend.Archive(
            url="https://files.example.invalid/llama_cpp_python-0.3.36.tar.gz", sha256=sha256(sdist_data)
        ),
        llama_cpp=backend.Archive(url=f"https://archive.example.invalid/{COMMIT}.tar.gz", sha256=sha256(llama_data)),
        commit=COMMIT,
        abi={path: sha256(data) for path, data in HEADERS.items()},
        dependencies=UPSTREAM_DEPENDENCIES,
        requires_python=">=3.8",
    )


class Downloads:
    """Serves archives by URL and records every download the backend asks for."""

    def __init__(self, archives: collections.abc.Mapping[str, bytes]) -> None:
        self.archives: dict[str, bytes] = dict(archives)
        self.requested: list[str] = []

    def __call__(self, url: str, target: pathlib.Path) -> None:
        self.requested.append(url)
        _ = target.write_bytes(self.archives[url])


def serve(pins: backend.Recipe, sdist_data: bytes, llama_data: bytes) -> Downloads:
    return Downloads({pins.sdist.url: sdist_data, pins.llama_cpp.url: llama_data})


def test_stage_rebuilds_the_sdist_on_the_pinned_llama_cpp_tree(tmp_path: pathlib.Path) -> None:
    sdist_data, llama_data = sdist(), llama_tree()
    pins = recipe(sdist_data, llama_data)

    tree = backend.stage(tmp_path, pins, sources=None, download=serve(pins, sdist_data, llama_data), llama_cpp=True)

    assert (tree / "vendor/llama.cpp/include/llama.h").read_bytes() == HEADERS["include/llama.h"]
    assert (tree / "vendor/llama.cpp/tools/mtmd/mtmd-helper.h").is_file()
    assert '__version__ = "0.3.36+mosfet.test"\n' in (tree / "llama_cpp/__init__.py").read_text()


def test_metadata_staging_never_fetches_the_llama_cpp_tree(tmp_path: pathlib.Path) -> None:
    sdist_data, llama_data = sdist(), llama_tree()
    pins = recipe(sdist_data, llama_data)
    downloads = serve(pins, sdist_data, llama_data)

    tree = backend.stage(tmp_path, pins, sources=None, download=downloads, llama_cpp=False)

    assert downloads.requested == [pins.sdist.url]
    assert not (tree / "vendor/llama.cpp").exists()
    assert '__version__ = "0.3.36+mosfet.test"\n' in (tree / "llama_cpp/__init__.py").read_text()


@pytest.mark.parametrize("archive", ["sdist", "llama_cpp"])
def test_a_download_that_does_not_match_its_pin_is_refused(tmp_path: pathlib.Path, archive: str) -> None:
    sdist_data, llama_data = sdist(), llama_tree()
    pins = recipe(sdist_data, llama_data)
    downloads = serve(pins, sdist_data, llama_data)
    tampered = pins.sdist.url if archive == "sdist" else pins.llama_cpp.url
    downloads.archives[tampered] += b"tampered"

    with pytest.raises(backend.RecipeError, match="sha256"):
        _ = backend.stage(tmp_path, pins, sources=None, download=downloads, llama_cpp=True)


@pytest.mark.parametrize("comment", [None, "0c1e57091ca1a8d4ad1ab4e1c47c2ab4c4b0a5ff"])
def test_a_llama_cpp_archive_from_another_commit_is_refused(tmp_path: pathlib.Path, comment: str | None) -> None:
    sdist_data, llama_data = sdist(), llama_tree(comment=comment)
    pins = recipe(sdist_data, llama_data)

    with pytest.raises(backend.RecipeError, match="commit"):
        _ = backend.stage(tmp_path, pins, sources=None, download=serve(pins, sdist_data, llama_data), llama_cpp=True)


def test_an_abi_header_that_does_not_match_its_pin_is_refused(tmp_path: pathlib.Path) -> None:
    sdist_data = sdist()
    llama_data = llama_tree(headers={**HEADERS, "tools/mtmd/mtmd.h": b"struct mtmd_context { int moved; };\n"})
    pins = recipe(sdist_data, llama_data)

    with pytest.raises(backend.RecipeError, match="tools/mtmd/mtmd.h"):
        _ = backend.stage(tmp_path, pins, sources=None, download=serve(pins, sdist_data, llama_data), llama_cpp=True)


def test_a_missing_abi_header_is_refused(tmp_path: pathlib.Path) -> None:
    sdist_data = sdist()
    llama_data = llama_tree(headers={path: data for path, data in HEADERS.items() if path != "include/llama.h"})
    pins = recipe(sdist_data, llama_data)

    with pytest.raises(backend.RecipeError, match="include/llama.h"):
        _ = backend.stage(tmp_path, pins, sources=None, download=serve(pins, sdist_data, llama_data), llama_cpp=True)


@pytest.mark.parametrize(
    ("files", "links"),
    [
        ({"llama_cpp_python-0.3.36/../escape.txt": b"x"}, None),
        ({"/llama_cpp_python-0.3.36/absolute.txt": b"x"}, None),
        ({}, {"llama_cpp_python-0.3.36/link": "/etc/passwd"}),
        ({"llama_cpp_python-0.3.36/a.txt": b"x", "second-root/b.txt": b"y"}, None),
    ],
)
def test_an_unsafe_archive_is_refused_before_extraction(
    tmp_path: pathlib.Path, files: dict[str, bytes], links: dict[str, str] | None
) -> None:
    sdist_data, llama_data = tarball(files, links=links), llama_tree()
    pins = recipe(sdist_data, llama_data)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(backend.RecipeError, match="archive"):
        _ = backend.stage(staging, pins, sources=None, download=serve(pins, sdist_data, llama_data), llama_cpp=True)

    assert not (tmp_path / "escape.txt").exists()
    assert not (staging / "sdist").exists()


def test_local_sources_are_used_when_their_hashes_match(tmp_path: pathlib.Path) -> None:
    sdist_data, llama_data = sdist(), llama_tree()
    pins = recipe(sdist_data, llama_data)
    sources = tmp_path / "sources"
    sources.mkdir()
    _ = (sources / "llama_cpp_python-0.3.36.tar.gz").write_bytes(sdist_data)
    _ = (sources / f"{COMMIT}.tar.gz").write_bytes(llama_data)
    downloads = Downloads({})
    staging = tmp_path / "staging"
    staging.mkdir()

    tree = backend.stage(staging, pins, sources=sources, download=downloads, llama_cpp=True)

    assert downloads.requested == []
    assert (tree / "vendor/llama.cpp/include/llama.h").is_file()


def test_a_local_source_that_does_not_match_its_pin_is_refused_and_never_replaced(tmp_path: pathlib.Path) -> None:
    sdist_data, llama_data = sdist(), llama_tree()
    pins = recipe(sdist_data, llama_data)
    sources = tmp_path / "sources"
    sources.mkdir()
    _ = (sources / "llama_cpp_python-0.3.36.tar.gz").write_bytes(sdist_data + b"tampered")
    downloads = serve(pins, sdist_data, llama_data)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(backend.RecipeError, match="sha256"):
        _ = backend.stage(staging, pins, sources=sources, download=downloads, llama_cpp=True)

    assert downloads.requested == []


def test_upstream_dependencies_that_drift_from_the_static_metadata_are_refused(tmp_path: pathlib.Path) -> None:
    sdist_data = sdist(pyproject=UPSTREAM_PYPROJECT.replace('"numpy>=1.20.0"', '"numpy>=2.0.0"'))
    llama_data = llama_tree()
    pins = recipe(sdist_data, llama_data)

    with pytest.raises(backend.RecipeError, match="dependencies"):
        _ = backend.stage(tmp_path, pins, sources=None, download=serve(pins, sdist_data, llama_data), llama_cpp=True)


def test_an_upstream_version_line_that_cannot_be_rewritten_is_refused(tmp_path: pathlib.Path) -> None:
    sdist_data = sdist(init=b'__version__ = "0.3.35"\n')
    llama_data = llama_tree()
    pins = recipe(sdist_data, llama_data)

    with pytest.raises(backend.RecipeError, match="__version__"):
        _ = backend.stage(tmp_path, pins, sources=None, download=serve(pins, sdist_data, llama_data), llama_cpp=True)


def test_the_recipe_pins_the_measured_sources_and_matches_the_static_metadata() -> None:
    pins = backend.Recipe.load(RECIPE_PROJECT)

    assert pins.version == "0.3.36"
    assert pins.local_version == "mosfet.4fbc76de"
    assert pins.sdist == backend.Archive(
        url="https://files.pythonhosted.org/packages/source/l/llama_cpp_python/llama_cpp_python-0.3.36.tar.gz",
        sha256="832db0699007f1be95a7e41ef12e88926b02ba836461e36a36372db2760c1a2e",
    )
    assert pins.commit == COMMIT
    assert pins.llama_cpp == backend.Archive(
        url=f"https://github.com/ggml-org/llama.cpp/archive/{COMMIT}.tar.gz",
        sha256="4ea37c28e9d6a74e6ffb34ff33a6555b33f0398e75814f0e2f362407d79ecb36",
    )
    assert pins.abi == {
        "include/llama.h": "c9a5f025ad7dbf07eda6660c8ff2c32cd80c07ba440d80728707a9c9d13c1bc3",
        "tools/mtmd/mtmd.h": "ff6798e8ec56a409b8f2e99821cdbdb857286ca085a762b7ed3e5599c1cc77d5",
        "tools/mtmd/mtmd-helper.h": "893b42990d4b386b8c52d8078499f5615c2d7b0c92d0c30501816d65cab8db6c",
    }
    assert pins.dependencies == UPSTREAM_DEPENDENCIES
    assert pins.requires_python == ">=3.8"


def test_the_build_backend_is_pinned_exactly_and_rebuilds_on_every_recipe_input() -> None:
    project = tomllib.loads((RECIPE_PROJECT / "pyproject.toml").read_text())
    recipe_pins = tomllib.loads((RECIPE_PROJECT / "recipe.toml").read_text())

    assert project["build-system"] == {
        "requires": [f"scikit-build-core=={recipe_pins['build']['scikit_build_core']}"],
        "build-backend": "backend",
        "backend-path": ["_build"],
    }
    assert project["project"]["version"] == "0.3.36+mosfet.4fbc76de"
    assert project["tool"]["uv"]["cache-keys"] == [
        {"file": "pyproject.toml"},
        {"file": "recipe.toml"},
        {"file": "_build/backend.py"},
        {"env": "CMAKE_ARGS"},
        {"env": "CMAKE_BUILD_PARALLEL_LEVEL"},
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sdist_url", "http://files.pythonhosted.org/llama_cpp_python-0.3.36.tar.gz", "https"),
        ("local_version", "mosfet.other", "version"),
    ],
)
def test_a_recipe_that_disagrees_with_itself_is_refused(
    tmp_path: pathlib.Path, field: str, value: str, message: str
) -> None:
    pyproject = (RECIPE_PROJECT / "pyproject.toml").read_text()
    recipe_text = (RECIPE_PROJECT / "recipe.toml").read_text()
    lines = [f'{field} = "{value}"' if line.startswith(f"{field} = ") else line for line in recipe_text.splitlines()]
    _ = (tmp_path / "pyproject.toml").write_text(pyproject)
    _ = (tmp_path / "recipe.toml").write_text("\n".join(lines) + "\n")

    with pytest.raises(backend.RecipeError, match=message):
        _ = backend.Recipe.load(tmp_path)


@pytest.mark.parametrize("hook", [backend.build_sdist, backend.build_editable])
def test_sdist_and_editable_builds_are_unsupported(hook: typing.Callable[[str], str]) -> None:
    with pytest.raises(NotImplementedError, match="wheel"):
        _ = hook("dist")
