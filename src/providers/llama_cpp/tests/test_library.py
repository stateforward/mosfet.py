from __future__ import annotations

import sys

import pytest

from mosfet.protocols import attachment
from mosfet.providers import llama_cpp

PINNED_BUILD = "0.3.36+mosfet.4fbc76de"
METAL_SYSTEM_INFO = (
    b"MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | DOTPROD = 1 | "
    b"ACCELERATE = 1 | REPACK = 1 | "
)
CUDA_SYSTEM_INFO = b"CUDA : ARCHS = 1210 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : NEON = 1 | REPACK = 1 | "
CPU_SYSTEM_INFO = b"CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | REPACK = 1 | "


class Bindings:
    """A stand-in for the ``llama_cpp`` module: the version it reports and its system-info value."""

    __version__: object
    system_info: object

    def __init__(self, version: object = PINNED_BUILD, system_info: object = METAL_SYSTEM_INFO) -> None:
        self.__version__ = version
        self.system_info = system_info

    def llama_print_system_info(self) -> object:
        return self.system_info


@pytest.mark.parametrize(
    ("system_info", "accelerator"),
    [
        (METAL_SYSTEM_INFO, llama_cpp.Accelerator.METAL),
        (b"Metal : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ", llama_cpp.Accelerator.METAL),
        (CUDA_SYSTEM_INFO, llama_cpp.Accelerator.CUDA),
        (CPU_SYSTEM_INFO, llama_cpp.Accelerator.CPU),
        (b"Vulkan : COOPMAT = 1 | CPU : AVX2 = 1 | ", llama_cpp.Accelerator.CPU),
    ],
)
def test_accelerator_is_read_from_the_registered_backends(
    system_info: bytes, accelerator: llama_cpp.Accelerator
) -> None:
    library = llama_cpp.Library(module=Bindings(system_info=system_info))

    assert library.accelerator() is accelerator


@pytest.mark.parametrize("version", ["0.3.36", "0.3.37", "0.3.36+mosfet.0c1e5709", "", None])
def test_any_build_other_than_the_pinned_one_is_refused(version: object) -> None:
    library = llama_cpp.Library(module=Bindings(version=version))

    with pytest.raises(llama_cpp.LibraryError, match=r"not the pinned build 0\.3\.36\+mosfet\.4fbc76de") as raised:
        _ = library.accelerator()

    assert raised.value.failure_kind == attachment.FailureKind.INITIALIZATION.value


def test_a_module_without_a_version_is_refused() -> None:
    library = llama_cpp.Library(module=object())

    with pytest.raises(llama_cpp.LibraryError, match="not the pinned build"):
        _ = library.accelerator()


@pytest.mark.parametrize("system_info", [b"", b"no backends here", "MTL : EMBED_LIBRARY = 1 | ", None])
def test_unreadable_system_info_is_refused(system_info: object) -> None:
    library = llama_cpp.Library(module=Bindings(system_info=system_info))

    with pytest.raises(llama_cpp.LibraryError, match="system info"):
        _ = library.accelerator()


class VersionOnly:
    """A ``llama_cpp`` stand-in of the pinned version that exposes no system info."""

    __version__: str = PINNED_BUILD


def test_bindings_without_system_info_are_refused() -> None:
    library = llama_cpp.Library(module=VersionOnly())

    with pytest.raises(llama_cpp.LibraryError, match="system info"):
        _ = library.accelerator()


@pytest.mark.live
def test_the_installed_build_is_the_pinned_one_and_reports_its_accelerator() -> None:
    accelerator = llama_cpp.Library().accelerator()

    if sys.platform == "darwin":
        assert accelerator is llama_cpp.Accelerator.METAL
    else:
        assert accelerator in set(llama_cpp.Accelerator)
