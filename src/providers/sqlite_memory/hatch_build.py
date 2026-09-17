from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import typing
import importlib


class _BuildHookInterface:
    root: str = ""


def _build_hook_interface() -> type[_BuildHookInterface]:
    try:
        module = importlib.import_module("hatchling.builders.hooks.plugin.interface")
    except ModuleNotFoundError:
        return _BuildHookInterface
    return typing.cast("type[_BuildHookInterface]", getattr(module, "BuildHookInterface"))


BuildHookInterface = _build_hook_interface()


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        del version
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        if os.environ.get("BOT_SQLITE_MEMORY_SKIP_NATIVE_BUILD") == "1":
            return
        root = pathlib.Path(self.root)
        output_dir = root / "src/mosfet/providers/sqlite_memory/_native"
        _ = build_objstore_extension(root, output_dir)
        _ = copy_sqlite_vec_extension(output_dir)


def build_objstore_extension(root: pathlib.Path, output_dir: pathlib.Path | None = None) -> pathlib.Path:
    vendor = root / "vendor/sqlite-objstore"
    if output_dir is None:
        output_dir = root / "src/mosfet/providers/sqlite_memory/_native"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / native_library_name()
    compiler = compiler_command()
    if compiler is None:
        raise RuntimeError("A C compiler is required to build the bundled sqlite-objstore extension.")

    sources = [str(path) for path in objstore_sources(vendor)]
    command = [
        *compiler,
        *shared_library_flags(),
        "-O2",
        "-fPIC",
        "-std=c17",
        "-DOBJSTORE_HAVE_BACKEND_FILE=1",
        "-DOBJSTORE_HAVE_BACKEND_SQLITE=1",
        "-DOBJSTORE_HAVE_BACKEND_OPFS=0",
        "-DOBJSTORE_HAVE_BACKEND_VFS=0",
        "-DBLAKE3_NO_SSE2=1",
        "-DBLAKE3_NO_SSE41=1",
        "-DBLAKE3_NO_AVX2=1",
        "-DBLAKE3_NO_AVX512=1",
        "-DBLAKE3_USE_NEON=0",
        "-I",
        str(root / "vendor/sqlite"),
        "-I",
        str(vendor / "include"),
        "-I",
        str(vendor / "src"),
        "-I",
        str(vendor / "third_party/blake3"),
        "-include",
        "src/mosfet/providers/sqlite_memory/_sqlite_extension_api.h",
        "-o",
        str(output),
        *sources,
        *sqlite_link_flags(),
    ]
    _ = subprocess.run(command, cwd=root, check=True)
    return output


def copy_sqlite_vec_extension(output_dir: pathlib.Path) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sqlite_vec = importlib.import_module("sqlite_vec")
    except ModuleNotFoundError as error:
        raise RuntimeError("sqlite-vec is required to bundle the provider's vector extension.") from error

    sqlite_vec_file = typing.cast(str | None, getattr(sqlite_vec, "__file__", None))
    if sqlite_vec_file is None:
        raise RuntimeError("sqlite-vec does not expose a package file path.")
    source_dir = pathlib.Path(sqlite_vec_file).resolve().parent
    source = source_dir / sqlite_vec_library_name()
    if not source.exists():
        suffixless_source = source_dir / "vec0"
        if suffixless_source.exists():
            source = suffixless_source
        else:
            raise RuntimeError(f"sqlite-vec extension artifact was not found in {source_dir}.")

    destination = output_dir / sqlite_vec_library_name()
    _ = shutil.copy2(source, destination)
    return destination


def sqlite_vec_library_name() -> str:
    if sys.platform == "darwin":
        return "vec0.dylib"
    if sys.platform == "win32":
        return "vec0.dll"
    return "vec0.so"


def compiler_command() -> list[str] | None:
    compiler = sysconfig.get_config_var("CC") or shutil.which("cc")
    if not compiler:
        return None
    return str(compiler).split()


def native_library_name() -> str:
    if sys.platform == "darwin":
        return "bot_objstore.dylib"
    if sys.platform == "win32":
        return "bot_objstore.dll"
    return "bot_objstore.so"


def shared_library_flags() -> list[str]:
    if sys.platform == "darwin":
        return ["-dynamiclib"]
    if sys.platform == "win32":
        return ["-shared"]
    return ["-shared"]


def sqlite_link_flags() -> list[str]:
    if sys.platform == "darwin":
        return ["-undefined", "dynamic_lookup"]
    if sys.platform == "win32":
        return ["-lsqlite3"]
    return []


def objstore_sources(vendor: pathlib.Path) -> tuple[pathlib.Path, ...]:
    return (
        pathlib.Path("src/mosfet/providers/sqlite_memory/_objstore_extension.c"),
        vendor / "src/objstore.c",
        vendor / "src/backend_registry.c",
        vendor / "src/backend_fs_common.c",
        vendor / "src/backend_portable.c",
        vendor / "src/backend_file.c",
        vendor / "src/backend_sqlite.c",
        vendor / "src/objstore_vtab.c",
        vendor / "src/objstore_txn.c",
        vendor / "src/object_manager.c",
        vendor / "src/blake3_hash.c",
        vendor / "third_party/blake3/blake3.c",
        vendor / "third_party/blake3/blake3_dispatch.c",
        vendor / "third_party/blake3/blake3_portable.c",
    )
