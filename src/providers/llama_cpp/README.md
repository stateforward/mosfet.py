# mosfet-provider-llama-cpp

In-process llama.cpp provider package for `stateforward.mosfet`. It drives the native
llama.cpp library through a pinned build of `llama-cpp-python`, so core `mosfet` stays
provider-agnostic and never imports `llama_cpp`.

## The pinned build

Upstream `llama-cpp-python` 0.3.36 vendors a llama.cpp that predates the models this
provider runs. This package depends on `llama-cpp-python==0.3.36+mosfet.4fbc76de`: the
same 0.3.36 bindings rebuilt against llama.cpp commit
`4fbc76dec51d0add466f0210855c0596589b60d4`. Nothing is published and no package index is
involved. The build recipe lives in this package at `vendor/llama-cpp-python/` and is a
`uv` path source:

- `recipe.toml` pins both upstream source archives by URL and sha256, the llama.cpp
  commit, the scikit-build-core version, and the sha256 of the three ABI headers
  (`llama.h`, `mtmd.h`, `mtmd-helper.h`) the 0.3.36 ctypes bindings were proven against.
- `_build/backend.py` is an in-tree PEP 517 backend. It fetches both archives, verifies
  each sha256 and the llama.cpp archive's commit, swaps the sdist's vendored llama.cpp for
  the pinned tree, verifies the ABI headers, stamps `__version__ = "0.3.36+mosfet.4fbc76de"`,
  and builds the wheel with scikit-build-core. Any mismatch stops the build.
- `pyproject.toml` declares static metadata, so `uv lock` never compiles. `uv.lock`
  records a directory source with no binary hash: every machine compiles once into its
  `uv` cache and rebuilds only when `pyproject.toml`, `recipe.toml`, `_build/backend.py`,
  `CMAKE_ARGS` or `CMAKE_BUILD_PARALLEL_LEVEL` changes.

`Library` refuses any other `llama_cpp` build at runtime, including the official 0.3.36
wheels, with `LibraryError`.

## Building

A C/C++ toolchain is required; scikit-build-core fetches `cmake` and `ninja` when the host
lacks them. From the workspace root:

```sh
# macOS arm64: Metal is on by default
uv sync --package mosfet-provider-llama-cpp

# Linux with an NVIDIA GPU: needs the CUDA toolkit (nvcc); pick the GPU's architecture
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=121" uv sync --package mosfet-provider-llama-cpp

# Linux CPU (CI): no flags
uv sync --package mosfet-provider-llama-cpp
```

Air-gapped and container builds can serve the two archives from a directory instead of the
network: `BOT_LLAMA_CPP_SOURCES=/path/to/sources`, holding
`llama_cpp_python-0.3.36.tar.gz` and `4fbc76dec51d0add466f0210855c0596589b60d4.tar.gz`. A
local file is used only when its sha256 matches the pin.

Check which accelerator the installed build computes on:

```sh
uv run --package mosfet-provider-llama-cpp python -c \
  "from mosfet.providers import llama_cpp; print(llama_cpp.Library().accelerator())"
```

## Tests

```sh
uv run --package mosfet-provider-llama-cpp python -m pytest src/providers/llama_cpp/tests -m 'not live'
```

The build-recipe tests use small synthetic archives and never touch the network. The one
`live` test loads the installed native library and checks that it is the pinned build.

## Publishing

This package is excluded from `uv publish`: PyPI cannot serve a dependency pinned to a
local version label (`+mosfet.4fbc76de`). When an upstream `llama-cpp-python` release
vendors llama.cpp at or after `4fbc76de`, one change deletes `vendor/llama-cpp-python/`,
pins that release exactly, updates the version `Library` asserts, and drops the exclusion.
