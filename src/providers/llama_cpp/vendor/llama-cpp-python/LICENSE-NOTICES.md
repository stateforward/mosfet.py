# Third-party notices

This build recipe holds no third-party source. At build time `_build/backend.py` fetches
the two upstream archives below, verifies them against the sha256 pins in `recipe.toml`,
and builds a wheel from them. That wheel carries each project's own license file. Both
projects are MIT licensed.

| Project | Source | Pin | License |
| --- | --- | --- | --- |
| llama-cpp-python 0.3.36 | <https://files.pythonhosted.org/packages/source/l/llama_cpp_python/llama_cpp_python-0.3.36.tar.gz> | sha256 `832db0699007f1be95a7e41ef12e88926b02ba836461e36a36372db2760c1a2e` | MIT, Copyright (c) 2023 Andrei Betlen |
| llama.cpp `4fbc76dec51d0add466f0210855c0596589b60d4` | <https://github.com/ggml-org/llama.cpp/archive/4fbc76dec51d0add466f0210855c0596589b60d4.tar.gz> | sha256 `4ea37c28e9d6a74e6ffb34ff33a6555b33f0398e75814f0e2f362407d79ecb36` | MIT, Copyright (c) 2023-2026 The ggml authors |

The build replaces the llama.cpp tree that the llama-cpp-python sdist vendors with the
pinned llama.cpp tree. It also rewrites `llama_cpp/__init__.py` to
`__version__ = "0.3.36+mosfet.4fbc76de"`. It changes no other upstream file.
