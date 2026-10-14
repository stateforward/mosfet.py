# Release and Versioning

This repo uses Hatchling dynamic version metadata with a package-owned
`__version__` constant for each distribution.

- `bot` reads its version from `src/mosfet/__init__.py`.
- `mosfet-provider-elevenlabs` reads its version from
  `src/providers/elevenlabs/src/mosfet/providers/elevenlabs/__init__.py`.
- `mosfet-provider-gemini` reads its version from
  `src/providers/gemini/src/mosfet/providers/gemini/__init__.py`.
- `mosfet-provider-livekit` reads its version from
  `src/providers/livekit/src/mosfet/providers/livekit/__init__.py`.
- `mosfet-provider-llama-cpp` reads its version from
  `src/providers/llama_cpp/src/mosfet/providers/llama_cpp/__init__.py`. It is built and
  tested like every provider but is **never published** (see below).
- `mosfet-provider-mlx-audio` reads its version from
  `src/providers/mlx_audio/src/mosfet/providers/mlx_audio/__init__.py`.
- `mosfet-provider-mlx-vlm` reads its version from
  `src/providers/mlx_vlm/src/mosfet/providers/mlx_vlm/__init__.py`.
- `mosfet-provider-moonshine` reads its version from
  `src/providers/moonshine/src/mosfet/providers/moonshine/__init__.py`.
- `mosfet-provider-openai-compat` reads its version from
  `src/providers/openai_compat/src/mosfet/providers/openai_compat/__init__.py`.
- `mosfet-provider-pyannote` reads its version from
  `src/providers/pyannote/src/mosfet/providers/pyannote/__init__.py`.
- `mosfet-provider-sqlite-memory` reads its version from
  `src/providers/sqlite_memory/src/mosfet/providers/sqlite_memory/__init__.py`.
- `scripts/check_release_version.py` discovers provider packages via
  `src/providers/*/pyproject.toml` (workspace glob, not a hardcoded list), validates that package `pyproject.toml`
  files declare `dynamic = ["version"]`, that Hatchling points at the expected
  version file, and that a release tag such as `v0.1.0` matches the package
  versions.

## Why This Shape

Python package metadata requires a version, and the PyPA `pyproject.toml`
specification allows the version to be supplied dynamically by the build backend.
Hatchling already builds this repo and supports reading a dynamic version from a
file containing `__version__`, so this setup avoids adding a versioning
dependency just to publish.

The CI cost tradeoff is deliberate:

- Do not derive versions from git history in CI. GitHub checkout fetches one
  commit by default, while full tag/history based schemes require `fetch-depth: 0`.
- Do validate the pushed tag against source-controlled package versions.
- Do use a single Python 3.13 job while the project declares `requires-python =
  ">=3.13"`. Add a matrix only when the package claims support for more Python
  minors.
- Do use `actions/setup-python` for the runner-cached Python and `setup-uv` for
  dependency caching.
- Do build release artifacts with `uv build --all-packages --no-sources` so the
  packages are checked in a publish-like mode.
- Do fail CI unless every wheel produced by that build installs with its local
  core wheel into a separate clean virtual environment and its public package
  imports with the repository and user site removed from Python's import path.
- Run that smoke gate through `scripts/smoke_wheels.sh`, so CI and a local run
  execute the same code; `scripts/smoke_wheels.sh --map-only dist` prints each
  wheel's import name without installing anything.
- Build the pinned `llama-cpp-python` wheel into `dist-native/` and pass
  `--find-links dist-native` to the smoke gate. `mosfet-provider-llama-cpp` depends on
  `llama-cpp-python==0.3.36+mosfet.4fbc76de`, which no index serves, and its build recipe
  is not a workspace member, so `uv build --all-packages` does not build it.
- Derive each wheel's import from the package naming contract: the
  `stateforward_mosfet` wheel imports `mosfet`, and a `mosfet_provider_<name>`
  wheel imports `mosfet.providers.<name>`. A built wheel outside that contract
  fails the smoke gate.
- Keep the workflow as a package publish-readiness lane until the broader repo
  Ruff and basedpyright baseline is intentionally cleaned up.

Primary references:

- [PyPA project metadata and dynamic fields](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
- [Hatch version source configuration](https://hatch.pypa.io/dev/version/)
- [uv package build and publish guide](https://docs.astral.sh/uv/guides/package/)
- [actions/checkout fetch-depth behavior](https://github.com/actions/checkout)
- [astral-sh/setup-uv caching and Python notes](https://github.com/astral-sh/setup-uv)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)

## Release Flow

1. Update every package `__version__` that will be released.
2. Run `uv run python scripts/check_release_version.py`.
3. Run `uv build --all-packages --no-sources`.
4. Commit the version bump.
5. Push a tag that matches the package versions, for example `v0.1.0`.

The `.github/workflows/release.yml` workflow validates the tag, runs scoped
Ruff, scoped basedpyright, package metadata tests, provider tests, builds all
packages, and publishes with `uv publish --trusted-publishing always`.
Configure the `pypi` GitHub environment and PyPI Trusted Publisher before
pushing a release tag.

### Packages excluded from publish

`mosfet-provider-llama-cpp` is removed from `dist/` before `uv publish`. It pins
`llama-cpp-python==0.3.36+mosfet.4fbc76de`, a local build of upstream 0.3.36 against
llama.cpp `4fbc76de` that its build recipe compiles on each machine. An index cannot
serve that requirement, so a published wheel could not be installed. The exclusion
step fails the release if it does not find exactly that provider's wheel and sdist.
The exclusion ends when an upstream `llama-cpp-python` release vendors llama.cpp at or
after `4fbc76de`. At that point one change pins that release exactly, deletes
`src/providers/llama_cpp/vendor/llama-cpp-python/`, and drops the exclusion step.

## Open Decisions Before External Publish

- Keep provider dependencies on core `bot` explicit when a provider imports
  current ability contracts. The OpenAI-compatible provider publishes with
  `stateforward.mosfet>=0.1.0,<0.2.0`; raise the lower bound in the provider package when a
  future release starts requiring newer core APIs.
- Keep the ElevenLabs provider spelling aligned across the distribution name,
  workspace path, and import path before publishing to a public index.
- Decide the release matrix for native SQLite memory wheels beyond the current
  Linux CI target and local macOS ARM verification. The provider packages
  `sqlite-objstore` from vendored source, copies the `sqlite-vec` loadable
  extension into the wheel, and tags the wheel as platform-specific.
