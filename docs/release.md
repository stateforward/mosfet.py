# Release and Versioning

This repo uses Hatchling dynamic version metadata with a package-owned
`__version__` constant for each distribution.

- `bot` reads its version from `src/bot/__init__.py`.
- `bot-provider-elevenlabs` reads its version from
  `src/providers/elevenlabs/src/bot/providers/elevenlabs/__init__.py`.
- `bot-provider-livekit` reads its version from
  `src/providers/livekit/src/bot/providers/livekit/__init__.py`.
- `bot-provider-mlx-audio` reads its version from
  `src/providers/mlx_audio/src/bot/providers/mlx_audio/__init__.py`.
- `bot-provider-mlx-vlm` reads its version from
  `src/providers/mlx_vlm/src/bot/providers/mlx_vlm/__init__.py`.
- `bot-provider-openai-compat` reads its version from
  `src/providers/openai_compat/src/bot/providers/openai_compat/__init__.py`.
- `bot-provider-sqlite-memory` reads its version from
  `src/providers/sqlite_memory/src/bot/providers/sqlite_memory/__init__.py`.
- `bot-provider-postgres-memory` reads its version from
  `src/providers/postgres_memory/src/bot/providers/postgres_memory/__init__.py`.
- `scripts/check_release_version.py` validates that package `pyproject.toml`
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
- Keep the wheel-to-import mapping exhaustive: an unknown built wheel fails the
  smoke gate until its public import is named explicitly.
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

## Open Decisions Before External Publish

- Keep provider dependencies on core `bot` explicit when a provider imports
  current ability contracts. The OpenAI-compatible provider publishes with
  `stateforward.bot>=0.1.0,<0.2.0`; raise the lower bound in the provider package when a
  future release starts requiring newer core APIs.
- Decide which optional hosted Postgres, PGlite socket, and S3-compatible client
  packages should become provider-owned dependencies. The initial Postgres
  memory provider keeps those adapters injected so the package can support both
  runtimes without changing root `bot` dependencies.
- Keep the ElevenLabs provider spelling aligned across the distribution name,
  workspace path, and import path before publishing to a public index.
- Decide the release matrix for native SQLite memory wheels beyond the current
  Linux CI target and local macOS ARM verification. The provider packages
  `sqlite-objstore` from vendored source, copies the `sqlite-vec` loadable
  extension into the wheel, and tags the wheel as platform-specific.
