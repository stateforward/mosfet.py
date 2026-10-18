# The builder bot's bench: the one image its sandbox allows (BL-5). docker.Host makes each machine from it with
# `sleep infinity` as uid 65532 on a read-only root, with no network unless the owner configures a bridge, so the
# bench has only what this image brings: python with stateforward.mosfet installed (bench image 2, W7-14), uv,
# git and pytest, plus the sh, timeout, mkdir, cat and base64 that docker.Host's exec wrappers need, and
# docs/bot.md and docs/bot_diagnostics.md at /usr/share/doc/mosfet/. So the bot can write a bot file,
# `from mosfet import bot`, and test it with pytest, reading bot.define's diagnostics.
# Built locally into the rootless daemon and never pushed or pulled. The build context is the repository root,
# because mosfet is installed from its source and its uv.lock. The image is built for one mosfet commit and
# labeled with it (org.opencontainers.image.revision), so the build needs that commit:
#
#   podman build --build-arg MOSFET_COMMIT=$(git rev-parse HEAD) --tag localhost/mosfet-bench:2 \
#     --file docker/bench/Dockerfile .
#
# Rebuild it after every pull or commit that changes mosfet: the builder bot checks the label when it starts and
# refuses an image built for another commit (or one with no label), naming both commits and this command.
# docker/bench/smoke.sh builds it the same way and checks it against a rootless Podman socket.

FROM ghcr.io/astral-sh/uv:0.9.18@sha256:5713fa8217f92b80223bc83aac7db36ec80a84437dbc0d04bbc659cae030d8c9 AS uv

FROM python:3.13.16-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.13 \
    UV_PROJECT_ENVIRONMENT=/opt/mosfet/venv
WORKDIR /src
COPY . .
# stateforward.mosfet and its locked dependencies (the root project only, none of the workspace's providers),
# without dev dependencies and not editable, so the source tree stays out of the bench; then pytest at one exact
# pin, with its own requirements pinned so the image rebuilds the same (the versions are the repository's
# uv.lock ones). Compiled at build time because the root is read-only at run time.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable \
    && uv pip install --python /opt/mosfet/venv/bin/python --compile-bytecode \
      pytest==9.0.3 iniconfig==2.3.0 packaging==26.2 pluggy==1.6.0 pygments==2.20.0

FROM python:3.13.16-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641
# The workspace tmpfs is root-owned and sticky (W5-15) while uid 65532 works in it, so git would refuse a
# repository at the workspace root as one of dubious ownership; the system config trusts that one path only.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system safe.directory /workspace
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=build /opt/mosfet/venv /opt/mosfet/venv
# The bench's `python` is the one with mosfet: a machine runs `python -m pytest` in it.
ENV PATH=/opt/mosfet/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/mosfet/venv
LABEL org.opencontainers.image.title="mosfet bench" \
      org.opencontainers.image.description="The builder bot's sandbox bench: python with mosfet, uv, git and pytest" \
      org.opencontainers.image.version="2" \
      org.opencontainers.image.source="https://github.com/stateforward/mosfet.py" \
      org.opencontainers.image.licenses="MIT"
# The bot definition guide and every diagnostic's explanation, for whoever writes a bot file on the bench; the
# builder's README tells its person where they are.
COPY docs/bot.md docs/bot_diagnostics.md /usr/share/doc/mosfet/
# The mosfet commit this image carries, last so a new commit rebuilds only this layer of the final stage. A build
# without it fails: an unlabeled bench is one the builder refuses.
ARG MOSFET_COMMIT
RUN case "$MOSFET_COMMIT" in \
      *[!0-9a-f]* | "") echo "MOSFET_COMMIT must be a commit: --build-arg MOSFET_COMMIT=\$(git rev-parse HEAD)" >&2; exit 1 ;; \
    esac \
    && [ "${#MOSFET_COMMIT}" -eq 40 ] \
    || { echo "MOSFET_COMMIT must be a full 40-character commit" >&2; exit 1; }
LABEL org.opencontainers.image.revision="$MOSFET_COMMIT"
