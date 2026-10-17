# The builder bot's bench: the one image its sandbox allows (BL-5). docker.Host makes each machine from it with
# `sleep infinity` as uid 65532 on a read-only root, with no network unless the owner configures a bridge, so the
# bench has only what this image brings: python, uv, git and pytest, plus the sh, timeout, mkdir, cat and base64
# that docker.Host's exec wrappers need. Built locally into the rootless daemon and never pushed or pulled:
#
#   podman build --tag localhost/mosfet-bench:1 --file docker/bench/Dockerfile docker/bench
#
# docker/bench/smoke.sh builds it and checks it against a rootless Podman socket.

FROM ghcr.io/astral-sh/uv:0.9.18@sha256:5713fa8217f92b80223bc83aac7db36ec80a84437dbc0d04bbc659cae030d8c9 AS uv

FROM python:3.13.16-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641
# The workspace tmpfs is root-owned and sticky (W5-15) while uid 65532 works in it, so git would refuse a
# repository at the workspace root as one of dubious ownership; the system config trusts that one path only.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system safe.directory /workspace
COPY --from=uv /uv /usr/local/bin/uv
# pytest at one exact pin, with its own requirements pinned so the image rebuilds the same; the versions are the
# repository's uv.lock ones. Compiled at build time because the root is read-only at run time.
RUN UV_PYTHON_DOWNLOADS=never uv pip install --system --no-cache --compile-bytecode \
      pytest==9.0.3 iniconfig==2.3.0 packaging==26.2 pluggy==1.6.0 pygments==2.20.0
LABEL org.opencontainers.image.title="mosfet bench" \
      org.opencontainers.image.description="The builder bot's sandbox bench: python, uv, git and pytest" \
      org.opencontainers.image.source="https://github.com/stateforward/mosfet.py" \
      org.opencontainers.image.licenses="MIT"
