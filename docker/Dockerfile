# One image recipe for any bot project with a `__main__`, run as `python -m <project>`. The build context is the
# repository root, because examples depend on the core package and providers by path; PROJECT names the project
# whose own uv.lock is installed, without dev dependencies and without editable installs, so the source tree stays
# out of the runtime image. The command names the project's package (default minimal_bot_example) and any flag
# bot.process takes after it.
#
#   docker build --build-arg PROJECT=examples/minimal_bot -f docker/Dockerfile .
#
# Build arguments:
#   PROJECT      the project to install (default examples/minimal_bot)
#   NATIVE       1 installs a C toolchain and cmake in the build stage, for providers that compile native code
#                (sqlite_memory, llama_cpp); default 0
#   RUNTIME_APT  Debian packages the runtime needs, for example "libgl1 libglib2.0-0" for opencv; default none
#
# docs/runtime.md ("Running in a container") has the layout, the fixed MOSFET_* options and the stop bound.

FROM ghcr.io/astral-sh/uv:0.9.18@sha256:5713fa8217f92b80223bc83aac7db36ec80a84437dbc0d04bbc659cae030d8c9 AS uv

FROM python:3.13.16-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641 AS build
ARG PROJECT=examples/minimal_bot
ARG NATIVE=0
RUN if [ "$NATIVE" = "1" ]; then \
      apt-get update \
      && apt-get install --yes --no-install-recommends build-essential cmake \
      && rm -rf /var/lib/apt/lists/*; \
    fi
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.13 \
    UV_PROJECT_ENVIRONMENT=/opt/mosfet/venv
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --project "$PROJECT"

FROM python:3.13.16-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641
ARG RUNTIME_APT=""
RUN if [ -n "$RUNTIME_APT" ]; then \
      apt-get update \
      && apt-get install --yes --no-install-recommends $RUNTIME_APT \
      && rm -rf /var/lib/apt/lists/*; \
    fi \
    && groupadd --gid 10001 mosfet \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /var/lib/mosfet --shell /usr/sbin/nologin mosfet \
    && install --directory --owner 10001 --group 10001 --mode 0700 /var/lib/mosfet /run/mosfet
COPY --from=build /opt/mosfet/venv /opt/mosfet/venv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOSFET_READY_FILE=/run/mosfet/ready \
    MOSFET_DATA_DIR=/var/lib/mosfet/data
USER 10001:10001
WORKDIR /var/lib/mosfet
LABEL org.opencontainers.image.title="mosfet bot" \
      org.opencontainers.image.description="A stateforward.mosfet bot project run with python -m" \
      org.opencontainers.image.source="https://github.com/stateforward/mosfet.py" \
      org.opencontainers.image.licenses="MIT"
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=2s --start-period=130s --retries=3 CMD ["test", "-f", "/run/mosfet/ready"]
ENTRYPOINT ["/opt/mosfet/venv/bin/python", "-m"]
CMD ["minimal_bot_example"]
