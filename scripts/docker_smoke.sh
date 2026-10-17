#!/usr/bin/env bash
# Build the generic image (docker/Dockerfile) for examples/minimal_bot and check what it promises a container:
# `python -m minimal_bot_example` becomes healthy only while the bot is ready, runs as uid 10001 on a read-only
# root with a 0700 data dir, stops with exit 0 inside Docker's 10 s SIGKILL, exits 1 (the bot file's own error)
# without its secret and 64 on a bad flag, and a ready file a SIGKILLed run left behind is gone once the next run
# boots. With --compose it instead brings up examples/minimal_bot/compose.yaml with its endpoint and secret
# overridden, waits for healthy, stops it and checks exit 0.
#
# Usage: scripts/docker_smoke.sh [--compose]
#
# Every docker command is argv form and takes only values this script made, so nothing untrusted is
# interpolated. The secret is a placeholder written into a labeled volume, so the check needs no bind mount and
# runs against any daemon. Every container and volume it makes is removed on exit; the image and the build cache
# stay.
set -euo pipefail

mode=image
if (( $# == 1 )) && [[ "$1" == "--compose" ]]; then
  mode=compose
elif (( $# > 0 )); then
  echo "Usage: scripts/docker_smoke.sh [--compose]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

image="mosfet-smoke/minimal_bot:local"
run_id="mosfet-smoke-$$"
label="org.stateforward.mosfet.smoke=$run_id"
secrets_volume="$run_id-secrets"
# minimal_bot has no devices, so it reaches ready without calling this closed local port.
endpoint=(--env OPENAI_BASE_URL=http://127.0.0.1:9/v1 --env OPENAI_MODEL=smoke)
secrets_mount=(--mount "type=volume,source=$secrets_volume,target=/run/secrets,readonly")
# The compose override and its placeholder secret, under the example's git- and docker-ignored secrets dir: a
# compose secret is a bind mount, and a path under the checkout is one a VM-backed daemon (colima) can see.
compose_dir=""

# Run docker compose on the example's compose file with this run's override and project name.
compose() {
  docker compose --project-name "$run_id" --file examples/minimal_bot/compose.yaml \
    --file "$compose_dir/override.yaml" "$@"
}

cleanup() {
  if [[ -n "$compose_dir" ]]; then
    compose down --volumes --rmi local --remove-orphans >/dev/null 2>&1 || true
    rm -rf "$compose_dir"
  fi
  docker ps --all --quiet --filter "label=$label" | while read -r container; do
    docker rm --force --volumes "$container" >/dev/null
  done
  docker volume rm --force "$secrets_volume" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() {
  echo "docker_smoke: $*" >&2
  exit 1
}

# Print a container's state field, for example .State.Status or .State.ExitCode.
state() {
  docker inspect --format "{{$2}}" "$1"
}

# Wait at most $2 seconds for container $1 to become healthy; fail at once when it exits instead.
wait_healthy() {
  local waited=0
  while [[ "$(state "$1" .State.Health.Status)" != "healthy" ]]; do
    if [[ "$(state "$1" .State.Status)" == "exited" ]]; then
      docker logs "$1" >&2 || true
      fail "container exited $(state "$1" .State.ExitCode) before it became healthy"
    fi
    if (( waited >= $2 )); then
      docker logs "$1" >&2 || true
      fail "container was not healthy within $2 s"
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

# Wait at most $2 seconds for container $1 to stop running.
wait_exited() {
  local waited=0
  while [[ "$(state "$1" .State.Status)" != "exited" ]]; do
    if (( waited >= $2 )); then
      fail "container did not exit within $2 s"
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

# Run a fixed command as root against the secrets volume, using the image's own shell.
secrets() {
  docker run --rm --label "$label" --user 0 --entrypoint sh \
    --mount "type=volume,source=$secrets_volume,target=/secrets" "$image" -c "$1"
}

if [[ "$mode" == "compose" ]]; then
  compose_dir="$(mktemp -d "$PWD/examples/minimal_bot/secrets/.smoke.XXXXXX")"
  chmod 0755 "$compose_dir"
  printf placeholder > "$compose_dir/openai_api_key"
  chmod 0444 "$compose_dir/openai_api_key"
  # The example's env_file is the operator's .env; the smoke replaces it with the closed local endpoint.
  printf '%s\n' \
    'services:' \
    '  minimal:' \
    '    env_file: !reset []' \
    '    environment:' \
    '      OPENAI_BASE_URL: http://127.0.0.1:9/v1' \
    '      OPENAI_MODEL: smoke' \
    'secrets:' \
    '  openai_api_key:' \
    "    file: $compose_dir/openai_api_key" > "$compose_dir/override.yaml"

  echo "docker_smoke: compose up becomes healthy" >&2
  compose config --quiet
  compose up --build --detach --wait --wait-timeout 60
  bot="$(compose ps --all --quiet minimal)"
  [[ -n "$bot" ]] || fail "compose started no minimal container"

  echo "docker_smoke: compose stop exits 0 within 10 s" >&2
  started="$(date +%s)"
  compose stop
  elapsed=$(($(date +%s) - started))
  (( elapsed < 10 )) || fail "compose stop took $elapsed s"
  code="$(state "$bot" .State.ExitCode)"
  [[ "$code" == "0" ]] || fail "compose stop exited $code, not 0"

  echo "docker_smoke: ok" >&2
  exit 0
fi

echo "docker_smoke: build $image" >&2
docker build --build-arg PROJECT=examples/minimal_bot --file docker/Dockerfile --tag "$image" .

docker volume create --label "$label" "$secrets_volume" >/dev/null
secrets 'printf placeholder > /secrets/openai_api_key && chmod 0444 /secrets/openai_api_key'

echo "docker_smoke: hardened run becomes healthy" >&2
bot="$(docker run --detach --label "$label" --init --read-only \
  --tmpfs /run/mosfet:uid=10001,gid=10001,mode=0700,size=64k --tmpfs /tmp:size=16m \
  --mount type=volume,target=/var/lib/mosfet "${secrets_mount[@]}" "${endpoint[@]}" \
  --cap-drop ALL --security-opt no-new-privileges:true "$image")"
# Importing mosfet and the provider before the bot can be ready is not under test; on a loaded host it takes
# over 20 s, so the bound matches the example's own start bound.
wait_healthy "$bot" 60

uid="$(docker exec "$bot" id -u)"
[[ "$uid" == "10001" ]] || fail "the bot runs as uid $uid, not 10001"
if docker exec "$bot" touch /smoke-probe 2>/dev/null; then
  fail "the root filesystem is writable"
fi
mode="$(docker exec "$bot" stat -c %a /var/lib/mosfet/data)"
[[ "$mode" == "700" ]] || fail "the data dir has mode $mode, not 700"

echo "docker_smoke: docker stop exits 0 within 10 s" >&2
started="$(date +%s)"
docker stop "$bot" >/dev/null
elapsed=$(($(date +%s) - started))
(( elapsed < 10 )) || fail "docker stop took $elapsed s"
code="$(state "$bot" .State.ExitCode)"
[[ "$code" == "0" ]] || fail "docker stop exited $code, not 0"

echo "docker_smoke: a missing secret exits 1, a bad flag 64" >&2
code=0
docker run --rm --label "$label" --init "${endpoint[@]}" "$image" >/dev/null 2>&1 || code=$?
[[ "$code" == "1" ]] || fail "a run without its secret exited $code, not 1"
code=0
docker run --rm --label "$label" --init "$image" minimal_bot_example --no-such-flag >/dev/null 2>&1 || code=$?
[[ "$code" == "64" ]] || fail "a run with a bad flag exited $code, not 64"

# No tmpfs here: the ready file lives in the container's writable layer, which a restart keeps, so the boot's
# own removal of a stale ready file is what is under test.
echo "docker_smoke: a ready file left by a SIGKILLed run is removed at the next boot" >&2
bot="$(docker run --detach --label "$label" --init "${secrets_mount[@]}" "${endpoint[@]}" "$image")"
waited=0
until docker exec "$bot" test -f /run/mosfet/ready; do
  (( waited < 60 )) || fail "the bot did not become ready within 60 s"
  sleep 1
  waited=$((waited + 1))
done
docker kill --signal KILL "$bot" >/dev/null
wait_exited "$bot" 30
docker cp "$bot:/run/mosfet/ready" - >/dev/null || fail "the SIGKILLed run left no ready file to test with"
# With its memory database replaced by a file that is not one, the next run boots, but its bot never activates:
# it spends its restart budget (70) and never becomes ready, so any ready file left once it exits is the stale one.
corrupt="$(mktemp)"
printf 'not a database' > "$corrupt"
docker cp "$corrupt" "$bot:/var/lib/mosfet/data/memory.sqlite3"
rm -f "$corrupt"
docker start "$bot" >/dev/null
wait_exited "$bot" 90
code="$(state "$bot" .State.ExitCode)"
[[ "$code" == "70" ]] || fail "the restarted run exited $code, not 70"
if docker cp "$bot:/run/mosfet/ready" - >/dev/null 2>&1; then
  fail "the stale ready file survived the next boot"
fi

echo "docker_smoke: ok" >&2
