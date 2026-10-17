#!/usr/bin/env bash
# Build the builder's bench image (docker/bench/Dockerfile) into a rootless Podman daemon and check what it
# promises the sandbox: docker.Host's own live check passes in full on it (src/providers/docker/scripts/
# live_check.py), and a one-off run under the sandbox's machine flags (src/providers/docker/README.md, "A machine")
# has git and pytest, runs a test and versions a file in the workspace with no network. Manual, not CI: it needs
# a rootless daemon, and the build pulls the pinned base images.
#
# Usage: docker/bench/smoke.sh --socket PATH [--python PATH]
#
#   --socket  the rootless Podman API socket (on macOS: podman machine inspect, ConnectionInfo.PodmanSocket.Path)
#   --python  an interpreter with stateforward-mosfet and mosfet-provider-docker installed (default python3)
#
# Every podman command is argv form and takes only values this script made, so nothing untrusted is
# interpolated. The one-off run carries this run's label and is removed on exit; the live check sweeps its own
# machines. The image and the build cache stay.
set -euo pipefail

usage() {
  echo "Usage: docker/bench/smoke.sh --socket PATH [--python PATH]" >&2
  exit 2
}

socket=""
python="python3"
while (( $# > 0 )); do
  case "$1" in
    --socket) (( $# >= 2 )) || usage; socket="$2"; shift 2 ;;
    --python) (( $# >= 2 )) || usage; python="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$socket" ]] || usage

cd "$(dirname "$0")/../.."

image="localhost/mosfet-bench:1"
label="org.stateforward.mosfet.smoke=mosfet-bench-smoke-$$"

podman_remote() {
  podman --url "unix://$socket" "$@"
}

cleanup() {
  podman_remote ps --all --quiet --filter "label=$label" | while read -r container; do
    podman_remote rm --force --volumes "$container" >/dev/null
  done
}
trap cleanup EXIT

fail() {
  echo "bench_smoke: $*" >&2
  exit 1
}

echo "bench_smoke: build $image" >&2
podman_remote build --file docker/bench/Dockerfile --tag "$image" docker/bench

echo "bench_smoke: docker.Host's live check on $image" >&2
"$python" src/providers/docker/scripts/live_check.py --socket "$socket" --image "$image"

# The flags docker.Host gives every machine, with the builder's spec (BL-3): uid 65532 on a read-only root, every
# capability dropped, no network, 1 CPU, 1 GiB without swap and 256 pids, a sticky tmpfs workspace that is also
# HOME, and only the spec's environment.
sandbox_run=(--rm --label "$label" --init --read-only --user 65532:65532 --network none
  --cap-drop ALL --security-opt no-new-privileges:true
  --cpus 1 --memory 1g --memory-swap 1g --pids-limit 256
  --tmpfs "/workspace:rw,nosuid,nodev,size=512m,mode=1777" --tmpfs "/tmp:rw,nosuid,nodev,size=64m"
  --workdir /workspace --env HOME=/workspace --env PYTHONDONTWRITEBYTECODE=1 --env PIP_NO_INPUT=1
  --entrypoint sh "$image" -c)

echo "bench_smoke: the bench has git and pytest" >&2
versions="$(podman_remote run "${sandbox_run[@]}" 'git --version && python -m pytest --version')" \
  || fail "git --version or python -m pytest --version failed in a sandbox run"
echo "$versions"
[[ "$versions" == *"git version"* ]] || fail "git --version printed no version"
[[ "$versions" == *"pytest "* ]] || fail "python -m pytest --version printed no version"

echo "bench_smoke: a test runs and a file is versioned in the workspace" >&2
worked="$(podman_remote run "${sandbox_run[@]}" '
  printf "def test_bench():\n    assert 1 + 1 == 2\n" > test_bench.py \
  && python -m pytest -q test_bench.py \
  && git init --quiet \
  && git add test_bench.py \
  && git -c user.name=bench -c user.email=bench@localhost commit --quiet --message bench \
  && git log --format=%s')" || fail "a pytest run or a git commit failed in a sandbox run"
echo "$worked"
[[ "$worked" == *"1 passed"* ]] || fail "pytest did not report 1 passed"
[[ "$worked" == *$'\n'"bench" ]] || fail "git log did not show the bench commit"

echo "bench_smoke: ok" >&2
