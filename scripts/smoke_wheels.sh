#!/usr/bin/env bash
# Install every built wheel with the local core wheel into its own clean virtual environment and import its public
# package with the repository and user site removed from the import path.
#
# Usage: scripts/smoke_wheels.sh [--map-only] [DIST_DIR]
#
#   DIST_DIR    directory holding the built wheels (default: dist)
#   --map-only  print "filename<TAB>import_name" for every wheel and install nothing
set -euo pipefail
shopt -s nullglob

map_only=false
dist_dir=""
for argument in "$@"; do
  case "$argument" in
    --map-only) map_only=true ;;
    -*)
      echo "Unknown option: $argument" >&2
      exit 2
      ;;
    *)
      if [[ -n "$dist_dir" ]]; then
        echo "Expected at most one wheel directory, got: $dist_dir $argument" >&2
        exit 2
      fi
      dist_dir="$argument"
      ;;
  esac
done
dist_dir="${dist_dir:-dist}"

wheels=("$dist_dir"/*.whl)
if (( ${#wheels[@]} == 0 )); then
  echo "No wheels were built in $dist_dir." >&2
  exit 1
fi

core_wheels=("$dist_dir"/stateforward_mosfet-*.whl)
if (( ${#core_wheels[@]} != 1 )); then
  echo "Expected exactly one stateforward.mosfet wheel, found ${#core_wheels[@]}." >&2
  exit 1
fi
core_wheel="${core_wheels[0]}"

if [[ "$map_only" == false ]]; then
  smoke_root="$(mktemp -d)"
  trap 'rm -rf "$smoke_root"' EXIT
fi

for wheel in "${wheels[@]}"; do
  filename="$(basename "$wheel")"
  case "$filename" in
    stateforward_mosfet-*.whl) import_name="mosfet" ;;
    mosfet_provider_*-*.whl)
      name="${filename#mosfet_provider_}"
      import_name="mosfet.providers.${name%%-*}"
      ;;
    *)
      echo "No import smoke mapping for built wheel: $filename" >&2
      exit 1
      ;;
  esac

  if [[ "$map_only" == true ]]; then
    printf '%s\t%s\n' "$filename" "$import_name"
    continue
  fi

  environment="$smoke_root/${filename%.whl}"
  uv venv --python "$(command -v python)" "$environment"
  python_bin="$environment/bin/python"
  if [[ "$wheel" == "$core_wheel" ]]; then
    uv pip install --python "$python_bin" "$wheel"
  else
    uv pip install --python "$python_bin" "$core_wheel" "$wheel"
  fi
  (
    cd "$environment"
    PYTHONNOUSERSITE=1 PYTHONPATH='' SMOKE_IMPORT="$import_name" \
      "$python_bin" -c 'import importlib, os; importlib.import_module(os.environ["SMOKE_IMPORT"])'
  )
done
