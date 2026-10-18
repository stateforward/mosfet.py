#!/usr/bin/env bash
# Thin wrapper — prefer: uv run python -m phone_bot_example
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run python -m phone_bot_example "$@"
