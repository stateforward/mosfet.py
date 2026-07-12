#!/usr/bin/env bash
# Thin wrapper — prefer: uv run phone-bot
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run phone-bot "$@"
