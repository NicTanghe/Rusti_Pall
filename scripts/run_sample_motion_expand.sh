#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$repo_root/target/release/animate_usd" expand "$@" --expand_overlap "${EXPAND_OVERLAP:-10}"
