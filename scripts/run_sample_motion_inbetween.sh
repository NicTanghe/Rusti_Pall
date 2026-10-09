#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$repo_root/target/release/animate_usd" inbetween "$@" --keep_frames "${KEEP_FRAMES:-0,-1}"
