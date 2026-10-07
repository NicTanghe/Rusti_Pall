#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 -m venv .venv
readonly python="$repo_root/.venv/bin/python"
# Python prepares weights on CPU; the Rust runtime uses WGPU/Vulkan on Linux.
if [[ "$(uname -s)" == Linux ]]; then
    "$python" -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cpu
fi
"$python" -m pip install -r scripts/requirements-export.txt

# The pinned upstream snapshot has no Python packaging metadata. Register it
# only in this virtual environment, so imports work from any working directory.
"$python" - <<'PY'
from pathlib import Path
import sysconfig

source = Path("reference/UniMate").resolve()
if not (source / "unimate/models/factory.py").is_file():
    raise SystemExit(f"Missing upstream UniMate source: {source}")
pth = Path(sysconfig.get_path("purelib")) / "unimate_reference.pth"
pth.write_text(str(source) + "\n")
print(f"Registered UniMate source: {source}")
PY
"$python" -c 'from unimate.models.factory import create_model; import unimate; print("UniMate import OK:", list(unimate.__path__))'
"$python" -m pip check
bash scripts/fetch_unimate_assets.sh
"$python" scripts/export_ema_state_dict.py
cargo build --locked
target/debug/rusty_pall check-weights \
    weights/unimate_uniml3d_f60_v3/config.json \
    weights/unimate_uniml3d_f60_v3/ema_named.pt
