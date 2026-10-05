#!/usr/bin/env bash
set -euo pipefail

# Download only the frozen inference assets used by this Rust port.
# Intentionally excludes checkpoints from other runs, training logs, samples,
# and the UniML3D dataset.

readonly model_revision="66d5240b4c4aded70040a2d67df4af261084b6e7"
readonly model_name="unimate_uniml3d_f60_v3"
readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly model_dir="${repo_root}/weights/${model_name}"
readonly hub_base="https://huggingface.co/Linzhan/UniMate/resolve/${model_revision}/${model_name}"

download_asset() {
    local relative_path="$1"
    local expected_sha256="$2"
    local destination="${model_dir}/${relative_path}"
    local partial="${destination}.part"

    mkdir -p "$(dirname "${destination}")"

    if [[ -f "${destination}" ]] && printf '%s  %s\n' "${expected_sha256}" "${destination}" | shasum -a 256 --check --status; then
        printf 'Already present and verified: %s\n' "${relative_path}"
        return
    fi

    : >> "${partial}"
    printf 'Downloading %s\n' "${relative_path}"
    curl --fail --location --retry 3 --retry-all-errors --continue-at - \
        --output "${partial}" "${hub_base}/${relative_path}"

    printf '%s  %s\n' "${expected_sha256}" "${partial}" | shasum -a 256 --check
    mv "${partial}" "${destination}"
}

download_asset \
    "checkpoints/checkpoint_step_150000.pt" \
    "f991ddd87262c251f4681d18297a6163ab4f139daa28c86d5ef0a66771df7bc6"

# The config and NumPy stats are small and are tied to the same immutable Hub
# revision. Their SHA-256 values are checked against the downloaded contents.
download_asset \
    "config.json" \
    "1ffbd379d3667de7d701ed6f3418f1c9d1a66f35d2653776255008c07c051aaa"
download_asset "dataset_stats.npy" "0ff2b9ef3ea9a15b024ab0670e55f659478251501e5596f9011d5bca4cf67773"

printf '\nInference assets are ready in %s\n' "${model_dir}"
