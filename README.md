# UniMate Rust inference port

This crate is the start of an inference-only Burn port of UniMate's released
`graph` + `adaln` flow model. It does not train or alter weights. The initial
runtime target is the resolved UniMate config and EMA weights, with prompt and
joint-name embeddings supplied as inputs.

## Current status

- Burn 0.20.1 is pinned for the ndarray CPU and WGPU (Metal on macOS) backends.
- The CLI parses and validates a resolved UniMate JSON config.
- The flow sampler contains Burn tensor operations for classifier-free
  guidance and an explicit Euler update.
- Burn's PyTorch reader inventories named tensors in the checkpoint without
  moving them to a device.
- Rust can read converted normalization JSON and select the root/local stats
  for each joint.
- The denoiser, EMA model loader, and reference fixtures are not implemented
  yet. No model weights are tracked in Git.

## Run

```sh
cargo run -- path/to/config.json
cargo run -- inspect-model path/to/checkpoint.pt path/to/model_manifest.json
```

Use the `config.json` written beside the checkpoint, since UniMate resolves
some architecture dimensions (including maximum joints and depth) at run time.
The model currently rejects other attention/conditioning variants and configs
that enable the optional GCN graph embedding.

## Required inference assets

Run `scripts/fetch_unimate_assets.sh` to download only the pinned current
release's checkpoint, resolved `config.json`, and `dataset_stats.npy`. It
places them in the ignored `weights/` directory and verifies file hashes. It
does not download the dataset, training logs, or sample renders. Inference must
select EMA weights from the checkpoint. For the initial Rust implementation,
provide precomputed T5 caption and joint-name embeddings so text tokenization
and FLAN-T5 are outside the denoiser port.

The released `.pt` checkpoint stores EMA values as an ordered `shadow_params`
list, without parameter names. Burn's reader cannot map that list directly.
Run `scripts/export_ema_state_dict.py` in a Python environment with the
upstream model dependencies (`torch`, `torch-geometric`, `einops`, and `numpy`)
to pair each unchanged EMA tensor with the corresponding upstream parameter
name and convert the pickled NumPy normalization stats to JSON. This is a
format conversion only; it does not train the model. Then inspect the named
tensors with:

```sh
python3 scripts/export_ema_state_dict.py
cargo run --locked -- inspect-weights \
  weights/unimate_uniml3d_f60_v3/ema_named.pt \
  weights/unimate_uniml3d_f60_v3/ema_manifest.json
```

The raw model state dictionary has been inventoried. Before motion generation
is implemented, the remaining correctness prerequisite is a PyTorch reference
output for one denoiser call. That fixture will anchor the Burn module parity
checks.

The downloaded model is released under CC-BY-NC-4.0. The upstream source code
in `reference/UniMate` is MIT licensed; see its included `LICENSE` file.
