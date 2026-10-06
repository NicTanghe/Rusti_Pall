# UniMate Rust inference port

This crate is the start of an inference-only Burn port of UniMate's released
`graph` + `adaln` flow model. It does not train or alter weights. The initial
runtime target is the resolved UniMate config and EMA weights, with prompt and
joint-name embeddings supplied as inputs.

## Current status

- Burn 0.20.1 is pinned for the ndarray CPU and WGPU (Metal on macOS) backends.
- The CLI parses and validates a resolved UniMate JSON config.
- Flow sampling supports adaptive Dormand-Prince 5(4) integration with the
  upstream `atol=1e-6`, `rtol=1e-3`, 50 requested points, and sequential CFG.
  Fixed-step Euler remains available for preview and editing workflows.
- Burn's PyTorch reader inventories named tensors in the checkpoint without
  moving them to a device.
- Rust can read converted normalization JSON and select the root/local stats
  for each joint.
- `src/model.rs` implements checkpoint-shaped RMSNorm, SwiGLU feed-forward, the
  graph distance/relation bias, SignNet spectral RoPE, temporal RoPE, and
  spatial graph attention with spectral rotations wired into Q/K. It also has
  temporal self-attention, the graph/adaLN spatiotemporal block, flow-time
  embedding, root/joint input embedding, t-pose conditioning pool, depth/name
  token embeddings, final root/joint output layer, and the assembled denoiser
  forward pass. The spectral joint RoPE encoder is shared across blocks, as in
  the upstream architecture.
- The strict EMA loader maps upstream PyTorch parameter paths into Burn module
  fields and refuses missing or unused tensors. Its name mapping was checked
  against the upstream module paths and checked-in 443-tensor model manifest;
  the converted 363-tensor EMA loads completely. A deterministic 8-joint,
  4-frame and a 16-joint, 8-frame PyTorch fixture compare through all ten
  blocks; their output max absolute errors are `3.0e-5` and `2.6e-5`, with
  RMSEs `7.5e-6` and `5.1e-6`. Full-shape and ODE parity are still outstanding.
  No model weights are tracked in Git.

## Run

```sh
cargo run -- path/to/config.json
cargo run -- inspect-model path/to/checkpoint.pt path/to/model_manifest.json
cargo run -- check-weights path/to/config.json path/to/ema_named.pt
cargo run -- inspect-burn-weights path/to/ema_named.pt
cargo run -- compare-forward path/to/config.json path/to/ema_named.pt path/to/fixture.pt
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
upstream model dependencies to pair each unchanged EMA tensor with its
parameter name and convert the NumPy normalization stats to JSON. On macOS,
the validated setup is:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements-export.txt
.venv/bin/python scripts/export_ema_state_dict.py
```

This is a format conversion only; it does not train the model. Inspect both
the upstream and Burn-remapped tensor names with:

```sh
cargo run --locked -- inspect-weights \
  weights/unimate_uniml3d_f60_v3/ema_named.pt \
  weights/unimate_uniml3d_f60_v3/ema_manifest.json
cargo run --locked -- inspect-burn-weights \
  weights/unimate_uniml3d_f60_v3/ema_named.pt
cargo run --locked -- check-weights \
  weights/unimate_uniml3d_f60_v3/config.json \
  weights/unimate_uniml3d_f60_v3/ema_named.pt
```

Generate the small deterministic forward fixture and compare it against Burn:

```sh
.venv/bin/python scripts/make_reference_fixture.py --joints 8 --frames 4
cargo run --locked -- compare-forward \
  weights/unimate_uniml3d_f60_v3/forward_fixture.config.json \
  weights/unimate_uniml3d_f60_v3/ema_named.pt \
  weights/unimate_uniml3d_f60_v3/forward_fixture.pt
```

The fixture records activations before/after token embedding, after every
transformer block, and after the final layer. `compare-forward-wgpu` runs the
same comparison on Burn WGPU/Metal when a GPU adapter is available. This
environment did not expose a Metal adapter, and its full-size CPU forward was
too slow for a practical comparison.

`sampler::standard_normal_noise` creates reproducible standard-normal initial
noise on the selected Burn device, and `UniMateDenoiser::sample_dopri5` accepts
that noise plus precomputed conditioning tensors from an asset-preparation
step. Burn seeding is backend-wide for the selected device; pass a saved noise
tensor when comparing backends. A user-facing rig/prompt preparation and
motion-file output path is not implemented yet. The Rust adaptive controller
follows the Dormand-Prince 5(4) tableau and tolerances, but is not guaranteed to
take the same internal steps as torchdiffeq. End-to-end ODE parity remains
unverified.

The downloaded model is released under CC-BY-NC-4.0. The upstream source code
in `reference/UniMate` is MIT licensed; see its included `LICENSE` file.
