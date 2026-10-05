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
- The denoiser, checkpoint reader, normalization loader, and reference fixtures
  are not implemented yet. No model weights are in this repository.

## Run

```sh
cargo run -- path/to/config.json
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

Before motion generation is implemented, the next prerequisites are a tensor
name/shape inventory for the released EMA checkpoint and PyTorch reference
outputs for one denoiser call. Those will define the weight mapping and the
parity checks for each Burn module.
