# rusty_pall — UniMate Rust inference port

This crate is the start of an inference-only Burn port of UniMate's released
`graph` + `adaln` flow model. It does not train or alter weights. The initial
runtime target is the resolved UniMate config and EMA weights, with prompt and
joint-name embeddings supplied as inputs.

See [PORT_ROADMAP.md](PORT_ROADMAP.md) for verified milestones and remaining
real-rig validation work.

## Current status

- The native `animate_usd` runner takes a USDZ rig, semantic joint labels and
  a prompt through Rust text encoding, inference and USD animation export.
- Burn 0.20.1 is pinned for the ndarray CPU and WGPU (Metal on macOS) backends.
- The CLI parses and validates a resolved UniMate JSON config.
- `sample` runs seeded flow inference on WGPU (Metal on macOS) and writes
  normalized float32 NumPy motion output.
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
  the converted 363-tensor EMA loads completely. Deterministic PyTorch
  fixtures have been compared through all ten blocks at 8 joints × 4 frames,
  16 × 8, 32 × 16, and the full 71 × 60 checkpoint shape. The full-size
  WGPU/Metal run completed on the Mac: final output max absolute error
  `3.98e-5`, RMSE `6.08e-6`. The adaptive sampler has also been compared with
  torchdiffeq on an 8-joint × 4-frame fixture at CFG scales 1 and 3; RMSEs were
  `3.58e-4` and `2.22e-3` respectively (the CFG 3 reference output RMS is
  `12.73`). A full-size Metal sample also completed at 71 joints × 60 frames
  in 56 ODE evaluations and produced finite output. Full-shape PyTorch sample
  parity remains to be measured. No model weights are tracked in Git.

## Run

### Native Rust rig animation

`animate_usd` reads USDZ with `mxpv/openusd`, builds rig conditioning, encodes
the prompt and joint labels with Candle FLAN-T5, samples with Burn/WGPU, and
writes USD animation. The runtime does not invoke Python or Blender. It uses
the converted EMA checkpoint from the initial setup described below.

```sh
cargo build --release --locked --bin animate_usd
target/release/animate_usd fetch-text
target/release/animate_usd run \
  /home/dude/Downloads/Megalania.usdz \
  examples/megalania.labels.json \
  outputs/megalania-walk \
  'A quadruped walks forward.'
```

Choose a new output directory for each run. `fetch-text` downloads the pinned
FLAN-T5 assets once (approximately 1 GB). `RUSTI_PALL_MODEL_DIR` and
`RUSTI_PALL_TEXT_DIR` override the model directories. Sampling selects discrete
WGPU adapter 0, which must be accessible to the process.

The output contains `animation.usdz` with the original mesh and texture,
`package/animation.usda`, the raw motion tensor in `motion.json`, saved
conditioning, validation reports, and a labelled `skeleton.svg`. Animation
has 60 frames at 30 fps. Open the USDZ in a viewer supporting USD skeletal
animation to assess the generated motion.

For separate preparation and sampling:

```sh
target/release/animate_usd inspect /home/dude/Downloads/Megalania.usdz
target/release/animate_usd prepare /home/dude/Downloads/Megalania.usdz \
  examples/megalania.labels.json outputs/megalania-test 'A quadruped walks forward.'
target/release/animate_usd sample outputs/megalania-test 10 3
# Only needed for an older export without a package:
target/release/animate_usd package outputs/megalania-test
# Rebuild USD/USDZ from saved motion, replacing exports without GPU sampling:
target/release/animate_usd reexport outputs/megalania-test
```

Preparation checks a saved rest-pose round trip before sampling. Export
reopens every frame and checks finite transforms, unit quaternions and fixed
bone lengths. Supported inputs currently have one skeleton, 4–71 joints in
parent-before-child order, rigid joint rest transforms, static ancestor
transforms, and Y or Z up. Semantic labels and a left/right facing pair are
required; the included annotation is specific to Megalania. This path is
experimental: numerical checks do not establish natural motion, correct
skinning in every viewer, or parity with upstream real-rig preprocessing.

The Megalania test with prompt `A quadruped walks forward.`, seed 10 and CFG 3
completed on the RTX 2070: 63 joints, 60 frames, 104 ODE evaluations (12
accepted and 5 rejected steps), approximately 722 seconds for sampling and
USD export. The USD and packaged USDZ passed transform validation. Playback
quality has not yet been visually verified.

Exports produced before the singleton `apiSchemas` serialization fix may fail
to open in Blender and other standard USD readers. Rebuild the executable and
run `reexport` on the affected run directory; the saved motion is preserved.
The native rig tests also check exported syntax with the independent `usdcat`
parser when that tool is installed.
Both saved Megalania runs were re-exported and imported into Blender 5.2.2:
63 bones, one animation action, 11,355 mesh vertices and the 1024×1024 texture
were present. This verifies import compatibility, not visual motion quality.

### Tensor inference CLI

```sh
cargo run -- path/to/config.json
cargo run -- inspect-model path/to/checkpoint.pt path/to/model_manifest.json
cargo run -- check-weights path/to/config.json path/to/ema_named.pt
cargo run -- inspect-burn-weights path/to/ema_named.pt
cargo run -- compare-forward path/to/config.json path/to/ema_named.pt path/to/fixture.pt
cargo run -- compare-forward-cpu path/to/config.json path/to/ema_named.pt path/to/fixture.pt
cargo run -- sample path/to/config.json path/to/ema_named.pt conditioning.pt motion.npy [seed] [cfg_scale] [stats.json dataset_type]
```

Use the `config.json` written beside the checkpoint, since UniMate resolves
some architecture dimensions (including maximum joints and depth) at run time.
The model currently rejects other attention/conditioning variants and configs
that enable the optional GCN graph embedding.

## Required inference assets

For a complete local setup (Python environment, upstream imports, downloaded
assets, EMA conversion, and Rust weight-loading check), run:

```sh
bash scripts/setup_environment.sh
source .venv/bin/activate
python -c 'from unimate.models.factory import create_model'
```

The Python import is `unimate`. Its pinned source already lives in
`reference/UniMate`; setup registers that directory in this virtual environment
without copying it or modifying the system Python installation. On Linux,
setup installs CPU PyTorch for conversion and reference fixtures. Rust GPU
inference uses WGPU/Vulkan and does not require CUDA PyTorch. This environment
covers the Rust port's helper scripts, not the upstream Blender/training stack.

On a Linux laptop with both Intel graphics and an NVIDIA GPU, select the
discrete GPU explicitly before running inference:

```sh
export CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(0)'
.venv/bin/python scripts/make_reference_fixture.py --joints 8 --frames 4
target/debug/rusty_pall compare-forward \
  weights/unimate_uniml3d_f60_v3/forward_fixture.config.json \
  weights/unimate_uniml3d_f60_v3/ema_named.pt \
  weights/unimate_uniml3d_f60_v3/forward_fixture.pt
```

The NVIDIA driver and Vulkan ICD must be installed, and the process must have
GPU device access. A sandbox may hide the GPU even when `nvidia-smi` works in
your normal terminal. Fixtures use synthetic conditioning; use `animate_usd`
above for the experimental real-rig pipeline.

For regular use, build with `cargo build --release --locked` and use
`target/release/rusty_pall` in place of `target/debug/rusty_pall`.

Validated on Linux with an RTX 2070 (8 GB), NVIDIA driver 615.71.09, and
the discrete WGPU adapter: the full 71-joint × 60-frame forward comparison
had max absolute error `5.25e-5` and RMSE `6.97e-6` against CPU PyTorch.
A debug-build sample with synthetic conditioning, seed 10 and CFG 3 completed
in 56 ODE evaluations (9 accepted steps, none rejected), producing finite
float32 output of shape `[1, 71, 12, 60]`. One-second GPU polling observed
1,165 MiB peak usage across these checks, including the desktop; this is not
a guaranteed memory ceiling. The successful forward run also emitted
`NVVM compilation failed: 3`; its cause has not been diagnosed. This verifies
inference on this GPU, not real-rig animation quality or full-sample parity.

Run `scripts/fetch_unimate_assets.sh` to download only the pinned current
release's checkpoint, resolved `config.json`, and `dataset_stats.npy`. It
places them in the ignored `weights/` directory and verifies file hashes. It
does not download the dataset, training logs, or sample renders. Inference must
select EMA weights from the checkpoint. For the initial Rust implementation,
the tensor CLI accepts precomputed T5 embeddings. The native `animate_usd`
runner computes these embeddings with Candle.

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

# Include the upstream torchdiffeq result to compare complete sampling.
.venv/bin/python scripts/make_reference_fixture.py \
  --joints 8 --frames 4 --include-sample --cfg-scale 3.0 \
  --output weights/unimate_uniml3d_f60_v3/sampling_fixture.pt
cargo run --locked -- compare-sample \
  weights/unimate_uniml3d_f60_v3/sampling_fixture.config.json \
  weights/unimate_uniml3d_f60_v3/ema_named.pt \
  weights/unimate_uniml3d_f60_v3/sampling_fixture.pt 3.0

# Generate a normalized motion tensor using Metal on macOS.
cargo run --locked -- sample \
  weights/unimate_uniml3d_f60_v3/sampling_fixture.config.json \
  weights/unimate_uniml3d_f60_v3/ema_named.pt \
  weights/unimate_uniml3d_f60_v3/sampling_fixture.pt \
  weights/unimate_uniml3d_f60_v3/motion.npy 10 3.0 \
  weights/unimate_uniml3d_f60_v3/dataset_stats.json objaverse
```

The fixture records activations before/after token embedding, after every
transformer block, and after the final layer. `compare-forward` uses Burn WGPU
by default; on macOS, Burn selects its Metal adapter. `compare-forward-metal`
and `compare-forward-wgpu` are equivalent explicit aliases, while
`compare-forward-cpu` selects the ndarray backend for diagnostics. The full
71-joint × 60-frame forward has been verified on Metal. If a sandboxed process
cannot see the adapter, run the command from a normal macOS Terminal session
with GPU access.

`sample` runs the full Burn flow sampler on WGPU (Metal on macOS), creates
seeded initial Gaussian noise, and writes feature values as a float32 NumPy
array with shape `[batch, joints, features, frames]`. Pass both `stats.json`
and a dataset type (`truebones`, `mixamo`, or `objaverse`) to inverse-normalize
the output; without them it remains normalized. Its input
`conditioning.pt` is a PyTorch tensor dictionary containing `caption_embedding`
`[B,768]`, `tpos_first_frame` and `tpos_first_frame_parents` `[B,J,12]`,
`n_joints` and `motion_lengths` `[B]`, `joint_names_emb` `[B,J,768]`,
`joint_depths` `[B,J]`, `graph_dist` and `joint_relations` `[B,J,J]`, and
`spectral_coords` `[B,J,max_freqs]`. Text embeddings and rig features must
already be prepared; `animate_usd` provides a separate native USDZ preparation
and animation export path. Burn seeding is
backend-wide, so pass the same saved noise tensor when comparing backends.
Rust uses the Dormand-Prince 5(4) tableau, torchdiffeq-style initial-step
selection, and the released tolerances. Reduced-shape end-to-end samples are
close; full-shape ODE parity against PyTorch and visual validation of real-rig
output remain outstanding.

The downloaded model is released under CC-BY-NC-4.0. The upstream source code
in `reference/UniMate` is MIT licensed; see its included `LICENSE` file.
