# UniMate Rust inference roadmap

## Current state

The frozen UniMate graph/adaLN denoiser and flow sampler are implemented in
Burn. On macOS, the default WGPU runtime selects the Metal adapter.

- The released EMA weights load strictly into the Burn model.
- A full 71-joint × 60-frame denoiser forward was compared against PyTorch on
  Metal. Final output max absolute error was `3.98e-5`; RMSE was `6.08e-6`.
- Full-size Metal sampling completed in 56 ODE evaluations and wrote finite
  output with shape `[1, 71, 12, 60]`.
- Reduced-shape complete samples were compared with the upstream
  `torchdiffeq` sampler at guidance scales 1 and 3.
- The `sample` CLI accepts precomputed conditioning tensors, runs inference,
  writes float32 NumPy output, and can optionally inverse-normalize it with
  UniMate's dataset statistics.

These results validate the inference computation, but the exercised
conditioning was synthetic fixture data. The project does not yet turn an
ordinary rig and prompt into useful motion.

## Remaining work

### 1. Prepare rig conditioning

Build a tool that takes a supported, canonicalized rig and produces the rig
inputs expected by the model:

- joint order, parent indices, joint count, and masks;
- hierarchy depths and graph-distance/relation matrices;
- sign-pinned spectral coordinates;
- normalized rest-pose features and parent features;
- joint-name embeddings.

The first version can keep Blender, FBX handling, and any existing UniMate rig
preprocessing in Python. Its output should use the conditioning tensor bundle
already consumed by the Rust CLI. Add shape/range checks so malformed rigs fail
with a useful error before inference.

### 2. Encode prompts and joint names

The denoiser consumes FLAN-T5 embeddings, not strings. Initially, use the
upstream Python text encoder to create caption and joint-name embeddings and
cache them with the prepared rig. Training or fine-tuning is not required.
Later, decide whether text encoding belongs in Rust or remains an offline
preparation step.

### 3. Validate a real rig and prompt

Create one conditioning bundle from a real rig and prompt, run it through the
Metal `sample` command, and check finite values, valid joint masks, expected
root motion, and plausible rotations. Compare a complete real-input sample
against upstream Python where practical. Full-shape ODE parity against
PyTorch has not yet been measured.

### 4. Decode motion and export animation

The model outputs 12 features per joint and frame. Inverse normalization is
available, but animation export still needs to:

- interpret the feature layout (root/RIFKE position, 6D rotation, local
  velocity);
- reconstruct root trajectory and joint transforms using the rig hierarchy;
- handle coordinate-system conventions and rest-pose offsets;
- write an animated glTF/GLB, with FBX support left for a later conversion
  step if needed.

Validate bone lengths, quaternion/rotation validity, root displacement, and
playback in a viewer before considering animation export complete.

### 5. Package and optimize

Once real-input output is correct, document the asset bundle format, make the
Metal inference command straightforward to run, benchmark representative
rigs, and consider performance changes. Preserve the validated FP32 path as
the reference when evaluating lower precision or batching.

## Immediate next milestone

Implement the rig/prompt conditioning-preparation tool and document the exact
bundle format. That bridges the already-working Burn sampler to actual user
inputs; animation reconstruction follows after a real sample is validated.
