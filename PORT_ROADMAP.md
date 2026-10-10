# UniMate Rust inference roadmap

## Current state

The frozen UniMate graph/adaLN denoiser and flow sampler are implemented in
Burn, with native USDZ rig preparation and animation export in `animate_usd`.

- Released EMA weights load strictly into the Burn model.
- Both CLIs prefer direct CUDA after a device test, with WGPU fallback and
  an explicit backend override. CUDA is enabled by default; HIP is not added.
- The full CUDA forward pass on the RTX 2070 matched the PyTorch fixture
  with maximum absolute output error `4.53e-5` and RMSE `6.36e-6`. A CUDA
  constrained-sampling smoke test preserved pinned features exactly and
  produced a valid USDZ. Backend preference/fallback unit tests pass.
- Full 71-joint × 60-frame denoiser forward comparisons passed against
  PyTorch on Metal and on an RTX 2070 using WGPU/Vulkan. RTX 2070 maximum
  absolute error was `5.25e-5`, with RMSE `6.97e-6`.
- Full-size synthetic sampling produced finite output. Reduced-shape complete
  samples were compared with upstream `torchdiffeq` at CFG scales 1 and 3.
- `animate_usd` uses the pure Rust `mxpv/openusd` crates to read USDZ skeletons,
  preserve the mesh and materials, and write skeletal animation.
- Native conditioning includes canonical rest positions, graph relations,
  distances, depths and normalized-Laplacian spectral coordinates.
- Candle FLAN-T5 encodes prompts and semantic joint labels on the CPU, with
  embeddings cached in the run's conditioning JSON. No Python runtime is
  used by `animate_usd`.
- Motion decoding reconstructs root trajectory and local joint rotations.
  Export checks finite transforms, unit quaternions and bone lengths by
  reopening the authored USD animation.
- Megalania's 63-joint USDZ passes a rest-pose export/reimport check, including
  its nested coordinate and scale transforms. Its canonical skeleton preview
  has been inspected.
- A full Megalania run with `A quadruped walks forward.`, seed 10 and CFG 3
  completed on the RTX 2070 in approximately 722 seconds, with 104 ODE
  evaluations. The 60-frame USD export and packaged USDZ passed transform
  validation; visual playback remains unverified.
- Five native rig tests cover hierarchy validation, graph features, rotation
  decoding, root velocity conventions and transformed rest-pose round trips.

## Remaining validation

Native `animate_usd inbetween`, `edit`, and `expand` now implement the
replacement sampler and multi-prompt stitching. CPU tests cover per-step
pinning, selectors, short clips, facing transforms and seam layout. See the
README for commands and shell wrappers. Reference inputs currently come from
saved native motion JSON; upstream dataset case lookup and original USD/FBX
clip-to-feature conversion remain to be implemented. Full default-step visual
validation and comparison against upstream constrained sampling remain open.
All three native modes passed RTX 2070 smoke tests (two steps for constrained
segments). The two-segment expansion produced 110 frames with exact overlap
features and stitch layout. All three exports opened with the reference USD
parser and imported into Blender with 63 bones and animation actions.

1. Review generated Megalania animation in a viewer with USD skeletal
   animation support. Numeric invariants alone do not verify natural motion,
   deformation quality, ground contact or the effect of semantic labels.
2. Compare native text embeddings, rig conditioning and decoded animation
   against upstream results for the same real rig. Full-shape ODE parity
   against PyTorch is still unmeasured.
3. Exercise additional rigs and reject unsupported asset conventions with
   clear diagnostics. Current support is one skeleton, 4–71 topologically
   ordered joints, rigid local rest transforms, static ancestor transforms,
   Y/Z up and a self-contained USDZ archive. Labels are supplied explicitly.
4. Add support for further import/export formats only as needed. FBX and
   glTF/GLB are not currently native runner formats.
5. Benchmark representative prompts and rigs before changing precision or
   inference scheduling. Preserve FP32 as the validated reference path.

See README.md for native build, preparation, sampling and packaging commands.
