use super::*;
use rusty_pall::motion_tasks::{self as mt, Motion, Replacement};
use serde_json::json;
use std::collections::BTreeMap;

pub const HELP: &str = "Native motion tasks (reference = a saved animate_usd run directory):
  animate_usd inbetween <reference-run> <new-output-root> <prompt> [--keep_frames 0,-1]
  animate_usd edit <reference-run> <new-output-root> <prompt> --keep_joints Neck,Head
  animate_usd expand <rig-run> <new-output-root> <prompts.json> [--expand_overlap 10]
Shared: --seed 10 --cfg_scale 3 --steps 50
Reference window: --gt_start_frame 0 (inbetween/edit only)
Outputs: <new-output-root>/inbetween, motion_edit, or motion_expand.
Reference motion is denormalized motion.json in the saved run's exact joint order.
An existing generated run is a reference, not an original ground-truth clip.
Expansion prompts.json is a JSON array of strings. Each model window remains 60 frames.
Constraints hold in UniMate feature space; selected joints are not world-space IK locks.";

pub fn run(a: &[String]) -> Result<()> {
    ensure!(a.len() >= 4, "{HELP}");
    let mode = a[0].as_str();
    let mut options = BTreeMap::new();
    for pair in a[4..].chunks(2) {
        ensure!(pair.len() == 2, "Option needs a value: {}", pair[0]);
        let key = pair[0].replace('-', "_");
        let key = key.trim_start_matches('_').to_owned();
        ensure!(
            matches!(
                key.as_str(),
                "seed"
                    | "cfg_scale"
                    | "steps"
                    | "gt_start_frame"
                    | "keep_frames"
                    | "keep_joints"
                    | "expand_overlap"
            ),
            "Unknown option {}",
            pair[0]
        );
        ensure!(
            options.insert(key, pair[1].clone()).is_none(),
            "Duplicate option {}",
            pair[0]
        );
    }
    let value =
        |key: &str, default: &str| options.get(key).cloned().unwrap_or_else(|| default.into());
    let seed: u64 = value("seed", "10").parse()?;
    let cfg_scale: f32 = value("cfg_scale", "3").parse()?;
    let steps: usize = value("steps", "50").parse()?;
    ensure!(
        cfg_scale.is_finite() && cfg_scale > 1.,
        "These modes require --cfg_scale > 1"
    );
    ensure!(steps > 0, "--steps must be positive");
    for key in options.keys() {
        let allowed = matches!(key.as_str(), "seed" | "cfg_scale" | "steps")
            || match mode {
                "inbetween" => matches!(key.as_str(), "gt_start_frame" | "keep_frames"),
                "edit" => matches!(key.as_str(), "gt_start_frame" | "keep_joints"),
                "expand" => key == "expand_overlap",
                _ => false,
            };
        ensure!(allowed, "--{key} is not valid for {mode}");
    }
    let source = Path::new(&a[1]);
    let mut p: Prepared = serde_json::from_slice(&fs::read(source.join("conditioning.json"))?)?;
    let cfg = config()?;
    let stats = stats()?;
    p.frames = cfg.dataset.max_motion_length;
    let frames = p.frames;
    let width = p.width;
    let mut reference = None;
    let mut selected = Vec::new();
    let start: usize = value("gt_start_frame", "0").parse()?;
    let mut valid = frames;
    let overlap: usize = value("expand_overlap", "10").parse()?;
    let prompts: Vec<String> = if mode == "expand" {
        ensure!(
            overlap > 0 && overlap < frames,
            "Overlap must be in 1..{frames}"
        );
        serde_json::from_slice(&fs::read(&a[3])?)?
    } else {
        vec![a[3].clone()]
    };
    ensure!(
        !prompts.is_empty() && prompts.iter().all(|p| !p.trim().is_empty()),
        "Prompts must not be empty"
    );
    seed.checked_add(prompts.len() as u64 - 1)
        .context("Segment seeds overflow u64")?;
    let folder = match mode {
        "inbetween" => "inbetween",
        "edit" => "motion_edit",
        "expand" => "motion_expand",
        _ => anyhow::bail!("{HELP}"),
    };
    if mode != "expand" {
        let motion: Motion = serde_json::from_slice(&fs::read(source.join("motion.json"))?)?;
        motion.validate(width)?;
        ensure!(
            start < motion.shape[3],
            "Reference window starts beyond the clip"
        );
        valid = frames.min(motion.shape[3] - start);
        let mut known = mt::window(&motion.values, width, motion.shape[3], start, frames)?;
        let q = mt::first_facing(&known, frames)?;
        mt::rotate_facing(&mut known, &p, frames, q, &stats)?;
        let keep = if mode == "inbetween" {
            selected = mt::keep_frames(&value("keep_frames", "0,-1"), frames)?;
            mt::frame_mask(width, p.joints.len(), frames, &selected)
        } else {
            selected = mt::keep_joints(&value("keep_joints", ""), &p)?;
            mt::joint_mask(width, frames, &selected)
        };
        reference = Some(Replacement {
            known,
            keep,
            steps,
            valid_frames: if mode == "edit" { valid } else { frames },
        });
        reference.as_ref().unwrap().validate(width, frames)?;
    }
    let output = Path::new(&a[2]).join(folder);
    let out = output.to_str().context("Output path must be UTF-8")?;
    ensure!(
        !output.exists(),
        "Output exists: {out}; choose a new output root"
    );
    // Rebuild/check the rig against this checkpoint before any GPU work.
    prepare_run(
        &p.input,
        source
            .join("labels.json")
            .to_str()
            .context("Invalid label path")?,
        out,
        &prompts[0],
    )?;
    let fresh: Prepared = serde_json::from_slice(&fs::read(output.join("conditioning.json"))?)?;
    ensure!(
        fresh.joints == p.joints && fresh.parents == p.parents && fresh.width == p.width,
        "Reference rig/config differs from prepared rig"
    );
    p.input = fresh.input;
    let rig = Rig::open(Path::new(&p.input))?;
    write_json(
        output.join("constraint.json"),
        &json!({"mode":mode,"reference_run":source.canonicalize()?,"prompts":prompts,"seed":seed,"cfg_scale":cfg_scale,"steps":steps,"gt_start_frame":start,"valid_reference_frames":valid,"resolved_indices":selected,"expand_overlap":if mode=="expand" {Some(overlap)} else {None},"constraint_space":"denormalized UniMate features","reference_kind":"saved motion.json; not necessarily ground truth"}),
    )?;
    if let Some(r) = &reference {
        write_json(
            output.join("reference-motion.json"),
            &Motion {
                shape: [1, width, 12, valid],
                values: mt::window(&r.known, width, frames, 0, valid)?,
            },
        )?;
    }
    let mut chain = Vec::new();
    let mut chain_frames = 0;
    let mut previous = Vec::new();
    let mut seam_errors = Vec::new();
    for (segment, prompt) in prompts.iter().enumerate() {
        eprintln!(
            "{mode}: segment {}/{}, prompt: {prompt}",
            segment + 1,
            prompts.len()
        );
        if p.caption.is_empty() || p.prompt != *prompt || p.joint_embeddings.is_empty() {
            p.prompt = prompt.clone();
            p.embed(&text_dir())?;
        }
        let mut q = None;
        let expansion_constraint;
        let constraint = if mode == "expand" && segment > 0 {
            let mut known = mt::window(&previous, width, frames, frames - overlap, frames)?;
            let facing = mt::first_facing(&known, frames)?;
            mt::rotate_facing(&mut known, &p, frames, facing, &stats)?;
            q = Some(facing);
            expansion_constraint = Replacement {
                known,
                keep: mt::frame_mask(
                    width,
                    p.joints.len(),
                    frames,
                    &(0..overlap).collect::<Vec<_>>(),
                ),
                steps,
                valid_frames: frames,
            };
            Some(&expansion_constraint)
        } else {
            reference.as_ref()
        };
        // Keep per-segment conditioning for reproduction and inspection.
        write_json(
            output.join(format!("segment-{segment:03}.conditioning.json")),
            &p,
        )?;
        let mut values = p.sample_constrained(
            &cfg,
            &model_dir().join("ema_named.pt"),
            &stats,
            seed + segment as u64,
            cfg_scale,
            constraint,
        )?;
        if let Some(r) = constraint {
            let max_error = values
                .iter()
                .zip(&r.known)
                .zip(&r.keep)
                .filter(|(_, keep)| **keep)
                .map(|((a, b), _)| (a - b).abs())
                .fold(0f32, f32::max);
            ensure!(
                max_error == 0.,
                "Replacement constraint changed at sampler output"
            );
        }
        if let Some(facing) = q {
            mt::rotate_facing(&mut values, &p, frames, facing.transpose(), &stats)?;
            let mut error = 0f32;
            for row in 0..p.joints.len() * 12 {
                for t in 0..overlap {
                    let expected = previous[row * frames + frames - overlap + t];
                    error = error.max((values[row * frames + t] - expected).abs());
                    // Facing conversion projects 6D axes onto rotations. Restore
                    // the original feature values exactly at the shared seam.
                    values[row * frames + t] = expected;
                }
            }
            seam_errors.push(error);
        }
        write_json(
            output.join(format!("segment-{segment:03}.motion.json")),
            &Motion {
                shape: [1, width, 12, frames],
                values: values.clone(),
            },
        )?;
        if segment == 0 {
            chain = values.clone();
            chain_frames = frames;
        } else {
            chain = mt::stitch(&chain, chain_frames, &values, width, frames, overlap)?;
            chain_frames += frames - overlap;
        }
        previous = values;
    }
    if mode == "edit" && valid < frames {
        chain = mt::window(&chain, width, frames, 0, valid)?;
        chain_frames = valid;
    }
    p.frames = chain_frames;
    p.prompt = prompts.join(" → ");
    write_json(output.join("conditioning.json"), &p)?;
    write_json(
        output.join("motion.json"),
        &Motion {
            shape: [1, width, 12, chain_frames],
            values: chain.clone(),
        },
    )?;
    p.export(&rig, &chain, &output.join("package/animation.usda"))?;
    package(out, false)?;
    write_json(
        output.join("constraint-validation.json"),
        &json!({"backend":rusty_pall::compute::selected()?.name(),"exact_feature_constraints":true,"seam_feature_delta_before_restore":seam_errors,"frames":chain_frames,"fps":30}),
    )?;
    eprintln!(
        "Saved {} ({chain_frames} frames at 30 fps)",
        output.join("animation.usdz").display()
    );
    Ok(())
}
