//! Motion constraints and expansion helpers in the denormalized UniMate layout.
use crate::{
    normalization::NormalizationStats,
    rig_motion::{Prepared, rotation6},
};
use anyhow::{Context, Result, ensure};
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Motion {
    pub shape: [usize; 4],
    pub values: Vec<f32>,
}
impl Motion {
    pub fn validate(&self, width: usize) -> Result<()> {
        let [b, j, d, t] = self.shape;
        ensure!(
            b == 1 && j == width && d == 12 && t > 0,
            "Expected motion shape [1,{width},12,T]"
        );
        ensure!(
            j.checked_mul(d).and_then(|v| v.checked_mul(t)) == Some(self.values.len()),
            "Motion data length differs from shape"
        );
        ensure!(
            self.values.iter().all(|v| v.is_finite()),
            "Non-finite reference motion"
        );
        Ok(())
    }
}

pub struct Replacement {
    pub known: Vec<f32>,
    pub keep: Vec<bool>,
    pub steps: usize,
    pub valid_frames: usize,
}
impl Replacement {
    pub fn validate(&self, width: usize, frames: usize) -> Result<()> {
        ensure!(
            self.steps > 0 && self.valid_frames > 0 && self.valid_frames <= frames,
            "Invalid replacement step count or clip length"
        );
        ensure!(
            self.known.len() == width * 12 * frames && self.keep.len() == self.known.len(),
            "Constraint dimensions differ"
        );
        ensure!(
            self.known.iter().all(|v| v.is_finite()),
            "Non-finite known motion"
        );
        ensure!(
            self.keep.iter().any(|v| *v),
            "Constraint selects no features"
        );
        Ok(())
    }
}

pub fn keep_frames(spec: &str, frames: usize) -> Result<Vec<usize>> {
    ensure!(frames > 0, "Empty generation window");
    let mut result = Vec::new();
    for s in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let i: i64 = s
            .parse()
            .with_context(|| format!("Invalid frame index {s}"))?;
        let i = if i < 0 {
            (frames as i64).saturating_add(i)
        } else {
            i
        };
        result.push(i.clamp(0, frames as i64 - 1) as usize);
    }
    result.sort_unstable();
    result.dedup();
    ensure!(!result.is_empty(), "keep_frames is empty");
    Ok(result)
}

pub fn keep_joints(spec: &str, p: &Prepared) -> Result<Vec<usize>> {
    let wanted: Vec<_> = spec
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    ensure!(!wanted.is_empty(), "keep_joints is empty");
    let mut result = Vec::new();
    for name in wanted {
        let mut matched = false;
        for (j, raw) in p.joints.iter().enumerate() {
            if [
                raw.as_str(),
                raw.rsplit('/').next().unwrap(),
                p.labels[j].as_str(),
            ]
            .iter()
            .any(|v| v.to_lowercase() == name)
            {
                result.push(j);
                matched = true;
            }
        }
        ensure!(
            matched,
            "No joint matches {name:?}; use a raw bone name or semantic label from rig.json/labels.json"
        );
    }
    result.sort_unstable();
    result.dedup();
    Ok(result)
}

pub fn frame_mask(width: usize, joints: usize, frames: usize, selected: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; width * 12 * frames];
    for j in 0..joints {
        for d in 0..12 {
            for t in selected {
                mask[(j * 12 + d) * frames + t] = true;
            }
        }
    }
    mask
}
pub fn joint_mask(width: usize, frames: usize, selected: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; width * 12 * frames];
    for j in selected {
        mask[j * 12 * frames..(j + 1) * 12 * frames].fill(true);
    }
    mask
}

/// Copy a temporal window, repeating the last actual frame for a short clip.
pub fn window(
    values: &[f32],
    width: usize,
    source_frames: usize,
    start: usize,
    frames: usize,
) -> Result<Vec<f32>> {
    ensure!(
        source_frames > 0
            && start < source_frames
            && frames > 0
            && values.len() == width * 12 * source_frames,
        "Invalid reference window"
    );
    let mut out = vec![0.; width * 12 * frames];
    for row in 0..width * 12 {
        for t in 0..frames {
            out[row * frames + t] =
                values[row * source_frames + (start + t).min(source_frames - 1)];
        }
    }
    Ok(out)
}

pub fn first_facing(values: &[f32], frames: usize) -> Result<Matrix3<f64>> {
    ensure!(
        frames > 0 && values.len() >= 12 * frames,
        "Missing root rotation"
    );
    rotation6(std::array::from_fn(|d| values[(d + 3) * frames] as f64))
}

/// Upstream rotate_unimate_facing: root R*q^-1, root-child q*R.
/// Positions/velocities stay in the per-frame facing coordinate system.
pub fn rotate_facing(
    values: &mut [f32],
    p: &Prepared,
    frames: usize,
    q: Matrix3<f64>,
    stats: &NormalizationStats,
) -> Result<()> {
    ensure!(
        values.len() == p.width * 12 * frames,
        "Invalid facing-rotation shape"
    );
    for j in 0..p.joints.len() {
        if j != 0 && p.parents[j] != Some(0) {
            continue;
        }
        let s = stats
            .for_joint::<12>("objaverse", j)
            .map_err(anyhow::Error::msg)?;
        for t in 0..frames {
            let r = rotation6(std::array::from_fn(|d| {
                values[(j * 12 + d + 3) * frames + t] as f64
            }))?;
            let r = if j == 0 { r * q.transpose() } else { q * r };
            for d in 0..6 {
                if s.std[d + 3] >= 1e-6 {
                    values[(j * 12 + d + 3) * frames + t] = r[(d % 3, d / 3)] as f32;
                }
            }
        }
    }
    Ok(())
}

pub fn stitch(
    chain: &[f32],
    chain_frames: usize,
    segment: &[f32],
    width: usize,
    frames: usize,
    overlap: usize,
) -> Result<Vec<f32>> {
    ensure!(
        overlap > 0
            && overlap < frames
            && chain_frames >= overlap
            && chain.len() == width * 12 * chain_frames
            && segment.len() == width * 12 * frames,
        "Invalid expansion seam"
    );
    let total = chain_frames + frames - overlap;
    let mut out = Vec::with_capacity(width * 12 * total);
    for row in 0..width * 12 {
        out.extend_from_slice(&chain[row * chain_frames..(row + 1) * chain_frames]);
        out.extend_from_slice(&segment[row * frames + overlap..(row + 1) * frames]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::NdArray,
        tensor::{Tensor, TensorData},
    };

    fn rig() -> Prepared {
        Prepared {
            input: String::new(),
            prompt: String::new(),
            labels: vec!["Hips".into(), "Neck".into(), "Head".into()],
            joints: vec!["root".into(), "root/n12".into(), "root/n12/n14".into()],
            parents: vec![None, Some(0), Some(1)],
            positions: vec![[0.; 3]; 3],
            canonical_rotation: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            shift: [0.; 3],
            scale: 1.,
            width: 4,
            frames: 4,
            frequencies: 2,
            tpos: vec![],
            parent_features: vec![],
            graph_dist: vec![],
            relations: vec![],
            depths: vec![],
            spectral: vec![],
            caption: vec![],
            joint_embeddings: vec![],
        }
    }
    #[test]
    fn replacement_pins_the_analytic_path_at_every_step() {
        type B = NdArray<f32>;
        let device = Default::default();
        let noise = Tensor::<B, 1>::from_data(TensorData::new(vec![2f32, 4.], [2]), &device);
        let known = Tensor::<B, 1>::from_data(TensorData::new(vec![10f32, 20.], [2]), &device);
        let keep = Tensor::from_data(TensorData::new(vec![true, false], [2]), &device);
        let mut calls = 0;
        let result = crate::sampler::sample_replacement(noise, known, keep, 5, |state, t| {
            calls += 1;
            let values = state.clone().into_data().to_vec::<f32>().unwrap();
            assert!((values[0] - (2. * (1. - t) + 10. * t)).abs() < 1e-5);
            assert!((values[1] - (4. + 2. * t)).abs() < 1e-5);
            Tensor::from_data(TensorData::new(vec![100f32, 2.], [2]), &device)
        })
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        assert_eq!(calls, 5);
        assert_eq!(result[0], 10.);
        assert!((result[1] - 6.).abs() < 1e-5);
    }
    #[test]
    fn selectors_resolve_negative_frames_aliases_and_reject_typos() {
        assert_eq!(keep_frames("0,-1,-60,500,-500", 60).unwrap(), vec![0, 59]);
        assert!(keep_frames(",", 60).is_err());
        assert!(keep_frames("head", 60).is_err());
        let p = rig();
        assert_eq!(keep_joints("n12,HEAD,root/n12", &p).unwrap(), vec![1, 2]);
        assert!(keep_joints("Typo", &p).is_err());
        let mask = frame_mask(4, 3, 4, &[0, 3]);
        assert_eq!(mask.iter().filter(|v| **v).count(), 3 * 12 * 2);
        assert!(mask[3 * 12 * 4..].iter().all(|v| !v));
    }
    #[test]
    fn short_clips_repeat_the_last_frame_and_stitch_without_duplicate_overlap() {
        let source: Vec<_> = (0..24).map(|v| v as f32).collect();
        let padded = window(&source, 1, 2, 1, 4).unwrap();
        assert_eq!(&padded[..8], &[1., 1., 1., 1., 3., 3., 3., 3.]);
        assert!(window(&source, 1, 2, 2, 4).is_err());
        let first: Vec<_> = (0..48).map(|v| v as f32).collect();
        let second: Vec<_> = (100..148).map(|v| v as f32).collect();
        let joined = stitch(&first, 4, &second, 1, 4, 2).unwrap();
        assert_eq!(
            &joined[..12],
            &[0., 1., 2., 3., 102., 103., 4., 5., 6., 7., 106., 107.]
        );
        assert_eq!(joined.len(), 12 * 6);
        assert!(stitch(&first, 4, &second, 1, 4, 4).is_err());
    }
    #[test]
    fn expansion_facing_roundtrip_preserves_motion_and_aligns_root() -> Result<()> {
        let p = rig();
        let path =
            std::env::temp_dir().join(format!("rusti-task-stats-{}.json", std::process::id()));
        fs_write_stats(&path)?;
        let stats = NormalizationStats::from_json(&path).map_err(anyhow::Error::msg)?;
        std::fs::remove_file(path)?;
        let yaw =
            nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), 0.7).into_inner();
        let mut values = vec![0.; p.width * 12 * p.frames];
        for j in 0..p.joints.len() {
            for t in 0..p.frames {
                let r = if j == 0 { yaw } else { Matrix3::identity() };
                values[(j * 12 + 1) * p.frames + t] = 2.;
                values[(j * 12 + 9) * p.frames + t] = 0.3;
                for d in 0..6 {
                    values[(j * 12 + d + 3) * p.frames + t] = r[(d % 3, d / 3)] as f32;
                }
            }
        }
        let original = values.clone();
        rotate_facing(&mut values, &p, p.frames, yaw, &stats)?;
        assert!((first_facing(&values, p.frames)? - Matrix3::identity()).norm() < 1e-6);
        rotate_facing(&mut values, &p, p.frames, yaw.transpose(), &stats)?;
        assert!(
            values
                .iter()
                .zip(original)
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        Ok(())
    }
    fn fs_write_stats(path: &std::path::Path) -> Result<()> {
        std::fs::write(
            path,
            serde_json::to_vec(
                &serde_json::json!({"objaverse":{"mean_root":vec![0.;12],"mean_local":vec![0.;12],"std_root":vec![1.;12],"std_local":vec![1.;12]}}),
            )?,
        )?;
        Ok(())
    }
    #[test]
    fn malformed_constraints_fail_before_sampling() {
        assert!(
            Replacement {
                known: vec![0.; 48],
                keep: vec![false; 48],
                steps: 50,
                valid_frames: 4
            }
            .validate(1, 4)
            .is_err()
        );
        assert!(
            Replacement {
                known: vec![0.; 48],
                keep: vec![true; 48],
                steps: 0,
                valid_frames: 4
            }
            .validate(1, 4)
            .is_err()
        );
        assert!(
            Motion {
                shape: [1, 1, 12, 4],
                values: vec![0.; 47]
            }
            .validate(1)
            .is_err()
        );
    }
}
