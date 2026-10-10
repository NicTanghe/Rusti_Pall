//! Native conditioning, feature decoding, and USD animation authoring.
use crate::{
    model,
    normalization::NormalizationStats,
    sampler,
    unimate::UniMateConfig,
    usd_rig::{Annotation, Rig, position, rotation},
};
use anyhow::{Context, Result, ensure};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use nalgebra::{DMatrix, Matrix3, Rotation3, UnitQuaternion, Vector3};
use openusd::{gf, sdf, usd};
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, fs, path::Path};

#[derive(Clone, Serialize, Deserialize)]
pub struct Prepared {
    pub input: String,
    pub prompt: String,
    pub labels: Vec<String>,
    pub joints: Vec<String>,
    pub parents: Vec<Option<usize>>,
    pub positions: Vec<[f64; 3]>,
    pub canonical_rotation: [[f64; 3]; 3],
    pub shift: [f64; 3],
    pub scale: f64,
    pub width: usize,
    pub frames: usize,
    pub frequencies: usize,
    pub tpos: Vec<f32>,
    pub parent_features: Vec<f32>,
    pub graph_dist: Vec<i64>,
    pub relations: Vec<i64>,
    pub depths: Vec<i64>,
    pub spectral: Vec<f32>,
    pub caption: Vec<f32>,
    pub joint_embeddings: Vec<f32>,
}

pub fn topology(
    parents: &[Option<usize>],
    width: usize,
    k: usize,
    max_depth: usize,
) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<f32>) {
    let n = parents.len();
    let mut adj = vec![Vec::new(); n];
    let mut depths = vec![0i64; width];
    let mut child_count = vec![0; n];
    for (i, p) in parents.iter().enumerate() {
        if let Some(p) = p {
            adj[i].push(*p);
            adj[*p].push(i);
            child_count[*p] += 1;
            depths[i] = (depths[*p] + 1).min(max_depth as i64);
        }
    }
    let mut distances = vec![0; width * width];
    let mut relations = vec![0; width * width];
    for i in 0..n {
        let mut d = vec![usize::MAX; n];
        d[i] = 0;
        let mut queue = VecDeque::from([i]);
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if d[v] == usize::MAX {
                    d[v] = d[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        for j in 0..n {
            distances[i * width + j] = d[j].min(5) as i64;
            relations[i * width + j] = if i == j {
                if child_count[i] == 0 { 5 } else { 0 }
            } else if parents[j] == Some(i) {
                2
            } else if parents[i] == Some(j) {
                1
            } else if parents[i].is_some() && parents[i] == parents[j] {
                3
            } else {
                4
            };
        }
    }
    let mut lap = DMatrix::<f64>::identity(n, n);
    for i in 0..n {
        for &j in &adj[i] {
            lap[(i, j)] = -1. / ((adj[i].len() * adj[j].len()) as f64).sqrt();
        }
    }
    let eigen = lap.symmetric_eigen();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    let mut spectral = vec![0.; width * k];
    for (f, &col) in order.iter().skip(1).take(k).enumerate() {
        let max = (0..n)
            .max_by(|&a, &b| {
                eigen.eigenvectors[(a, col)]
                    .abs()
                    .total_cmp(&eigen.eigenvectors[(b, col)].abs())
            })
            .unwrap();
        let sign = if eigen.eigenvectors[(max, col)] < 0. {
            -1.
        } else {
            1.
        };
        for i in 0..n {
            spectral[i * k + f] = (sign * eigen.eigenvectors[(i, col)]) as f32;
        }
    }
    (distances, relations, depths, spectral)
}

pub fn prepare(
    rig: &Rig,
    annotation: &Annotation,
    input: &Path,
    prompt: &str,
    cfg: &UniMateConfig,
    stats: &NormalizationStats,
) -> Result<Prepared> {
    let n = rig.joints.len();
    let width = cfg.dataset.max_joints;
    ensure!(
        (4..=width).contains(&n),
        "Rig has {n} joints; supported range is 4..={width}"
    );
    let mut labels = vec![String::new(); n];
    for (joint, label) in &annotation.labels {
        let i = rig.index(joint)?;
        ensure!(labels[i].is_empty(), "Joint {joint} has duplicate labels");
        ensure!(!label.trim().is_empty(), "Empty label for {joint}");
        labels[i] = label.clone();
    }
    for (i, label) in labels.iter().enumerate() {
        ensure!(!label.is_empty(), "Missing label for {}", rig.joints[i]);
    }
    let right = rig.index(&annotation.right)?;
    let left = rig.index(&annotation.left)?;
    ensure!(right != left, "Facing joints must differ");
    let up = if rig.up == "Z" {
        Rotation3::from_axis_angle(&Vector3::x_axis(), -std::f64::consts::FRAC_PI_2).into_inner()
    } else {
        Matrix3::identity()
    };
    let world: Vec<_> = rig.world_rest.iter().map(position).collect();
    let across = up * (world[right] - world[left]);
    let forward = Vector3::y().cross(&across);
    ensure!(
        forward.norm() > 1e-7,
        "Facing pair has no horizontal separation"
    );
    let heading =
        Rotation3::from_axis_angle(&Vector3::y_axis(), -forward.x.atan2(forward.z)).into_inner();
    let c = heading * up;
    let mut diameter = 0f64;
    let mut adj = vec![Vec::new(); n];
    for (i, p) in rig.parents.iter().enumerate() {
        if let Some(p) = p {
            let length = (world[i] - world[*p]).norm();
            adj[i].push((*p, length));
            adj[*p].push((i, length));
        }
    }
    for start in 0..n {
        let mut queue = VecDeque::from([(start, usize::MAX, 0.)]);
        while let Some((u, p, d)) = queue.pop_front() {
            diameter = diameter.max(d);
            for &(v, w) in &adj[u] {
                if v != p {
                    queue.push_back((v, u, d + w));
                }
            }
        }
    }
    ensure!(diameter > 1e-7, "Degenerate skeleton diameter");
    let scale = 2. / diameter;
    let rotated: Vec<_> = world.iter().map(|p| c * p * scale).collect();
    let floor = rotated.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let shift = Vector3::new(-rotated[0].x, -floor, -rotated[0].z);
    let positions: Vec<[f64; 3]> = rotated.iter().map(|p| (p + shift).into()).collect();
    let mut tpos = vec![0f32; width * 12];
    for i in 0..n {
        let s = stats
            .for_joint::<12>("objaverse", i)
            .map_err(anyhow::Error::msg)?;
        let mut raw = [0f32; 12];
        for j in 0..3 {
            raw[j] = positions[i][j] as f32;
        }
        raw[3] = 1.;
        raw[7] = 1.; // first two columns of identity rotation
        for j in 0..12 {
            ensure!(
                s.std[j] > 0. && s.std[j].is_finite(),
                "Invalid normalization std"
            );
            tpos[i * 12 + j] = (raw[j] - s.mean[j]) / s.std[j];
        }
    }
    let mut parent_features = tpos.clone();
    for (i, p) in rig.parents.iter().enumerate() {
        if let Some(p) = p {
            parent_features[i * 12..(i + 1) * 12].copy_from_slice(&tpos[p * 12..(p + 1) * 12]);
        }
    }
    let k = cfg.model.max_freqs;
    let (graph_dist, relations, depths, spectral) =
        topology(&rig.parents, width, k, cfg.dataset.max_depth);
    Ok(Prepared {
        input: input.canonicalize()?.to_string_lossy().into(),
        prompt: prompt.into(),
        labels,
        joints: rig.joints.clone(),
        parents: rig.parents.clone(),
        positions,
        canonical_rotation: std::array::from_fn(|i| std::array::from_fn(|j| c[(i, j)])),
        shift: shift.into(),
        scale,
        width,
        frames: cfg.dataset.max_motion_length,
        frequencies: k,
        tpos,
        parent_features,
        graph_dist,
        relations,
        depths,
        spectral,
        caption: Vec::new(),
        joint_embeddings: Vec::new(),
    })
}

impl Prepared {
    pub fn embed(&mut self, dir: &Path) -> Result<()> {
        let mut texts = vec![self.prompt.clone()];
        texts.extend(self.labels.clone());
        let embeddings = crate::text_encoder::encode(dir, &texts)?;
        self.caption = embeddings[0].clone();
        self.joint_embeddings = vec![0.; self.width * 768];
        for (i, e) in embeddings.iter().skip(1).enumerate() {
            self.joint_embeddings[i * 768..(i + 1) * 768].copy_from_slice(e);
        }
        Ok(())
    }

    pub fn sample(
        &self,
        cfg: &UniMateConfig,
        weights: &Path,
        stats: &NormalizationStats,
        seed: u64,
        scale: f32,
    ) -> Result<Vec<f32>> {
        self.sample_constrained(cfg, weights, stats, seed, scale, None)
    }

    pub fn sample_constrained(
        &self,
        cfg: &UniMateConfig,
        weights: &Path,
        stats: &NormalizationStats,
        seed: u64,
        scale: f32,
        replacement: Option<&crate::motion_tasks::Replacement>,
    ) -> Result<Vec<f32>> {
        match crate::compute::selected()? {
            #[cfg(all(feature = "cuda", not(target_os = "macos")))]
            crate::compute::Compute::Cuda => self.sample_with::<burn_cuda::Cuda>(
                cfg,
                weights,
                stats,
                seed,
                scale,
                replacement,
                Default::default(),
            ),
            #[cfg(not(all(feature = "cuda", not(target_os = "macos"))))]
            crate::compute::Compute::Cuda => anyhow::bail!("CUDA not compiled"),
            crate::compute::Compute::Metal | crate::compute::Compute::Wgpu => self
                .sample_with::<burn::backend::Wgpu>(
                cfg,
                weights,
                stats,
                seed,
                scale,
                replacement,
                Default::default(),
            ),
        }
    }

    fn sample_with<B: Backend>(
        &self,
        cfg: &UniMateConfig,
        weights: &Path,
        stats: &NormalizationStats,
        seed: u64,
        scale: f32,
        replacement: Option<&crate::motion_tasks::Replacement>,
        device: B::Device,
    ) -> Result<Vec<f32>> {
        ensure!(
            scale.is_finite() && scale >= 1.,
            "CFG scale must be finite and >= 1"
        );
        let w = self.width;
        let f = self.frames;
        let k = self.frequencies;
        if let Some(r) = replacement {
            r.validate(w, f)?;
            ensure!(scale > 1., "Constrained sampling requires CFG > 1");
        }
        ensure!(
            w == cfg.dataset.max_joints
                && f == cfg.dataset.max_motion_length
                && k == cfg.model.max_freqs,
            "Conditioning/config dimensions differ"
        );
        for (name, values, len) in [
            ("caption", &self.caption, 768),
            ("joint embeddings", &self.joint_embeddings, w * 768),
            ("rest pose", &self.tpos, w * 12),
            ("parent features", &self.parent_features, w * 12),
            ("spectral", &self.spectral, w * k),
        ] {
            ensure!(
                values.len() == len && values.iter().all(|v| v.is_finite()),
                "Invalid {name}"
            );
        }
        ensure!(
            self.graph_dist.len() == w * w
                && self.relations.len() == w * w
                && self.depths.len() == w,
            "Invalid topology dimensions"
        );
        let mut model =
            model::UniMateDenoiser::<B>::from_config(cfg, &device).map_err(anyhow::Error::msg)?;
        model
            .load_ema_weights(weights)
            .map_err(anyhow::Error::msg)?;
        let condition = model::DenoiserCondition {
            caption_embedding: Tensor::from_data(
                TensorData::new(self.caption.clone(), [1, 768]),
                &device,
            ),
            tpos_first_frame: Tensor::from_data(
                TensorData::new(self.tpos.clone(), [1, w, 12]),
                &device,
            ),
            tpos_first_frame_parents: Tensor::from_data(
                TensorData::new(self.parent_features.clone(), [1, w, 12]),
                &device,
            ),
            n_joints: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![self.joints.len() as i64], [1]),
                &device,
            ),
            motion_lengths: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![replacement.map_or(f, |r| r.valid_frames) as i64], [1]),
                &device,
            ),
            joint_names_emb: Tensor::from_data(
                TensorData::new(self.joint_embeddings.clone(), [1, w, 768]),
                &device,
            ),
            joint_depths: Tensor::<B, 2, Int>::from_data(
                TensorData::new(self.depths.clone(), [1, w]),
                &device,
            ),
            graph_dist: Tensor::<B, 3, Int>::from_data(
                TensorData::new(self.graph_dist.clone(), [1, w, w]),
                &device,
            ),
            joint_relations: Tensor::<B, 3, Int>::from_data(
                TensorData::new(self.relations.clone(), [1, w, w]),
                &device,
            ),
            spectral_coords: Tensor::from_data(
                TensorData::new(self.spectral.clone(), [1, w, k]),
                &device,
            ),
        };
        let noise = sampler::standard_normal_noise::<B, 4>([1, w, 12, f], seed, &device);
        let sample = if let Some(r) = replacement {
            r.validate(w, f)?;
            let mut known = r.known.clone();
            for j in 0..w {
                let s = stats
                    .for_joint::<12>("objaverse", j)
                    .map_err(anyhow::Error::msg)?;
                for d in 0..12 {
                    for t in 0..f {
                        let i = (j * 12 + d) * f + t;
                        known[i] = (known[i] - s.mean[d]) / s.std[d];
                    }
                }
            }
            model
                .sample_replacement(
                    noise,
                    Tensor::from_data(TensorData::new(known, [1, w, 12, f]), &device),
                    Tensor::from_data(TensorData::new(r.keep.clone(), [1, w, 12, f]), &device),
                    &condition,
                    scale,
                    r.steps,
                )
                .map_err(anyhow::Error::msg)?
        } else {
            let (sample, info) = model
                .sample_dopri5(noise, &condition, scale)
                .map_err(anyhow::Error::msg)?;
            eprintln!(
                "ODE evaluations={}, accepted={}, rejected={}",
                info.evaluations, info.accepted_steps, info.rejected_steps
            );
            sample
        };
        let mut values = sample
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        for j in 0..w {
            let s = stats
                .for_joint::<12>("objaverse", j)
                .map_err(anyhow::Error::msg)?;
            for d in 0..12 {
                for t in 0..f {
                    let index = (j * 12 + d) * f + t;
                    values[index] = values[index] * s.std[d] + s.mean[d];
                }
            }
        }
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "Non-finite generated motion"
        );
        if let Some(r) = replacement {
            // Preserve the caller's raw representation exactly after inverse normalization.
            for (i, keep) in r.keep.iter().enumerate() {
                if *keep {
                    values[i] = r.known[i];
                }
            }
        }
        Ok(values)
    }

    pub fn export(&self, rig: &Rig, values: &[f32], output: &Path) -> Result<()> {
        ensure!(
            rig.joints == self.joints,
            "USD skeleton changed since preparation"
        );
        ensure!(
            values.len() == self.width * 12 * self.frames && values.iter().all(|v| v.is_finite()),
            "Invalid motion tensor"
        );
        let n = rig.joints.len();
        let frames = self.frames;
        let c = Matrix3::from_fn(|i, j| self.canonical_rotation[i][j]);
        let inverse_skeleton = rig
            .skeleton_world
            .try_inverse()
            .context("Singular skeleton transform")?;
        let skeleton_rotation = rotation(&rig.skeleton_world)?;
        let rest_rot: Vec<_> = rig
            .world_rest
            .iter()
            .map(|m| rotation(m).map(|r| c * r))
            .collect::<Result<_>>()?;
        let anim = format!(
            "{}/UniMateMotion",
            rig.skeleton
                .rsplit_once('/')
                .context("Invalid skeleton path")?
                .0
        );
        rig.stage
            .define_prim(anim.as_str())?
            .set_type_name("SkelAnimation")?;
        rig.stage
            .create_attribute(format!("{anim}.joints"), "token[]")?
            .set(sdf::Value::token_vec(rig.joints.clone()))?;
        rig.stage
            .create_attribute(format!("{anim}.scales"), "half3[]")?
            .set(sdf::Value::Vec3hVec(vec![
                [gf::f16::from_f32(1.); 3].into();
                n
            ]))?;
        let root_positions = decode_root(values, frames)?;
        let mut max_bone_error = 0f64;
        for t in 0..frames {
            let get = |j: usize, d: usize| values[(j * 12 + d) * frames + t] as f64;
            let hml: Vec<_> = (0..n)
                .map(|j| rotation6(std::array::from_fn(|d| get(j, d + 3))))
                .collect::<Result<_>>()?;
            let root_pos = root_positions[t];
            let mut sum = vec![Matrix3::zeros(); n];
            let mut counts = vec![0; n];
            for (j, p) in rig.parents.iter().enumerate() {
                if let Some(p) = p {
                    sum[*p] += hml[j];
                    counts[*p] += 1;
                }
            }
            let mut translations = Vec::new();
            let mut quats = Vec::new();
            for j in 0..n {
                let delta = if counts[j] > 0 {
                    project_rotation(sum[j])
                } else {
                    Matrix3::identity()
                };
                let q = if let Some(p) = rig.parents[j] {
                    rest_rot[p].transpose() * delta * rest_rot[j]
                } else {
                    skeleton_rotation.transpose() * c.transpose() * delta * rest_rot[j]
                };
                let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(q));
                let q = q.quaternion();
                quats.push(gf::Quatf {
                    w: q.w as f32,
                    x: q.i as f32,
                    y: q.j as f32,
                    z: q.k as f32,
                });
                let pos = if j == 0 {
                    let world = c.transpose() * (root_pos - Vector3::from(self.shift)) / self.scale;
                    let p = inverse_skeleton * world.push(1.);
                    Vector3::new(p.x, p.y, p.z)
                } else {
                    position(&rig.rest[j])
                };
                if j > 0 {
                    max_bone_error =
                        max_bone_error.max((pos.norm() - position(&rig.rest[j]).norm()).abs());
                }
                translations.push([pos.x as f32, pos.y as f32, pos.z as f32]);
            }
            let time = usd::TimeCode::new(t as f64);
            rig.stage
                .create_attribute(format!("{anim}.translations"), "float3[]")?
                .set_at(
                    sdf::Value::Vec3fVec(translations.into_iter().map(Into::into).collect()),
                    time,
                )?;
            rig.stage
                .create_attribute(format!("{anim}.rotations"), "quatf[]")?
                .set_at(sdf::Value::QuatfVec(quats), time)?;
        }
        for path in &rig.meshes {
            rig.stage
                .prim(path.as_str())?
                .apply_api("MaterialBindingAPI")?;
        }
        for path in std::iter::once(&rig.skeleton).chain(&rig.meshes) {
            rig.stage.prim(path.as_str())?.apply_api("SkelBindingAPI")?;
            rig.stage
                .create_relationship(format!("{path}.skel:animationSource"))?
                .set_targets([anim.as_str()])?;
        }
        rig.stage.set_start_time_code(0.)?;
        rig.stage.set_end_time_code((frames - 1) as f64)?;
        rig.stage.set_time_codes_per_second(30.)?;
        rig.stage.set_frames_per_second(30.)?;
        rig.stage
            .root_layer()
            .export(output.to_str().context("Output path must be UTF-8")?)?;
        // openusd 0.7's text writer omits brackets on singleton token list ops.
        // Its own parser accepts that spelling, but the reference USD parser
        // rejects it for apiSchemas. Preserve the list-edit operation and token.
        let text = fs::read_to_string(output)?;
        fs::write(output, fix_api_schema_lists(&text))?;
        validate_export(rig, output, frames, false)?;
        fs::write(
            output.with_extension("validation.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
            "frames":frames,"joints":n,"finite":true,"max_local_bone_length_error":max_bone_error,
            "prompt":self.prompt,"text_encoder_revision":crate::text_encoder::REVISION,
            "note":"Native USD import/export and real-rig conditioning are experimental; inspect playback."}))?,
        )?;
        Ok(())
    }
}

pub fn fix_api_schema_lists(text: &str) -> String {
    text.split_inclusive('\n')
        .map(|line| {
            let Some((left, right)) = line.split_once(" = ") else {
                return line.to_owned();
            };
            let field = left.trim();
            if !matches!(
                field,
                "apiSchemas"
                    | "prepend apiSchemas"
                    | "append apiSchemas"
                    | "delete apiSchemas"
                    | "add apiSchemas"
                    | "reorder apiSchemas"
            ) {
                return line.to_owned();
            }
            let value = right.trim_end();
            if value.starts_with('"') && value.ends_with('"') {
                format!("{left} = [{value}]{}", &right[value.len()..])
            } else {
                line.to_owned()
            }
        })
        .collect()
}

/// Reopen the saved USD and verify its authored TRS and fixed bone lengths.
pub fn validate_export(rig: &Rig, path: &Path, frames: usize, rest_pose: bool) -> Result<()> {
    let stage = usd::Stage::open(path.to_str().context("Invalid output path")?)?;
    let targets = stage
        .relationship(format!("{}.skel:animationSource", rig.skeleton))?
        .targets()?;
    ensure!(
        targets.len() == 1,
        "Exported skeleton needs one animation source"
    );
    let anim = &targets[0];
    for t in 0..frames {
        let time = usd::TimeCode::new(t as f64);
        let positions = stage
            .attribute(format!("{anim}.translations"))?
            .get_at::<Vec<gf::Vec3f>>(time)?
            .context("Missing exported translations")?;
        let rotations = stage
            .attribute(format!("{anim}.rotations"))?
            .get_at::<Vec<gf::Quatf>>(time)?
            .context("Missing exported rotations")?;
        ensure!(
            positions.len() == rig.joints.len() && rotations.len() == rig.joints.len(),
            "Exported joint counts differ"
        );
        for j in 0..rig.joints.len() {
            let q = rotations[j];
            let norm = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
            ensure!(
                norm.is_finite() && (norm - 1.).abs() < 1e-4,
                "Invalid exported quaternion"
            );
            let m = crate::usd_rig::matrix(gf::Matrix4d::from_trs(
                positions[j],
                q,
                gf::vec3f(1., 1., 1.),
            ));
            ensure!(
                m.iter().all(|v| v.is_finite()),
                "Non-finite exported transform"
            );
            if j > 0 {
                ensure!(
                    (position(&m).norm() - position(&rig.rest[j]).norm()).abs()
                        < 1e-4 * (1. + position(&rig.rest[j]).norm()),
                    "Export changed bone length"
                );
            }
            if rest_pose {
                ensure!(
                    (m - rig.rest[j]).norm() < 1e-4 * (1. + rig.rest[j].norm()),
                    "Rest-pose round trip failed for {}",
                    rig.joints[j]
                );
            }
        }
    }
    Ok(())
}

pub fn decode_root(values: &[f32], frames: usize) -> Result<Vec<Vector3<f64>>> {
    ensure!(
        frames > 0 && values.len() >= 12 * frames,
        "Missing root features"
    );
    let mut pos = Vector3::zeros();
    let mut result = Vec::new();
    for t in 0..frames {
        let facing = rotation6(std::array::from_fn(|d| values[(d + 3) * frames + t] as f64))?;
        if t > 0 {
            pos += facing.transpose()
                * Vector3::new(
                    values[9 * frames + t - 1] as f64,
                    0.,
                    values[11 * frames + t - 1] as f64,
                );
        }
        pos.y = values[frames + t] as f64;
        ensure!(
            pos.iter().all(|v| v.is_finite()),
            "Non-finite root trajectory"
        );
        result.push(pos);
    }
    Ok(result)
}

pub fn rotation6(v: [f64; 6]) -> Result<Matrix3<f64>> {
    let x = Vector3::new(v[0], v[1], v[2]);
    let y = Vector3::new(v[3], v[4], v[5]);
    ensure!(x.norm() > 1e-10, "Degenerate 6D rotation");
    let x = x.normalize();
    let z = x.cross(&y);
    ensure!(z.norm() > 1e-10, "Collinear 6D rotation axes");
    let z = z.normalize();
    let y = z.cross(&x);
    Ok(Matrix3::from_columns(&[x, y, z]))
}

pub fn project_rotation(sum: Matrix3<f64>) -> Matrix3<f64> {
    let svd = sum.svd(true, true);
    let mut u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    if (u * vt).determinant() < 0. {
        u.set_column(2, &(-u.column(2)));
    }
    u * vt
}

pub fn extract_package(input: &Path, dir: &Path) -> Result<()> {
    if input.extension().and_then(|e| e.to_str()) != Some("usdz") {
        return Ok(());
    }
    let mut zip = zip::ZipArchive::new(fs::File::open(input)?)?;
    let mut total = 0u64;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let path = entry.enclosed_name().context("Unsafe USDZ member path")?;
        total = total
            .checked_add(entry.size())
            .context("USDZ size overflow")?;
        ensure!(
            total < 4 * 1024 * 1024 * 1024,
            "USDZ exceeds 4 GiB uncompressed"
        );
        ensure!(
            entry.unix_mode().unwrap_or(0) & 0o170000 != 0o120000,
            "USDZ symlinks are unsupported"
        );
        let dest = dir.join(path);
        if entry.is_dir() {
            fs::create_dir_all(dest)?;
            continue;
        }
        fs::create_dir_all(dest.parent().unwrap())?;
        let mut out = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dest)?;
        std::io::copy(&mut entry, &mut out)?;
    }
    Ok(())
}
