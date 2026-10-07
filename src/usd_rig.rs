//! Native USD skeleton import. Matrices cross the USD row-vector / nalgebra
//! column-vector boundary exactly once, in `matrix`.
use anyhow::{Context, Result, bail, ensure};
use nalgebra::{Matrix3, Matrix4, Vector3};
use openusd::{gf, usd};
use openusd_schemas::geom::{Imageable, Xformable};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

struct Transform(usd::Prim);
impl usd::SchemaBase for Transform {
    const KIND: usd::SchemaKind = usd::SchemaKind::AbstractTyped;
    fn prim(&self) -> &usd::Prim {
        &self.0
    }
}
impl Imageable for Transform {}
impl Xformable for Transform {}

pub fn matrix(m: gf::Matrix4d) -> Matrix4<f64> {
    Matrix4::from_row_slice(&m.0).transpose()
}

pub fn rotation(m: &Matrix4<f64>) -> Result<Matrix3<f64>> {
    ensure!(m.iter().all(|v| v.is_finite()), "Non-finite USD transform");
    let mut r = m.fixed_view::<3, 3>(0, 0).into_owned();
    let lengths = [r.column(0).norm(), r.column(1).norm(), r.column(2).norm()];
    ensure!(lengths[0] > 1e-10, "Singular USD transform");
    ensure!(
        lengths.iter().all(|v| (v / lengths[0] - 1.).abs() < 1e-4),
        "Non-uniform scale is unsupported; apply scale to the rig before importing"
    );
    r /= lengths[0];
    ensure!(
        (r.transpose() * r - Matrix3::identity()).norm() < 1e-4 && r.determinant() > 0.,
        "Shear or reflected USD transform is unsupported"
    );
    Ok(r)
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Annotation {
    /// Exact joint path or unique final path component -> training vocabulary.
    pub labels: BTreeMap<String, String>,
    pub right: String,
    pub left: String,
}

pub struct Rig {
    pub stage: usd::Stage,
    pub skeleton: String,
    pub meshes: Vec<String>,
    pub joints: Vec<String>,
    pub parents: Vec<Option<usize>>,
    pub rest: Vec<Matrix4<f64>>,
    pub world_rest: Vec<Matrix4<f64>>,
    pub skeleton_world: Matrix4<f64>,
    pub up: String,
}

impl Rig {
    pub fn open(input: &Path) -> Result<Self> {
        let stage = usd::Stage::open(input.to_str().context("Input path must be UTF-8")?)?;
        let mut paths = Vec::new();
        stage.traverse(usd::PrimPredicate::DEFAULT, |p| paths.push(p.to_string()))?;
        let mut skeletons = Vec::new();
        let mut meshes = Vec::new();
        for path in &paths {
            let prim = stage.prim(path.as_str())?;
            match prim.type_name()?.as_deref() {
                Some("Skeleton") => skeletons.push(path.clone()),
                Some("Mesh") => meshes.push(path.clone()),
                _ => (),
            }
        }
        ensure!(
            skeletons.len() == 1,
            "Expected one USD Skeleton, found {}",
            skeletons.len()
        );
        let skeleton = skeletons.remove(0);
        let joints = stage
            .attribute(format!("{skeleton}.joints"))?
            .get::<Vec<openusd::tf::Token>>()?
            .context("Skeleton has no joints")?
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let parents = parents(&joints)?;
        let rest: Vec<_> = stage
            .attribute(format!("{skeleton}.restTransforms"))?
            .get::<Vec<gf::Matrix4d>>()?
            .context("Skeleton has no restTransforms")?
            .into_iter()
            .map(matrix)
            .collect();
        ensure!(
            rest.len() == joints.len(),
            "restTransforms count differs from joint count"
        );
        let binds = stage
            .attribute(format!("{skeleton}.bindTransforms"))?
            .get::<Vec<gf::Matrix4d>>()?
            .context("Skeleton has no bindTransforms")?;
        ensure!(
            binds.len() == joints.len(),
            "bindTransforms count differs from joint count"
        );
        let mut chain: Vec<_> = stage
            .prim(skeleton.as_str())?
            .path()
            .ancestors_below_root()
            .collect();
        chain.reverse();
        let mut skeleton_world = Matrix4::identity();
        for path in chain {
            let x = Transform(stage.prim(path)?);
            for op in x.xform_op_order()?.unwrap_or_default() {
                if op.starts_with("xformOp:") {
                    ensure!(
                        !x.0.attribute(op).value_might_be_time_varying()?,
                        "Animated ancestor transforms must be baked before import"
                    );
                }
            }
            if x.resets_xform_stack()? {
                skeleton_world = Matrix4::identity();
            }
            skeleton_world *= matrix(x.local_to_parent_transform(0.)?);
        }
        rotation(&skeleton_world)?;
        let mut world_rest = Vec::new();
        for (i, m) in rest.iter().enumerate() {
            rotation(m)?;
            // Local joint scaling complicates skinning and conditioning; reject it.
            ensure!(
                (m.fixed_view::<3, 1>(0, 0).norm() - 1.).abs() < 1e-4,
                "Joint {} has a non-unit local scale",
                joints[i]
            );
            let g = parents[i].map_or(skeleton_world, |p| world_rest[p]) * m;
            world_rest.push(g);
        }
        let mut skinned = Vec::new();
        for mesh in meshes {
            let mut targets = Vec::new();
            for ancestor in stage.prim(mesh.as_str())?.path().ancestors_below_root() {
                let found = stage
                    .relationship(format!("{ancestor}.skel:skeleton"))?
                    .targets()?;
                if !found.is_empty() {
                    targets = found;
                    break;
                }
            }
            if targets.iter().any(|p| p.as_str() == skeleton) {
                let weights = stage
                    .attribute(format!("{mesh}.primvars:skel:jointWeights"))?
                    .get::<Vec<f32>>()?
                    .context("Bound mesh has no jointWeights")?;
                let indices = stage
                    .attribute(format!("{mesh}.primvars:skel:jointIndices"))?
                    .get::<openusd::sdf::Value>()?
                    .context("Bound mesh has no jointIndices")?;
                let openusd::sdf::Value::IntVec(indices) = indices else {
                    bail!("Expected int[] jointIndices on {mesh}");
                };
                ensure!(
                    weights.len() == indices.len() && !weights.is_empty(),
                    "Invalid skin arrays on {mesh}"
                );
                ensure!(
                    weights.iter().all(|w| w.is_finite() && *w >= 0.),
                    "Invalid skin weights on {mesh}"
                );
                let skin_joints = stage
                    .attribute(format!("{mesh}.skel:joints"))?
                    .get::<Vec<openusd::tf::Token>>()?;
                let count = skin_joints.as_ref().map_or(joints.len(), Vec::len);
                ensure!(
                    indices.iter().all(|i| *i >= 0 && (*i as usize) < count),
                    "Skin indices out of range on {mesh}"
                );
                skinned.push(mesh);
            }
        }
        ensure!(!skinned.is_empty(), "No mesh is skinned to the skeleton");
        let up = match stage.stage_metadata("upAxis")? {
            Some(openusd::sdf::Value::Token(t)) => t.to_string(),
            None => "Y".to_owned(),
            _ => bail!("Invalid USD upAxis"),
        };
        ensure!(
            up == "Y" || up == "Z",
            "Only Y-up and Z-up USD scenes are supported"
        );
        Ok(Self {
            stage,
            skeleton,
            meshes: skinned,
            joints,
            parents,
            rest,
            world_rest,
            skeleton_world,
            up,
        })
    }

    pub fn index(&self, name: &str) -> Result<usize> {
        let matches: Vec<_> = self
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.as_str() == name || j.rsplit('/').next() == Some(name))
            .map(|(i, _)| i)
            .collect();
        ensure!(matches.len() == 1, "Joint {name:?} is missing or ambiguous");
        Ok(matches[0])
    }

    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"skeleton":self.skeleton,"joint_count":self.joints.len(),
            "fits_model":self.joints.len()<=71,"skinned_meshes":self.meshes,"up_axis":self.up,
            "joints":self.joints.iter().enumerate().map(|(i,name)|serde_json::json!({
                "name":name,"parent":self.parents[i],"world_position":position(&self.world_rest[i]).as_slice()
            })).collect::<Vec<_>>()})
    }
}

pub fn position(m: &Matrix4<f64>) -> Vector3<f64> {
    m.fixed_view::<3, 1>(0, 3).into_owned()
}

pub fn parents(joints: &[String]) -> Result<Vec<Option<usize>>> {
    ensure!(!joints.is_empty(), "Empty skeleton");
    let mut known = BTreeMap::new();
    let mut parents = Vec::new();
    for (i, j) in joints.iter().enumerate() {
        ensure!(!known.contains_key(j), "Duplicate joint {j}");
        let parent = if let Some((p, _)) = j.rsplit_once('/') {
            Some(
                *known
                    .get(p)
                    .with_context(|| format!("Parent {p} must appear before {j}"))?,
            )
        } else {
            None
        };
        ensure!(
            i == 0 || parent.is_some(),
            "Multiple skeleton roots are unsupported"
        );
        parents.push(parent);
        known.insert(j.clone(), i);
    }
    Ok(parents)
}
