//! Reader for the upstream global/per-dataset motion normalization values.

use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug, Deserialize)]
pub struct NormalizationStats {
    datasets: BTreeMap<String, DatasetStats>,
}

#[derive(Debug, Deserialize)]
struct DatasetStats {
    mean_root: Vec<f64>,
    std_root: Vec<f64>,
    mean_local: Vec<f64>,
    std_local: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct JointNormalization<const FEATURES: usize> {
    pub mean: [f32; FEATURES],
    pub std: [f32; FEATURES],
}

impl NormalizationStats {
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let stats: Self = serde_json::from_str(&contents)
            .map_err(|error| format!("could not parse {}: {error}", path.display()))?;
        if stats.datasets.is_empty() {
            return Err("normalization file contains no dataset entries".to_owned());
        }
        Ok(stats)
    }

    /// Select root-joint statistics for joint zero and local-joint statistics
    /// for all other joints, matching UniMate's motion normalization layout.
    pub fn for_joint<const FEATURES: usize>(
        &self,
        dataset_type: &str,
        joint_index: usize,
    ) -> Result<JointNormalization<FEATURES>, String> {
        let dataset = self
            .datasets
            .get(dataset_type)
            .or_else(|| (self.datasets.len() == 1).then(|| self.datasets.values().next().unwrap()))
            .ok_or_else(|| format!("no normalization stats for dataset {dataset_type:?}"))?;
        let (mean_values, std_values) = if joint_index == 0 {
            (&dataset.mean_root, &dataset.std_root)
        } else {
            (&dataset.mean_local, &dataset.std_local)
        };

        Ok(JointNormalization {
            mean: to_feature_array(mean_values, "mean", dataset_type)?,
            std: to_feature_array(std_values, "std", dataset_type)?,
        })
    }
}

fn to_feature_array<const FEATURES: usize>(
    values: &[f64],
    field: &str,
    dataset_type: &str,
) -> Result<[f32; FEATURES], String> {
    if values.len() != FEATURES {
        return Err(format!(
            "{dataset_type} {field} has {} values; expected {FEATURES}",
            values.len()
        ));
    }
    let mut result = [0.0; FEATURES];
    for (output, value) in result.iter_mut().zip(values) {
        *output = *value as f32;
    }
    Ok(result)
}
