//! Inspect the frozen PyTorch checkpoint without loading tensor data onto a
//! device. UniMate checkpoints wrap the model and EMA weights in separate
//! top-level dictionaries; this reader inventories the EMA dictionary.

use burn_store::pytorch::PytorchReader;
use serde::Serialize;
use std::{fs, path::Path};

#[derive(Debug, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Serialize)]
pub struct CheckpointManifest {
    pub source: String,
    pub top_level_key: String,
    pub tensor_count: usize,
    pub tensors: Vec<TensorInfo>,
}

pub fn inspect_ema(path: impl AsRef<Path>) -> Result<CheckpointManifest, String> {
    let path = path.as_ref();
    let reader = PytorchReader::with_top_level_key(path, "ema_state_dict")
        .map_err(|error| format!("could not read EMA tensors: {error}"))?;

    let mut names = reader.keys();
    names.sort();
    let tensors = names
        .into_iter()
        .map(|name| {
            let tensor = reader
                .get(&name)
                .expect("key came from the same reader's key list");
            TensorInfo {
                name,
                shape: tensor.shape.clone(),
                dtype: format!("{:?}", tensor.dtype),
            }
        })
        .collect::<Vec<_>>();

    Ok(CheckpointManifest {
        source: path.display().to_string(),
        top_level_key: "ema_state_dict".to_owned(),
        tensor_count: tensors.len(),
        tensors,
    })
}

pub fn write_manifest(
    manifest: &CheckpointManifest,
    output_path: impl AsRef<Path>,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(manifest).map_err(|error| error.to_string())?;
    fs::write(output_path, json).map_err(|error| error.to_string())
}
