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

    if reader.is_empty() {
        return Err(
            "this UniMate checkpoint stores EMA values as a positional shadow_params list, which Burn's PyTorch reader cannot name directly; run scripts/export_ema_state_dict.py first".to_owned(),
        );
    }

    manifest_from_reader(path, "ema_state_dict", reader)
}

pub fn inspect_state_dict(
    path: impl AsRef<Path>,
    top_level_key: Option<&str>,
) -> Result<CheckpointManifest, String> {
    let path = path.as_ref();
    let reader = if let Some(key) = top_level_key {
        PytorchReader::with_top_level_key(path, key)
    } else {
        PytorchReader::new(path)
    }
    .map_err(|error| format!("could not read named tensors: {error}"))?;

    manifest_from_reader(path, top_level_key.unwrap_or("root"), reader)
}

fn manifest_from_reader(
    path: &Path,
    top_level_key: &str,
    reader: PytorchReader,
) -> Result<CheckpointManifest, String> {
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
        top_level_key: top_level_key.to_owned(),
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
