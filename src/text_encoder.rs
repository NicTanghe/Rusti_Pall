//! FLAN-T5 encoding using Candle (Metal on macOS, CPU elsewhere). Each
//! sequence is encoded unpadded,
//! since Candle's T5 encoder does not expose an encoder padding mask.
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::t5::{Config, T5EncoderModel};
use std::{collections::BTreeMap, fs, path::Path};

pub const REVISION: &str = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2";

pub fn fetch(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir)?;
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(1200))
        .build()?;
    for name in ["config.json", "tokenizer.json", "model.safetensors"] {
        let dst = dir.join(name);
        if dst.is_file() {
            continue;
        }
        eprintln!("Downloading FLAN-T5 {name} ({REVISION})...");
        let mut response = client
            .get(format!(
                "https://huggingface.co/google/flan-t5-base/resolve/{REVISION}/{name}"
            ))
            .send()?
            .error_for_status()?;
        let partial = dir.join(format!("{name}.part"));
        response.copy_to(&mut fs::File::create(&partial)?)?;
        fs::rename(partial, dst)?;
    }
    Ok(())
}

pub fn encode(dir: &Path, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let device = crate::gpu::text_device();
    match encode_on(dir, texts, &device) {
        Err(e) if !device.is_cpu() => {
            eprintln!("GPU text encoding failed ({e:#}); retrying on CPU");
            encode_on(dir, texts, &Device::Cpu)
        }
        result => result,
    }
}

fn encode_on(dir: &Path, texts: &[String], device: &Device) -> Result<Vec<Vec<f32>>> {
    let config: Config = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
    ensure!(
        config.d_model == 768,
        "Expected FLAN-T5-base with width 768"
    );
    let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    tokenizer.with_padding(None);
    // Loading to owned tensors avoids exposing mmap lifetime to an overwritten cache.
    let tensors = candle_core::safetensors::load(dir.join("model.safetensors"), device)?;
    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, device);
    let mut model = T5EncoderModel::load(vb, &config)?;
    let mut cache = BTreeMap::new();
    for text in texts {
        if cache.contains_key(text) {
            continue;
        }
        let enc = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        ensure!(
            !enc.is_empty() && enc.len() <= 512,
            "Text must tokenize to 1–512 tokens"
        );
        let ids = Tensor::new(enc.get_ids(), device)?.unsqueeze(0)?;
        let embedding = model.forward(&ids)?.mean(1)?.squeeze(0)?.to_vec1::<f32>()?;
        ensure!(
            embedding.len() == 768 && embedding.iter().all(|v| v.is_finite()),
            "Invalid text embedding"
        );
        cache.insert(text.clone(), embedding);
        model.clear_kv_cache();
    }
    texts
        .iter()
        .map(|t| cache.get(t).cloned().context("Missing text embedding"))
        .collect()
}
