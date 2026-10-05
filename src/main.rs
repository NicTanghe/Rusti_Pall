mod checkpoint;
pub mod model;
pub mod normalization;
pub mod sampler;
pub mod unimate;

use std::{env, fs, process::ExitCode};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("inspect-ema") => inspect_checkpoint(args.collect()),
        Some("inspect-model") => inspect_named_checkpoint(args.collect(), Some("model_state_dict")),
        Some("inspect-weights") => inspect_named_checkpoint(args.collect(), None),
        Some("check-weights") => check_weights(args.collect()),
        Some(config_path) => validate_config(config_path.to_owned()),
        None => {
            eprintln!(
                "Usage:\n  rusty_uni_pall <resolved-config.json>\n  rusty_uni_pall inspect-model <checkpoint.pt> [manifest.json]\n  rusty_uni_pall inspect-ema <checkpoint.pt> [manifest.json]\n  rusty_uni_pall inspect-weights <named-weights.pt> [manifest.json]\n  rusty_uni_pall check-weights <resolved-config.json> <ema_named.pt>"
            );
            ExitCode::from(2)
        }
    }
}

fn check_weights(args: Vec<String>) -> ExitCode {
    let (Some(config_path), Some(weights_path)) = (args.first(), args.get(1)) else {
        eprintln!("Usage: rusty_uni_pall check-weights <resolved-config.json> <ema_named.pt>");
        return ExitCode::from(2);
    };
    let result = fs::read_to_string(config_path)
        .map_err(anyhow_io)
        .and_then(|contents| {
            serde_json::from_str::<unimate::UniMateConfig>(&contents).map_err(|e| e.to_string())
        })
        .and_then(|config| {
            type CpuBackend = burn::backend::NdArray<f32>;
            let device = Default::default();
            let mut model = model::UniMateDenoiser::<CpuBackend>::from_config(&config, &device)?;
            model.load_ema_weights(weights_path)
        });
    match result {
        Ok(report) => {
            println!(
                "Loaded {} EMA tensors into the Burn model",
                report.applied.len()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Could not load EMA weights: {error}");
            ExitCode::FAILURE
        }
    }
}

fn inspect_checkpoint(args: Vec<String>) -> ExitCode {
    let Some(checkpoint_path) = args.first() else {
        eprintln!("Usage: rusty_uni_pall inspect-ema <checkpoint.pt> [manifest.json]");
        return ExitCode::from(2);
    };

    match checkpoint::inspect_ema(checkpoint_path) {
        Ok(manifest) => {
            println!("EMA tensors: {}", manifest.tensor_count);
            if let Some(output_path) = args.get(1) {
                match checkpoint::write_manifest(&manifest, output_path) {
                    Ok(()) => println!("Wrote tensor manifest to {output_path}"),
                    Err(error) => {
                        eprintln!("Could not write manifest: {error}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                for tensor in &manifest.tensors {
                    println!("{}\t{:?}\t{}", tensor.name, tensor.shape, tensor.dtype);
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Could not inspect checkpoint: {error}");
            ExitCode::FAILURE
        }
    }
}

fn inspect_named_checkpoint(args: Vec<String>, top_level_key: Option<&str>) -> ExitCode {
    let Some(checkpoint_path) = args.first() else {
        eprintln!("Expected a checkpoint or named weight file");
        return ExitCode::from(2);
    };

    match checkpoint::inspect_state_dict(checkpoint_path, top_level_key) {
        Ok(manifest) => {
            println!("Named tensors: {}", manifest.tensor_count);
            if let Some(output_path) = args.get(1) {
                match checkpoint::write_manifest(&manifest, output_path) {
                    Ok(()) => println!("Wrote tensor manifest to {output_path}"),
                    Err(error) => {
                        eprintln!("Could not write manifest: {error}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                for tensor in &manifest.tensors {
                    println!("{}\t{:?}\t{}", tensor.name, tensor.shape, tensor.dtype);
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Could not inspect weights: {error}");
            ExitCode::FAILURE
        }
    }
}

fn validate_config(config_path: String) -> ExitCode {
    let result = fs::read_to_string(&config_path)
        .map_err(anyhow_io)
        .and_then(|contents| {
            serde_json::from_str::<unimate::UniMateConfig>(&contents).map_err(|e| e.to_string())
        })
        .and_then(|config| {
            config
                .validate()
                .map(|()| config)
                .map_err(|e| e.to_string())
        });

    match result {
        Ok(config) => {
            println!(
                "Validated UniMate inference config: {}",
                config.experiment.name
            );
            println!(
                "{} frames, up to {} joints, {} layers, width {}, {} heads",
                config.dataset.max_motion_length,
                config.dataset.max_joints,
                config.model.num_layers,
                config.model.latent_dim,
                config.model.num_heads
            );
            println!("Burn inference model and EMA weight loader are available.");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Could not load inference config: {error}");
            ExitCode::FAILURE
        }
    }
}

fn anyhow_io(error: std::io::Error) -> String {
    error.to_string()
}
