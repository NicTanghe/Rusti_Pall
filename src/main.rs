mod checkpoint;
pub mod model;
pub mod normalization;
pub mod sampler;
pub mod unimate;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use burn_store::pytorch::PytorchReader;
use std::{env, fs, process::ExitCode};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("inspect-ema") => inspect_checkpoint(args.collect()),
        Some("inspect-model") => inspect_named_checkpoint(args.collect(), Some("model_state_dict")),
        Some("inspect-weights") => inspect_named_checkpoint(args.collect(), None),
        Some("inspect-burn-weights") => inspect_burn_weight_keys(args.collect()),
        Some("compare-forward") => compare_forward(args.collect()),
        Some("compare-forward-wgpu") => compare_forward_wgpu(args.collect()),
        Some("check-weights") => check_weights(args.collect()),
        Some(config_path) => validate_config(config_path.to_owned()),
        None => {
            eprintln!(
                "Usage:\n  rusty_uni_pall <resolved-config.json>\n  rusty_uni_pall inspect-model <checkpoint.pt> [manifest.json]\n  rusty_uni_pall inspect-ema <checkpoint.pt> [manifest.json]\n  rusty_uni_pall inspect-weights <named-weights.pt> [manifest.json]\n  rusty_uni_pall inspect-burn-weights <ema_named.pt>\n  rusty_uni_pall check-weights <resolved-config.json> <ema_named.pt>\n  rusty_uni_pall compare-forward <config.json> <ema_named.pt> <fixture.pt>\n  rusty_uni_pall compare-forward-wgpu <config.json> <ema_named.pt> <fixture.pt>"
            );
            ExitCode::from(2)
        }
    }
}

fn compare_forward(args: Vec<String>) -> ExitCode {
    compare_forward_with::<burn::backend::NdArray<f32>>(args, Default::default())
}

fn compare_forward_wgpu(args: Vec<String>) -> ExitCode {
    compare_forward_with::<burn::backend::Wgpu>(args, burn::backend::wgpu::WgpuDevice::default())
}

fn compare_forward_with<B: Backend>(args: Vec<String>, device: B::Device) -> ExitCode {
    let (Some(config_path), Some(weights_path), Some(fixture_path)) =
        (args.first(), args.get(1), args.get(2))
    else {
        eprintln!(
            "Usage: rusty_uni_pall compare-forward <config.json> <ema_named.pt> <fixture.pt>"
        );
        return ExitCode::from(2);
    };

    let result = fs::read_to_string(config_path)
        .map_err(anyhow_io)
        .and_then(|contents| {
            serde_json::from_str::<unimate::UniMateConfig>(&contents).map_err(|e| e.to_string())
        })
        .and_then(|config| {
            let mut model = model::UniMateDenoiser::<B>::from_config(&config, &device)?;
            model.load_ema_weights(weights_path)?;
            let reader = PytorchReader::new(fixture_path)
                .map_err(|error| format!("could not open forward fixture: {error}"))?;
            let motion = fixture_f32::<4, B>(&reader, "motion", &device)?;
            let timesteps = fixture_f32::<1, B>(&reader, "timesteps", &device)?;
            let caption = fixture_f32::<2, B>(&reader, "caption_embedding", &device)?;
            let tpos = fixture_f32::<3, B>(&reader, "tpos_first_frame", &device)?;
            let parents = fixture_f32::<3, B>(&reader, "tpos_first_frame_parents", &device)?;
            let n_joints = fixture_int::<1, B>(&reader, "n_joints", &device)?;
            let motion_lengths = fixture_int::<1, B>(&reader, "motion_lengths", &device)?;
            let joint_names = fixture_f32::<3, B>(&reader, "joint_names_emb", &device)?;
            let joint_depths = fixture_int::<2, B>(&reader, "joint_depths", &device)?;
            let graph_dist = fixture_int::<3, B>(&reader, "graph_dist", &device)?;
            let joint_relations = fixture_int::<3, B>(&reader, "joint_relations", &device)?;
            let spectral_coords = fixture_f32::<3, B>(&reader, "spectral_coords", &device)?;
            let expected = fixture_f32::<4, B>(&reader, "expected", &device)?;
            let (actual, traces) = model.forward_with_trace(
                motion,
                timesteps,
                caption,
                tpos,
                parents,
                n_joints,
                motion_lengths,
                joint_names,
                joint_depths,
                graph_dist,
                joint_relations,
                spectral_coords,
            );
            let mut trace_metrics = Vec::with_capacity(traces.len());
            for (index, actual_trace) in traces.into_iter().enumerate() {
                let name = format!("trace_{index}");
                let expected_trace = fixture_f32::<4, B>(&reader, &name, &device)?;
                if actual_trace.dims() != expected_trace.dims() {
                    return Err(format!("shape mismatch for {name}"));
                }
                trace_metrics.push((name, tensor_metrics(expected_trace, actual_trace)?));
            }
            let output_metrics = tensor_metrics(expected, actual)?;
            Ok((trace_metrics, output_metrics))
        });

    match result {
        Ok((traces, (max_absolute, max_relative, rmse))) => {
            for (name, (max_absolute, max_relative, rmse)) in traces {
                println!(
                    "{name}: max_abs={max_absolute:.8e}, max_rel={max_relative:.8e}, rmse={rmse:.8e}"
                );
            }
            println!(
                "output: max_abs={max_absolute:.8e}, max_rel={max_relative:.8e}, rmse={rmse:.8e}"
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Could not compare forward pass: {error}");
            ExitCode::FAILURE
        }
    }
}

fn tensor_metrics<B: burn::tensor::backend::Backend>(
    expected: Tensor<B, 4>,
    actual: Tensor<B, 4>,
) -> Result<(f32, f32, f64), String> {
    let expected_values = expected
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| format!("could not read PyTorch output: {error}"))?;
    let actual_values = actual
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| format!("could not read Burn output: {error}"))?;
    if expected_values.len() != actual_values.len() {
        return Err(format!(
            "output size mismatch: PyTorch {}, Burn {}",
            expected_values.len(),
            actual_values.len()
        ));
    }
    let mut max_absolute = 0.0_f32;
    let mut max_relative = 0.0_f32;
    let mut sum_squared = 0.0_f64;
    for (&reference, &candidate) in expected_values.iter().zip(&actual_values) {
        if !reference.is_finite() || !candidate.is_finite() {
            return Err("forward output contains a non-finite value".into());
        }
        let error = (reference - candidate).abs();
        max_absolute = max_absolute.max(error);
        max_relative = max_relative.max(error / reference.abs().max(1e-8));
        sum_squared += (error as f64) * (error as f64);
    }
    let rmse = (sum_squared / expected_values.len() as f64).sqrt();
    Ok((max_absolute, max_relative, rmse))
}

fn fixture_f32<const D: usize, B: burn::tensor::backend::Backend>(
    reader: &PytorchReader,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, D>, String> {
    let snapshot = reader
        .get(name)
        .ok_or_else(|| format!("fixture is missing tensor `{name}`"))?;
    let shape: [usize; D] = snapshot
        .shape
        .clone()
        .try_into()
        .map_err(|_| format!("fixture tensor `{name}` has the wrong rank"))?;
    let values = snapshot
        .to_data()
        .map_err(|error| error.to_string())?
        .to_vec::<f32>()
        .map_err(|error| error.to_string())?;
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

fn fixture_int<const D: usize, B: burn::tensor::backend::Backend>(
    reader: &PytorchReader,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, D, Int>, String> {
    let snapshot = reader
        .get(name)
        .ok_or_else(|| format!("fixture is missing tensor `{name}`"))?;
    let shape: [usize; D] = snapshot
        .shape
        .clone()
        .try_into()
        .map_err(|_| format!("fixture tensor `{name}` has the wrong rank"))?;
    let values = snapshot
        .to_data()
        .map_err(|error| error.to_string())?
        .to_vec::<i64>()
        .map_err(|error| error.to_string())?;
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

fn inspect_burn_weight_keys(args: Vec<String>) -> ExitCode {
    let Some(weights_path) = args.first() else {
        eprintln!("Usage: rusty_uni_pall inspect-burn-weights <ema_named.pt>");
        return ExitCode::from(2);
    };
    match model::UniMateDenoiser::<burn::backend::NdArray<f32>>::remapped_ema_weight_keys(
        weights_path,
    ) {
        Ok(mut keys) => {
            keys.sort();
            for key in keys {
                println!("{key}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Could not inspect remapped weights: {error}");
            ExitCode::FAILURE
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
