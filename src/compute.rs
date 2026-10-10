//! Runtime backend selection. Probe in a child so a missing CUDA library or
//! driver panic cannot poison the inference process or prevent WGPU fallback.
use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use std::{
    process::{Command, ExitCode},
    sync::OnceLock,
};

pub const PROBE_ARG: &str = "--internal-backend-probe";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compute {
    Cuda,
    Wgpu,
}
impl Compute {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
        }
    }
}

pub fn selected() -> Result<Compute> {
    static SELECTED: OnceLock<Result<Compute, String>> = OnceLock::new();
    SELECTED
        .get_or_init(|| {
            let requested = std::env::var("RUSTI_PALL_BACKEND").unwrap_or_else(|_| "auto".into());
            choose(&requested, cfg!(feature = "cuda"), probe_process)
                .map(|backend| {
                    eprintln!("Compute backend: {}", backend.name());
                    backend
                })
                .map_err(|e| format!("{e:#}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

fn choose(
    requested: &str,
    cuda_compiled: bool,
    mut probe: impl FnMut(Compute) -> Result<()>,
) -> Result<Compute> {
    match requested.to_ascii_lowercase().as_str() {
        "auto" => {
            if cuda_compiled {
                match probe(Compute::Cuda) {
                    Ok(()) => return Ok(Compute::Cuda),
                    Err(e) => eprintln!("CUDA unavailable; trying WGPU: {e:#}"),
                }
            }
            probe(Compute::Wgpu).context("WGPU also unavailable; check GPU access and drivers")?;
            Ok(Compute::Wgpu)
        }
        "cuda" => {
            ensure!(
                cuda_compiled,
                "CUDA was not compiled in; rebuild without --no-default-features"
            );
            probe(Compute::Cuda)
                .context("CUDA was explicitly requested but its device test failed")?;
            Ok(Compute::Cuda)
        }
        "wgpu" => {
            probe(Compute::Wgpu)?;
            Ok(Compute::Wgpu)
        }
        _ => bail!("RUSTI_PALL_BACKEND must be auto, cuda, or wgpu; got {requested:?}"),
    }
}

fn probe_process(backend: Compute) -> Result<()> {
    let output = Command::new(std::env::current_exe()?)
        .args([PROBE_ARG, backend.name()])
        .output()
        .context("Could not start backend device test")?;
    ensure!(
        output.status.success(),
        "{} probe failed ({}): {}",
        backend.name(),
        output.status,
        String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(1800)
            .collect::<String>()
            .trim()
    );
    ensure!(
        String::from_utf8_lossy(&output.stdout).trim() == "backend-probe-ok",
        "Unexpected backend probe response"
    );
    Ok(())
}

/// Called at the entry point of both executables, before normal CLI dispatch.
pub fn handle_probe() -> Option<ExitCode> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some(PROBE_ARG) {
        return None;
    }
    let result = match args.next().as_deref() {
        #[cfg(feature = "cuda")]
        Some("cuda") => device_test::<burn::backend::Cuda>(&Default::default()),
        Some("wgpu") => device_test::<burn::backend::Wgpu>(&Default::default()),
        _ => Err(anyhow::anyhow!("Backend not compiled or unknown")),
    };
    Some(match result {
        Ok(()) => {
            println!("backend-probe-ok");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e:#}");
            ExitCode::FAILURE
        }
    })
}

fn device_test<B: Backend>(device: &B::Device) -> Result<()> {
    // Exercise device initialization, allocation, kernel compilation, matmul,
    // elementwise arithmetic and readback, not just driver enumeration.
    let a = Tensor::<B, 2>::from_data(TensorData::new(vec![1f32, 2., 3., 4.], [2, 2]), device);
    let values = (a.clone().matmul(a) + 1.)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    ensure!(
        values == [8., 11., 16., 23.],
        "GPU arithmetic test failed: {values:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefers_cuda_and_falls_back_only_in_auto_mode() {
        let mut calls = vec![];
        assert_eq!(
            choose("auto", true, |b| {
                calls.push(b);
                Ok(())
            })
            .unwrap(),
            Compute::Cuda
        );
        assert_eq!(calls, vec![Compute::Cuda]);
        let mut calls = vec![];
        assert_eq!(
            choose("auto", true, |b| {
                calls.push(b);
                if b == Compute::Cuda {
                    bail!("no driver")
                }
                Ok(())
            })
            .unwrap(),
            Compute::Wgpu
        );
        assert_eq!(calls, vec![Compute::Cuda, Compute::Wgpu]);
        assert!(choose("cuda", true, |_| bail!("no driver")).is_err());
        assert!(choose("auto", true, |_| bail!("no device")).is_err());
    }
    #[test]
    fn explicit_wgpu_and_build_without_cuda_do_not_probe_cuda() {
        for (request, compiled) in [("wgpu", true), ("auto", false)] {
            assert_eq!(
                choose(request, compiled, |b| {
                    assert_eq!(b, Compute::Wgpu);
                    Ok(())
                })
                .unwrap(),
                Compute::Wgpu
            );
        }
        assert!(choose("cuda", false, |_| panic!("must not probe")).is_err());
        assert!(choose("hip", true, |_| panic!("must not probe")).is_err());
    }
}
