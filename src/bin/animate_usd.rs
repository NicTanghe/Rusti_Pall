//! A native Rust USDZ → UniMate → animated USD runner. No Python processes.
#[path = "animate_usd/tasks.rs"]
mod tasks;
use anyhow::{Context, Result, ensure};
use rusty_pall::{
    normalization::NormalizationStats,
    rig_motion::{Prepared, extract_package, prepare},
    text_encoder,
    unimate::UniMateConfig,
    usd_rig::{Annotation, Rig},
};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn model_dir() -> PathBuf {
    std::env::var_os("RUSTI_PALL_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("weights/unimate_uniml3d_f60_v3"))
}
fn text_dir() -> PathBuf {
    std::env::var_os("RUSTI_PALL_TEXT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            root()
                .join("weights/flan-t5-base")
                .join(text_encoder::REVISION)
        })
}
fn config() -> Result<UniMateConfig> {
    let c: UniMateConfig = serde_json::from_slice(&fs::read(model_dir().join("config.json"))?)?;
    c.validate().map_err(anyhow::Error::msg)?;
    Ok(c)
}
fn stats() -> Result<NormalizationStats> {
    NormalizationStats::from_json(model_dir().join("dataset_stats.json"))
        .map_err(anyhow::Error::msg)
}
fn write_json(path: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn prepare_run(input: &str, labels: &str, out: &str, prompt: &str) -> Result<()> {
    let input = Path::new(input);
    let output = Path::new(out);
    ensure!(
        input.extension().and_then(|s| s.to_str()) == Some("usdz"),
        "The animation runner currently needs a self-contained .usdz package; inspect also accepts plain USD"
    );
    ensure!(
        !output.exists(),
        "Output directory already exists; use 'sample {out}' to resume or choose another directory"
    );
    let rig = Rig::open(input)?;
    let annotation: Annotation = serde_json::from_slice(&fs::read(labels)?)?;
    let mut prepared = prepare(&rig, &annotation, input, prompt, &config()?, &stats()?)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(output)?;
    let extension = input
        .extension()
        .and_then(|e| e.to_str())
        .context("Input needs a USD extension")?;
    let source = output.join(format!("source.{extension}"));
    fs::copy(input, &source)?;
    prepared.input = source.canonicalize()?.to_string_lossy().into();
    let package = output.join("package");
    fs::create_dir(&package)?;
    extract_package(&source, &package)?;
    write_json(output.join("rig.json"), &rig.report())?;
    write_json(output.join("labels.json"), &annotation)?;
    write_json(output.join("conditioning.json"), &prepared)?;
    let mut rest = vec![0f32; prepared.width * 12 * prepared.frames];
    for j in 0..prepared.joints.len() {
        for t in 0..prepared.frames {
            for d in 0..3 {
                rest[(j * 12 + d) * prepared.frames + t] = prepared.positions[j][d] as f32;
            }
            rest[(j * 12 + 3) * prepared.frames + t] = 1.;
            rest[(j * 12 + 7) * prepared.frames + t] = 1.;
        }
    }
    prepared.export(&rig, &rest, &package.join("rest-pose.usda"))?;
    rusty_pall::rig_motion::validate_export(
        &rig,
        &package.join("rest-pose.usda"),
        prepared.frames,
        true,
    )?;
    preview(&prepared, &output.join("skeleton.svg"))?;
    eprintln!(
        "Prepared {} joints in {}. Inspect skeleton.svg and package/rest-pose.usda.",
        prepared.joints.len(),
        output.display()
    );
    Ok(())
}

fn sample(out: &str, seed: u64, cfg: f32) -> Result<()> {
    let output = Path::new(out);
    ensure!(
        !output.join("package/animation.usda").exists(),
        "Animation already exists; choose a new run directory"
    );
    let path = output.join("conditioning.json");
    let mut prepared: Prepared = serde_json::from_slice(&fs::read(&path)?)?;
    let rig = Rig::open(Path::new(&prepared.input))?;
    if prepared.caption.is_empty() {
        eprintln!("Encoding prompt and joint labels with native CPU FLAN-T5...");
        prepared
            .embed(&text_dir())
            .context("Text encoding failed. Run 'animate_usd fetch-text' if weights are missing")?;
        write_json(&path, &prepared)?;
    }
    let start = std::time::Instant::now();
    eprintln!(
        "Sampling {} frames on discrete WGPU adapter 0...",
        prepared.frames
    );
    let values = prepared.sample(
        &config()?,
        &model_dir().join("ema_named.pt"),
        &stats()?,
        seed,
        cfg,
    )?;
    write_json(
        output.join("motion.json"),
        &serde_json::json!({"shape":[1,prepared.width,12,prepared.frames],"values":values,"seed":seed,"cfg":cfg}),
    )?;
    prepared.export(&rig, &values, &output.join("package/animation.usda"))?;
    package(out, false)?;
    write_json(
        output.join("run.json"),
        &serde_json::json!({"seed":seed,"cfg":cfg,"sampling_export_seconds":start.elapsed().as_secs_f64(),"prompt":prepared.prompt}),
    )?;
    eprintln!(
        "Saved {} ({:.1}s)",
        output.join("package/animation.usda").display(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn reexport(out: &str) -> Result<()> {
    let output = Path::new(out);
    let prepared: Prepared = serde_json::from_slice(&fs::read(output.join("conditioning.json"))?)?;
    #[derive(serde::Deserialize)]
    struct Motion {
        shape: [usize; 4],
        values: Vec<f32>,
    }
    let motion: Motion = serde_json::from_slice(&fs::read(output.join("motion.json"))?)?;
    ensure!(
        motion.shape == [1, prepared.width, 12, prepared.frames],
        "Saved motion shape differs from conditioning"
    );
    let rig = Rig::open(Path::new(&prepared.input))?;
    prepared.export(&rig, &motion.values, &output.join("package/animation.usda"))?;
    package(out, true)
}

fn package(out: &str, replace: bool) -> Result<()> {
    let output = Path::new(out);
    let prepared: Prepared = serde_json::from_slice(&fs::read(output.join("conditioning.json"))?)?;
    let destination = output.join("animation.usdz");
    ensure!(
        replace || !destination.exists(),
        "Animation package already exists"
    );
    let source = Path::new(&prepared.input);
    let rig = Rig::open(source)?;
    let mut original = zip::ZipArchive::new(fs::File::open(source)?)?;
    // Use a distinct root name without overwriting any source asset.
    let names: std::collections::BTreeSet<String> =
        original.file_names().map(str::to_owned).collect();
    let mut root_name = "unimate-animation.usda".to_owned();
    while names.contains(&root_name) {
        root_name.insert_str(0, "_");
    }
    let temporary = output.join("animation.usdz.part");
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let mut archive = openusd::usdz::ArchiveWriter::new(file);
    archive.add_layer(
        &root_name,
        &fs::read(output.join("package/animation.usda"))?,
    )?;
    for i in 0..original.len() {
        let mut entry = original.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut bytes)?;
        archive.add_layer(entry.name(), &bytes)?;
    }
    archive.finish()?;
    // The resolver selects archive handling by extension.
    fs::rename(&temporary, &destination)?;
    rusty_pall::rig_motion::validate_export(&rig, &destination, prepared.frames, false)?;
    eprintln!("Packaged {}", destination.display());
    Ok(())
}

fn preview(p: &Prepared, path: &Path) -> Result<()> {
    let mut s = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"850\" viewBox=\"0 0 1200 850\"><rect width=\"1200\" height=\"850\" fill=\"white\"/>",
    );
    let esc = |v: &str| {
        v.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    };
    for (panel, (a, b, title)) in [
        (2, 1, "Side: +Z forward →, +Y up"),
        (0, 2, "Top: +X →, +Z up"),
    ]
    .into_iter()
    .enumerate()
    {
        let x0 = 300. + panel as f64 * 600.;
        let y0 = 320.;
        s += &format!(
            "<text x=\"{}\" y=\"25\" font-family=\"sans-serif\">{title}</text>",
            panel * 600 + 20
        );
        let xy = |i: usize| (x0 + p.positions[i][a] * 240., y0 - p.positions[i][b] * 240.);
        for (i, parent) in p.parents.iter().enumerate() {
            if let Some(parent) = parent {
                let (x, y) = xy(i);
                let (px, py) = xy(*parent);
                s += &format!(
                    "<line x1=\"{x}\" y1=\"{y}\" x2=\"{px}\" y2=\"{py}\" stroke=\"#486b9c\"/>"
                );
            }
        }
        for i in 0..p.joints.len() {
            let (x, y) = xy(i);
            s += &format!(
                "<circle cx=\"{x}\" cy=\"{y}\" r=\"3\" fill=\"#be4336\"/><text x=\"{}\" y=\"{}\" font-size=\"9\">{}</text>",
                x + 3.,
                y - 3.,
                i
            );
        }
    }
    for i in 0..p.joints.len() {
        let x = 20 + (i / 21) * 395;
        let y = 570 + (i % 21) * 13;
        let name = p.joints[i].rsplit('/').next().unwrap();
        s += &format!(
            "<text x=\"{x}\" y=\"{y}\" font-size=\"11\" font-family=\"sans-serif\">{i}: {} — {}</text>",
            esc(name),
            esc(&p.labels[i])
        );
    }
    s += "</svg>";
    fs::write(path, s)?;
    Ok(())
}

fn run() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.iter().any(|v| v == "-h" || v == "--help") {
        println!("{}", tasks::HELP);
        return Ok(());
    }
    match a.first().map(String::as_str) {
        Some("inbetween" | "edit" | "expand") => tasks::run(&a)?,
        Some("inspect") if a.len() == 2 => {
            let rig = Rig::open(Path::new(&a[1]))?;
            println!("{}", serde_json::to_string_pretty(&rig.report())?);
        }
        Some("fetch-text") if a.len() == 1 => text_encoder::fetch(&text_dir())?,
        Some("package") if a.len() == 2 => package(&a[1], false)?,
        Some("reexport") if a.len() == 2 => reexport(&a[1])?,
        Some("prepare" | "run") if a.len() == 5 => {
            prepare_run(&a[1], &a[2], &a[3], &a[4])?;
            if a[0] == "run" {
                sample(&a[3], 10, 3.)?;
            }
        }
        Some("sample") if (2..=4).contains(&a.len()) => sample(
            &a[1],
            a.get(2).map(|s| s.parse()).transpose()?.unwrap_or(10),
            a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(3.),
        )?,
        _ => anyhow::bail!(
            "Usage:\n  animate_usd inspect <rig.usdz>\n  animate_usd fetch-text\n  animate_usd package <output-dir>\n  animate_usd reexport <output-dir>\n  animate_usd prepare <rig.usdz> <labels.json> <new-output-dir> <prompt>\n  animate_usd sample <output-dir> [seed=10] [cfg=3]\n  animate_usd run <rig.usdz> <labels.json> <new-output-dir> <prompt>\nModel paths can be overridden with RUSTI_PALL_MODEL_DIR and RUSTI_PALL_TEXT_DIR."
        ),
    }
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
