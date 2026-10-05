//! Configuration schema for the released UniMate graph/adaLN inference model.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct UniMateConfig {
    pub experiment: ExperimentConfig,
    pub dataset: DatasetConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub sampling: SamplingConfig,
}

#[derive(Debug, Deserialize)]
pub struct ExperimentConfig {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct DatasetConfig {
    pub max_motion_length: usize,
    pub max_joints: usize,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub attention: String,
    pub text_cond: String,
    pub latent_dim: usize,
    pub ff_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub concat_parent_features: bool,
    pub use_graph_emb: bool,
    pub use_depth_emb: bool,
    pub use_spectral_rope: bool,
    pub max_freqs: usize,
    pub use_signnet: bool,
    pub inject_tpos_to_adaln: bool,
    pub num_tpos_queries: usize,
}

#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    pub diff_model: String,
    pub use_ema: bool,
}

#[derive(Debug, Deserialize)]
pub struct SamplingConfig {
    pub cfg_scale: f32,
}

impl UniMateConfig {
    /// Reject configurations that require a different architecture than this port.
    pub fn validate(&self) -> Result<(), &'static str> {
        let model = &self.model;
        if model.attention != "graph" || model.text_cond != "adaln" {
            return Err("this port currently targets the graph attention + adaLN model");
        }
        if self.training.diff_model != "flow" {
            return Err("this port currently supports flow inference only");
        }
        if model.num_heads == 0 || model.latent_dim % model.num_heads != 0 {
            return Err("latent_dim must be divisible by a nonzero num_heads");
        }
        if !model.use_spectral_rope
            || !model.use_signnet
            || !model.use_depth_emb
            || !model.concat_parent_features
        {
            return Err("config does not match the released graph/adaLN architecture being ported");
        }
        if model.use_graph_emb {
            return Err("GCN graph embeddings are outside the initial inference target");
        }
        if !self.training.use_ema {
            return Err("inference expects EMA checkpoint weights");
        }
        if self.dataset.max_motion_length == 0
            || self.dataset.max_joints == 0
            || model.num_layers == 0
        {
            return Err("frame, joint, and layer limits must be nonzero");
        }
        Ok(())
    }
}
