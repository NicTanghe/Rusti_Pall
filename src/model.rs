//! Burn implementations of the small reusable UniMate network layers.
//!
//! These module field names intentionally mirror the upstream PyTorch names so
//! the converted EMA state dict can be loaded without opaque positional maps.

use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{
        DType, Tensor, TensorData,
        activation::{gelu, silu, softmax},
        backend::Backend,
    },
};
use burn_store::{ApplyResult, ModuleSnapshot, PytorchStore};
use std::path::Path;

#[derive(Module, Debug)]
pub struct LlamaRmsNorm<B: Backend> {
    /// Matches the upstream `weight` parameter name.
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> LlamaRmsNorm<B> {
    pub fn new(width: usize, device: &B::Device) -> Self {
        Self {
            weight: burn::module::Initializer::Ones.init([width], device),
            eps: 1e-6,
        }
    }

    /// RMS normalization over the final feature dimension, matching UniMate's
    /// LlamaRMSNorm (including FP32 variance accumulation).
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = input.dtype();
        let rms = (input.clone().cast(DType::F32).square().mean_dim(D - 1) + self.eps).sqrt();
        (input / rms.cast(dtype)) * self.weight.val().unsqueeze()
    }
}

#[derive(Module, Debug)]
pub struct SwiGLUFFN<B: Backend> {
    /// Upstream `w12`: a single projection split into gate and value halves.
    pub w12: Linear<B>,
    /// Upstream `w3`: projection back to the model width.
    pub w3: Linear<B>,
}

impl<B: Backend> SwiGLUFFN<B> {
    pub fn new(width: usize, hidden_width: usize, device: &B::Device) -> Self {
        Self {
            w12: LinearConfig::new(width, hidden_width * 2).init(device),
            w3: LinearConfig::new(hidden_width, width).init(device),
        }
    }

    /// Computes `w3(silu(gate) * value)` for an arbitrary leading shape.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let halves = self.w12.forward(input).chunk(2, D - 1);
        let gate = halves[0].clone();
        let value = halves[1].clone();
        self.w3.forward(silu(gate) * value)
    }
}

/// Two-linear SiLU MLP used by UniMate's sequential embedders.
#[derive(Module, Debug)]
pub struct SiluMlp<B: Backend> {
    pub linear_0: Linear<B>,
    pub linear_2: Linear<B>,
}

impl<B: Backend> SiluMlp<B> {
    pub fn new(input: usize, hidden: usize, output: usize, device: &B::Device) -> Self {
        Self {
            linear_0: LinearConfig::new(input, hidden).init(device),
            linear_2: LinearConfig::new(hidden, output).init(device),
        }
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.linear_2.forward(silu(self.linear_0.forward(input)))
    }
}

/// Sinusoidal flow-time embedding followed by UniMate's learned MLP.
#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    pub mlp_0: Linear<B>,
    pub mlp_2: Linear<B>,
    hidden_size: usize,
    frequency_size: usize,
}

impl<B: Backend> TimestepEmbedder<B> {
    pub fn new(hidden_size: usize, device: &B::Device) -> Self {
        let frequency_size = 256;
        Self {
            mlp_0: LinearConfig::new(frequency_size, hidden_size).init(device),
            mlp_2: LinearConfig::new(hidden_size, hidden_size).init(device),
            hidden_size,
            frequency_size,
        }
    }

    /// `timesteps` is `(B,)`; output is `(B, hidden_size)`.
    pub fn forward(&self, timesteps: Tensor<B, 1>) -> Tensor<B, 2> {
        let half = self.frequency_size / 2;
        let frequencies: Vec<f32> = (0..half)
            .map(|index| (-10000.0_f64.ln() * index as f64 / half as f64).exp() as f32)
            .collect();
        let frequency =
            Tensor::<B, 1>::from_data(TensorData::new(frequencies, [half]), &timesteps.device())
                .reshape([1, half]);
        let args = timesteps.unsqueeze_dim::<2>(1) * frequency;
        let embedding = Tensor::cat(vec![args.clone().cos(), args.sin()], 1);
        let hidden = silu(self.mlp_0.forward(embedding));
        let output = self.mlp_2.forward(hidden);
        debug_assert_eq!(output.dims()[1], self.hidden_size);
        output
    }
}

/// Root/joint-separated motion and first-pose encoder.
#[derive(Module, Debug)]
pub struct InputLayer<B: Backend> {
    pub root_tpos_embedder: SiluMlp<B>,
    pub root_x_embedder: SiluMlp<B>,
    pub joint_tpos_embedder: SiluMlp<B>,
    pub parent_tpos_embedder: SiluMlp<B>,
    pub tpos_fuse: SiluMlp<B>,
    pub joint_x_embedder: SiluMlp<B>,
    max_joints: usize,
    feature_len: usize,
    latent_dim: usize,
}

impl<B: Backend> InputLayer<B> {
    pub fn new(
        feature_len: usize,
        latent_dim: usize,
        max_joints: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            root_tpos_embedder: SiluMlp::new(feature_len, latent_dim, latent_dim, device),
            root_x_embedder: SiluMlp::new(feature_len, latent_dim, latent_dim, device),
            joint_tpos_embedder: SiluMlp::new(feature_len, latent_dim, latent_dim, device),
            parent_tpos_embedder: SiluMlp::new(feature_len, latent_dim, latent_dim, device),
            tpos_fuse: SiluMlp::new(latent_dim * 2, latent_dim, latent_dim, device),
            joint_x_embedder: SiluMlp::new(feature_len, latent_dim, latent_dim, device),
            max_joints,
            feature_len,
            latent_dim,
        }
    }

    /// Encodes motion `[B,J,features,F]`, t-pose `[B,J,features]`, parent
    /// t-pose features, and joint counts. Returns embedded tokens `[B,F+1,J,D]`,
    /// pooled-source t-pose tokens `[B,J,D]`, and the valid-joint mask `[B,J]`.
    pub fn forward(
        &self,
        motion: Tensor<B, 4>,
        tpos_first_frame: Tensor<B, 3>,
        tpos_first_frame_parents: Tensor<B, 3>,
        n_joints: Tensor<B, 1, burn::tensor::Int>,
    ) -> (Tensor<B, 4>, Tensor<B, 3>, Tensor<B, 2, burn::tensor::Bool>) {
        let [batch, joints, feature_len, _frames] = motion.dims();
        assert_eq!(joints, self.max_joints);
        assert_eq!(feature_len, self.feature_len);
        assert_eq!(tpos_first_frame.dims(), [batch, joints, feature_len]);

        let tpose = tpos_first_frame.unsqueeze_dim::<4>(1);
        let root_tpos = self
            .root_tpos_embedder
            .forward(tpose.clone().narrow(2, 0, 1));
        let joint_tpos = self
            .joint_tpos_embedder
            .forward(tpose.clone().narrow(2, 1, joints - 1));
        let parents = tpos_first_frame_parents.unsqueeze_dim::<4>(1);
        let parent_tpos = self
            .parent_tpos_embedder
            .forward(parents.narrow(2, 1, joints - 1));
        let joint_tpos = self
            .tpos_fuse
            .forward(Tensor::cat(vec![joint_tpos, parent_tpos], 3));
        let tpos_tokens = Tensor::cat(vec![root_tpos, joint_tpos], 2);
        let tpos_tokens_3d = tpos_tokens
            .clone()
            .reshape([batch, joints, self.latent_dim]);

        let joint_positions =
            Tensor::<B, 1, burn::tensor::Int>::arange(0..self.max_joints as i64, &motion.device())
                .reshape([1, self.max_joints]);
        let valid_joints = joint_positions.lower(n_joints.reshape([batch, 1]));

        let motion = motion.permute([0, 3, 1, 2]);
        let root_motion = self.root_x_embedder.forward(motion.clone().narrow(2, 0, 1));
        let joint_motion = self
            .joint_x_embedder
            .forward(motion.narrow(2, 1, joints - 1));
        let motion_tokens = Tensor::cat(vec![root_motion, joint_motion], 2);
        let tokens = Tensor::cat(vec![tpos_tokens, motion_tokens], 1);
        (tokens, tpos_tokens_3d, valid_joints)
    }
}

/// Query-based t-pose pooling that injects rig conditioning into adaLN.
#[derive(Module, Debug)]
pub struct TposCrossAttentionPool<B: Backend> {
    pub queries: Param<Tensor<B, 3>>,
    pub norm_q: LlamaRmsNorm<B>,
    pub norm_kv: LlamaRmsNorm<B>,
    pub cross_attn: PackedCrossAttention<B>,
    pub norm_ff: LlamaRmsNorm<B>,
    pub ffn: SwiGLUFFN<B>,
    pub norm_out: LlamaRmsNorm<B>,
    pub out_proj: Linear<B>,
    num_queries: usize,
    width: usize,
}

impl<B: Backend> TposCrossAttentionPool<B> {
    pub fn new(width: usize, num_queries: usize, device: &B::Device) -> Self {
        let queries = burn::module::Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        }
        .init([1, num_queries, width], device);
        Self {
            queries,
            norm_q: LlamaRmsNorm::new(width, device),
            norm_kv: LlamaRmsNorm::new(width, device),
            cross_attn: PackedCrossAttention::new(width, 4, device),
            norm_ff: LlamaRmsNorm::new(width, device),
            ffn: SwiGLUFFN::new(width, width * 2, device),
            norm_out: LlamaRmsNorm::new(width, device),
            out_proj: LinearConfig::new(width, width).init(device),
            num_queries,
            width,
        }
    }

    pub fn forward(
        &self,
        tpos_emb: Tensor<B, 3>,
        valid_joints: Tensor<B, 2, burn::tensor::Bool>,
    ) -> Tensor<B, 2> {
        let batch = tpos_emb.dims()[0];
        let query = self.queries.val().repeat_dim(0, batch);
        let residual = query.clone();
        let query = self.norm_q.forward(query);
        let key_value = self.norm_kv.forward(tpos_emb);
        let query = query.reshape([batch, self.num_queries, self.width]);
        let pooled = residual.reshape([batch, self.num_queries, self.width])
            + self.cross_attn.forward(query, key_value, valid_joints);
        let pooled = pooled.clone() + self.ffn.forward(self.norm_ff.forward(pooled));
        let pooled = pooled.mean_dim(1).reshape([batch, self.width]);
        self.out_proj.forward(self.norm_out.forward(pooled))
    }
}

/// Module wrapper matching upstream `tpos_pool.pool` parameter prefixes.
#[derive(Module, Debug)]
pub struct TposPool<B: Backend> {
    pub pool: TposCrossAttentionPool<B>,
}

impl<B: Backend> TposPool<B> {
    pub fn new(width: usize, num_queries: usize, device: &B::Device) -> Self {
        Self {
            pool: TposCrossAttentionPool::new(width, num_queries, device),
        }
    }

    pub fn forward(
        &self,
        tpos_emb: Tensor<B, 3>,
        valid_joints: Tensor<B, 2, burn::tensor::Bool>,
    ) -> Tensor<B, 2> {
        self.pool.forward(tpos_emb, valid_joints)
    }
}

/// The released checkpoint adds these per-joint embeddings to every frame.
#[derive(Module, Debug)]
pub struct TokenEmbeddings<B: Backend> {
    pub joint_name_embedder: Linear<B>,
    pub depth_embedding: Embedding<B>,
    max_depth: usize,
}

impl<B: Backend> TokenEmbeddings<B> {
    pub fn new(text_dim: usize, width: usize, max_depth: usize, device: &B::Device) -> Self {
        Self {
            joint_name_embedder: LinearConfig::new(text_dim, width).init(device),
            depth_embedding: EmbeddingConfig::new(max_depth + 1, width).init(device),
            max_depth,
        }
    }

    pub fn forward(
        &self,
        tokens: Tensor<B, 4>,
        joint_names_emb: Tensor<B, 3>,
        joint_depths: Tensor<B, 2, burn::tensor::Int>,
    ) -> Tensor<B, 4> {
        let frames = tokens.dims()[1];
        let depths = joint_depths;
        let depth_mask = depths.clone().greater_elem(self.max_depth as i64);
        let depths = depths.mask_fill(depth_mask, self.max_depth as i64);
        let depth = self.depth_embedding.forward(depths).unsqueeze_dim::<4>(1);
        let names = self
            .joint_name_embedder
            .forward(joint_names_emb)
            .unsqueeze_dim::<4>(1);
        tokens + depth.repeat_dim(1, frames) + names.repeat_dim(1, frames)
    }
}

/// Complete released graph/adaLN flow denoiser. Text encoder output and
/// canonical skeleton features are inputs; this struct performs neural
/// inference only and leaves ODE integration to `sampler`.
#[derive(Module, Debug)]
pub struct UniMateDenoiser<B: Backend> {
    pub time_embedder: TimestepEmbedder<B>,
    pub cond_embedder: Linear<B>,
    pub input_layer: InputLayer<B>,
    pub tpos_pool: TposPool<B>,
    pub token_embeddings: TokenEmbeddings<B>,
    pub transformer_blocks: Vec<SpatioTemporalBlock<B>>,
    pub final_layer: FinalLayer<B>,
    max_motion_length: usize,
    max_joints: usize,
    latent_dim: usize,
    text_dim: usize,
    feature_len: usize,
}

impl<B: Backend> UniMateDenoiser<B> {
    pub fn from_config(
        config: &crate::unimate::UniMateConfig,
        device: &B::Device,
    ) -> Result<Self, String> {
        config.validate().map_err(str::to_owned)?;
        let model = &config.model;
        Ok(Self::new(
            config.dataset.feature_len,
            config.dataset.max_motion_length,
            config.dataset.max_joints,
            config.dataset.max_depth,
            model.latent_dim,
            model.ff_size,
            model.num_layers,
            model.num_heads,
            model.max_freqs,
            768, // google/flan-t5-base hidden width
            model.num_tpos_queries,
            device,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        feature_len: usize,
        max_motion_length: usize,
        max_joints: usize,
        max_depth: usize,
        latent_dim: usize,
        ff_size: usize,
        num_layers: usize,
        num_heads: usize,
        max_freqs: usize,
        text_dim: usize,
        num_tpos_queries: usize,
        device: &B::Device,
    ) -> Self {
        let max_frames = max_motion_length + 1;
        let transformer_blocks = (0..num_layers)
            .map(|_| {
                SpatioTemporalBlock::new(
                    latent_dim, ff_size, num_heads, max_freqs, max_frames, device,
                )
            })
            .collect();
        Self {
            time_embedder: TimestepEmbedder::new(latent_dim, device),
            cond_embedder: LinearConfig::new(text_dim, latent_dim).init(device),
            input_layer: InputLayer::new(feature_len, latent_dim, max_joints, device),
            tpos_pool: TposPool::new(latent_dim, num_tpos_queries, device),
            token_embeddings: TokenEmbeddings::new(text_dim, latent_dim, max_depth, device),
            transformer_blocks,
            final_layer: FinalLayer::new(latent_dim, feature_len, max_joints, device),
            max_motion_length,
            max_joints,
            latent_dim,
            text_dim,
            feature_len,
        }
    }

    /// Inputs use the same canonical forms as UniMate's graph/adaLN forward:
    /// motion `[B,J,D,F]`, prompt/joint embeddings `[B,(J),text_dim]`,
    /// topology matrices `[B,J,J]`, and per-joint spectral/depth features.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        motion: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        caption_embedding: Tensor<B, 2>,
        tpos_first_frame: Tensor<B, 3>,
        tpos_first_frame_parents: Tensor<B, 3>,
        n_joints: Tensor<B, 1, burn::tensor::Int>,
        motion_lengths: Tensor<B, 1, burn::tensor::Int>,
        joint_names_emb: Tensor<B, 3>,
        joint_depths: Tensor<B, 2, burn::tensor::Int>,
        graph_dist: Tensor<B, 3, burn::tensor::Int>,
        joint_relations: Tensor<B, 3, burn::tensor::Int>,
        spectral_coords: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch, joints, feature_len, frames] = motion.dims();
        assert_eq!(joints, self.max_joints);
        assert_eq!(feature_len, self.feature_len);
        assert_eq!(frames, self.max_motion_length);
        assert_eq!(caption_embedding.dims(), [batch, self.text_dim]);

        let (mut tokens, tpos_tokens, valid_joints) =
            self.input_layer
                .forward(motion, tpos_first_frame, tpos_first_frame_parents, n_joints);
        let condition =
            self.time_embedder.forward(timesteps) + self.cond_embedder.forward(caption_embedding);
        let condition = condition + self.tpos_pool.forward(tpos_tokens, valid_joints.clone());
        tokens = self
            .token_embeddings
            .forward(tokens, joint_names_emb, joint_depths);

        let positions =
            Tensor::<B, 1, burn::tensor::Int>::arange(0..(frames + 1) as i64, &tokens.device())
                .reshape([1, frames + 1]);
        let valid_frames = positions.lower(motion_lengths.reshape([batch, 1]) + 1);
        for block in &self.transformer_blocks {
            tokens = block.forward(
                tokens,
                condition.clone(),
                valid_frames.clone(),
                graph_dist.clone(),
                joint_relations.clone(),
                valid_joints.clone(),
                spectral_coords.clone(),
            );
        }
        self.final_layer
            .forward(tokens, condition, valid_joints)
            .narrow(2, 1, frames)
            .permute([0, 3, 1, 2])
    }

    /// Loads the named EMA state dictionary produced by
    /// `scripts/export_ema_state_dict.py`. Missing parameters are rejected by
    /// Burn's strict store; extra checkpoint parameters are rejected here.
    pub fn load_ema_weights(&mut self, path: impl AsRef<Path>) -> Result<ApplyResult, String> {
        let mut store = PytorchStore::from_file(path.as_ref().to_path_buf())
            .map_indices_contiguous(false)
            // Sequential PyTorch modules retain indices 0 and 2 around SiLU.
            .with_key_remapping(r"\.linear_0\.", ".0.")
            .with_key_remapping(r"\.linear_2\.", ".2.")
            .with_key_remapping(r"\.mlp_0\.", ".mlp.0.")
            .with_key_remapping(r"\.mlp_2\.", ".mlp.2.")
            .with_key_remapping(r"\.phi_0\.", ".phi.0.")
            .with_key_remapping(r"\.phi_2\.", ".phi.2.")
            .with_key_remapping(r"\.rho_0\.", ".rho.0.")
            .with_key_remapping(r"\.rho_2\.", ".rho.2.")
            .with_key_remapping(r"\.ada_ln_linear\.", ".adaLN_modulation.1.")
            .with_key_remapping(r"^token_embeddings\.", "");
        let result = self
            .load_from(&mut store)
            .map_err(|error| format!("could not load EMA weights: {error}"))?;
        if !result.unused.is_empty() {
            return Err(format!(
                "EMA file contains {} unconsumed tensors; refusing a partial or mismatched load:\n{}",
                result.unused.len(),
                result.unused.join("\n")
            ));
        }
        Ok(result)
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    /// Matches the upstream block's `mlp` field.
    pub mlp: SwiGLUFFN<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(width: usize, ff_size: usize, device: &B::Device) -> Self {
        Self {
            // This is the exact hidden-width rule in SpatioTemporalBlock.
            mlp: SwiGLUFFN::new(width, (2 * ff_size) / 3, device),
        }
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.mlp.forward(input)
    }
}

/// Per-layer Graphormer bias used by UniMate's spatial attention.
#[derive(Module, Debug)]
pub struct GraphAttnBias<B: Backend> {
    pub graph_dist_embedding: Embedding<B>,
    pub graph_dist_proj: Linear<B>,
    pub graph_dist_scale: Param<Tensor<B, 1>>,
    pub graph_rel_embedding: Embedding<B>,
    pub graph_rel_proj: Linear<B>,
    pub graph_rel_scale: Param<Tensor<B, 1>>,
    graph_feature_dim: usize,
}

/// Manual spatial self-attention. This mirrors UniMate's graph-biased path
/// and keeps the graph bias shared across frames instead of expanding it.
#[derive(Module, Debug)]
pub struct SpatialGraphSelfAttention<B: Backend> {
    pub qkv: Linear<B>,
    pub proj: Linear<B>,
    pub q_norm: LlamaRmsNorm<B>,
    pub k_norm: LlamaRmsNorm<B>,
    pub rope: SpectralJointRoPE<B>,
    pub graph_dist_embedding: Embedding<B>,
    pub graph_dist_proj: Linear<B>,
    pub graph_dist_scale: Param<Tensor<B, 1>>,
    pub graph_rel_embedding: Embedding<B>,
    pub graph_rel_proj: Linear<B>,
    pub graph_rel_scale: Param<Tensor<B, 1>>,
    heads: usize,
    head_dim: usize,
    scale: f64,
    graph_feature_dim: usize,
}

impl<B: Backend> SpatialGraphSelfAttention<B> {
    pub fn new(width: usize, heads: usize, max_freqs: usize, device: &B::Device) -> Self {
        assert!(heads > 0 && width % heads == 0);
        let head_dim = width / heads;
        let graph_feature_dim = width / 4;
        let scale_init = burn::module::Initializer::Constant { value: 0.02 };
        Self {
            qkv: LinearConfig::new(width, width * 3).init(device),
            proj: LinearConfig::new(width, width).init(device),
            q_norm: LlamaRmsNorm::new(head_dim, device),
            k_norm: LlamaRmsNorm::new(head_dim, device),
            rope: SpectralJointRoPE::new(head_dim, max_freqs, device),
            graph_dist_embedding: EmbeddingConfig::new(6, graph_feature_dim).init(device),
            graph_dist_proj: LinearConfig::new(graph_feature_dim, heads).init(device),
            graph_dist_scale: scale_init.init([1], device),
            graph_rel_embedding: EmbeddingConfig::new(6, graph_feature_dim).init(device),
            graph_rel_proj: LinearConfig::new(graph_feature_dim, heads).init(device),
            graph_rel_scale: scale_init.init([1], device),
            heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            graph_feature_dim,
        }
    }

    /// `input`: `(B,F,J,D)`, `graph_bias`: `(B,H,J,J)`.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        graph_dist: Tensor<B, 3, burn::tensor::Int>,
        joint_relations: Tensor<B, 3, burn::tensor::Int>,
        valid_joints: Tensor<B, 2, burn::tensor::Bool>,
        spectral_coords: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch, frames, joints, width] = input.dims();
        let pair_count = joints * joints;
        let dist = self
            .graph_dist_embedding
            .forward(graph_dist.reshape([batch, pair_count]))
            .reshape([batch, joints, joints, self.graph_feature_dim]);
        let rel = self
            .graph_rel_embedding
            .forward(joint_relations.reshape([batch, pair_count]))
            .reshape([batch, joints, joints, self.graph_feature_dim]);
        let dist =
            self.graph_dist_proj.forward(dist) * self.graph_dist_scale.val().reshape([1, 1, 1, 1]);
        let rel =
            self.graph_rel_proj.forward(rel) * self.graph_rel_scale.val().reshape([1, 1, 1, 1]);
        let valid = valid_joints
            .reshape([batch, 1, joints])
            .unsqueeze_dim::<4>(1);
        let graph_bias = (dist + rel)
            .permute([0, 3, 1, 2])
            .mask_fill(valid.bool_not(), f32::NEG_INFINITY);
        let qkv = self.qkv.forward(input).chunk(3, 3);
        let to_heads = |tensor: Tensor<B, 4>| {
            tensor
                .reshape([batch, frames, joints, self.heads, self.head_dim])
                .permute([0, 1, 3, 2, 4])
        };
        let query = self.q_norm.forward(to_heads(qkv[0].clone()));
        let key = self.k_norm.forward(to_heads(qkv[1].clone()));
        let value = to_heads(qkv[2].clone());
        let (query, key) = self.rope.forward_per_frame(query, key, spectral_coords);

        let weights =
            (query.matmul(key.swap_dims(3, 4)) * self.scale) + graph_bias.unsqueeze_dim::<5>(1);
        let weights = softmax(weights.cast(DType::F32), 4).cast(value.dtype());
        let attended = weights
            .matmul(value)
            .permute([0, 1, 3, 2, 4])
            .reshape([batch, frames, joints, width]);
        self.proj.forward(attended)
    }
}

impl<B: Backend> GraphAttnBias<B> {
    pub fn new(width: usize, num_heads: usize, device: &B::Device) -> Self {
        let graph_feature_dim = width / 4;
        let scale_init = burn::module::Initializer::Constant { value: 0.02 };
        Self {
            graph_dist_embedding: EmbeddingConfig::new(6, graph_feature_dim).init(device),
            graph_dist_proj: LinearConfig::new(graph_feature_dim, num_heads).init(device),
            graph_dist_scale: scale_init.init([1], device),
            graph_rel_embedding: EmbeddingConfig::new(6, graph_feature_dim).init(device),
            graph_rel_proj: LinearConfig::new(graph_feature_dim, num_heads).init(device),
            graph_rel_scale: scale_init.init([1], device),
            graph_feature_dim,
        }
    }

    /// Computes `(batch, heads, joints, joints)` additive attention bias.
    /// Distances and relation ids are integer square matrices from the
    /// skeleton preprocessing; valid_joint marks real joints.
    pub fn forward(
        &self,
        graph_dist: Tensor<B, 3, burn::tensor::Int>,
        joint_relations: Tensor<B, 3, burn::tensor::Int>,
        valid_joint: Tensor<B, 2, burn::tensor::Bool>,
    ) -> Tensor<B, 4> {
        let [batch, joints, _] = graph_dist.dims();
        let pair_count = joints * joints;
        let dist = self
            .graph_dist_embedding
            .forward(graph_dist.reshape([batch, pair_count]))
            .reshape([batch, joints, joints, self.graph_feature_dim]);
        let relation = self
            .graph_rel_embedding
            .forward(joint_relations.reshape([batch, pair_count]))
            .reshape([batch, joints, joints, self.graph_feature_dim]);

        let dist =
            self.graph_dist_proj.forward(dist) * self.graph_dist_scale.val().reshape([1, 1, 1, 1]);
        let relation = self.graph_rel_proj.forward(relation)
            * self.graph_rel_scale.val().reshape([1, 1, 1, 1]);
        let bias = (dist + relation).permute([0, 3, 1, 2]);
        let key_mask = valid_joint
            .reshape([batch, 1, joints])
            .unsqueeze_dim::<4>(1);
        bias.mask_fill(key_mask.bool_not(), f32::NEG_INFINITY)
    }
}

/// Sign invariant spectral projection used by the released checkpoint.
#[derive(Module, Debug)]
pub struct SignNetSpectralEncoder<B: Backend> {
    pub phi_0: Linear<B>,
    pub phi_2: Linear<B>,
    pub rho_0: Linear<B>,
    pub rho_2: Linear<B>,
    hidden_dim: usize,
    num_eigvecs: usize,
}

impl<B: Backend> SignNetSpectralEncoder<B> {
    pub fn new(num_eigvecs: usize, hidden_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self {
            phi_0: LinearConfig::new(1, hidden_dim).init(device),
            phi_2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            rho_0: LinearConfig::new(num_eigvecs * hidden_dim, hidden_dim).init(device),
            rho_2: LinearConfig::new(hidden_dim, out_dim).init(device),
            hidden_dim,
            num_eigvecs,
        }
    }

    /// Maps `(B,J,K)` Laplacian eigenvectors to `(B,J,out_dim)` angles.
    pub fn forward(&self, spectral_coords: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, joints, eigvecs] = spectral_coords.dims();
        assert_eq!(eigvecs, self.num_eigvecs, "spectral feature count mismatch");
        let values = spectral_coords.unsqueeze_dim::<4>(3);
        let phi = |x| {
            let hidden = gelu(self.phi_0.forward(x));
            self.phi_2.forward(hidden)
        };
        let encoded = phi(values.clone()) + phi(-values);
        let encoded = encoded.reshape([batch, joints, eigvecs * self.hidden_dim]);
        let hidden = gelu(self.rho_0.forward(encoded));
        self.rho_2.forward(hidden)
    }
}

/// Spectral joint RoPE, evaluated once per sample and broadcast across frames.
#[derive(Module, Debug)]
pub struct SpectralJointRoPE<B: Backend> {
    pub spectral_encoder: SignNetSpectralEncoder<B>,
    head_dim: usize,
}

/// One-axis sinusoidal RoPE used by UniMate's temporal attention.
#[derive(Debug, Clone)]
pub struct TemporalRoPE {
    max_len: usize,
    head_dim: usize,
    base: f64,
}

/// Temporal self-attention applied independently to each joint.
#[derive(Module, Debug)]
pub struct TemporalSelfAttention<B: Backend> {
    pub qkv: Linear<B>,
    pub proj: Linear<B>,
    pub q_norm: LlamaRmsNorm<B>,
    pub k_norm: LlamaRmsNorm<B>,
    max_frames: usize,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl<B: Backend> TemporalSelfAttention<B> {
    pub fn new(width: usize, heads: usize, max_frames: usize, device: &B::Device) -> Self {
        assert!(heads > 0 && width % heads == 0);
        let head_dim = width / heads;
        Self {
            qkv: LinearConfig::new(width, width * 3).init(device),
            proj: LinearConfig::new(width, width).init(device),
            q_norm: LlamaRmsNorm::new(head_dim, device),
            k_norm: LlamaRmsNorm::new(head_dim, device),
            max_frames,
            heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        }
    }

    /// `input`: `(B,F,J,D)`, `valid_frames`: `(B,F)` (true means valid).
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        valid_frames: Tensor<B, 2, burn::tensor::Bool>,
    ) -> Tensor<B, 4> {
        let [batch, frames, joints, width] = input.dims();
        let joint_major = input.permute([0, 2, 1, 3]);
        let qkv = self.qkv.forward(joint_major).chunk(3, 3);
        let to_heads = |tensor: Tensor<B, 4>| {
            tensor
                .reshape([batch, joints, frames, self.heads, self.head_dim])
                .permute([0, 1, 3, 2, 4])
        };
        let query = self.q_norm.forward(to_heads(qkv[0].clone()));
        let key = self.k_norm.forward(to_heads(qkv[1].clone()));
        let value = to_heads(qkv[2].clone());

        let query_flat = query.reshape([batch * joints, self.heads, frames, self.head_dim]);
        let key_flat = key.reshape([batch * joints, self.heads, frames, self.head_dim]);
        let (query, key) =
            TemporalRoPE::new(self.max_frames, self.head_dim).forward(query_flat, key_flat);
        let query = query.reshape([batch, joints, self.heads, frames, self.head_dim]);
        let key = key.reshape([batch, joints, self.heads, frames, self.head_dim]);

        let logits = (query.matmul(key.swap_dims(3, 4)) * self.scale).cast(DType::F32);
        let key_mask = valid_frames.reshape([batch, 1, 1, 1, frames]).bool_not();
        let logits = logits.mask_fill(key_mask, f32::NEG_INFINITY);
        let weights = softmax(logits, 4).cast(value.dtype());
        let attended = weights
            .matmul(value)
            .permute([0, 1, 3, 2, 4])
            .reshape([batch, joints, frames, width]);
        self.proj.forward(attended).permute([0, 2, 1, 3])
    }
}

/// UniMate graph attention + temporal attention + SwiGLU adaLN block.
#[derive(Module, Debug)]
pub struct SpatioTemporalBlock<B: Backend> {
    pub norm_s: LlamaRmsNorm<B>,
    pub s_attn: SpatialGraphSelfAttention<B>,
    pub norm_t: LlamaRmsNorm<B>,
    pub t_attn: TemporalSelfAttention<B>,
    pub norm_mlp: LlamaRmsNorm<B>,
    pub mlp: SwiGLUFFN<B>,
    /// PyTorch checkpoint's `adaLN_modulation.1` projection.
    pub ada_ln_linear: Linear<B>,
    width: usize,
}

impl<B: Backend> SpatioTemporalBlock<B> {
    pub fn new(
        width: usize,
        ff_size: usize,
        heads: usize,
        max_freqs: usize,
        max_frames: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            norm_s: LlamaRmsNorm::new(width, device),
            s_attn: SpatialGraphSelfAttention::new(width, heads, max_freqs, device),
            norm_t: LlamaRmsNorm::new(width, device),
            t_attn: TemporalSelfAttention::new(width, heads, max_frames, device),
            norm_mlp: LlamaRmsNorm::new(width, device),
            mlp: SwiGLUFFN::new(width, (2 * ff_size) / 3, device),
            ada_ln_linear: LinearConfig::new(width, 9 * width).init(device),
            width,
        }
    }

    /// `x`: `(B,F,J,D)`, `condition`: `(B,D)`, `graph_bias`: `(B,H,J,J)`.
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        condition: Tensor<B, 2>,
        valid_frames: Tensor<B, 2, burn::tensor::Bool>,
        graph_dist: Tensor<B, 3, burn::tensor::Int>,
        joint_relations: Tensor<B, 3, burn::tensor::Int>,
        valid_joints: Tensor<B, 2, burn::tensor::Bool>,
        spectral_coords: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch, _, _, width] = x.dims();
        assert_eq!(width, self.width, "transformer block width mismatch");
        let modulation = silu(condition);
        let chunks = self.ada_ln_linear.forward(modulation).chunk(9, 1);
        let expand = |index: usize| chunks[index].clone().reshape([batch, 1, 1, width]);

        let dtype = x.dtype();
        let spatial_input = self.norm_s.forward(x.clone().cast(DType::F32)).cast(dtype);
        let spatial_input = spatial_input * (expand(1) + 1.0) + expand(0);
        let spatial = self.s_attn.forward(
            spatial_input,
            graph_dist,
            joint_relations,
            valid_joints,
            spectral_coords,
        );
        let x = x + expand(2) * spatial;

        let temporal_input = self.norm_t.forward(x.clone().cast(DType::F32)).cast(dtype);
        let temporal_input = temporal_input * (expand(4) + 1.0) + expand(3);
        let temporal = self.t_attn.forward(temporal_input, valid_frames);
        let x = x + expand(5) * temporal;

        let mlp_input = self
            .norm_mlp
            .forward(x.clone().cast(DType::F32))
            .cast(dtype);
        let mlp_input = mlp_input * (expand(7) + 1.0) + expand(6);
        let mlp = self.mlp.forward(mlp_input);
        let output = x + expand(8) * mlp;
        output
    }
}

impl TemporalRoPE {
    pub fn new(max_len: usize, head_dim: usize) -> Self {
        assert_eq!(head_dim % 2, 0, "temporal RoPE requires an even head width");
        // Matches RopeND's auto_base rule with one position axis.
        let base = ((8.0 * max_len as f64 / std::f64::consts::PI) as usize / 100 + 1) * 100;
        Self {
            max_len,
            head_dim,
            base: base as f64,
        }
    }

    /// Rotates q/k shaped `(batch, heads, sequence, head_dim)`.
    /// Frequency tables are built from the same deterministic scalar formula
    /// as upstream RopeND and materialized on the selected Burn device.
    pub fn forward<B: Backend>(
        &self,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_, _, sequence, head_dim] = query.dims();
        assert_eq!(head_dim, self.head_dim, "temporal RoPE head width mismatch");
        assert!(
            sequence <= self.max_len,
            "sequence exceeds configured temporal RoPE length"
        );
        let device = query.device();
        let half = head_dim / 2;
        let mut cos_values = Vec::with_capacity(sequence * head_dim);
        let mut sin_values = Vec::with_capacity(sequence * head_dim);
        for position in 0..sequence {
            let mut angles = Vec::with_capacity(half);
            for index in 0..half {
                let inverse_frequency = self.base.powf(-(index as f64) / head_dim as f64);
                angles.push(position as f64 * inverse_frequency);
            }
            for angle in angles.iter().chain(angles.iter()) {
                cos_values.push(angle.cos() as f32);
                sin_values.push(angle.sin() as f32);
            }
        }
        let cos =
            Tensor::<B, 2>::from_data(TensorData::new(cos_values, [sequence, head_dim]), &device)
                .reshape([1, 1, sequence, head_dim]);
        let sin =
            Tensor::<B, 2>::from_data(TensorData::new(sin_values, [sequence, head_dim]), &device)
                .reshape([1, 1, sequence, head_dim]);
        let dtype = query.dtype();
        let rotate = |x: Tensor<B, 4>| {
            let x = x.cast(DType::F32);
            let halves = x.clone().chunk(2, 3);
            let rotated = Tensor::cat(vec![-halves[1].clone(), halves[0].clone()], 3);
            (x * cos.clone() + rotated * sin.clone()).cast(dtype)
        };
        (rotate(query), rotate(key))
    }
}

impl<B: Backend> SpectralJointRoPE<B> {
    pub fn new(head_dim: usize, max_freqs: usize, device: &B::Device) -> Self {
        assert_eq!(head_dim % 2, 0, "spectral RoPE requires an even head width");
        Self {
            spectral_encoder: SignNetSpectralEncoder::new(max_freqs, 64, head_dim / 2, device),
            head_dim,
        }
    }

    /// Applies spectral rotations to `(B,F,H,J,D)` query and key tensors.
    pub fn forward_per_frame(
        &self,
        query: Tensor<B, 5>,
        key: Tensor<B, 5>,
        spectral_coords: Tensor<B, 3>,
    ) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let [batch, _, _, joints, head_dim] = query.dims();
        assert_eq!(head_dim, self.head_dim, "attention head width mismatch");
        let dtype = query.dtype();
        let angles = self.spectral_encoder.forward(spectral_coords);
        let angles =
            Tensor::cat(vec![angles.clone(), angles], 2).reshape([batch, 1, 1, joints, head_dim]);
        let cos = angles.clone().cos();
        let sin = angles.sin();
        let rotate = |x: Tensor<B, 5>| {
            let x = x.cast(DType::F32);
            let halves = x.clone().chunk(2, 4);
            let rotated = Tensor::cat(vec![-halves[1].clone(), halves[0].clone()], 4);
            (x * cos.clone() + rotated * sin.clone()).cast(dtype)
        };
        (rotate(query), rotate(key))
    }
}

/// PyTorch-compatible packed-QKV multi-head attention used by final root aggregation.
#[derive(Module, Debug)]
pub struct PackedCrossAttention<B: Backend> {
    /// PyTorch `MultiheadAttention` stores Q, K, and V in one `[3D,D]` matrix.
    pub in_proj_weight: Param<Tensor<B, 2>>,
    pub in_proj_bias: Param<Tensor<B, 1>>,
    pub out_proj: Linear<B>,
    heads: usize,
    head_dim: usize,
    width: usize,
}

impl<B: Backend> PackedCrossAttention<B> {
    pub fn new(width: usize, heads: usize, device: &B::Device) -> Self {
        assert!(heads > 0 && width % heads == 0);
        Self {
            in_proj_weight: burn::module::Initializer::Zeros.init([3 * width, width], device),
            in_proj_bias: burn::module::Initializer::Zeros.init([3 * width], device),
            out_proj: LinearConfig::new(width, width).init(device),
            heads,
            head_dim: width / heads,
            width,
        }
    }

    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key_value: Tensor<B, 3>,
        key_valid: Tensor<B, 2, burn::tensor::Bool>,
    ) -> Tensor<B, 3> {
        let [batch_frames, query_len, _] = query.dims();
        let key_len = key_value.dims()[1];
        let weights = self.in_proj_weight.val().chunk(3, 0);
        let biases = self.in_proj_bias.val().chunk(3, 0);
        let linear = |input: Tensor<B, 3>, weight: Tensor<B, 2>, bias: Tensor<B, 1>| {
            input.matmul(weight.transpose().unsqueeze_dim::<3>(0))
                + bias.reshape([1, 1, self.width])
        };
        let q = linear(query, weights[0].clone(), biases[0].clone());
        let k = linear(key_value.clone(), weights[1].clone(), biases[1].clone());
        let v = linear(key_value, weights[2].clone(), biases[2].clone());
        let to_heads = |x: Tensor<B, 3>, length: usize| {
            x.reshape([batch_frames, length, self.heads, self.head_dim])
                .permute([0, 2, 1, 3])
        };
        let q = to_heads(q, query_len);
        let k = to_heads(k, key_len);
        let v = to_heads(v, key_len);
        let logits =
            (q.matmul(k.swap_dims(2, 3)) * (self.head_dim as f64).powf(-0.5)).cast(DType::F32);
        let mask = key_valid.reshape([batch_frames, 1, 1, key_len]).bool_not();
        let weights = softmax(logits.mask_fill(mask, f32::NEG_INFINITY), 3).cast(v.dtype());
        let output =
            weights
                .matmul(v)
                .permute([0, 2, 1, 3])
                .reshape([batch_frames, query_len, self.width]);
        self.out_proj.forward(output)
    }
}

/// Final modulation, root aggregation, and distinct root/joint output heads.
#[derive(Module, Debug)]
pub struct FinalLayer<B: Backend> {
    pub norm_final: LlamaRmsNorm<B>,
    pub root_cross_attn: PackedCrossAttention<B>,
    pub root_cross_norm_q: LlamaRmsNorm<B>,
    pub root_cross_norm_kv: LlamaRmsNorm<B>,
    pub root_out: SiluMlp<B>,
    pub joint_out: SiluMlp<B>,
    pub ada_ln_linear: Linear<B>,
    joint_count: usize,
    width: usize,
    feature_len: usize,
}

impl<B: Backend> FinalLayer<B> {
    pub fn new(width: usize, feature_len: usize, joint_count: usize, device: &B::Device) -> Self {
        Self {
            norm_final: LlamaRmsNorm::new(width, device),
            root_cross_attn: PackedCrossAttention::new(width, 4, device),
            root_cross_norm_q: LlamaRmsNorm::new(width, device),
            root_cross_norm_kv: LlamaRmsNorm::new(width, device),
            root_out: SiluMlp::new(width, width, feature_len, device),
            joint_out: SiluMlp::new(width, width, feature_len, device),
            ada_ln_linear: LinearConfig::new(width, 2 * width).init(device),
            joint_count,
            width,
            feature_len,
        }
    }

    /// Input tokens `[B,F,J,D]`, condition `[B,D]`, joint mask `[B,J]`.
    /// Output is `[B,J,feature_len,F]`, including the prepended t-pose frame.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        condition: Tensor<B, 2>,
        valid_joints: Tensor<B, 2, burn::tensor::Bool>,
    ) -> Tensor<B, 4> {
        let [batch, frames, joints, width] = input.dims();
        assert_eq!(joints, self.joint_count);
        assert_eq!(width, self.width);
        let modulation = silu(condition);
        let chunks = self.ada_ln_linear.forward(modulation).chunk(2, 1);
        let x = self
            .norm_final
            .forward(input.reshape([batch, frames * joints, width]));
        let x = x * (chunks[1].clone().unsqueeze_dim::<3>(1) + 1.0)
            + chunks[0].clone().unsqueeze_dim::<3>(1);
        let x = x.reshape([batch, frames, joints, width]);
        let root = x.clone().narrow(2, 0, 1);
        let joints_x = x.narrow(2, 1, joints - 1);
        let root_flat = root.clone().reshape([batch * frames, 1, width]);
        let joints_flat = joints_x
            .clone()
            .reshape([batch * frames, joints - 1, width]);

        let valid = valid_joints
            .narrow(1, 1, joints - 1)
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, frames)
            .reshape([batch * frames, joints - 1]);
        let root_query = self.root_cross_norm_q.forward(root_flat);
        let joint_keys = self.root_cross_norm_kv.forward(joints_flat);
        let root_agg = self
            .root_cross_attn
            .forward(root_query, joint_keys, valid)
            .reshape([batch, frames, 1, width]);
        let root = self.root_out.forward(root + root_agg);
        let joints_x = self.joint_out.forward(joints_x);
        Tensor::cat(vec![root, joints_x], 2).permute([0, 3, 1, 2])
    }
}
