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
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl<B: Backend> SpatialGraphSelfAttention<B> {
    pub fn new(width: usize, heads: usize, max_freqs: usize, device: &B::Device) -> Self {
        assert!(heads > 0 && width % heads == 0);
        let head_dim = width / heads;
        Self {
            qkv: LinearConfig::new(width, width * 3).init(device),
            proj: LinearConfig::new(width, width).init(device),
            q_norm: LlamaRmsNorm::new(head_dim, device),
            k_norm: LlamaRmsNorm::new(head_dim, device),
            rope: SpectralJointRoPE::new(head_dim, max_freqs, device),
            heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        }
    }

    /// `input`: `(B,F,J,D)`, `graph_bias`: `(B,H,J,J)`.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        graph_bias: Tensor<B, 4>,
        spectral_coords: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch, frames, joints, width] = input.dims();
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
        graph_bias: Tensor<B, 4>,
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
        let spatial = self
            .s_attn
            .forward(spatial_input, graph_bias, spectral_coords);
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
