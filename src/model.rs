//! Burn implementations of the small reusable UniMate network layers.
//!
//! These module field names intentionally mirror the upstream PyTorch names so
//! the converted EMA state dict can be loaded without opaque positional maps.

use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{
        DType, Tensor,
        activation::{silu, softmax},
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
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl<B: Backend> SpatialGraphSelfAttention<B> {
    pub fn new(width: usize, heads: usize, device: &B::Device) -> Self {
        assert!(heads > 0 && width % heads == 0);
        let head_dim = width / heads;
        Self {
            qkv: LinearConfig::new(width, width * 3).init(device),
            proj: LinearConfig::new(width, width).init(device),
            q_norm: LlamaRmsNorm::new(head_dim, device),
            k_norm: LlamaRmsNorm::new(head_dim, device),
            heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        }
    }

    /// `input`: `(B,F,J,D)`, `graph_bias`: `(B,H,J,J)`.
    pub fn forward(&self, input: Tensor<B, 4>, graph_bias: Tensor<B, 4>) -> Tensor<B, 4> {
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
