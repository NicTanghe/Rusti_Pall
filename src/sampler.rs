//! Burn tensor operations used by flow sampling.
//!
//! The model call stays outside this module so the sampler can be validated
//! independently of the architecture port.

use burn::tensor::{Tensor, backend::Backend};

/// Classifier-free guidance: `uncond + scale * (cond - uncond)`.
pub fn classifier_free_guidance<B: Backend, const D: usize>(
    unconditional: Tensor<B, D>,
    conditional: Tensor<B, D>,
    scale: f32,
) -> Tensor<B, D> {
    unconditional.clone() + (conditional - unconditional) * scale
}

/// One explicit Euler update for the flow ODE.
pub fn euler_step<B: Backend, const D: usize>(
    state: Tensor<B, D>,
    velocity: Tensor<B, D>,
    dt: f32,
) -> Tensor<B, D> {
    state + velocity * dt
}
