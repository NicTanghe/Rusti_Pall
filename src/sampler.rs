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

/// Runge-Kutta statistics useful for performance reporting.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntegrationStats {
    pub evaluations: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
}

/// Integrates the velocity field from noise time 0 to data time 1 using an
/// adaptive Dormand-Prince 5(4) method. UniMate's Python inference uses
/// torchdiffeq `dopri5` with atol=1e-6, rtol=1e-3 and 50 requested output
/// points; this implementation uses the same tableau and tolerances, with a
/// standard adaptive controller. It is intended as the native inference path;
/// numerical parity still requires comparison against a PyTorch fixture.
pub fn sample_dopri5<B, const D: usize, F>(
    initial: Tensor<B, D>,
    output_points: usize,
    velocity: F,
) -> Result<(Tensor<B, D>, IntegrationStats), String>
where
    B: Backend,
    F: FnMut(&Tensor<B, D>, f32) -> Tensor<B, D>,
{
    sample_dopri5_interval(initial, 0.0, 1.0, output_points, 1e-6, 1e-3, velocity)
}

pub fn sample_dopri5_interval<B, const D: usize, F>(
    mut state: Tensor<B, D>,
    start: f32,
    end: f32,
    output_points: usize,
    atol: f32,
    rtol: f32,
    mut velocity: F,
) -> Result<(Tensor<B, D>, IntegrationStats), String>
where
    B: Backend,
    F: FnMut(&Tensor<B, D>, f32) -> Tensor<B, D>,
{
    if !(start < end) || output_points < 2 || atol <= 0.0 || rtol <= 0.0 {
        return Err("invalid adaptive ODE interval, point count, or tolerance".into());
    }

    let mut t = start;
    let mut h = (end - start) / (output_points - 1) as f32;
    let mut stats = IntegrationStats::default();
    const MAX_STEPS: usize = 10_000;

    while t < end {
        if stats.accepted_steps + stats.rejected_steps >= MAX_STEPS {
            return Err("Dormand-Prince exceeded 10,000 attempted steps".into());
        }
        h = h.min(end - t);
        let k1 = velocity(&state, t);
        let k2 = velocity(
            &combine(&state, h, &[(&k1, 1.0 / 5.0)]),
            t + h * (1.0 / 5.0),
        );
        let k3 = velocity(
            &combine(&state, h, &[(&k1, 3.0 / 40.0), (&k2, 9.0 / 40.0)]),
            t + h * (3.0 / 10.0),
        );
        let k4 = velocity(
            &combine(
                &state,
                h,
                &[(&k1, 44.0 / 45.0), (&k2, -56.0 / 15.0), (&k3, 32.0 / 9.0)],
            ),
            t + h * (4.0 / 5.0),
        );
        let k5 = velocity(
            &combine(
                &state,
                h,
                &[
                    (&k1, 19372.0 / 6561.0),
                    (&k2, -25360.0 / 2187.0),
                    (&k3, 64448.0 / 6561.0),
                    (&k4, -212.0 / 729.0),
                ],
            ),
            t + h * (8.0 / 9.0),
        );
        let k6 = velocity(
            &combine(
                &state,
                h,
                &[
                    (&k1, 9017.0 / 3168.0),
                    (&k2, -355.0 / 33.0),
                    (&k3, 46732.0 / 5247.0),
                    (&k4, 49.0 / 176.0),
                    (&k5, -5103.0 / 18656.0),
                ],
            ),
            t + h,
        );
        let fifth = combine(
            &state,
            h,
            &[
                (&k1, 35.0 / 384.0),
                (&k3, 500.0 / 1113.0),
                (&k4, 125.0 / 192.0),
                (&k5, -2187.0 / 6784.0),
                (&k6, 11.0 / 84.0),
            ],
        );
        let k7 = velocity(&fifth, t + h);
        stats.evaluations += 7;

        let fourth = combine(
            &state,
            h,
            &[
                (&k1, 5179.0 / 57600.0),
                (&k3, 7571.0 / 16695.0),
                (&k4, 393.0 / 640.0),
                (&k5, -92097.0 / 339200.0),
                (&k6, 187.0 / 2100.0),
                (&k7, 1.0 / 40.0),
            ],
        );
        let scale = state.clone().abs().max_pair(fifth.clone().abs()) * rtol + atol;
        let normalized_error = (fifth.clone() - fourth) / scale;
        let error_squared = normalized_error.square().mean();
        let error = error_squared
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("could not read adaptive error estimate: {e}"))?[0]
            .sqrt();

        let factor = if error == 0.0 {
            10.0
        } else {
            (0.9 * error.powf(-0.2)).clamp(0.2, 10.0)
        };
        if error <= 1.0 {
            state = fifth;
            t += h;
            stats.accepted_steps += 1;
        } else {
            stats.rejected_steps += 1;
        }
        h *= factor;
        if h < f32::EPSILON * (1.0 + t.abs()) {
            return Err("Dormand-Prince step size underflow".into());
        }
    }

    Ok((state, stats))
}

/// Compose UniMate inference's classifier-free behavior with a velocity
/// callback. `unconditional=true` requests the zero-caption branch. At scale
/// 1 the upstream inference CLI samples unconditionally; above 1 it evaluates
/// both branches and applies `uncond + scale * (cond - uncond)`.
pub fn predict_cfg<B, const D: usize, F>(
    state: &Tensor<B, D>,
    time: f32,
    scale: f32,
    mut predict: F,
) -> Tensor<B, D>
where
    B: Backend,
    F: FnMut(&Tensor<B, D>, f32, bool) -> Tensor<B, D>,
{
    if scale <= 1.0 {
        predict(state, time, true)
    } else {
        let conditional = predict(state, time, false);
        let unconditional = predict(state, time, true);
        classifier_free_guidance(unconditional, conditional, scale)
    }
}

fn combine<B: Backend, const D: usize>(
    state: &Tensor<B, D>,
    step: f32,
    terms: &[(&Tensor<B, D>, f32)],
) -> Tensor<B, D> {
    terms
        .iter()
        .fold(state.clone(), |acc, (derivative, coefficient)| {
            acc + (*derivative).clone() * (step * *coefficient)
        })
}
