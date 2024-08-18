use crate::{
  noises::gn::gn,
  processes::cpoisson::{compound_poisson, CompoundPoisson},
};
use ndarray::Array1;
use rand_distr::Distribution;

/// Generates a path of the Lévy diffusion process.
///
/// The Lévy diffusion process incorporates both Gaussian and jump components, often used in financial modeling.
///
/// # Parameters
///
/// - `gamma`: Drift parameter.
/// - `sigma`: Volatility parameter.
/// - `lambda`: Jump intensity.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated Lévy diffusion process path.
///
/// # Example
///
/// ```
/// let levy_path = levy_diffusion(0.1, 0.2, 0.5, 1000, Some(0.0), Some(1.0));
/// ```

#[derive(Default)]
pub struct LevyDiffusion {
  pub gamma: f64,
  pub sigma: f64,
  pub lambda: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn levy_diffusion<D>(params: &LevyDiffusion, jdistr: D) -> Array1<f64>
where
  D: Distribution<f64> + Copy,
{
  let LevyDiffusion {
    gamma,
    sigma,
    lambda,
    n,
    x0,
    t,
  } = *params;

  let dt = t.unwrap_or(1.0) / n as f64;
  let mut levy = Array1::<f64>::zeros(n + 1);
  levy[0] = x0.unwrap_or(0.0);
  let gn = gn(n, t);

  for i in 1..(n + 1) {
    let [.., jumps] = compound_poisson(
      &CompoundPoisson {
        lambda,
        t_max: Some(dt),
        n: None,
      },
      jdistr,
    );

    levy[i] = levy[i - 1] + gamma * dt + sigma * gn[i - 1] + jumps.sum();
  }

  levy
}
