use ndarray::Array1;
use rand_distr::Distribution;

use crate::{
  noises::gn::gn,
  processes::cpoisson::{compound_poisson, CompoundPoisson},
};

/// Generates a path of the Merton jump diffusion process.
///
/// The Merton jump diffusion process combines a continuous diffusion process with jumps, commonly used in financial modeling.
///
/// # Parameters
///
/// - `alpha`: Drift parameter.
/// - `sigma`: Volatility parameter.
/// - `lambda`: Jump intensity.
/// - `theta`: Jump size.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated Merton jump diffusion process path.
///
/// # Example
///
/// ```
/// let merton_path = merton(0.1, 0.2, 0.5, 0.05, 1000, Some(0.0), Some(1.0));
/// ```

#[derive(Default)]
pub struct Merton {
  pub alpha: f64,
  pub sigma: f64,
  pub lambda: f64,
  pub theta: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn merton<D>(params: &Merton, jdistr: D) -> Array1<f64>
where
  D: Distribution<f64> + Copy,
{
  let Merton {
    alpha,
    sigma,
    lambda,
    theta,
    n,
    x0,
    t,
  } = *params;
  let dt = t.unwrap_or(1.0) / n as f64;
  let mut merton = Array1::<f64>::zeros(n + 1);
  merton[0] = x0.unwrap_or(0.0);
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

    merton[i] = merton[i - 1]
      + (alpha * sigma.powf(2.0) / 2.0 - lambda * theta) * dt
      + sigma * gn[i - 1]
      + jumps.sum();
  }

  merton
}
