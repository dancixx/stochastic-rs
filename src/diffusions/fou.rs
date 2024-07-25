use crate::{noises::fgn::FgnFft, utils::Generator};
use ndarray::Array1;

/// Generates a path of the fractional Ornstein-Uhlenbeck (fOU) process.
///
/// The fOU process incorporates fractional Brownian motion, which introduces long-range dependence.
///
/// # Parameters
///
/// - `hurst`: Hurst parameter for fractional Brownian motion, must be in (0, 1).
/// - `mu`: Long-term mean level.
/// - `sigma`: Volatility parameter.
/// - `theta`: Speed of mean reversion.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated fOU process path.
///
/// # Panics
///
/// Panics if `hurst` is not in (0, 1).
///
/// # Example
///
/// ```
/// let fou_path = fou(0.75, 0.0, 0.1, 0.5, 1000, Some(0.0), Some(1.0));
/// ```

#[derive(Default)]
pub struct Fou {
  pub hurst: f64,
  pub mu: f64,
  pub sigma: f64,
  pub theta: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn fou(params: &Fou) -> Array1<f64> {
  let Fou {
    hurst,
    mu,
    sigma,
    theta,
    n,
    x0,
    t,
  } = *params;

  assert!(
    hurst > 0.0 && hurst < 1.0,
    "Hurst parameter must be in (0, 1)"
  );

  let fgn = FgnFft::new(hurst, n - 1, t, None).sample();
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut fou = Array1::<f64>::zeros(n);
  fou[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    fou[i] = fou[i - 1] + theta * (mu - fou[i - 1]) * dt + sigma * fgn[i - 1]
  }

  fou
}
