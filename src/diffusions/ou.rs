use crate::{
  noises::{fgn::FgnFft, gn},
  utils::Generator,
};
use ndarray::Array1;

/// Generates a path of the Ornstein-Uhlenbeck (OU) process.
///
/// The OU process is a mean-reverting stochastic process used in various fields such as finance and physics.
///
/// # Parameters
///
/// - `mu`: Long-term mean level.
/// - `sigma`: Volatility parameter.
/// - `theta`: Speed of mean reversion.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated OU process path.
///
/// # Example
///
/// ```
/// let ou_path = ou(0.0, 0.1, 0.5, 1000, Some(0.0), Some(1.0));
/// ```
pub fn ou(
  mu: f64,
  sigma: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Array1<f64> {
  let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut ou = Array1::<f64>::zeros(n);
  ou[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    ou[i] = ou[i - 1] + theta * (mu - ou[i - 1]) * dt + sigma * gn[i - 1]
  }

  ou
}

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
#[allow(clippy::too_many_arguments)]
pub fn fou(
  hurst: f64,
  mu: f64,
  sigma: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Array1<f64> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  let fgn = FgnFft::new(hurst, n - 1, t, None).sample();
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut fou = Array1::<f64>::zeros(n);
  fou[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    fou[i] = fou[i - 1] + theta * (mu - fou[i - 1]) * dt + sigma * fgn[i - 1]
  }

  fou
}
