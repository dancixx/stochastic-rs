use crate::{
  noises::{fgn::FgnFft, gn::gn},
  utils::Generator,
};
use ndarray::Array1;

/// Generates a path of the Geometric Brownian Motion (GBM) process.
///
/// The GBM process is commonly used in financial mathematics to model stock prices.
///
/// # Parameters
///
/// - `mu`: Drift parameter.
/// - `sigma`: Volatility parameter.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 100.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated GBM process path.
///
/// # Example
///
/// ```
/// let gbm_path = gbm(0.05, 0.2, 1000, Some(100.0), Some(1.0));
/// ```
pub fn gbm(mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>) -> Array1<f64> {
  let gn = gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut gbm = Array1::<f64>::zeros(n);
  gbm[0] = x0.unwrap_or(100.0);

  for i in 1..n {
    gbm[i] = gbm[i - 1] + mu * gbm[i - 1] * dt + sigma * gbm[i - 1] * gn[i - 1]
  }

  gbm
}

/// Generates a path of the fractional Geometric Brownian Motion (fGBM) process.
///
/// The fGBM process incorporates fractional Brownian motion, which introduces long-range dependence.
///
/// # Parameters
///
/// - `hurst`: Hurst parameter for fractional Brownian motion, must be in (0, 1).
/// - `mu`: Drift parameter.
/// - `sigma`: Volatility parameter.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 100.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated fGBM process path.
///
/// # Panics
///
/// Panics if `hurst` is not in (0, 1).
///
/// # Example
///
/// ```
/// let fgbm_path = fgbm(0.75, 0.05, 0.2, 1000, Some(100.0), Some(1.0));
/// ```
pub fn fgbm(
  hurst: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Array1<f64> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  let fgn = FgnFft::new(hurst, n - 1, t, None).sample();
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut fgbm = Array1::<f64>::zeros(n);
  fgbm[0] = x0.unwrap_or(100.0);

  for i in 1..n {
    fgbm[i] = fgbm[i - 1] + mu * fgbm[i - 1] * dt + sigma * fgbm[i - 1] * fgn[i - 1]
  }

  fgbm
}
