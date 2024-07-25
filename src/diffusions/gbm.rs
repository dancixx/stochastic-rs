use crate::noises::gn::gn;
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

#[derive(Default)]
pub struct Gbm {
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn gbm(params: &Gbm) -> Array1<f64> {
  let Gbm {
    mu,
    sigma,
    n,
    x0,
    t,
  } = *params;

  let gn = gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut gbm = Array1::<f64>::zeros(n);
  gbm[0] = x0.unwrap_or(100.0);

  for i in 1..n {
    gbm[i] = gbm[i - 1] + mu * gbm[i - 1] * dt + sigma * gbm[i - 1] * gn[i - 1]
  }

  gbm
}
