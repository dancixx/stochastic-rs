use derive_builder::Builder;
use ndarray::Array1;

use crate::{noises::fgn::FgnFft, utils::Generator};

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

#[derive(Default, Builder)]
#[builder(setter(into))]
pub struct Fgbm {
  pub hurst: f64,
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn fgbm(params: &Fgbm) -> Array1<f64> {
  let Fgbm {
    hurst,
    mu,
    sigma,
    n,
    x0,
    t,
  } = *params;

  assert!(
    hurst > 0.0 && hurst < 1.0,
    "Hurst parameter must be in (0, 1)"
  );

  let dt = t.unwrap_or(1.0) / n as f64;
  let fgn = FgnFft::new(hurst, n, t, None).sample();

  let mut fgbm = Array1::<f64>::zeros(n + 1);
  fgbm[0] = x0.unwrap_or(100.0);

  for i in 1..(n + 1) {
    fgbm[i] = fgbm[i - 1] + mu * fgbm[i - 1] * dt + sigma * fgbm[i - 1] * fgn[i - 1]
  }

  fgbm
}
