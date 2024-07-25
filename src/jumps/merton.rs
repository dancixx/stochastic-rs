use ndarray::Array1;

use crate::{noises::gn::gn, processes::poisson::compound_poisson};

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
pub fn merton(
  alpha: f64,
  sigma: f64,
  lambda: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Array1<f64> {
  let dt = t.unwrap_or(1.0) / n as f64;
  let mut merton = Array1::<f64>::zeros(n);
  merton[0] = x0.unwrap_or(0.0);
  let gn = gn(n - 1, t);

  for i in 1..n {
    let [.., jumps] = compound_poisson(None, lambda, t, None, None);

    merton[i] = merton[i - 1]
      + (alpha * sigma.powf(2.0) / 2.0 - lambda * theta) * dt
      + sigma * gn[i - 1]
      + jumps.sum();
  }

  merton
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_merton() {
    let alpha = 0.0;
    let sigma = 1.0;
    let lambda = 10.0;
    let theta = 0.0;
    let n = 1000;
    let t = 10.0;
    let m = merton(alpha, sigma, lambda, theta, n, None, Some(t));
    assert_eq!(m.len(), n);
  }
}
