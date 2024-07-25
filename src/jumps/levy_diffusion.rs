use crate::{noises::gn::gn, processes::poisson::compound_poisson};
use ndarray::Array1;

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

pub fn levy_diffusion(
  gamma: f64,
  sigma: f64,
  lambda: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Array1<f64> {
  let dt = t.unwrap_or(1.0) / n as f64;
  let mut levy = Array1::<f64>::zeros(n);
  levy[0] = x0.unwrap_or(0.0);
  let gn = gn(n - 1, t);

  for i in 1..n {
    let [.., jumps] = compound_poisson(None, lambda, Some(dt), None, Some(10.));

    levy[i] = levy[i - 1] + gamma * dt + sigma * gn[i - 1] + jumps.sum();
  }

  levy
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_levy_diffusion() {
    let gamma = 0.0;
    let sigma = 1.0;
    let lambda = 10.0;
    let n = 1000;
    let t = 10.0;
    let l = levy_diffusion(gamma, sigma, lambda, n, None, Some(t));
    assert_eq!(l.len(), n);
  }
}
