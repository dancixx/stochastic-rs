use ndarray::Array1;

use crate::prelude::{correlated::correlated_bms, poisson::compound_poisson};

/// Generates paths for the Bates (1996) model.
///
/// The Bates model combines a stochastic volatility model with jump diffusion, commonly used in financial mathematics to model asset prices.
///
/// # Parameters
///
/// - `mu`: Drift parameter of the asset price.
/// - `kappa`: Rate of mean reversion of the volatility.
/// - `theta`: Long-term mean level of the volatility.
/// - `eta`: Volatility of the volatility (vol of vol).
/// - `rho`: Correlation between the asset price and its volatility.
/// - `lambda`: Jump intensity.
/// - `n`: Number of time steps.
/// - `s0`: Initial value of the asset price (optional, defaults to 0.0).
/// - `v0`: Initial value of the volatility (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
/// - `use_sym`: Whether to use symmetric noise for the volatility (optional, defaults to false).
///
/// # Returns
///
/// A `[Array1<f64>; 2]` where the first vector represents the asset price path and the second vector represents the volatility path.
///
/// # Example
///
/// ```
/// let paths = bates_1996(0.05, 1.5, 0.04, 0.3, -0.7, 0.1, 1000, Some(100.0), Some(0.04), Some(1.0), Some(false));
/// let asset_prices = paths[0];
/// let volatilities = paths[1];
/// ```
///
/// # Panics
///
/// This function will panic if the `correlated_bms` or `compound_poisson` functions return invalid lengths or if there are issues with array indexing.
#[allow(clippy::too_many_arguments)]
pub fn bates_1996(
  mu: f64,
  kappa: f64,
  theta: f64,
  eta: f64,
  rho: f64,
  lambda: f64,
  n: usize,
  s0: Option<f64>,
  v0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
) -> [Array1<f64>; 2] {
  let correlated_bms = correlated_bms(rho, n, t);
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut s = Array1::<f64>::zeros(n);
  let mut v = Array1::<f64>::zeros(n);
  let z = compound_poisson(n, lambda, None, t, None);

  s[0] = s0.unwrap_or(0.0);
  v[0] = v0.unwrap_or(0.0);

  for i in 1..n {
    let jump_idx = z[0]
      .iter()
      .position(|&x| x > i as f64)
      .unwrap_or(z[0].len() - 1);

    s[i] = s[i - 1]
      + mu * s[i - 1] * dt
      + s[i - 1] * v[i - 1].sqrt() * correlated_bms[0][i - 1]
      + z[2][jump_idx];

    let random: f64 = match use_sym.unwrap_or(false) {
      true => eta * (v[i]).abs().sqrt() * correlated_bms[1][i - 1],
      false => eta * (v[i]).max(0.0).sqrt() * correlated_bms[1][i - 1],
    };

    v[i] = v[i - 1] + kappa * (theta - v[i - 1]) * dt + random;
  }

  [s, v]
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_bates_1996() {
    let mu = 0.0;
    let kappa = 0.0;
    let theta = 0.0;
    let eta = 0.0;
    let rho = 0.0;
    let lambda = 1.0;
    let n = 1000;
    let t = 10.0;
    let b = bates_1996(
      mu,
      kappa,
      theta,
      eta,
      rho,
      lambda,
      n,
      None,
      None,
      Some(t),
      None,
    );
    assert_eq!(b[0].len(), n);
    assert_eq!(b[1].len(), n);
  }
}
