use ndarray::Array1;

use crate::prelude::correlated::correlated_bms;

/// Generates a path of the SABR (Stochastic Alpha, Beta, Rho) model.
///
/// The SABR model is widely used in financial mathematics for modeling stochastic volatility.
/// It incorporates correlated Brownian motions to simulate the underlying asset price and volatility.
///
/// # Parameters
///
/// - `alpha`: The volatility of volatility.
/// - `beta`: The elasticity parameter, must be in the range (0, 1).
/// - `rho`: The correlation between the asset price and volatility, must be in the range (-1, 1).
/// - `n`: Number of time steps.
/// - `f0`: Initial value of the forward rate (optional, defaults to 0.0).
/// - `v0`: Initial value of the volatility (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A tuple of two `Vec<f64>` representing the generated paths for the forward rate and volatility.
///
/// # Example
///
/// ```
/// let (forward_rate_path, volatility_path) = sabr(0.2, 0.5, -0.3, 1000, Some(0.04), Some(0.2), Some(1.0));
/// ```
pub fn sabr(
  alpha: f64,
  beta: f64,
  rho: f64,
  n: usize,
  f0: Option<f64>,
  v0: Option<f64>,
  t: Option<f64>,
) -> [Array1<f64>; 2] {
  if !(0.0..1.0).contains(&beta) {
    panic!("Beta parameter must be in (0, 1)")
  }

  if !(-1.0..1.0).contains(&rho) {
    panic!("Rho parameter must be in (-1, 1)")
  }

  if alpha < 0.0 {
    panic!("Alpha parameter must be positive")
  }

  let correlated_bms = correlated_bms(rho, n, t);

  let mut f = Array1::<f64>::zeros(n);
  let mut v = Array1::<f64>::zeros(n);

  f[0] = f0.unwrap_or(0.0);
  v[0] = v0.unwrap_or(0.0);

  for i in 0..n {
    f[i] = f[i - 1] + v[i - 1] * f[i - 1].powf(beta) * correlated_bms[0][i - 1];
    v[i] = v[i - 1] + alpha * v[i - 1] * correlated_bms[1][i - 1];
  }

  [f, v]
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sabr() {
    let alpha = 0.2;
    let beta = 0.5;
    let rho = -0.3;
    let n = 1000;
    let f0 = Some(0.04);
    let v0 = Some(0.2);
    let t = Some(1.0);
    let [f, v] = sabr(alpha, beta, rho, n, f0, v0, t);
    assert_eq!(f.len(), n);
    assert_eq!(v.len(), n);
  }
}
