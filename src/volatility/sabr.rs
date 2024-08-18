use ndarray::Array1;

use crate::noises::cgns::{cgns, Cgns};

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
/// A tuple of two `Array1<f64>` representing the generated paths for the forward rate and volatility.
///
/// # Example
///
/// ```
/// let (forward_rate_path, volatility_path) = sabr(0.2, 0.5, -0.3, 1000, Some(0.04), Some(0.2), Some(1.0));
/// ```

#[derive(Default)]
pub struct Sabr {
  pub alpha: f64,
  pub beta: f64,
  pub rho: f64,
  pub n: usize,
  pub f0: Option<f64>,
  pub v0: Option<f64>,
  pub t: Option<f64>,
}

pub fn sabr(params: &Sabr) -> [Array1<f64>; 2] {
  let Sabr {
    alpha,
    beta,
    rho,
    n,
    f0,
    v0,
    t,
  } = *params;

  assert!(0.0 < beta && beta < 1.0, "Beta parameter must be in (0, 1)");
  assert!(-1.0 < rho && rho < 1.0, "Rho parameter must be in (-1, 1)");
  assert!(alpha > 0.0, "Alpha parameter must be positive");

  let [cgn1, cgn2] = cgns(&Cgns { rho, n, t });

  let mut f = Array1::<f64>::zeros(n + 1);
  let mut v = Array1::<f64>::zeros(n + 1);

  f[0] = f0.unwrap_or(0.0);
  v[0] = v0.unwrap_or(0.0);

  for i in 1..(n + 1) {
    f[i] = f[i - 1] + v[i - 1] * f[i - 1].powf(beta) * cgn1[i - 1];
    v[i] = v[i - 1] + alpha * v[i - 1] * cgn2[i - 1];
  }

  [f, v]
}
