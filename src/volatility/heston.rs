use ndarray::Array1;

use crate::noises::cgns::{cgns, Cgns};

/// Generates paths for the Heston model.
///
/// The Heston model is a stochastic volatility model used to describe the evolution of the volatility of an underlying asset.
///
/// # Parameters
///
/// - `mu`: Drift parameter of the asset price.
/// - `kappa`: Rate of mean reversion of the volatility.
/// - `theta`: Long-term mean level of the volatility.
/// - `eta`: Volatility of the volatility (vol of vol).
/// - `rho`: Correlation between the asset price and its volatility.
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
/// let paths = heston(0.05, 1.5, 0.04, 0.3, -0.7, 1000, Some(100.0), Some(0.04), Some(1.0), Some(false));
/// let asset_prices = paths[0];
/// let volatilities = paths[1];
/// ```

#[derive(Default)]
pub struct Heston {
  pub mu: f64,
  pub kappa: f64,
  pub theta: f64,
  pub eta: f64,
  pub rho: f64,
  pub n: usize,
  pub s0: Option<f64>,
  pub v0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
}

pub fn heston(params: &Heston) -> [Array1<f64>; 2] {
  let Heston {
    mu,
    kappa,
    theta,
    eta,
    rho,
    n,
    s0,
    v0,
    t,
    use_sym,
  } = *params;

  let [cgn1, cgn2] = cgns(&Cgns { rho, n, t });
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut s = Array1::<f64>::zeros(n);
  let mut v = Array1::<f64>::zeros(n);

  s[0] = s0.unwrap_or(0.0);
  v[0] = v0.unwrap_or(0.0);

  for i in 1..n {
    s[i] = s[i - 1] + mu * s[i - 1] * dt + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1];

    let random: f64 = match use_sym.unwrap_or(false) {
      true => eta * (v[i - 1]).abs().sqrt() * cgn2[i - 1],
      false => eta * (v[i - 1]).max(0.0).sqrt() * cgn2[i - 1],
    };
    v[i] = v[i - 1] + kappa * (theta - v[i - 1]) * dt + random;
  }

  [s, v]
}
