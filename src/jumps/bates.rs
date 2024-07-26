use ndarray::Array1;
use rand_distr::Distribution;

use crate::prelude::{
  cbms::{correlated_bms, CorrelatedBms},
  cpoisson::{compound_poisson, CompoundPoisson},
};

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

#[derive(Default)]
pub struct Bates1996 {
  pub mu: f64,
  pub kappa: f64,
  pub theta: f64,
  pub eta: f64,
  pub rho: f64,
  pub lambda: f64,
  pub n: usize,
  pub s0: Option<f64>,
  pub v0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
}

pub fn bates_1996(
  params: &Bates1996,
  jump_distr: impl Distribution<f64> + Copy,
) -> [Array1<f64>; 2] {
  let Bates1996 {
    mu,
    kappa,
    theta,
    eta,
    rho,
    lambda,
    n,
    s0,
    v0,
    t,
    use_sym,
  } = *params;

  let correlated_bms = correlated_bms(&CorrelatedBms { rho, n, t });
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut s = Array1::<f64>::zeros(n);
  let mut v = Array1::<f64>::zeros(n);

  s[0] = s0.unwrap_or(0.0);
  v[0] = v0.unwrap_or(0.0);

  for i in 1..n {
    let [.., jumps] = compound_poisson(
      &CompoundPoisson {
        n: None,
        lambda,
        t_max: Some(dt),
      },
      jump_distr,
    );

    s[i] = s[i - 1]
      + mu * s[i - 1] * dt
      + s[i - 1] * v[i - 1].sqrt() * correlated_bms[0][i - 1]
      + jumps.sum();

    let random: f64 = match use_sym.unwrap_or(false) {
      true => eta * (v[i]).abs().sqrt() * correlated_bms[1][i - 1],
      false => eta * (v[i]).max(0.0).sqrt() * correlated_bms[1][i - 1],
    };

    v[i] = v[i - 1] + kappa * (theta - v[i - 1]) * dt + random;
  }

  [s, v]
}
