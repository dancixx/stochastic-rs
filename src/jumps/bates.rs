use derive_builder::Builder;
use ndarray::Array1;
use rand_distr::Distribution;

use crate::{
  noises::cgns::{cgns, Cgns},
  processes::cpoisson::{compound_poisson, CompoundPoisson},
};

/// Generates paths for the Bates (1996) model.
///
/// The Bates model combines a stochastic volatility model with jump diffusion,
/// commonly used in financial mathematics to model asset prices.
///
/// # Parameters
///
/// - `mu`: Drift parameter of the asset price.
/// - `b`: The continuously compounded domestic/foreign interest rate differential.
/// - `r`: The continuously compounded risk-free interest rate.
/// - `r_f`: The continuously compounded foreign interest rate.
/// - `lambda`: Jump intensity.
/// - `k`: Mean jump size.
/// - `alpha`: Rate of mean reversion of the volatility.
/// - `beta`: Long-term mean level of the volatility.
/// - `sigma`: Volatility of the volatility (vol of vol).
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
/// let params = Bates1996 {
///     mu: 0.05,
///     lambda: 0.1,
///     k: 0.2,
///     alpha: 1.5,
///     beta: 0.04,
///     sigma: 0.3,
///     rho: -0.7,
///     n: 1000,
///     s0: Some(100.0),
///     v0: Some(0.04),
///     t: Some(1.0),
///     use_sym: Some(false),
/// };
///
/// let jump_distr = Normal::new(0.0, 1.0); // Example jump distribution
/// let paths = bates_1996(&params, jump_distr);
/// let asset_prices = paths[0];
/// let volatilities = paths[1];
/// ```
///
/// # Panics
///
/// This function will panic if the `correlated_bms` or `compound_poisson` functions return invalid lengths or if there are issues with array indexing.

#[derive(Default, Builder)]
#[builder(setter(into))]
pub struct Bates1996 {
  pub mu: Option<f64>,
  pub b: Option<f64>,
  pub r: Option<f64>,
  pub r_f: Option<f64>,
  pub lambda: f64,
  pub k: f64,
  pub alpha: f64,
  pub beta: f64,
  pub sigma: f64,
  pub rho: f64,
  pub n: usize,
  pub s0: Option<f64>,
  pub v0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
}

pub fn bates_1996<D>(params: &Bates1996, jdistr: D) -> [Array1<f64>; 2]
where
  D: Distribution<f64> + Copy,
{
  let Bates1996 {
    mu,
    b,
    r,
    r_f,
    lambda,
    k,
    alpha,
    beta,
    sigma,
    rho,
    n,
    s0,
    v0,
    t,
    use_sym,
  } = *params;

  let [cgn1, cgn2] = cgns(&Cgns { rho, n, t });
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut s = Array1::<f64>::zeros(n + 1);
  let mut v = Array1::<f64>::zeros(n + 1);

  s[0] = s0.unwrap_or(0.0);
  v[0] = v0.unwrap_or(0.0);

  let drift = match (mu, b, r, r_f) {
    (Some(r), Some(r_f), ..) => r - r_f,
    (Some(b), ..) => b,
    _ => mu.unwrap(),
  };

  for i in 1..(n + 1) {
    let [.., jumps] = compound_poisson(
      &CompoundPoisson {
        n: None,
        lambda,
        t_max: Some(dt),
      },
      jdistr,
    );

    let sqrt_v = use_sym
      .unwrap_or(false)
      .then(|| v[i - 1].abs())
      .unwrap_or(v[i - 1].max(0.0))
      .sqrt();

    s[i] = s[i - 1]
      + (drift - lambda * k) * s[i - 1] * dt
      + s[i - 1] * sqrt_v * cgn1[i - 1]
      + jumps.sum();

    v[i] = v[i - 1] + (alpha - beta * v[i - 1]) * dt + sigma * v[i - 1] * cgn2[i - 1];
  }

  [s, v]
}
