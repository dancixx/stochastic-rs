use ndarray::Array1;

use crate::{diffusions::fou::fou, diffusions::fou::Fou};

/// Generates a path of the fractional Vasicek (fVasicek) model.
///
/// The fVasicek model incorporates fractional Brownian motion into the Vasicek model.
///
/// # Parameters
///
/// - `hurst`: Hurst parameter for fractional Brownian motion, must be in (0, 1).
/// - `mu`: Long-term mean level, must be non-zero.
/// - `sigma`: Volatility parameter.
/// - `theta`: Speed of mean reversion.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated fVasicek process path.
///
/// # Panics
///
/// Panics if `mu` is zero.
///
/// # Example
///
/// ```
/// let fvasicek_path = fvasicek(0.75, 0.1, 0.02, 0.3, 1000, Some(0.0), Some(1.0));
/// ```

#[derive(Default)]
pub struct Fvasicek {
  pub hurst: f64,
  pub mu: f64,
  pub sigma: f64,
  pub theta: Option<f64>,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn fvasicek(params: &Fvasicek) -> Array1<f64> {
  let Fvasicek {
    hurst,
    mu,
    sigma,
    theta,
    n,
    x0,
    t,
  } = *params;

  assert!(mu != 0.0, "mu must be non-zero");

  fou(&Fou {
    hurst,
    mu,
    sigma,
    theta: theta.unwrap_or(1.0),
    n,
    x0,
    t,
  })
}
