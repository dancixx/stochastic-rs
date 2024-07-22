use ndarray::Array1;

use crate::diffusions::ou::{fou, ou};

/// Generates a path of the Vasicek model.
///
/// The Vasicek model is a type of Ornstein-Uhlenbeck process used to model interest rates.
///
/// # Parameters
///
/// - `mu`: Long-term mean level, must be non-zero.
/// - `sigma`: Volatility parameter.
/// - `theta`: Speed of mean reversion.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated Vasicek process path.
///
/// # Panics
///
/// Panics if `mu` is zero.
///
/// # Example
///
/// ```
/// let vasicek_path = vasicek(0.1, 0.02, 0.3, 1000, Some(0.0), Some(1.0));
/// ```
pub fn vasicek(
  mu: f64,
  sigma: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Array1<f64> {
  if mu == 0.0 {
    panic!("mu must be non-zero");
  }

  ou(mu, sigma, theta, n, x0, t)
}

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
#[allow(clippy::too_many_arguments)]
pub fn fvasicek(
  hurst: f64,
  mu: f64,
  sigma: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Array1<f64> {
  if mu == 0.0 {
    panic!("mu must be non-zero");
  }

  fou(hurst, mu, sigma, theta, n, x0, t)
}
