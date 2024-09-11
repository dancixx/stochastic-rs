use derive_builder::Builder;
use ndarray::Array1;

use crate::{diffusions::ou::ou, diffusions::ou::Ou};

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

#[derive(Default, Builder)]
#[builder(setter(into))]
pub struct Vasicek {
  pub mu: f64,
  pub sigma: f64,
  pub theta: Option<f64>,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn vasicek(params: &Vasicek) -> Array1<f64> {
  let Vasicek {
    mu,
    sigma,
    theta,
    n,
    x0,
    t,
  } = *params;

  assert!(mu != 0.0, "mu must be non-zero");

  ou(&Ou {
    mu,
    sigma,
    theta: theta.unwrap_or(1.0),
    n,
    x0,
    t,
  })
}
