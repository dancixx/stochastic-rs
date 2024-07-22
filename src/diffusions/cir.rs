use crate::{
  noises::{fgn::FgnFft, gn},
  utils::Generator,
};
use ndarray::Array1;

/// Generates a path of the Cox-Ingersoll-Ross (CIR) process.
///
/// The CIR process is commonly used in financial mathematics to model interest rates.
///
/// # Parameters
///
/// - `theta`: Speed of mean reversion.
/// - `mu`: Long-term mean level.
/// - `sigma`: Volatility parameter.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
/// - `use_sym`: Whether to use symmetric noise (optional, defaults to false).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated CIR process path.
///
/// # Panics
///
/// Panics if `2 * theta * mu < sigma^2`.
///
/// # Example
///
/// ```
/// let cir_path = cir(0.5, 0.02, 0.1, 1000, Some(0.01), Some(1.0), Some(false));
/// ```
pub fn cir(
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
) -> Array1<f64> {
  if 2.0 * theta * mu < sigma.powi(2) {
    panic!("2 * theta * mu < sigma^2")
  }

  let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut cir = Array1::<f64>::zeros(n);
  cir[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    let random = match use_sym.unwrap_or(false) {
      true => sigma * (cir[i - 1]).abs().sqrt() * gn[i - 1],
      false => sigma * (cir[i - 1]).max(0.0).sqrt() * gn[i - 1],
    };
    cir[i] = cir[i - 1] + theta * (mu - cir[i - 1]) * dt + random
  }

  cir
}

/// Generates a path of the fractional Cox-Ingersoll-Ross (fCIR) process.
///
/// The fCIR process incorporates fractional Brownian motion, which introduces long-range dependence.
///
/// # Parameters
///
/// - `hurst`: Hurst parameter for fractional Brownian motion, must be in (0, 1).
/// - `theta`: Speed of mean reversion.
/// - `mu`: Long-term mean level.
/// - `sigma`: Volatility parameter.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
/// - `use_sym`: Whether to use symmetric noise (optional, defaults to false).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated fCIR process path.
///
/// # Panics
///
/// Panics if `hurst` is not in (0, 1).
/// Panics if `2 * theta * mu < sigma^2`.
///
/// # Example
///
/// ```
/// let fcir_path = fcir(0.75, 0.5, 0.02, 0.1, 1000, Some(0.01), Some(1.0), Some(false));
/// ```
#[allow(clippy::too_many_arguments)]
pub fn fcir(
  hurst: f64,
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
) -> Array1<f64> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  if 2.0 * theta * mu < sigma.powi(2) {
    panic!("2 * theta * mu < sigma^2")
  }

  let fgn = FgnFft::new(hurst, n - 1, t, None).sample();
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut fcir = Array1::<f64>::zeros(n);
  fcir[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    let random = match use_sym.unwrap_or(false) {
      true => sigma * (fcir[i - 1]).abs().sqrt() * fgn[i - 1],
      false => sigma * (fcir[i - 1]).max(0.0) * fgn[i - 1],
    };
    fcir[i] = fcir[i - 1] + theta * (mu - fcir[i - 1]) * dt + random
  }

  fcir
}
