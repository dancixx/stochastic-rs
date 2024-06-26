use ndarray::Array1;

use crate::{
  noises::{fgn::FgnFft, gn},
  utils::Generator,
};

/// Generates a path of the Jacobi process.
///
/// The Jacobi process is a mean-reverting process used in various fields such as finance and biology.
///
/// # Parameters
///
/// - `alpha`: Speed of mean reversion.
/// - `beta`: Long-term mean level.
/// - `sigma`: Volatility parameter.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Vec<f64>` representing the generated Jacobi process path.
///
/// # Panics
///
/// Panics if `alpha`, `beta`, or `sigma` are not positive.
/// Panics if `alpha` is greater than `beta`.
///
/// # Example
///
/// ```
/// let jacobi_path = jacobi(0.5, 1.0, 0.2, 1000, Some(0.5), Some(1.0));
/// ```
pub fn jacobi(
  alpha: f64,
  beta: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<f64> {
  if alpha < 0.0 || beta < 0.0 || sigma < 0.0 {
    panic!("alpha, beta, and sigma must be positive")
  }

  if alpha > beta {
    panic!("alpha must be less than beta")
  }

  let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut jacobi = Array1::<f64>::zeros(n);
  jacobi[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    jacobi[i] = match jacobi[i] {
      _ if jacobi[i - 1] <= 0.0 && i > 0 => 0.0,
      _ if jacobi[i - 1] >= 1.0 && i > 0 => 1.0,
      _ => {
        jacobi[i - 1]
          + (alpha - beta * jacobi[i - 1]) * dt
          + sigma * (jacobi[i - 1] * (1.0 - jacobi[i - 1])).sqrt() * gn[i - 1]
      }
    }
  }

  jacobi.to_vec()
}

/// Generates a path of the fractional Jacobi (fJacobi) process.
///
/// The fJacobi process incorporates fractional Brownian motion, which introduces long-range dependence.
///
/// # Parameters
///
/// - `hurst`: Hurst parameter for fractional Brownian motion, must be in (0, 1).
/// - `alpha`: Speed of mean reversion.
/// - `beta`: Long-term mean level.
/// - `sigma`: Volatility parameter.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Vec<f64>` representing the generated fJacobi process path.
///
/// # Panics
///
/// Panics if `hurst` is not in (0, 1).
/// Panics if `alpha`, `beta`, or `sigma` are not positive.
/// Panics if `alpha` is greater than `beta`.
///
/// # Example
///
/// ```
/// let fjacobi_path = fjacobi(0.75, 0.5, 1.0, 0.2, 1000, Some(0.5), Some(1.0));
/// ```
#[allow(clippy::too_many_arguments)]
pub fn fjacobi(
  hurst: f64,
  alpha: f64,
  beta: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<f64> {
  if !(0.0..1.0).contains(&hurst) {
    panic!("Hurst parameter must be in (0, 1)")
  }

  if alpha < 0.0 || beta < 0.0 || sigma < 0.0 {
    panic!("alpha, beta, and sigma must be positive")
  }

  if alpha > beta {
    panic!("alpha must be less than beta")
  }

  let fgn = FgnFft::new(hurst, n - 1, t, None).sample();
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut fjacobi = Array1::<f64>::zeros(n);
  fjacobi[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    fjacobi[i] = match fjacobi[i - 1] {
      _ if fjacobi[i - 1] <= 0.0 && i > 0 => 0.0,
      _ if fjacobi[i - 1] >= 1.0 && i > 0 => 1.0,
      _ => {
        fjacobi[i - 1]
          + (alpha - beta * fjacobi[i - 1]) * dt
          + sigma * (fjacobi[i - 1] * (1.0 - fjacobi[i - 1])).sqrt() * fgn[i - 1]
      }
    }
  }

  fjacobi.to_vec()
}
