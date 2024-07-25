use ndarray::Array1;

use crate::{noises::fgn::FgnFft, utils::Generator};

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
/// A `Array1<f64>` representing the generated fJacobi process path.
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

#[derive(Default)]
pub struct Fjacobi {
  pub hurst: f64,
  pub alpha: f64,
  pub beta: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn fjacobi(params: &Fjacobi) -> Array1<f64> {
  let Fjacobi {
    hurst,
    alpha,
    beta,
    sigma,
    n,
    x0,
    t,
  } = *params;

  assert!(
    hurst > 0.0 && hurst < 1.0,
    "Hurst parameter must be in (0, 1)"
  );
  assert!(alpha > 0.0, "alpha must be positive");
  assert!(beta > 0.0, "beta must be positive");
  assert!(sigma > 0.0, "sigma must be positive");
  assert!(alpha < beta, "alpha must be less than beta");

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

  fjacobi
}
