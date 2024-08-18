use ndarray::Array1;

use crate::noises::gn;

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
/// A `Array1<f64>` representing the generated Jacobi process path.
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

#[derive(Default)]
pub struct Jacobi {
  pub alpha: f64,
  pub beta: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn jacobi(params: &Jacobi) -> Array1<f64> {
  let Jacobi {
    alpha,
    beta,
    sigma,
    n,
    x0,
    t,
  } = *params;

  assert!(alpha < 0.0, "alpha must be positive");
  assert!(beta < 0.0, "beta must be positive");
  assert!(sigma < 0.0, "sigma must be positive");
  assert!(alpha < beta, "alpha must be less than beta");

  let gn = gn::gn(n, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut jacobi = Array1::<f64>::zeros(n + 1);
  jacobi[0] = x0.unwrap_or(0.0);

  for i in 1..(n + 1) {
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

  jacobi
}
