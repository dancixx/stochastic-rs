use ndarray::Array1;
use rand_distr::Distribution;

use crate::{diffusions::ou, prelude::poisson::compound_poisson};

/// Generates a path of the jump fractional Ornstein-Uhlenbeck (FOU) process.
///
/// The jump FOU process incorporates both the fractional Ornstein-Uhlenbeck dynamics and compound Poisson jumps,
/// which can be useful in various financial and physical modeling contexts.
///
/// # Parameters
///
/// - `hurst`: The Hurst parameter for the fractional Ornstein-Uhlenbeck process.
/// - `mu`: The mean reversion level.
/// - `sigma`: The volatility parameter.
/// - `theta`: The mean reversion speed.
/// - `lambda`: The jump intensity of the compound Poisson process.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated jump FOU process path.
///
/// # Example
///
/// ```
/// let jump_fou_path = jump_fou(0.1, 0.2, 0.5, 0.3, 0.5, 1000, None, Some(1.0));
/// ```

#[allow(clippy::too_many_arguments)]
pub fn jump_fou(
  hurst: f64,
  mu: f64,
  sigma: f64,
  theta: f64,
  lambda: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  jump_distr: impl Distribution<f64> + Copy,
) -> Array1<f64> {
  let dt = t.unwrap_or(1.0) / n as f64;
  let fou = ou::fou(hurst, mu, sigma, theta, n, x0, t);
  let mut jump_fou = Array1::<f64>::zeros(n);
  jump_fou[0] = fou[0];

  for i in 1..n {
    let [.., jumps] = compound_poisson(None, lambda, Some(dt), jump_distr);

    jump_fou[i] = fou[i - 1] + jumps.sum();
  }

  jump_fou
}

#[cfg(test)]
mod tests {
  use rand_distr::Normal;

  use super::*;

  #[test]
  fn test_jump_fou() {
    let hurst = 0.1;
    let mu = 0.2;
    let sigma = 0.5;
    let theta = 0.3;
    let lambda = 0.5;
    let n = 1000;
    let t = 1.0;
    let jf = jump_fou(
      hurst,
      mu,
      sigma,
      theta,
      lambda,
      n,
      None,
      Some(t),
      Normal::new(0.0, 1.0).unwrap(),
    );
    assert_eq!(jf.len(), n);
  }
}
