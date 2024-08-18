use crate::noises::gn;
use ndarray::Array1;

/// Generates a path of the Ornstein-Uhlenbeck (OU) process.
///
/// The OU process is a mean-reverting stochastic process used in various fields such as finance and physics.
///
/// # Parameters
///
/// - `mu`: Long-term mean level.
/// - `sigma`: Volatility parameter.
/// - `theta`: Speed of mean reversion.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated OU process path.
///
/// # Example
///
/// ```
/// let ou_path = ou(0.0, 0.1, 0.5, 1000, Some(0.0), Some(1.0));
/// ```

#[derive(Default)]
pub struct Ou {
  pub mu: f64,
  pub sigma: f64,
  pub theta: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn ou(params: &Ou) -> Array1<f64> {
  let Ou {
    mu,
    sigma,
    theta,
    n,
    x0,
    t,
  } = *params;

  let gn = gn::gn(n, Some(t.unwrap_or(1.0)));
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut ou = Array1::<f64>::zeros(n + 1);
  ou[0] = x0.unwrap_or(0.0);

  for i in 1..(n + 1) {
    ou[i] = ou[i - 1] + theta * (mu - ou[i - 1]) * dt + sigma * gn[i - 1]
  }

  ou
}
