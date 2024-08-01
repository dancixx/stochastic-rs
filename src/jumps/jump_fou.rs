use ndarray::Array1;
use rand_distr::Distribution;

use crate::{
  noises::fgn::FgnFft,
  prelude::cpoisson::{compound_poisson, CompoundPoisson},
  utils::Generator,
};

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

#[derive(Default)]
pub struct JumpFou {
  pub hurst: f64,
  pub mu: f64,
  pub sigma: f64,
  pub theta: f64,
  pub lambda: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn jump_fou(params: &JumpFou, jump_distr: impl Distribution<f64> + Copy) -> Array1<f64> {
  let JumpFou {
    hurst,
    mu,
    sigma,
    theta,
    lambda,
    n,
    x0,
    t,
  } = *params;
  let dt = t.unwrap_or(1.0) / n as f64;
  let fgn = FgnFft::new(hurst, n - 1, t, None).sample();
  let mut jump_fou = Array1::<f64>::zeros(n);
  jump_fou[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    let [.., jumps] = compound_poisson(
      &CompoundPoisson {
        lambda,
        t_max: Some(dt),
        n: None,
      },
      jump_distr,
    );

    jump_fou[i] =
      jump_fou[i - 1] + theta * (mu - jump_fou[i - 1]) * dt + sigma * fgn[i - 1] + jumps.sum();
  }

  jump_fou
}
