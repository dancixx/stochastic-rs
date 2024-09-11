use derive_builder::Builder;
use ndarray::Array1;
use rand_distr::Distribution;

use crate::{
  noises::fgn::FgnFft,
  processes::cpoisson::{compound_poisson, CompoundPoissonBuilder},
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

#[derive(Default, Builder)]
#[builder(setter(into))]
pub struct JumpFou {
  pub hurst: f64,
  pub mu: f64,
  pub sigma: f64,
  pub theta: f64,
  pub lambda: Option<f64>,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn jump_fou<D>(params: &JumpFou, jdistr: D) -> Array1<f64>
where
  D: Distribution<f64> + Copy,
{
  let JumpFou {
    hurst,
    mu,
    sigma,
    theta,
    lambda,
    n,
    x0,
    t,
  } = params;
  let dt = t.unwrap_or(1.0) / *n as f64;
  let fgn = FgnFft::new(*hurst, *n, *t, None).sample();
  let mut jump_fou = Array1::<f64>::zeros(*n + 1);
  jump_fou[0] = x0.unwrap_or(0.0);

  for i in 1..(*n + 1) {
    let [.., jumps] = compound_poisson(
      &CompoundPoissonBuilder::default()
        .lambda(lambda.unwrap())
        .t_max(dt)
        .n(*n)
        .build()
        .unwrap(),
      jdistr,
    );

    jump_fou[i] =
      jump_fou[i - 1] + theta * (mu - jump_fou[i - 1]) * dt + sigma * fgn[i - 1] + jumps.sum();
  }

  jump_fou
}
