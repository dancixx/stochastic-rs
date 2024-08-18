use ndarray::Array1;

use crate::{noises::fgn::FgnFft, utils::Generator};

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

#[derive(Default)]
pub struct Fcir {
  pub hurst: f64,
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
}

pub fn fcir(params: &Fcir) -> Array1<f64> {
  let Fcir {
    hurst,
    theta,
    mu,
    sigma,
    n,
    x0,
    t,
    use_sym,
  } = *params;

  assert!(
    hurst > 0.0 && hurst < 1.0,
    "Hurst parameter must be in (0, 1)"
  );
  assert!(2.0 * theta * mu < sigma.powi(2), "2 * theta * mu < sigma^2");

  let fgn = FgnFft::new(hurst, n, t, None).sample();
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut fcir = Array1::<f64>::zeros(n + 1);
  fcir[0] = x0.unwrap_or(0.0);

  for i in 1..(n + 1) {
    let random = match use_sym.unwrap_or(false) {
      true => sigma * (fcir[i - 1]).abs().sqrt() * fgn[i - 1],
      false => sigma * (fcir[i - 1]).max(0.0) * fgn[i - 1],
    };
    fcir[i] = fcir[i - 1] + theta * (mu - fcir[i - 1]) * dt + random
  }

  fcir
}
