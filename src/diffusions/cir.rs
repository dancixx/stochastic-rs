use crate::noises::gn;
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

#[derive(Default)]
pub struct Cir {
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
}

pub fn cir(params: &Cir) -> Array1<f64> {
  let Cir {
    theta,
    mu,
    sigma,
    n,
    x0,
    t,
    use_sym,
  } = *params;

  assert!(2.0 * theta * mu < sigma.powi(2), "2 * theta * mu < sigma^2");

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
