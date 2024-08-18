use crate::noises::gn::gn;

use derive_builder::Builder;
use ndarray::Array1;

/// Generates a path of the Inverse Gaussian (IG) process.
///
/// The IG process is used in various fields such as finance and engineering.
///
/// # Parameters
///
/// - `gamma`: Drift parameter.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated IG process path.
///
/// # Example
///
/// ```
/// let ig_path = ig(0.1, 1000, Some(0.0), Some(1.0));
/// ```

#[derive(Default, Builder)]
#[builder(setter(into))]
pub struct Ig {
  pub gamma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
}

pub fn ig(params: &Ig) -> Array1<f64> {
  let Ig { gamma, n, x0, t } = *params;
  let dt = t.unwrap_or(1.0) / n as f64;
  let gn = gn(n, t);
  let mut ig = Array1::zeros(n + 1);
  ig[0] = x0.unwrap_or(0.0);

  for i in 1..(n + 1) {
    ig[i] = ig[i - 1] + gamma * dt + gn[i - 1]
  }

  ig
}
