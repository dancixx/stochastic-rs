use crate::noises::gn::gn;

use derive_builder::Builder;
use ndarray::Array1;
use ndarray_rand::{rand_distr::InverseGaussian, RandomExt};

/// Generates a path of the Normal Inverse Gaussian (NIG) process.
///
/// The NIG process is used in financial mathematics to model stock returns.
///
/// # Parameters
///
/// - `theta`: Drift parameter.
/// - `sigma`: Volatility parameter.
/// - `kappa`: Shape parameter of the Inverse Gaussian distribution.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated NIG process path.
///
/// # Example
///
/// ```
/// let nig_path = nig(0.1, 0.2, 0.5, 1000, Some(0.0), Some(1.0));
/// ```

#[derive(Default, Builder)]
#[builder(setter(into))]
pub struct Nig {
  theta: f64,
  sigma: f64,
  kappa: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
}

pub fn nig(params: &Nig) -> Array1<f64> {
  let Nig {
    theta,
    sigma,
    kappa,
    n,
    x0,
    t,
  } = *params;

  let dt = t.unwrap_or(1.0) / n as f64;
  let scale = dt.powf(2.0) / kappa;
  let mean = dt / scale;
  let ig = Array1::random(n, InverseGaussian::new(mean, scale).unwrap());
  let gn = gn(n, t);
  let mut nig = Array1::zeros(n + 1);
  nig[0] = x0.unwrap_or(0.0);

  for i in 1..(n + 1) {
    nig[i] = nig[i - 1] + theta * ig[i - 1] + sigma * ig[i - 1].sqrt() * gn[i - 1]
  }

  nig
}
