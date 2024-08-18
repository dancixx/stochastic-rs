use crate::noises::gn;
use ndarray::Array1;
use ndarray_rand::rand_distr::Gamma;
use ndarray_rand::RandomExt;

/// Generates a path of the Variance Gamma (VG) process.
///
/// The VG process is used in financial modeling to capture excess kurtosis and skewness in asset returns.
///
/// # Parameters
///
/// - `mu`: Drift parameter.
/// - `sigma`: Volatility parameter.
/// - `nu`: Variance rate parameter.
/// - `n`: Number of time steps.
/// - `x0`: Initial value of the process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated VG process path.
///
/// # Example
///
/// ```
/// let vg_path = vg(0.1, 0.2, 0.5, 1000, Some(0.0), Some(1.0));
/// ```

#[derive(Default)]
pub struct Vg {
  mu: f64,
  sigma: f64,
  nu: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
}

pub fn vg(params: &Vg) -> Array1<f64> {
  let Vg {
    mu,
    sigma,
    nu,
    n,
    x0,
    t,
  } = *params;

  let dt = t.unwrap_or(1.0) / n as f64;

  let shape = dt / nu;
  let scale = nu;

  let mut vg = Array1::<f64>::zeros(n + 1);
  vg[0] = x0.unwrap_or(0.0);

  let gn = gn::gn(n, t);
  let gammas = Array1::random(n, Gamma::new(shape, scale).unwrap());

  for i in 1..(n + 1) {
    vg[i] = vg[i - 1] + mu * gammas[i - 1] + sigma * gammas[i - 1].sqrt() * gn[i - 1];
  }

  vg
}
