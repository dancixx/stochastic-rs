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
/// A `Vec<f64>` representing the generated VG process path.
///
/// # Example
///
/// ```
/// use stochastic_rs::jumps::vg::vg;
///
/// let vg_path = vg(0.1, 0.2, 0.5, 1000, Some(0.0), Some(1.0));
/// ```
pub fn vg(mu: f64, sigma: f64, nu: f64, n: usize, x0: Option<f64>, t: Option<f64>) -> Vec<f64> {
  let dt = t.unwrap_or(1.0) / n as f64;

  let shape = dt / nu;
  let scale = nu;

  let mut vg = Array1::<f64>::zeros(n);
  vg[0] = x0.unwrap_or(0.0);

  let gn = gn::gn(n - 1, t);
  let gammas = Array1::random(n - 1, Gamma::new(shape, scale).unwrap());

  for i in 1..n {
    vg[i] = vg[i - 1] + mu * gammas[i - 1] + sigma * gammas[i - 1].sqrt() * gn[i - 1];
  }

  vg.to_vec()
}
