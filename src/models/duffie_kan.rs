use ndarray::Array1;

use crate::processes::correlated::correlated_bms;

/// Generates paths for the Duffie-Kan multifactor interest rate model.
///
/// The Duffie-Kan model is a multifactor interest rate model incorporating correlated Brownian motions,
/// used in financial mathematics for modeling interest rates.
///
/// # Parameters
///
/// - `alpha`: The drift term coefficient for the Brownian motion.
/// - `beta`: The drift term coefficient for the Brownian motion.
/// - `gamma`: The drift term coefficient for the Brownian motion.
/// - `rho`: The correlation between the two Brownian motions.
/// - `a1`: The coefficient for the `r` term in the drift of `r`.
/// - `b1`: The coefficient for the `x` term in the drift of `r`.
/// - `c1`: The constant term in the drift of `r`.
/// - `sigma1`: The diffusion coefficient for the `r` process.
/// - `a2`: The coefficient for the `r` term in the drift of `x`.
/// - `b2`: The coefficient for the `x` term in the drift of `x`.
/// - `c2`: The constant term in the drift of `x`.
/// - `sigma2`: The diffusion coefficient for the `x` process.
/// - `n`: Number of time steps.
/// - `r0`: Initial value of the `r` process (optional, defaults to 0.0).
/// - `x0`: Initial value of the `x` process (optional, defaults to 0.0).
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A tuple of two `Vec<f64>` representing the generated paths for the `r` and `x` processes.
///
/// # Example
///
/// ```
/// let (r_path, x_path) = duffie_kan(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1000, Some(0.05), Some(0.02), Some(1.0));
/// ```
#[allow(clippy::many_single_char_names)]
pub fn duffie_kan(
  alpha: f64,
  beta: f64,
  gamma: f64,
  rho: f64,
  a1: f64,
  b1: f64,
  c1: f64,
  sigma1: f64,
  a2: f64,
  b2: f64,
  c2: f64,
  sigma2: f64,
  n: usize,
  r0: Option<f64>,
  x0: Option<f64>,
  t: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
  let correlated_bms = correlated_bms(rho, n, t);
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut r = Array1::<f64>::zeros(n);
  let mut x = Array1::<f64>::zeros(n);

  r[0] = r0.unwrap_or(0.0);
  x[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    r[i] = r[i - 1]
      + (a1 * r[i - 1] + b1 * x[i - 1] + c1) * dt
      + sigma1 * (alpha * r[i - 1] + beta * x[i - 1] + gamma) * correlated_bms[0][i - 1];
    x[i] = x[i - 1]
      + (a2 * r[i - 1] + b2 * x[i - 1] + c2) * dt
      + sigma2 * (alpha * r[i - 1] + beta * x[i - 1] + gamma) * correlated_bms[1][i - 1];
  }

  (r.to_vec(), x.to_vec())
}
