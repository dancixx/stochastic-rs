use ndarray::Array1;

use crate::prelude::{correlated::correlated_bms, poisson::compound_poisson};

/// Bates (1996) model
// TODO: under development
#[allow(clippy::too_many_arguments)]
pub fn bates_1996(
  r: f64,
  lambda: usize,
  k: f64,
  rho: f64,
  eta: f64,
  kappa: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
) -> [Vec<f64>; 2] {
  let dt = t.unwrap_or(1.0) / n as f64;
  let mut s = Array1::<f64>::zeros(n + 1);
  let mut v = Array1::<f64>::zeros(n + 1);
  s[0] = x0.unwrap_or(0.0);
  v[0] = x0.unwrap_or(0.0);

  let z = compound_poisson(n, lambda, None);
  let correlated_bms = correlated_bms(rho, n, t);

  for i in 1..n {
    let random: f64 = match use_sym.unwrap_or(false) {
      true => eta * (v[i]).abs().sqrt() * correlated_bms[1][i - 1],
      false => eta * (v[i]).max(0.0).sqrt() * correlated_bms[1][i - 1],
    };
    v[i + 1] = v[i] + kappa * (theta - v[i]) * dt + random;
    s[i + 1] =
      (r - lambda as f64 * k - 0.5 * v[i]) * dt + v[i].sqrt() * correlated_bms[0][i - 1] + z[i];
  }

  [s.to_vec(), v.to_vec()]
}
