use ndarray::Array1;

use crate::processes::correlated;

#[allow(clippy::too_many_arguments)]
pub fn heston(
  mu: f64,
  kappa: f64,
  theta: f64,
  eta: f64,
  rho: f64,
  n: usize,
  s0: Option<f64>,
  v0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
) -> [Vec<f64>; 2] {
  let correlated_bms = correlated::correlated_bms(rho, n, t);
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut s = Array1::<f64>::zeros(n + 1);
  let mut v = Array1::<f64>::zeros(n + 1);

  s[0] = s0.unwrap_or(0.0);
  v[0] = v0.unwrap_or(0.0);

  for i in 1..n + 1 {
    s[i + 1] = s[i] + mu * s[i] * dt + s[i] * v[i].sqrt() * correlated_bms[0][i - 1];

    let random: f64 = match use_sym.unwrap_or(false) {
      true => eta * (v[i]).abs().sqrt() * correlated_bms[1][i - 1],
      false => eta * (v[i]).max(0.0).sqrt() * correlated_bms[1][i - 1],
    };
    v[i + 1] = v[i] + kappa * (theta - v[i]) * dt + random;
  }

  [s.to_vec(), v.to_vec()]
}
