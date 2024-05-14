use ndarray::Array1;

use crate::prelude::correlated::correlated_bms;

#[allow(clippy::too_many_arguments)]
pub fn heston(
  mu: f32,
  kappa: f32,
  theta: f32,
  eta: f32,
  rho: f32,
  n: usize,
  s0: Option<f32>,
  v0: Option<f32>,
  t: Option<f32>,
  use_sym: Option<bool>,
) -> [Vec<f32>; 2] {
  let correlated_bms = correlated_bms(rho, n, t);
  let dt = t.unwrap_or(1.0) / n as f32;

  let mut s = Array1::<f32>::zeros(n);
  let mut v = Array1::<f32>::zeros(n);

  s[0] = s0.unwrap_or(0.0);
  v[0] = v0.unwrap_or(0.0);

  for i in 1..n {
    s[i] = s[i - 1] + mu * s[i - 1] * dt + s[i - 1] * v[i - 1].sqrt() * correlated_bms[0][i - 1];

    let random: f32 = match use_sym.unwrap_or(false) {
      true => eta * (v[i - 1]).abs().sqrt() * correlated_bms[1][i - 1],
      false => eta * (v[i - 1]).max(0.0).sqrt() * correlated_bms[1][i - 1],
    };
    v[i] = v[i - 1] + kappa * (theta - v[i - 1]) * dt + random;
  }

  [s.to_vec(), v.to_vec()]
}
