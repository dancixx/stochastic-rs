use ndarray::Array1;

use crate::prelude::{correlated::correlated_bms, poisson::compound_poisson};

/// Bates (1996) model
// TODO: under development
#[allow(clippy::too_many_arguments)]
pub fn bates_1996(
  mu: f64,
  kappa: f64,
  theta: f64,
  eta: f64,
  rho: f64,
  lambda: usize,
  n: usize,
  s0: Option<f64>,
  v0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
) -> [Vec<f64>; 2] {
  let correlated_bms = correlated_bms(rho, n, t);
  let dt = t.unwrap_or(1.0) / n as f64;

  let mut s = Array1::<f64>::zeros(n);
  let mut v = Array1::<f64>::zeros(n);

  let [jump_times, jumps] = compound_poisson(n, lambda, None, Some(t.unwrap_or(1.0)), None, None);

  s[0] = s0.unwrap_or(0.0);
  v[0] = v0.unwrap_or(0.0);

  for i in 1..n {
    // find the index of the last jump before time t_i
    let mut j = 0;
    while jump_times[j] < i as f64 * dt {
      j += 1;
    }

    s[i] = s[i - 1]
      + mu * s[i - 1] * dt
      + s[i - 1] * v[i - 1].sqrt() * correlated_bms[0][i - 1]
      + jumps[j];

    let random: f64 = match use_sym.unwrap_or(false) {
      true => eta * (v[i]).abs().sqrt() * correlated_bms[1][i - 1],
      false => eta * (v[i]).max(0.0).sqrt() * correlated_bms[1][i - 1],
    };
    v[i] = v[i - 1] + kappa * (theta - v[i - 1]) * dt + random;
  }

  [s.to_vec(), v.to_vec()]
}
