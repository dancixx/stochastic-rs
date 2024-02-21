use ndarray::Array1;

use crate::prelude::{correlated::correlated_bms, poisson::compound_poisson};

/// Bates (1996) model
#[allow(clippy::too_many_arguments)]
pub fn bates_1996(
  mu: f64,
  kappa: f64,
  theta: f64,
  eta: f64,
  rho: f64,
  lambda: f64,
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
  let z = compound_poisson(n, lambda, None, t, None, None);
  println!("{:?}", z.len());

  s[0] = s0.unwrap_or(0.0);
  v[0] = v0.unwrap_or(0.0);

  for i in 1..n {
    s[i] = s[i - 1]
      + mu * s[i - 1] * dt
      + s[i - 1] * v[i - 1].sqrt() * correlated_bms[0][i - 1]
      + z[i - 1];

    let random: f64 = match use_sym.unwrap_or(false) {
      true => eta * (v[i]).abs().sqrt() * correlated_bms[1][i - 1],
      false => eta * (v[i]).max(0.0).sqrt() * correlated_bms[1][i - 1],
    };
    v[i] = v[i - 1] + kappa * (theta - v[i - 1]) * dt + random;
  }

  [s.to_vec(), v.to_vec()]
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_bates_1996() {
    let mu = 0.0;
    let kappa = 0.0;
    let theta = 0.0;
    let eta = 0.0;
    let rho = 0.0;
    let lambda = 1.0;
    let n = 1000;
    let t = 10.0;
    let b = bates_1996(
      mu,
      kappa,
      theta,
      eta,
      rho,
      lambda,
      n,
      None,
      None,
      Some(t),
      None,
    );
    assert_eq!(b[0].len(), n);
    assert_eq!(b[1].len(), n);
  }
}
