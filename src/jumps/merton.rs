use ndarray::Array1;

use crate::{noises::gn::gn, processes::poisson::compound_poisson};

pub fn merton(
  alpha: f64,
  sigma: f64,
  lambda: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<f64> {
  let dt = t.unwrap_or(1.0) / n as f64;
  let mut merton = Array1::<f64>::zeros(n);
  merton[0] = x0.unwrap_or(0.0);
  let gn = gn(n - 1, t);
  let z = compound_poisson(n, lambda, None, t, None, None);

  for i in 1..n {
    merton[i] = merton[i - 1]
      + (alpha * sigma.powf(2.0) / 2.0 - lambda * theta) * dt
      + sigma * gn[i - 1]
      + z[i - 1];
  }

  merton.to_vec()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_merton() {
    let alpha = 0.0;
    let sigma = 1.0;
    let lambda = 10.0;
    let theta = 0.0;
    let n = 1000;
    let t = 10.0;
    let m = merton(alpha, sigma, lambda, theta, n, None, Some(t));
    assert_eq!(m.len(), n);
  }
}
