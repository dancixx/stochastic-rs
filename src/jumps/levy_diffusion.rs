use crate::{noises::gn::gn, processes::poisson::compound_poisson};
use ndarray::Array1;

pub fn levy_diffusion(
  gamma: f64,
  sigma: f64,
  lambda: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<f64> {
  let dt = t.unwrap_or(1.0) / n as f64;
  let mut levy = Array1::<f64>::zeros(n);
  levy[0] = x0.unwrap_or(0.0);
  let gn = gn(n - 1, t);
  let z = compound_poisson(n, lambda, None, t, None, None);

  for i in 1..n {
    levy[i] = levy[i - 1] + gamma * dt + sigma * gn[i - 1] * z[i - 1];
  }

  levy.to_vec()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_levy_diffusion() {
    let gamma = 0.0;
    let sigma = 1.0;
    let lambda = 10.0;
    let n = 1000;
    let t = 10.0;
    let l = levy_diffusion(gamma, sigma, lambda, n, None, Some(t));
    assert_eq!(l.len(), n);
  }
}
