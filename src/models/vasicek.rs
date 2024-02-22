use crate::diffusions::ou::{fou, ou};

pub fn vasicek(
  mu: f64,
  sigma: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<f64> {
  if mu == 0.0 {
    panic!("mu must be non-zero");
  }

  ou(mu, sigma, theta, n, x0, t)
}

#[allow(clippy::too_many_arguments)]
pub fn fvasicek(
  hurst: f64,
  mu: f64,
  sigma: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<f64> {
  if mu == 0.0 {
    panic!("mu must be non-zero");
  }

  fou(hurst, mu, sigma, theta, n, x0, t)
}
