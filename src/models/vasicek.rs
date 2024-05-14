use crate::diffusions::ou::{fou, ou};

pub fn vasicek(
  mu: f32,
  sigma: f32,
  theta: f32,
  n: usize,
  x0: Option<f32>,
  t: Option<f32>,
) -> Vec<f32> {
  if mu == 0.0 {
    panic!("mu must be non-zero");
  }

  ou(mu, sigma, theta, n, x0, t)
}

#[allow(clippy::too_many_arguments)]
pub fn fvasicek(
  hurst: f32,
  mu: f32,
  sigma: f32,
  theta: f32,
  n: usize,
  x0: Option<f32>,
  t: Option<f32>,
) -> Vec<f32> {
  if mu == 0.0 {
    panic!("mu must be non-zero");
  }

  fou(hurst, mu, sigma, theta, n, x0, t)
}
