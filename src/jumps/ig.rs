use crate::noises::gn::gn;

use ndarray::Array1;
use ndarray_rand::{rand_distr::InverseGaussian, RandomExt};

pub fn ig(gamma: f64, n: usize, x0: Option<f64>, t: Option<f64>) -> Vec<f64> {
  let dt = t.unwrap_or(1.0) / n as f64;
  let gn = gn(n - 1, t);
  let mut ig = Array1::zeros(n);
  ig[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    ig[i] = ig[i - 1] + gamma * dt + gn[i - 1]
  }

  ig.to_vec()
}

pub fn nig(
  theta: f64,
  sigma: f64,
  kappa: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
) -> Vec<f64> {
  let dt = t.unwrap_or(1.0) / n as f64;
  let scale = dt.powf(2.0) / kappa;
  let mean = dt / scale;
  let ig = Array1::random(n - 1, InverseGaussian::new(mean, scale).unwrap());
  let gn = gn(n - 1, t);
  let mut nig = Array1::zeros(n);
  nig[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    nig[i] = nig[i - 1] + theta * ig[i - 1] + sigma * ig[i - 1].sqrt() * gn[i - 1]
  }

  nig.to_vec()
}
