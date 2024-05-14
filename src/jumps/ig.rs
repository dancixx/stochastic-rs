use crate::noises::gn::gn;

use ndarray::Array1;
use ndarray_rand::{rand_distr::InverseGaussian, RandomExt};

pub fn ig(gamma: f32, n: usize, x0: Option<f32>, t: Option<f32>) -> Vec<f32> {
  let dt = t.unwrap_or(1.0) / n as f32;
  let gn = gn(n - 1, t);
  let mut ig = Array1::zeros(n);
  ig[0] = x0.unwrap_or(0.0);

  for i in 1..n {
    ig[i] = ig[i - 1] + gamma * dt + gn[i - 1]
  }

  ig.to_vec()
}

pub fn nig(
  theta: f32,
  sigma: f32,
  kappa: f32,
  n: usize,
  x0: Option<f32>,
  t: Option<f32>,
) -> Vec<f32> {
  let dt = t.unwrap_or(1.0) / n as f32;
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
