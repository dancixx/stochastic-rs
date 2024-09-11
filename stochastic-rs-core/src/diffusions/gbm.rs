use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::Sampling;

pub struct Gbm {
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl Gbm {
  pub fn new(params: &Self) -> Self {
    Self {
      mu: params.mu,
      sigma: params.sigma,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
    }
  }
}

impl Sampling<f64> for Gbm {
  fn sample(&self) -> Array1<f64> {
    let gn = Array1::random(
      self.n,
      Normal::new(0.0, (self.t.unwrap_or(1.0) / self.n as f64).sqrt()).unwrap(),
    );

    let dt = self.t.unwrap_or(1.0) / self.n as f64;

    let mut gbm = Array1::<f64>::zeros(self.n + 1);
    gbm[0] = self.x0.unwrap_or(100.0);

    for i in 1..(self.n + 1) {
      gbm[i] = gbm[i - 1] + self.mu * gbm[i - 1] * dt + self.sigma * gbm[i - 1] * gn[i - 1]
    }

    gbm
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
