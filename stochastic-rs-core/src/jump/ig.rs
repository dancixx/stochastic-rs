use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::Sampling;

#[derive(Default)]

pub struct Ig {
  pub gamma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl Ig {
  #[must_use]
  pub fn new(params: &Ig) -> Self {
    Self {
      gamma: params.gamma,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
    }
  }
}

impl Sampling<f64> for Ig {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let gn = Array1::random(
      self.n,
      Normal::new(0.0, (self.t.unwrap_or(1.0) / self.n as f64).sqrt()).unwrap(),
    );
    let mut ig = Array1::zeros(self.n + 1);
    ig[0] = self.x0.unwrap_or(0.0);

    for i in 1..(self.n + 1) {
      ig[i] = ig[i - 1] + self.gamma * dt + gn[i - 1]
    }

    ig
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
