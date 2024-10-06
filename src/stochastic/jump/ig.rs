use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling;

#[derive(ImplNew)]

pub struct IG {
  pub gamma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl Sampling<f64> for IG {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut ig = Array1::zeros(self.n);
    ig[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      ig[i] = ig[i - 1] + self.gamma * dt + gn[i - 1]
    }

    ig
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
