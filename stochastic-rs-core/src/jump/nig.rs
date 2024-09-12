use ndarray::Array1;
use ndarray_rand::{rand_distr::InverseGaussian, RandomExt};
use rand_distr::Normal;

use crate::Sampling;

#[derive(Default)]

pub struct Nig {
  pub theta: f64,
  pub sigma: f64,
  pub kappa: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl Nig {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      theta: params.theta,
      sigma: params.sigma,
      kappa: params.kappa,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
    }
  }
}

impl Sampling<f64> for Nig {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let scale = dt.powf(2.0) / self.kappa;
    let mean = dt / scale;
    let ig = Array1::random(self.n, InverseGaussian::new(mean, scale).unwrap());
    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut nig = Array1::zeros(self.n + 1);
    nig[0] = self.x0.unwrap_or(0.0);

    for i in 1..=self.n {
      nig[i] = nig[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * gn[i - 1]
    }

    nig
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
