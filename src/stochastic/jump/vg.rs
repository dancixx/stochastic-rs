use ndarray::Array1;
use ndarray_rand::rand_distr::Gamma;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling;

#[derive(Default)]
pub struct Vg {
  pub mu: f64,
  pub sigma: f64,
  pub nu: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl Vg {
  #[must_use]
  pub fn new(params: &Vg) -> Self {
    Self {
      mu: params.mu,
      sigma: params.sigma,
      nu: params.nu,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
    }
  }
}

impl Sampling<f64> for Vg {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / self.n as f64;

    let shape = dt / self.nu;
    let scale = self.nu;

    let mut vg = Array1::<f64>::zeros(self.n + 1);
    vg[0] = self.x0.unwrap_or(0.0);

    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());
    let gammas = Array1::random(self.n, Gamma::new(shape, scale).unwrap());

    for i in 1..=self.n {
      vg[i] = vg[i - 1] + self.mu * gammas[i - 1] + self.sigma * gammas[i - 1].sqrt() * gn[i - 1];
    }

    vg
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
