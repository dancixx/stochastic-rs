use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::rand_distr::Gamma;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling;

#[derive(ImplNew)]
pub struct VG {
  pub mu: f64,
  pub sigma: f64,
  pub nu: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl Sampling<f64> for VG {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;

    let shape = dt / self.nu;
    let scale = self.nu;

    let mut vg = Array1::<f64>::zeros(self.n);
    vg[0] = self.x0.unwrap_or(0.0);

    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let gammas = Array1::random(self.n - 1, Gamma::new(shape, scale).unwrap());

    for i in 1..self.n {
      vg[i] = vg[i - 1] + self.mu * gammas[i - 1] + self.sigma * gammas[i - 1].sqrt() * gn[i - 1];
    }

    vg
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
