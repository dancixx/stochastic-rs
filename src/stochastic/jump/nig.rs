use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::{rand_distr::InverseGaussian, RandomExt};
use rand_distr::Normal;

use crate::stochastic::Sampling;

#[derive(ImplNew)]

pub struct NIG {
  pub theta: f64,
  pub sigma: f64,
  pub kappa: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl Sampling<f64> for NIG {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let scale = dt.powf(2.0) / self.kappa;
    let mean = dt / scale;
    let ig = Array1::random(self.n - 1, InverseGaussian::new(mean, scale).unwrap());
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut nig = Array1::zeros(self.n);
    nig[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      nig[i] = nig[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * gn[i - 1]
    }

    nig
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

#[cfg(test)]
mod tests {
  use crate::{
    plot_1d,
    stochastic::{N, X0},
  };

  use super::*;

  #[test]
  fn nig_length_equals_n() {
    let nig = NIG::new(2.25, 2.5, 1.0, N, Some(X0), Some(100.0), None);
    assert_eq!(nig.sample().len(), N);
  }

  #[test]
  fn nig_starts_with_x0() {
    let nig = NIG::new(2.25, 2.5, 1.0, N, Some(X0), Some(100.0), None);
    assert_eq!(nig.sample()[0], X0);
  }

  #[test]
  fn nig_plot() {
    let nig = NIG::new(2.25, 2.5, 1.0, N, Some(X0), Some(100.0), None);
    plot_1d!(nig.sample(), "Normal Inverse Gaussian (NIG)");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn nig_malliavin() {
    unimplemented!()
  }
}
