use ndarray::Array1;
use stochastic_rs_macros::ImplNew;

use crate::stochastic::{diffusion::ou::OU, Sampling};

#[derive(ImplNew)]
pub struct Vasicek {
  pub mu: f64,
  pub sigma: f64,
  pub theta: Option<f64>,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub ou: OU,
}

impl Sampling<f64> for Vasicek {
  fn sample(&self) -> Array1<f64> {
    self.ou.sample()
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
