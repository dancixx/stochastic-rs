use ndarray::Array1;
use stochastic_rs_macros::ImplNew;

use crate::stochastic::{diffusion::fou::FOU, Sampling};

#[derive(ImplNew)]
pub struct FVasicek {
  pub hurst: f64,
  pub mu: f64,
  pub sigma: f64,
  pub theta: Option<f64>,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fou: FOU,
}

impl Sampling<f64> for FVasicek {
  fn sample(&self) -> Array1<f64> {
    self.fou.sample()
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
