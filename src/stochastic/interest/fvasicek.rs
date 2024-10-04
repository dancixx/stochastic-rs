use ndarray::Array1;

use crate::stochastic::{diffusion::fou::FOU, Sampling};

#[derive(Default)]
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

impl FVasicek {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let fou = FOU::new(&FOU {
      hurst: params.hurst,
      mu: params.mu,
      sigma: params.sigma,
      theta: params.theta.unwrap_or(1.0),
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
      ..Default::default()
    });

    Self {
      hurst: params.hurst,
      mu: params.mu,
      sigma: params.sigma,
      theta: params.theta,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
      fou,
    }
  }
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
