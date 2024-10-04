use ndarray::Array1;

use crate::stochastic::{diffusion::ou::OU, Sampling};

#[derive(Default)]
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

impl Vasicek {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let ou = OU::new(&OU {
      mu: params.mu,
      sigma: params.sigma,
      theta: params.theta.unwrap_or(1.0),
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
    });

    Self {
      mu: params.mu,
      sigma: params.sigma,
      theta: params.theta,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
      ou,
    }
  }
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
