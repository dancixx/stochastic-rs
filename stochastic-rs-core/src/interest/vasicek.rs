use ndarray::Array1;

use crate::{diffusions::ou::Ou, Sampling};

#[derive(Default)]
pub struct Vasicek {
  pub mu: f64,
  pub sigma: f64,
  pub theta: Option<f64>,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  ou: Ou,
}

impl Vasicek {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let ou = Ou::new(&Ou {
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

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
