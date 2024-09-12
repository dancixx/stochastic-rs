use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::Sampling;

#[derive(Default)]
pub struct Jacobi {
  pub alpha: f64,
  pub beta: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl Jacobi {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      alpha: params.alpha,
      beta: params.beta,
      sigma: params.sigma,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
    }
  }
}

impl Sampling<f64> for Jacobi {
  fn sample(&self) -> Array1<f64> {
    assert!(self.alpha > 0.0, "alpha must be positive");
    assert!(self.beta > 0.0, "beta must be positive");
    assert!(self.sigma > 0.0, "sigma must be positive");
    assert!(self.alpha < self.beta, "alpha must be less than beta");

    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut jacobi = Array1::<f64>::zeros(self.n + 1);
    jacobi[0] = self.x0.unwrap_or(0.0);

    for i in 1..=self.n {
      jacobi[i] = match jacobi[i - 1] {
        _ if jacobi[i - 1] <= 0.0 && i > 0 => 0.0,
        _ if jacobi[i - 1] >= 1.0 && i > 0 => 1.0,
        _ => {
          jacobi[i - 1]
            + (self.alpha - self.beta * jacobi[i - 1]) * dt
            + self.sigma * (jacobi[i - 1] * (1.0 - jacobi[i - 1])).sqrt() * gn[i - 1]
        }
      }
    }

    jacobi
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
