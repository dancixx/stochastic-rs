use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{noise::cgns::CGNS, Sampling2D};

/// Hull-White 2-factor model
/// dX(t) = (k(t) + U(t) - theta * X(t)) dt + sigma_1 dW1(t) x(0) = x0
/// dU(t) = b * U(t) dt + sigma_2 dW2(t) u(0) = 0
#[derive(ImplNew)]
pub struct HullWhite2F {
  pub k: fn(f64) -> f64,
  pub theta: f64,
  pub sigma1: f64,
  pub sigma2: f64,
  pub rho: f64,
  pub b: f64,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub n: usize,
  pub m: Option<usize>,
  pub cgns: CGNS,
}

impl Sampling2D<f64> for HullWhite2F {
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;

    let mut x = Array1::<f64>::zeros(self.n);
    let mut u = Array1::<f64>::zeros(self.n);

    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      x[i] = x[i - 1]
        + ((self.k)(i as f64 * dt) + u[i - 1] - self.theta * x[i - 1]) * dt
        + self.sigma1 * cgn1[i - 1];

      u[i] = u[i - 1] + self.b * u[i - 1] * dt + self.sigma2 * cgn2[i - 1];
    }

    [x, u]
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
