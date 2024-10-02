use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling;

/// Cox-Ingersoll-Ross (CIR) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW(t)
/// where X(t) is the CIR process.
#[derive(Default)]
pub struct CIR {
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
  pub m: Option<usize>,
}

impl CIR {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      theta: params.theta,
      mu: params.mu,
      sigma: params.sigma,
      n: params.n,
      x0: params.x0,
      t: params.t,
      use_sym: params.use_sym,
      m: params.m,
    }
  }
}

impl Sampling<f64> for CIR {
  fn sample(&self) -> Array1<f64> {
    assert!(
      2.0 * self.theta * self.mu < self.sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut cir = Array1::<f64>::zeros(self.n + 1);
    cir[0] = self.x0.unwrap_or(0.0);

    for i in 1..=self.n {
      let dcir = self.theta * (self.mu - cir[i - 1]) * dt
        + self.sigma * (cir[i - 1]).abs().sqrt() * gn[i - 1];

      cir[i] = match self.use_sym.unwrap_or(false) {
        true => (cir[i - 1] + dcir).abs(),
        false => (cir[i - 1] + dcir).max(0.0),
      };
    }

    cir
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
