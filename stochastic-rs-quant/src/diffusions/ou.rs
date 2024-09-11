use crate::quant::traits::Process;

pub struct OU<T> {
  pub theta: T,
  pub mu: T,
  pub sigma: T,
}

impl OU<f64> {
  #[must_use]
  #[inline(always)]
  pub fn new(theta: f64, mu: f64, sigma: f64) -> Self {
    Self { theta, mu, sigma }
  }
}

impl Process<f64> for OU<f64> {
  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.theta * (self.mu - x)
  }

  fn diffusion(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }
}
