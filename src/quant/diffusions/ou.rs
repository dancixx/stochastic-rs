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

#[cfg(feature = "f32")]
impl OU<f32> {
  #[must_use]
  #[inline(always)]
  pub fn new_f32(theta: f32, mu: f32, sigma: f32) -> Self {
    Self { theta, mu, sigma }
  }
}

#[cfg(feature = "f32")]
impl Process<f32> for OU<f32> {
  fn drift(&self, x: f32, _t: f32) -> f32 {
    self.theta * (self.mu - x)
  }

  fn diffusion(&self, _x: f32, _t: f32) -> f32 {
    self.sigma
  }
}
