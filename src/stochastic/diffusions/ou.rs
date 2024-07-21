use crate::stochastic::traits::Process;

#[cfg(feature = "f32")]
use crate::stochastic::traits::ProcessF32;

pub struct OU {
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
}

impl Process for OU {
  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.theta * (self.mu - x)
  }

  fn diffusion(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }
}

#[cfg(feature = "f32")]
pub struct OUF32 {
  pub theta: f32,
  pub mu: f32,
  pub sigma: f32,
}

#[cfg(feature = "f32")]
impl ProcessF32 for OUF32 {
  fn drift(&self, x: f32, _t: f32) -> f32 {
    self.theta * (self.mu - x)
  }

  fn diffusion(&self, _x: f32, _t: f32) -> f32 {
    self.sigma
  }
}
