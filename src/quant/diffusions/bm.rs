use crate::quant::traits::Process;

pub struct BM<T> {
  mu: T,
  sigma: T,
}

impl BM<f64> {
  #[must_use]
  #[inline(always)]
  pub fn new() -> Self {
    Self {
      mu: 0.0,
      sigma: 1.0,
    }
  }
}

impl Process<f64> for BM<f64> {
  fn drift(&self, _x: f64, _t: f64) -> f64 {
    self.mu
  }

  fn diffusion(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }
}

#[cfg(feature = "f32")]
impl BM<f32> {
  #[must_use]
  #[inline(always)]
  pub fn new_f32() -> Self {
    Self {
      mu: 0.0,
      sigma: 1.0,
    }
  }
}

#[cfg(feature = "f32")]
impl Process<f32> for BM<f32> {
  fn drift(&self, _x: f32, _t: f32) -> f32 {
    self.mu
  }

  fn diffusion(&self, _x: f32, _t: f32) -> f32 {
    self.sigma
  }
}
