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
