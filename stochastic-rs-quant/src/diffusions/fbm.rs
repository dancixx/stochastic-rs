use crate::quant::{noises::fgn::FGN, traits_f::FractionalProcess};

pub struct FBM<T> {
  mu: T,
  sigma: T,
  pub hurst: T,
  n: usize,
  x_0: T,
  t_0: T,
  t: T,
  fgn: FGN<T>,
}

impl FBM<f64> {
  #[must_use]
  #[inline(always)]
  pub fn new(hurst: f64, n: usize, x_0: f64, t_0: f64, t: f64) -> Self {
    Self {
      mu: 0.0,
      sigma: 1.0,
      hurst,
      n,
      x_0,
      t_0,
      t,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl FractionalProcess<f64> for FBM<f64> {
  fn drift(&self, _x: f64, _t: f64) -> f64 {
    self.mu
  }

  fn diffusion(&self, _x: f64, _t: f64) -> f64 {
    self.sigma
  }

  fn hurst(&self) -> f64 {
    self.hurst
  }

  fn fgn(&self) -> FGN<f64> {
    self.fgn.clone()
  }

  fn params(&self) -> (usize, f64, f64, f64) {
    (self.n, self.x_0, self.t_0, self.t)
  }
}
