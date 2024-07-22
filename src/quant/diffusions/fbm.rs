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
      mu: 0_f64,
      sigma: 1_f64,
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

#[cfg(feature = "f32")]
impl FBM<f32> {
  #[must_use]
  #[inline(always)]
  pub fn new_f32(hurst: f32, n: usize, x_0: f32, t_0: f32, t: f32) -> Self {
    Self {
      mu: 0_f32,
      sigma: 1_f32,
      hurst,
      n,
      x_0,
      t_0,
      t,
      fgn: FGN::new_f32(hurst, n - 1, t),
    }
  }
}

#[cfg(feature = "f32")]
impl FractionalProcess<f32> for FBM<f32> {
  fn drift(&self, _x: f32, _t: f32) -> f32 {
    self.mu
  }

  fn diffusion(&self, _x: f32, _t: f32) -> f32 {
    self.sigma
  }

  fn hurst(&self) -> f32 {
    self.hurst
  }

  fn fgn(&self) -> FGN<f32> {
    self.fgn.clone()
  }

  fn params(&self) -> (usize, f32, f32, f32) {
    (self.n, self.x_0, self.t_0, self.t)
  }
}
