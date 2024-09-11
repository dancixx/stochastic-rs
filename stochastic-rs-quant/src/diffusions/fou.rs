use crate::quant::{noises::fgn::FGN, traits_f::FractionalProcess};

pub struct FOU<T> {
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub hurst: T,
  n: usize,
  x_0: T,
  t_0: T,
  t: T,
  fgn: FGN<T>,
}

impl FOU<f64> {
  #[must_use]
  #[inline(always)]
  pub fn new(
    theta: f64,
    mu: f64,
    sigma: f64,
    hurst: f64,
    n: usize,
    x_0: f64,
    t_0: f64,
    t: f64,
  ) -> Self {
    Self {
      theta,
      mu,
      sigma,
      hurst,
      n,
      x_0,
      t_0,
      t,
      fgn: FGN::new(hurst, n, t),
    }
  }
}

impl FractionalProcess<f64> for FOU<f64> {
  fn drift(&self, x: f64, _t: f64) -> f64 {
    self.theta * (self.mu - x)
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
