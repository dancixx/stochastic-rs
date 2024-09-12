use ndarray::{s, Array1};

use crate::{noise::fgn::Fgn, Sampling};

#[derive(Default)]
pub struct Fgbm {
  pub hurst: f64,
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  fgn: Fgn,
}

impl Fgbm {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let fgn = Fgn::new(params.hurst, params.n, params.t, params.m);

    Self {
      hurst: params.hurst,
      mu: params.mu,
      sigma: params.sigma,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
      fgn,
    }
  }
}

impl Sampling<f64> for Fgbm {
  fn sample(&self) -> Array1<f64> {
    assert!(
      self.hurst > 0.0 && self.hurst < 1.0,
      "Hurst parameter must be in (0, 1)"
    );

    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let fgn = self.fgn.sample();

    let mut fgbm = Array1::<f64>::zeros(self.n + 1);
    fgbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..(self.n + 1) {
      fgbm[i] = fgbm[i - 1] + self.mu * fgbm[i - 1] * dt + self.sigma * fgbm[i - 1] * fgn[i - 1]
    }

    fgbm.slice(s![..self.n()]).to_owned()
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
