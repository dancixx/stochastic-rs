use ndarray::Array1;

use crate::{noises::fgn::Fgn, Sampling};

#[derive(Default)]
pub struct Fcir {
  pub hurst: f64,
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
  pub m: Option<usize>,
  fgn: Fgn,
}

impl Fcir {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let fgn = Fgn::new(&Fgn {
      hurst: params.hurst,
      n: params.n,
      t: params.t,
      m: params.m,
      ..Default::default()
    });

    Self {
      hurst: params.hurst,
      theta: params.theta,
      mu: params.mu,
      sigma: params.sigma,
      n: params.n,
      x0: params.x0,
      t: params.t,
      use_sym: params.use_sym,
      m: params.m,
      fgn,
    }
  }
}

impl Sampling<f64> for Fcir {
  fn sample(&self) -> Array1<f64> {
    assert!(
      2.0 * self.theta * self.mu < self.sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    let fgn = self.sample();
    let dt = self.t.unwrap_or(1.0) / self.n as f64;

    let mut fcir = Array1::<f64>::zeros(self.n + 1);
    fcir[0] = self.x0.unwrap_or(0.0);

    for i in 1..(self.n + 1) {
      let random = match self.use_sym.unwrap_or(false) {
        true => self.sigma * (fcir[i - 1]).abs().sqrt() * fgn[i - 1],
        false => self.sigma * (fcir[i - 1]).max(0.0) * fgn[i - 1],
      };
      fcir[i] = fcir[i - 1] + self.theta * (self.mu - fcir[i - 1]) * dt + random
    }

    fcir
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
