use ndarray::{s, Array1};

use crate::stochastic::{noise::fgn::FGN, Sampling};

#[derive(Default)]
pub struct FCIR {
  pub hurst: f64,
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
  pub m: Option<usize>,
  pub fgn: FGN,
}

impl FCIR {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let fgn = FGN::new(params.hurst, params.n, params.t, params.m);

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

impl Sampling<f64> for FCIR {
  /// Sample the Fractional Cox-Ingersoll-Ross (FCIR) process
  fn sample(&self) -> Array1<f64> {
    assert!(
      2.0 * self.theta * self.mu < self.sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    let fgn = self.fgn.sample();
    let dt = self.t.unwrap_or(1.0) / self.n as f64;

    let mut fcir = Array1::<f64>::zeros(self.n + 1);
    fcir[0] = self.x0.unwrap_or(0.0);

    for i in 1..=self.n {
      let dfcir = self.theta * (self.mu - fcir[i - 1]) * dt
        + self.sigma * (fcir[i - 1]).abs().sqrt() * fgn[i - 1];

      fcir[i] = match self.use_sym.unwrap_or(false) {
        true => (fcir[i - 1] + dfcir).abs(),
        false => (fcir[i - 1] + dfcir).max(0.0),
      };
    }

    fcir.slice(s![..self.n()]).to_owned()
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
