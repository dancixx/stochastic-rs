use ndarray::{s, Array1};

use crate::stochastic::{noise::fgn::FGN, Sampling};

#[derive(Default)]
pub struct FOU {
  pub hurst: f64,
  pub mu: f64,
  pub sigma: f64,
  pub theta: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fgn: FGN,
}

impl FOU {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let fgn = FGN::new(params.hurst, params.n, params.t, params.m);

    Self {
      hurst: params.hurst,
      mu: params.mu,
      sigma: params.sigma,
      theta: params.theta,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
      fgn,
    }
  }
}

impl Sampling<f64> for FOU {
  /// Sample the Fractional Ornstein-Uhlenbeck (FOU) process
  fn sample(&self) -> Array1<f64> {
    assert!(
      self.hurst > 0.0 && self.hurst < 1.0,
      "Hurst parameter must be in (0, 1)"
    );

    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let fgn = self.fgn.sample();

    let mut fou = Array1::<f64>::zeros(self.n + 1);
    fou[0] = self.x0.unwrap_or(0.0);

    for i in 1..=self.n {
      fou[i] = fou[i - 1] + self.theta * (self.mu - fou[i - 1]) * dt + self.sigma * fgn[i - 1]
    }

    fou.slice(s![..self.n()]).to_owned()
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of paths
  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
