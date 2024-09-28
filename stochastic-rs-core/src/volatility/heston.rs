use ndarray::Array1;

use crate::{noise::cgns::Cgns, Sampling2D};

#[derive(Default)]

pub struct Heston {
  /// Initial stock price
  pub s0: Option<f64>,
  /// Initial volatility
  pub v0: Option<f64>,
  /// Mean reversion rate
  pub kappa: f64,
  /// Long-run average volatility
  pub theta: f64,
  /// Volatility of volatility
  pub sigma: f64,
  /// Correlation between the stock price and its volatility
  pub rho: f64,
  /// Drift of the stock price
  pub mu: f64,
  /// Number of time steps
  pub n: usize,
  /// Time to maturity
  pub t: Option<f64>,
  /// Use the symmetric method for the variance to avoid negative values
  pub use_sym: Option<bool>,
  /// Number of paths for multithreading
  pub m: Option<usize>,
  /// Noise generator
  pub cgns: Cgns,
}

impl Heston {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let cgns = Cgns::new(&Cgns {
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
    });

    Self {
      s0: params.s0,
      v0: params.v0,
      kappa: params.kappa,
      theta: params.theta,
      sigma: params.sigma,
      rho: params.rho,
      mu: params.mu,
      n: params.n,
      t: params.t,
      use_sym: params.use_sym,
      m: params.m,
      cgns,
    }
  }
}

impl Sampling2D<f64> for Heston {
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / self.n as f64;

    let mut s = Array1::<f64>::zeros(self.n + 1);
    let mut v = Array1::<f64>::zeros(self.n + 1);

    s[0] = self.s0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    for i in 1..=self.n {
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1];

      let dv =
        self.kappa * (self.theta - v[i - 1]) * dt + self.sigma * v[i - 1].sqrt() * cgn2[i - 1];

      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(0.0),
      }
    }
    println!("S: {:?}", s);

    [s, v]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
