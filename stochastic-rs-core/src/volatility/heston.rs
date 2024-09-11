use ndarray::Array1;

use crate::{noises::cgns::Cgns, Sampling2D};

#[derive(Default)]

pub struct Heston {
  pub mu: f64,
  pub kappa: f64,
  pub theta: f64,
  pub eta: f64,
  pub rho: f64,
  pub n: usize,
  pub s0: Option<f64>,
  pub v0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
  pub m: Option<usize>,
  cgns: Cgns,
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
      mu: params.mu,
      kappa: params.kappa,
      theta: params.theta,
      eta: params.eta,
      rho: params.rho,
      n: params.n,
      s0: params.s0,
      v0: params.v0,
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

    for i in 1..(self.n + 1) {
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1];

      let random: f64 = match self.use_sym.unwrap_or(false) {
        true => self.eta * (v[i - 1]).abs().sqrt() * cgn2[i - 1],
        false => self.eta * (v[i - 1]).max(0.0).sqrt() * cgn2[i - 1],
      };
      v[i] = v[i - 1] + self.kappa * (self.theta - v[i - 1]) * dt + random;
    }

    [s, v]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
