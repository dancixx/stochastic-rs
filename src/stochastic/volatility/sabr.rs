use ndarray::Array1;

use crate::stochastic::{noise::cgns::CGNS, Sampling2D};

#[derive(Default)]

pub struct Sabr {
  pub alpha: f64,
  pub beta: f64,
  pub rho: f64,
  pub n: usize,
  pub f0: Option<f64>,
  pub v0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub cgns: CGNS,
}

impl Sabr {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let cgns = CGNS::new(&CGNS {
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
    });

    Self {
      alpha: params.alpha,
      beta: params.beta,
      rho: params.rho,
      n: params.n,
      f0: params.f0,
      v0: params.v0,
      t: params.t,
      m: params.m,
      cgns,
    }
  }
}

impl Sampling2D<f64> for Sabr {
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();

    let mut f = Array1::<f64>::zeros(self.n + 1);
    let mut v = Array1::<f64>::zeros(self.n + 1);

    f[0] = self.f0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    for i in 1..=self.n {
      f[i] = f[i - 1] + v[i - 1] * f[i - 1].powf(self.beta) * cgn1[i - 1];
      v[i] = v[i - 1] + self.alpha * v[i - 1] * cgn2[i - 1];
    }

    [f, v]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
