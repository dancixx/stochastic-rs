use ndarray::Array1;

use crate::stochastic::{noise::cgns::Cgns, Sampling2D};

#[derive(Default)]

pub struct DuffieKan {
  pub alpha: f64,
  pub beta: f64,
  pub gamma: f64,
  pub rho: f64,
  pub a1: f64,
  pub b1: f64,
  pub c1: f64,
  pub sigma1: f64,
  pub a2: f64,
  pub b2: f64,
  pub c2: f64,
  pub sigma2: f64,
  pub n: usize,
  pub r0: Option<f64>,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub cgns: Cgns,
}

impl DuffieKan {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let cgns = Cgns::new(&Cgns {
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
    });

    Self {
      alpha: params.alpha,
      beta: params.beta,
      gamma: params.gamma,
      rho: params.rho,
      a1: params.a1,
      b1: params.b1,
      c1: params.c1,
      sigma1: params.sigma1,
      a2: params.a2,
      b2: params.b2,
      c2: params.c2,
      sigma2: params.sigma2,
      n: params.n,
      r0: params.r0,
      x0: params.x0,
      t: params.t,
      m: params.m,
      cgns,
    }
  }
}

impl Sampling2D<f64> for DuffieKan {
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / self.n as f64;

    let mut r = Array1::<f64>::zeros(self.n + 1);
    let mut x = Array1::<f64>::zeros(self.n + 1);

    r[0] = self.r0.unwrap_or(0.0);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..=self.n {
      r[i] = r[i - 1]
        + (self.a1 * r[i - 1] + self.b1 * x[i - 1] + self.c1) * dt
        + self.sigma1 * (self.alpha * r[i - 1] + self.beta * x[i - 1] + self.gamma) * cgn1[i - 1];
      x[i] = x[i - 1]
        + (self.a2 * r[i - 1] + self.b2 * x[i - 1] + self.c2) * dt
        + self.sigma2 * (self.alpha * r[i - 1] + self.beta * x[i - 1] + self.gamma) * cgn2[i - 1];
    }

    [r, x]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
