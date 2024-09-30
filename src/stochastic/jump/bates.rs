use ndarray::Array1;

use crate::stochastic::{
  noise::cgns::Cgns, process::cpoisson::CompoundPoisson, ProcessDistribution, Sampling2D,
  Sampling3D,
};

#[derive(Default)]
pub struct Bates1996<D>
where
  D: ProcessDistribution,
{
  pub mu: Option<f64>,
  pub b: Option<f64>,
  pub r: Option<f64>,
  pub r_f: Option<f64>,
  pub lambda: f64,
  pub k: f64,
  pub alpha: f64,
  pub beta: f64,
  pub sigma: f64,
  pub rho: f64,
  pub n: usize,
  pub s0: Option<f64>,
  pub v0: Option<f64>,
  pub t: Option<f64>,
  pub use_sym: Option<bool>,
  pub m: Option<usize>,
  pub jumps_distribution: D,
  pub cgns: Cgns,
  pub cpoisson: CompoundPoisson<D>,
}

impl<D: ProcessDistribution> Bates1996<D> {
  #[must_use]
  pub fn new(params: &Bates1996<D>) -> Self {
    let cgns = Cgns::new(&Cgns {
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
    });

    let cpoisson = CompoundPoisson::new(&CompoundPoisson {
      n: None,
      lambda: params.lambda,
      t_max: Some(params.t.unwrap_or(1.0) / params.n as f64),
      distribution: params.jumps_distribution,
      m: params.m,
      ..Default::default()
    });

    Self {
      mu: params.mu,
      b: params.b,
      r: params.r,
      r_f: params.r_f,
      lambda: params.lambda,
      k: params.k,
      alpha: params.alpha,
      beta: params.beta,
      sigma: params.sigma,
      rho: params.rho,
      n: params.n,
      s0: params.s0,
      v0: params.v0,
      t: params.t,
      use_sym: params.use_sym,
      m: params.m,
      jumps_distribution: params.jumps_distribution,
      cgns,
      cpoisson,
    }
  }
}

impl<D: ProcessDistribution> Sampling2D<f64> for Bates1996<D> {
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / self.n as f64;

    let mut s = Array1::<f64>::zeros(self.n + 1);
    let mut v = Array1::<f64>::zeros(self.n + 1);

    s[0] = self.s0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    let drift = match (self.mu, self.b, self.r, self.r_f) {
      (Some(r), Some(r_f), ..) => r - r_f,
      (Some(b), ..) => b,
      _ => self.mu.unwrap(),
    };

    for i in 1..=self.n {
      let [.., jumps] = self.cpoisson.sample();

      s[i] = s[i - 1]
        + (drift - self.lambda * self.k) * s[i - 1] * dt
        + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1]
        + jumps.sum();

      let dv = (self.alpha - self.beta * v[i - 1]) * dt + self.sigma * v[i - 1] * cgn2[i - 1];

      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(0.0),
      }
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
