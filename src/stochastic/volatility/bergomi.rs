use ndarray::{s, Array1};

use crate::stochastic::{noise::cgns::CGNS, Sampling2D};

#[derive(Default)]
pub struct Bergomi {
  pub nu: f64,
  pub v0: Option<f64>,
  pub s0: Option<f64>,
  pub r: f64,
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub cgns: CGNS,
}

impl Bergomi {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let cgns = CGNS::new(&CGNS {
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
    });

    Self {
      nu: params.nu,
      v0: params.v0,
      s0: params.s0,
      r: params.r,
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
      cgns,
    }
  }
}

impl Sampling2D<f64> for Bergomi {
  fn sample(&self) -> [Array1<f64>; 2] {
    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let [cgn1, z] = self.cgns.sample();

    let mut s = Array1::<f64>::zeros(self.n + 1);
    let mut v2 = Array1::<f64>::zeros(self.n + 1);
    s[0] = self.s0.unwrap_or(100.0);
    v2[0] = self.v0.unwrap_or(1.0).powi(2);

    for i in 0..=self.n {
      s[i] = s[i - 1] + self.r * s[i - 1] * dt + v2[i - 1].sqrt() * s[i - 1] * cgn1[i - 1];

      let sum_z = z.slice(s![..i]).sum();
      let t = i as f64 * dt;
      v2[i] =
        self.v0.unwrap_or(1.0).powi(2) * (self.nu * t * sum_z - 0.5 * self.nu.powi(2) * t.powi(2))
    }

    [s, v2]
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
