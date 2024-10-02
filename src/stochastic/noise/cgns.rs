use ndarray::{s, Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling2D;

#[derive(Default)]
pub struct CGNS {
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl CGNS {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
    }
  }
}

impl Sampling2D<f64> for CGNS {
  fn sample(&self) -> [Array1<f64>; 2] {
    assert!(
      (-1.0..=1.0).contains(&self.rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let mut cgns = Array2::<f64>::zeros((2, self.n + 1));
    let gn1 = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());
    let gn2 = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

    for i in 1..=self.n {
      cgns[[0, i]] = gn1[i - 1];
      cgns[[1, i]] = self.rho * gn1[i - 1] + (1.0 - self.rho.powi(2)).sqrt() * gn2[i - 1];
    }

    [
      cgns.row(0).slice(s![..self.n()]).into_owned(),
      cgns.row(1).slice(s![..self.n()]).into_owned(),
    ]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
