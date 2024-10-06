use impl_new_derive::ImplNew;
use ndarray::{Array1, Array2};

use crate::stochastic::{noise::cgns::CGNS, Sampling2D};

#[derive(ImplNew)]
pub struct CBMS {
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub cgns: CGNS,
}

impl Sampling2D<f64> for CBMS {
  fn sample(&self) -> [Array1<f64>; 2] {
    assert!(
      (-1.0..=1.0).contains(&self.rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    let mut bms = Array2::<f64>::zeros((2, self.n));
    let [cgn1, cgn2] = self.cgns.sample();

    for i in 1..self.n {
      bms[[0, i]] = bms[[0, i - 1]] + cgn1[i - 1];
      bms[[1, i]] =
        bms[[1, i - 1]] + self.rho * cgn1[i - 1] + (1.0 - self.rho.powi(2)).sqrt() * cgn2[i - 1];
    }

    [bms.row(0).into_owned(), bms.row(1).into_owned()]
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
