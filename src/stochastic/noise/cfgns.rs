use impl_new_derive::ImplNew;
use ndarray::{Array1, Array2};

use crate::stochastic::{Sampling, Sampling2D};

use super::fgn::FGN;

#[derive(ImplNew)]
pub struct CFGNS {
  pub hurst: f64,
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fgn: FGN,
}

impl Sampling2D<f64> for CFGNS {
  fn sample(&self) -> [Array1<f64>; 2] {
    assert!(
      (0.0..=1.0).contains(&self.hurst),
      "Hurst parameter must be in (0, 1)"
    );
    assert!(
      (-1.0..=1.0).contains(&self.rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    let mut cfgns = Array2::<f64>::zeros((2, self.n));
    let fgn1 = self.fgn.sample();
    let fgn2 = self.fgn.sample();

    for i in 1..self.n {
      cfgns[[0, i]] = fgn1[i - 1];
      cfgns[[1, i]] = self.rho * fgn1[i - 1] + (1.0 - self.rho.powi(2)).sqrt() * fgn2[i - 1];
    }

    [cfgns.row(0).into_owned(), cfgns.row(1).into_owned()]
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
