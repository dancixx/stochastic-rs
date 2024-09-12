use ndarray::{Array1, Array2};

use crate::{noise::cgns::Cgns, Sampling2D};

#[derive(Default)]
pub struct Cbms {
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub cgns: Cgns,
}

impl Cbms {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let cgns = Cgns::new(&Cgns {
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
    });

    Self {
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
      cgns,
    }
  }
}

impl Sampling2D<f64> for Cbms {
  fn sample(&self) -> [Array1<f64>; 2] {
    assert!(
      !(-1.0..=1.0).contains(&self.rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    let mut bms = Array2::<f64>::zeros((2, self.n + 1));
    let [cgn1, cgn2] = self.cgns.sample();

    for i in 1..=self.n {
      bms[[0, i]] = bms[[0, i - 1]] + cgn1[i - 1];
      bms[[1, i]] =
        bms[[1, i - 1]] + self.rho * cgn1[i - 1] + (1.0 - self.rho.powi(2)).sqrt() * cgn2[i - 1];
    }

    [bms.row(0).into_owned(), bms.row(1).into_owned()]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
