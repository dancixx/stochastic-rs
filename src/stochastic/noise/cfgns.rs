use ndarray::{s, Array1, Array2};

use crate::stochastic::{Sampling, Sampling2D};

use super::fgn::Fgn;

#[derive(Default)]
pub struct Cfgns {
  pub hurst: f64,
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fgn: Fgn,
}

impl Cfgns {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let fgn = Fgn::new(params.hurst, params.n, params.t, params.m);

    Self {
      hurst: params.hurst,
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
      fgn,
    }
  }
}

impl Sampling2D<f64> for Cfgns {
  fn sample(&self) -> [Array1<f64>; 2] {
    assert!(
      (0.0..=1.0).contains(&self.hurst),
      "Hurst parameter must be in (0, 1)"
    );
    assert!(
      (-1.0..=1.0).contains(&self.rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    let mut cfgns = Array2::<f64>::zeros((2, self.n + 1));
    let fgn1 = self.fgn.sample();
    let fgn2 = self.fgn.sample();

    for i in 1..=self.n {
      cfgns[[0, i]] = fgn1[i - 1];
      cfgns[[1, i]] = self.rho * fgn1[i - 1] + (1.0 - self.rho.powi(2)).sqrt() * fgn2[i - 1];
    }

    [
      cfgns.row(0).slice(s![..self.n()]).into_owned(),
      cfgns.row(1).slice(s![..self.n()]).into_owned(),
    ]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
