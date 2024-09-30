use ndarray::{Array1, Array2};

use crate::stochastic::{noise::cfgns::Cfgns, Sampling2D};

#[derive(Default)]
pub struct Cfbms {
  pub hurst1: f64,
  pub hurst2: Option<f64>,
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub cfgns: Cfgns,
}

impl Cfbms {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let cfgns = Cfgns::new(&Cfgns {
      hurst: params.hurst1,
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
      ..Default::default()
    });

    Self {
      hurst1: params.hurst1,
      hurst2: params.hurst2,
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
      cfgns,
    }
  }
}

impl Sampling2D<f64> for Cfbms {
  fn sample(&self) -> [Array1<f64>; 2] {
    assert!(
      (0.0..=1.0).contains(&self.hurst1),
      "Hurst parameter for the first fBM must be in (0, 1)"
    );

    if let Some(hurst2) = self.hurst2 {
      assert!(
        (0.0..=1.0).contains(&hurst2),
        "Hurst parameter for the second fBM must be in (0, 1)"
      );
    }
    assert!(
      (-1.0..=1.0).contains(&self.rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    let mut fbms = Array2::<f64>::zeros((2, self.n + 1));
    let [fgn1, fgn2] = self.cfgns.sample();

    for i in 1..=self.n {
      fbms[[0, i]] = fbms[[0, i - 1]] + fgn1[i - 1];
      fbms[[1, i]] =
        fbms[[1, i - 1]] + self.rho * fgn1[i - 1] + (1.0 - self.rho.powi(2)).sqrt() * fgn2[i - 1];
    }

    [fbms.row(0).to_owned(), fbms.row(1).to_owned()]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
