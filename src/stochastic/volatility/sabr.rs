use std::sync::Mutex;

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
  pub calculate_malliavin: Option<bool>,
  malliavin_of_vol: Mutex<Option<Array1<f64>>>,
  malliavin_of_price: Mutex<Option<Array1<f64>>>,
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
      calculate_malliavin: Some(false),
      malliavin_of_vol: Mutex::new(None),
      malliavin_of_price: Mutex::new(None),
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

    if self.calculate_malliavin.is_some() && self.calculate_malliavin.unwrap() {
      // Only volatility Malliavin derivative is supported
      let mut malliavin_of_vol = Array1::<f64>::zeros(self.n + 1);
      for i in 1..=self.n {
        malliavin_of_vol[i] = self.alpha * v[i - 1];
      }

      let _ = std::mem::replace(
        &mut *self.malliavin_of_vol.lock().unwrap(),
        Some(malliavin_of_vol),
      );
    }

    [f, v]
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  /// Calculate the Malliavin derivative of the SABR model
  ///
  /// The Malliavin derivative of the volaility process in the SABR model is given by:
  /// D_r \sigma_t = \alpha \sigma_t 1_{[0, T]}(r)
  fn malliavin(&self) -> [Array1<f64>; 2] {
    [
      Array1::zeros(self.n + 1),
      self
        .malliavin_of_vol
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone(),
    ]
  }
}
