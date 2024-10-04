use std::sync::Mutex;

use ndarray::Array1;
use ndarray_rand::RandomExt;
use num_complex::Complex64;
use rand_distr::Normal;
use statrs::{
  distribution::{Continuous, ContinuousCDF, LogNormal},
  statistics::{Distribution as StatDistribution, Median, Mode},
};

use crate::stochastic::{Distribution, Sampling};

#[derive(Default)]
pub struct GBM {
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub distribution: Option<LogNormal>,
  pub calculate_malliavin: Option<bool>,
  malliavin: Mutex<Option<Array1<f64>>>,
}

impl GBM {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      mu: params.mu,
      sigma: params.sigma,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
      distribution: None,
      calculate_malliavin: Some(false),
      malliavin: Mutex::new(None),
    }
  }
}

impl Sampling<f64> for GBM {
  /// Sample the GBM process
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut gbm = Array1::<f64>::zeros(self.n + 1);
    gbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..=self.n {
      gbm[i] = gbm[i - 1] + self.mu * gbm[i - 1] * dt + self.sigma * gbm[i - 1] * gn[i - 1]
    }

    if self.calculate_malliavin.is_some() && self.calculate_malliavin.unwrap() {
      let mut malliavin = Array1::zeros(self.n + 1);
      for i in 1..=self.n {
        malliavin[i] = self.sigma * gbm[i - 1];
      }

      // This equivalent to the following:
      // self.malliavin.lock().unwrap().replace(Some(malliavin));
      let _ = std::mem::replace(&mut *self.malliavin.lock().unwrap(), Some(malliavin));
    }

    gbm
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  /// Distribution of the GBM process
  fn distribution(&mut self) {
    let mu = self.x0.unwrap() * (self.mu * self.t.unwrap()).exp();
    let sigma = (self.x0.unwrap().powi(2)
      * (2.0 * self.mu * self.t.unwrap()).exp()
      * ((self.sigma.powi(2) * self.t.unwrap()).exp() - 1.0))
      .sqrt();

    self.distribution = Some(LogNormal::new(mu, sigma).unwrap());
  }

  /// Mallaivin derivative of the GBM process
  ///
  /// The Malliavin derivative of the CEV process is given by
  /// D_r S_t = \sigma S_t * 1_[0, r](r)
  ///
  /// The Malliavin derivate of the GBM shows the sensitivity of the stock price with respect to the Wiener process.
  fn malliavin(&self) -> Array1<f64> {
    self.malliavin.lock().unwrap().as_ref().unwrap().clone()
  }
}

impl Distribution for GBM {
  /// Characteristic function of the distribution
  fn characteristic_function(&self, _t: f64) -> Complex64 {
    unimplemented!()
  }

  /// Probability density function of the distribution
  fn pdf(&self, x: f64) -> f64 {
    self.distribution.as_ref().unwrap().pdf(x)
  }

  /// Cumulative distribution function of the distribution
  fn cdf(&self, x: f64) -> f64 {
    self.distribution.as_ref().unwrap().cdf(x)
  }

  /// Inverse cumulative distribution function of the distribution
  fn inv_cdf(&self, p: f64) -> f64 {
    self.distribution.as_ref().unwrap().inverse_cdf(p)
  }

  /// Mean of the distribution
  fn mean(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .mean()
      .expect("Mean not found")
  }

  /// Mode of the distribution
  fn mode(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .mode()
      .expect("Mode not found")
  }

  /// Median of the distribution
  fn median(&self) -> f64 {
    self.distribution.as_ref().unwrap().median()
  }

  /// Variance of the distribution
  fn variance(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .variance()
      .expect("Variance not found")
  }

  /// Skewness of the distribution
  fn skewness(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .skewness()
      .expect("Skewness not found")
  }

  /// Kurtosis of the distribution
  fn kurtosis(&self) -> f64 {
    unimplemented!()
  }

  /// Entropy of the distribution
  fn entropy(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .entropy()
      .expect("Entropy not found")
  }

  /// Moment generating function of the distribution
  fn moment_generating_function(&self, _t: f64) -> f64 {
    unimplemented!()
  }
}
