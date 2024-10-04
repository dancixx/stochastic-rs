use std::sync::Mutex;

use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling;

#[derive(Default)]
pub struct CEV {
  pub mu: f64,
  pub sigma: f64,
  pub gamma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub calculate_malliavin: Option<bool>,
  malliavin: Mutex<Option<Array1<f64>>>,
}

impl CEV {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      mu: params.mu,
      sigma: params.sigma,
      gamma: params.gamma,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
      calculate_malliavin: Some(false),
      malliavin: Mutex::new(None),
    }
  }
}

impl Sampling<f64> for CEV {
  /// Sample the CEV process
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut cev = Array1::<f64>::zeros(self.n + 1);
    cev[0] = self.x0.unwrap_or(0.0);

    for i in 1..=self.n {
      cev[i] = cev[i - 1]
        + self.mu * cev[i - 1] * dt
        + self.sigma * cev[i - 1].powf(self.gamma) * gn[i - 1]
    }

    if self.calculate_malliavin.is_some() && self.calculate_malliavin.unwrap() {
      let mut det_term = Array1::zeros(self.n + 1);
      let mut stochastic_term = Array1::zeros(self.n + 1);
      let mut malliavin = Array1::zeros(self.n + 1);

      for i in 1..=self.n {
        det_term[i] = (self.mu
          - (self.gamma.powi(2) * self.sigma.powi(2) * cev[i - 1].powf(2.0 * self.gamma - 2.0)
            / 2.0))
          * dt;
        stochastic_term[i] =
          self.sigma * self.gamma * cev[i - 1].powf(self.gamma - 1.0) * gn[i - 1];
        malliavin[i] =
          self.sigma * cev[i - 1].powf(self.gamma) * (det_term[i] + stochastic_term[i]).exp();
      }

      let _ = std::mem::replace(&mut *self.malliavin.lock().unwrap(), Some(malliavin));
    }

    cev
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  /// Calculate the Malliavin derivative of the CEV process
  ///
  /// The Malliavin derivative of the CEV process is given by
  /// D_r S_t = \sigma S_t^{\gamma} * 1_{[0, r]}(r) exp(\int_0^r (\mu - \frac{\gamma^2 \sigma^2 S_u^{2\gamma - 2}}{2}) du + \int_0^r \gamma \sigma S_u^{\gamma - 1} dW_u)
  ///
  /// The Malliavin derivative of the CEV process shows the sensitivity of the stock price with respect to the Wiener process.
  fn malliavin(&self) -> Array1<f64> {
    self.malliavin.lock().unwrap().clone().unwrap()
  }
}
