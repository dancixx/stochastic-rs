use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use statrs::function::gamma::gamma;

use crate::stochastic::Sampling;

#[derive(ImplNew)]
pub struct RoughHeston {
  pub v0: Option<f64>,
  pub theta: f64,
  pub kappa: f64,
  pub nu: f64,
  pub hurst: f64,
  pub c1: Option<f64>,
  pub c2: Option<f64>,
  pub t: Option<f64>,
  pub n: usize,
  pub m: Option<usize>,
}

impl Sampling<f64> for RoughHeston {
  fn sample(&self) -> ndarray::Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut yt = Array1::<f64>::zeros(self.n);
    let mut zt = Array1::<f64>::zeros(self.n);
    let mut v2 = Array1::zeros(self.n);

    yt[0] = self.theta + (self.v0.unwrap_or(1.0).powi(2) - self.theta) * (-self.kappa * 0.0).exp();
    zt[0] = 0.0; // Initial condition for Z_t, typically 0 for such integrals.
    v2[0] = self.v0.unwrap_or(1.0).powi(2);

    for i in 1..self.n {
      let t = dt * i as f64;
      yt[i] = self.theta + (yt[i - 1] - self.theta) * (-self.kappa * dt).exp();
      zt[i] = zt[i - 1] * (-self.kappa * dt).exp() + (v2[i - 1].powi(2)).sqrt() * gn[i - 1];

      let integral = (0..i)
        .map(|j| {
          let tj = j as f64 * dt;
          ((t - tj).powf(self.hurst - 0.5) * zt[j]) * dt
        })
        .sum::<f64>();

      v2[i] = yt[i]
        + self.c1.unwrap_or(1.0) * self.nu * zt[i]
        + self.c2.unwrap_or(1.0) * self.nu * integral / gamma(self.hurst + 0.5);
    }

    v2
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
