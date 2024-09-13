use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use statrs::function::gamma::gamma;

use crate::Sampling;

#[derive(Default)]
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

impl RoughHeston {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      v0: params.v0,
      theta: params.theta,
      kappa: params.kappa,
      nu: params.nu,
      hurst: params.hurst,
      c1: params.c1,
      c2: params.c2,
      t: params.t,
      n: params.n,
      m: params.m,
    }
  }
}

impl Sampling<f64> for RoughHeston {
  fn sample(&self) -> ndarray::Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut yt = Array1::<f64>::zeros(self.n + 1);
    let mut zt = Array1::<f64>::zeros(self.n + 1);
    let mut v2 = Array1::zeros(self.n + 1);

    yt[0] = self.theta + (self.v0.unwrap_or(1.0).powi(2) - self.theta) * (-self.kappa * 0.0).exp();
    zt[0] = 0.0; // Initial condition for Z_t, typically 0 for such integrals.
    v2[0] = self.v0.unwrap_or(1.0).powi(2);

    for i in 1..=self.n {
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

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use plotly::{common::Line, Plot, Scatter};

  use super::*;

  #[test]
  fn test_rough_heston() {
    let params = RoughHeston {
      v0: Some(1.2),
      theta: 1.0,
      kappa: 1.0,
      nu: 0.2,
      hurst: 0.75,
      c1: Some(1.0),
      c2: Some(1.0),
      t: Some(1.0),
      n: 1000,
      m: Some(1000),
    };

    let rh = RoughHeston::new(&params);
    let _sample = rh.sample();

    let mut plot = Plot::new();
    let trace = Scatter::new((0.._sample.len()).collect::<Vec<_>>(), _sample.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("orange")
          .shape(plotly::common::LineShape::Linear),
      )
      .name("Fbm");
    plot.add_trace(trace);
    plot.show();
  }
}
