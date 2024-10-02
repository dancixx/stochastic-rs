use ndarray::Array1;

use crate::stochastic::{noise::cgns::CGNS, Sampling2D};

use super::HestonPow;

#[derive(Default)]

pub struct Heston {
  /// Initial stock price
  pub s0: Option<f64>,
  /// Initial volatility
  pub v0: Option<f64>,
  /// Mean reversion rate
  pub kappa: f64,
  /// Long-run average volatility
  pub theta: f64,
  /// Volatility of volatility
  pub sigma: f64,
  /// Correlation between the stock price and its volatility
  pub rho: f64,
  /// Drift of the stock price
  pub mu: f64,
  /// Number of time steps
  pub n: usize,
  /// Time to maturity
  pub t: Option<f64>,
  /// Power of the variance
  /// If 0.5 then it is the original Heston model
  /// If 1.5 then it is the 3/2 model
  pub pow: HestonPow,
  /// Use the symmetric method for the variance to avoid negative values
  pub use_sym: Option<bool>,
  /// Number of paths for multithreading
  pub m: Option<usize>,
  /// Noise generator
  pub cgns: CGNS,
}

impl Heston {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let cgns = CGNS::new(&CGNS {
      rho: params.rho,
      n: params.n,
      t: params.t,
      m: params.m,
    });

    Self {
      s0: params.s0,
      v0: params.v0,
      kappa: params.kappa,
      theta: params.theta,
      sigma: params.sigma,
      rho: params.rho,
      mu: params.mu,
      n: params.n,
      t: params.t,
      pow: params.pow,
      use_sym: params.use_sym,
      m: params.m,
      cgns,
    }
  }
}

impl Sampling2D<f64> for Heston {
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / self.n as f64;

    let mut s = Array1::<f64>::zeros(self.n + 1);
    let mut v = Array1::<f64>::zeros(self.n + 1);

    s[0] = self.s0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    for i in 1..=self.n {
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1];

      let dv = self.kappa * (self.theta - v[i - 1]) * dt
        + self.sigma
          * v[i - 1].powf(match self.pow {
            HestonPow::Sqrt => 0.5,
            HestonPow::ThreeHalves => 1.5,
          })
          * cgn2[i - 1];

      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(0.0),
      }
    }

    [s, v]
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
  fn plot() {
    let heston = Heston::new(&Heston {
      s0: Some(0.05),
      v0: Some(0.04),
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.1,
      rho: -0.7,
      mu: 0.05,
      n: 1000,
      t: Some(1.0),
      pow: HestonPow::default(),
      use_sym: Some(true),
      m: Some(1),
      cgns: CGNS::default(),
    });
    let mut plot = Plot::new();
    let [s, v] = heston.sample();
    let price = Scatter::new((0..s.len()).collect::<Vec<_>>(), s.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("blue")
          .shape(plotly::common::LineShape::Linear),
      )
      .name("Heston");
    plot.add_trace(price);
    let vol = Scatter::new((0..v.len()).collect::<Vec<_>>(), v.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("orange")
          .shape(plotly::common::LineShape::Linear),
      )
      .name("Heston");
    plot.add_trace(vol);
    plot.show();
  }
}
