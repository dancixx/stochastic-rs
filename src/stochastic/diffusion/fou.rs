use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{noise::fgn::FGN, Sampling};

#[derive(ImplNew)]
pub struct FOU {
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fgn: FGN,
}

impl Sampling<f64> for FOU {
  /// Sample the Fractional Ornstein-Uhlenbeck (FOU) process
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let fgn = self.fgn.sample();

    let mut fou = Array1::<f64>::zeros(self.n);
    fou[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      fou[i] = fou[i - 1] + self.theta * (self.mu - fou[i - 1]) * dt + self.sigma * fgn[i - 1]
    }

    fou
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

#[cfg(test)]
mod tests {
  use crate::{
    plot_1d,
    stochastic::{noise::fgn::FGN, Sampling, N, X0},
  };

  use super::*;

  #[test]
  fn fou_length_equals_n() {
    let fou = FOU::new(
      2.0,
      1.0,
      0.8,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
    );

    assert_eq!(fou.sample().len(), N);
  }

  #[test]
  fn fou_starts_with_x0() {
    let fou = FOU::new(
      2.0,
      1.0,
      0.8,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
    );

    assert_eq!(fou.sample()[0], X0);
  }

  #[test]
  fn fou_plot() {
    let fou = FOU::new(
      2.0,
      1.0,
      0.8,
      N,
      Some(X0),
      Some(1.0),
      None,
      FGN::new(0.7, N - 1, Some(1.0), None),
    );

    plot_1d!(fou.sample(), "Fractional Ornstein-Uhlenbeck (FOU) Process");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn fou_malliavin() {
    unimplemented!();
  }
}
