use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::{Distribution, Normal};

use crate::stochastic::{process::cpoisson::CompoundPoisson, Sampling, Sampling3D};

#[derive(ImplNew)]
pub struct Merton<D>
where
  D: Distribution<f64> + Send + Sync,
{
  pub alpha: f64,
  pub sigma: f64,
  pub lambda: f64,
  pub theta: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub cpoisson: CompoundPoisson<D>,
}

impl<D> Sampling<f64> for Merton<D>
where
  D: Distribution<f64> + Send + Sync,
{
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let mut merton = Array1::<f64>::zeros(self.n);
    merton[0] = self.x0.unwrap_or(0.0);
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      merton[i] = merton[i - 1]
        + (self.alpha * self.sigma.powf(2.0) / 2.0 - self.lambda * self.theta) * dt
        + self.sigma * gn[i - 1]
        + jumps.sum();
    }

    merton
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
    stochastic::{process::poisson::Poisson, N, X0},
  };

  use super::*;

  #[test]
  fn merton_length_equals_n() {
    let merton = Merton::new(
      2.25,
      2.5,
      1.0,
      1.0,
      N,
      Some(X0),
      Some(1.0),
      None,
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    assert_eq!(merton.sample().len(), N);
  }

  #[test]
  fn merton_starts_with_x0() {
    let merton = Merton::new(
      2.25,
      2.5,
      1.0,
      1.0,
      N,
      Some(X0),
      Some(1.0),
      None,
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    assert_eq!(merton.sample()[0], X0);
  }

  #[test]
  fn merton_plot() {
    let merton = Merton::new(
      2.25,
      2.5,
      1.0,
      1.0,
      N,
      Some(X0),
      Some(1.0),
      None,
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    plot_1d!(merton.sample(), "Merton process");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn merton_malliavin() {
    unimplemented!()
  }
}
