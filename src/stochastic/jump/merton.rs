use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::{
  process::cpoisson::CompoundPoisson, ProcessDistribution, Sampling, Sampling3D,
};

#[derive(Default)]
pub struct Merton<D>
where
  D: ProcessDistribution,
{
  pub alpha: f64,
  pub sigma: f64,
  pub lambda: f64,
  pub theta: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub jump_distribution: D,
  pub cpoisson: CompoundPoisson<D>,
}

impl<D: ProcessDistribution> Merton<D> {
  #[must_use]
  pub fn new(params: &Merton<D>) -> Self {
    let cpoisson = CompoundPoisson::new(&CompoundPoisson {
      n: None,
      lambda: params.lambda,
      t_max: Some(params.t.unwrap_or(1.0) / params.n as f64),
      distribution: params.jump_distribution,
      m: params.m,
      ..Default::default()
    });

    Self {
      alpha: params.alpha,
      sigma: params.sigma,
      lambda: params.lambda,
      theta: params.theta,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
      jump_distribution: params.jump_distribution,
      cpoisson,
    }
  }
}

impl<D: ProcessDistribution> Sampling<f64> for Merton<D> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let mut merton = Array1::<f64>::zeros(self.n + 1);
    merton[0] = self.x0.unwrap_or(0.0);
    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

    for i in 1..=self.n {
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
