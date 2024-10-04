use ndarray::{Array1, Axis};
use rand::thread_rng;

use crate::stochastic::{ProcessDistribution, Sampling, Sampling3D};

use super::poisson::Poisson;

#[derive(Default)]
pub struct CompoundPoisson<D>
where
  D: ProcessDistribution,
{
  pub n: Option<usize>,
  pub lambda: f64,
  pub t_max: Option<f64>,
  pub m: Option<usize>,
  pub distribution: D,
  pub poisson: Poisson,
}

impl<D: ProcessDistribution> CompoundPoisson<D> {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let poisson = Poisson::new(&Poisson {
      lambda: params.lambda,
      n: params.n,
      t_max: params.t_max,
      m: params.m,
    });

    Self {
      n: params.n,
      lambda: params.lambda,
      t_max: params.t_max,
      m: params.m,
      distribution: params.distribution,
      poisson,
    }
  }
}

impl<D: ProcessDistribution> Sampling3D<f64> for CompoundPoisson<D> {
  fn sample(&self) -> [Array1<f64>; 3] {
    if self.n.is_none() && self.t_max.is_none() {
      panic!("n or t_max must be provided");
    }

    let poisson = self.poisson.sample();
    let mut jumps = Array1::<f64>::zeros(poisson.len());
    for i in 1..poisson.len() {
      jumps[i] = self.distribution.sample(&mut thread_rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [poisson, cum_jupms, jumps]
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n.unwrap_or(0)
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
