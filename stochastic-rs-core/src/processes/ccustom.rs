use ndarray::{Array1, Axis};
use rand::thread_rng;
use rand_distr::Distribution;

use crate::{ProcessDistribution, Sampling, Sampling3D};

use super::customjt::CustomJt;

#[derive(Default)]
pub struct CompoundCustom<D, E>
where
  D: ProcessDistribution,
  E: ProcessDistribution,
{
  pub n: Option<usize>,
  pub t_max: Option<f64>,
  pub m: Option<usize>,
  pub jumps_distribution: D,
  pub jump_times_distribution: E,
  customjt: CustomJt<E>,
}

impl<D, E> CompoundCustom<D, E>
where
  D: ProcessDistribution,
  E: ProcessDistribution,
{
  #[must_use]
  pub fn new(params: &Self) -> Self {
    let customjt = CustomJt::new(&CustomJt {
      n: params.n,
      t_max: params.t_max,
      m: params.m,
      distribution: params.jump_times_distribution,
    });

    Self {
      n: params.n,
      t_max: params.t_max,
      m: params.m,
      jumps_distribution: params.jumps_distribution,
      jump_times_distribution: params.jump_times_distribution,
      customjt,
    }
  }
}

impl<D, E> Sampling3D<f64> for CompoundCustom<D, E>
where
  D: ProcessDistribution,
  E: ProcessDistribution,
{
  fn sample(&self) -> [Array1<f64>; 3] {
    if self.n.is_none() && self.t_max.is_none() {
      panic!("n or t_max must be provided");
    }

    let p = self.customjt.sample();
    let mut jumps = Array1::<f64>::zeros(self.n.unwrap_or(p.len()));
    for i in 1..p.len() {
      jumps[i] = self.jumps_distribution.sample(&mut thread_rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [p, cum_jupms, jumps]
  }

  fn n(&self) -> usize {
    self.n.unwrap_or(0)
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
