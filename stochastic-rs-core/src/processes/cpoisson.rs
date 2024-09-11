use ndarray::{Array1, Axis};
use rand::thread_rng;
use rand_distr::Distribution;

use crate::{Sampling, Sampling3D};

use super::poisson::Poisson;

/// Generates a compound Poisson process.
///
/// The compound Poisson process models the occurrence of events over time, where each event has a random magnitude (jump). It is commonly used in insurance and finance.
///
/// # Parameters
///
/// - `n`: Number of time steps.
/// - `lambda`: Rate parameter (average number of events per unit time).
/// - `jumps`: Vector of jump sizes (optional).
/// - `t_max`: Maximum time (optional, defaults to 1.0).
/// - `jump_mean`: Mean of the jump sizes (optional, defaults to 0.0).
/// - `jump_std`: Standard deviation of the jump sizes (optional, defaults to 1.0).
///
/// # Returns
///
/// A `(Array1<f64>, Array1<f64>, Array1<f64>)` representing the exponetial times from Poisson, generated compound Poisson cumulative process path and the jumps.
///
/// # Panics
///
/// Panics if `n` is zero.
///
/// # Example
///
/// ```
/// let (p, cum_cp, cp) = compound_poisson(1000, 2.0, None, Some(10.0), Some(0.0), Some(1.0));
/// ```

pub struct CompoundPoisson<D>
where
  D: Distribution<f64> + Copy + Send + Sync,
{
  pub n: Option<usize>,
  pub lambda: f64,
  pub t_max: Option<f64>,
  pub m: Option<usize>,
  pub distribution: D,
  poisson: Poisson,
}

impl<D: Distribution<f64> + Copy + Send + Sync> CompoundPoisson<D> {
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

impl<D: Distribution<f64> + Copy + Send + Sync> Sampling3D<f64> for CompoundPoisson<D> {
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

  fn n(&self) -> usize {
    self.n.unwrap_or(0)
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
