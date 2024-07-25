use ndarray::{Array1, Axis};
use rand::thread_rng;
use rand_distr::Distribution;

use super::poisson::{poisson, Poisson};

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

pub struct CompoundPoisson {
  pub n: Option<usize>,
  pub lambda: f64,
  pub t_max: Option<f64>,
}

pub fn compound_poisson(
  params: &CompoundPoisson,
  distribution: impl Distribution<f64> + Copy,
) -> [Array1<f64>; 3] {
  let CompoundPoisson { n, lambda, t_max } = *params;
  if n.is_none() && t_max.is_none() {
    panic!("n or t_max must be provided");
  }

  let p = poisson(&Poisson { lambda, n, t_max });
  let mut jumps = Array1::<f64>::zeros(n.unwrap_or(p.len()));

  for i in 1..p.len() {
    jumps[i] = distribution.sample(&mut thread_rng());
  }

  let mut cum_jupms = jumps.clone();
  cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

  [p, cum_jupms, jumps]
}
