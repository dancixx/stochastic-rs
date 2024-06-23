use ndarray::{Array1, Axis};

use crate::noises::gn;

/// Generates a path of Brownian Motion (BM).
///
/// The BM process is a fundamental continuous-time stochastic process used in various fields such as finance and physics.
///
/// # Parameters
///
/// - `n`: Number of time steps.
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Vec<f64>` representing the generated Brownian Motion path.
///
/// # Example
///
/// ```
/// use stochastic_rs::processes::bm::bm;
///
/// let bm_path = bm(1000, Some(1.0));
/// ```

pub fn bm(n: usize, t: Option<f64>) -> Vec<f64> {
  let gn = gn::gn(n - 1, Some(t.unwrap_or(1.0)));
  let mut bm = Array1::<f64>::from(gn);
  bm.accumulate_axis_inplace(Axis(0), |&x, y| *y += x);
  vec![0.0].into_iter().chain(bm).collect()
}
