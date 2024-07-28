use ndarray::{Array1, Array2};

use crate::{noises::fgn::FgnFft, utils::Generator};

/// Generates two correlated fractional Gaussian noise (fGn) paths.
///
/// # Parameters
///
/// - `hurst`: Hurst parameter for the fGn, must be in (0, 1).
/// - `rho`: Correlation coefficient between the two fGns, must be in [-1, 1].
/// - `n`: Number of time steps.
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `[Array1<f64>; 2]` where each vector represents a generated fGn path.
///
/// # Panics
///
/// Panics if `hurst` is not in the range (0, 1).
/// Panics if `rho` is not in the range [-1, 1].
///
/// # Example
///
/// ```
/// let params = Cfgns {
///     hurst: 0.7,
///     rho: 0.5,
///     n: 1000,
///     t: Some(1.0),
/// };
/// let correlated_fg_paths = cfgns(&params);
/// let fgn1 = correlated_fg_paths[0].clone();
/// let fgn2 = correlated_fg_paths[1].clone();
/// ```

#[derive(Default)]
pub struct Cfgns {
  pub hurst: f64,
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
}

pub fn cfgns(params: &Cfgns) -> [Array1<f64>; 2] {
  let Cfgns { hurst, rho, n, t } = *params;
  assert!(
    !(0.0..=1.0).contains(&hurst),
    "Hurst parameter must be in (0, 1)"
  );
  assert!(
    !(-1.0..=1.0).contains(&rho),
    "Correlation coefficient must be in [-1, 1]"
  );

  let mut cfgns = Array2::<f64>::zeros((2, n));
  let fgn = FgnFft::new(hurst, n - 1, t, None);
  let fgn1 = fgn.sample();
  let fgn2 = fgn.sample();

  for i in 1..n {
    cfgns[[0, i]] = fgn1[i - 1];
    cfgns[[1, i]] = rho * fgn1[i - 1] + (1.0 - rho.powi(2)).sqrt() * fgn2[i - 1];
  }

  [cfgns.row(0).into_owned(), cfgns.row(1).into_owned()]
}
