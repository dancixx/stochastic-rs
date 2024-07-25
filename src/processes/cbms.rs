use crate::noises::gn;
use ndarray::{Array1, Array2};

/// Generates two correlated Brownian motion (BM) paths.
///
/// # Parameters
///
/// - `rho`: Correlation coefficient between the two BMs, must be in [-1, 1].
/// - `n`: Number of time steps.
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `[Array1<f64>; 2]` where each vector represents a generated BM path.
///
/// # Panics
///
/// Panics if `rho` is not in the range [-1, 1].
///
/// # Example
///
/// ```
/// let correlated_paths = correlated_bms(0.5, 1000, Some(1.0));
/// let bm1 = correlated_paths[0];
/// let bm2 = correlated_paths[1];
/// ```

#[derive(Default)]
pub struct CorrelatedBms {
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
}

pub fn correlated_bms(params: &CorrelatedBms) -> [Array1<f64>; 2] {
  let CorrelatedBms { rho, n, t } = *params;
  assert!(
    rho >= -1.0 && rho <= 1.0,
    "Correlation coefficient must be in [-1, 1]"
  );

  let mut bms = Array2::<f64>::zeros((n, 2));

  let gn1 = gn::gn(n, Some(t.unwrap_or(1.0)));
  let gn2 = gn::gn(n, Some(t.unwrap_or(1.0)));

  for i in 1..n {
    bms[[i, 0]] = bms[[i - 1, 0]] + gn1[i - 1];
    bms[[i, 1]] = rho * gn1[i - 1] + (1.0 - rho.powi(2)).sqrt() * gn2[i - 1];
  }

  [bms.column(0).into_owned(), bms.column(1).into_owned()]
}
