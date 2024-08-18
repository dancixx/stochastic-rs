use crate::noises::cgns::{cgns, Cgns};
use derive_builder::Builder;
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
/// let params = Cbms {
///     rho: 0.5,
///     n: 1000,
///     t: Some(1.0),
/// };
/// let correlated_paths = cbms(&params);
/// let bm1 = correlated_paths[0].clone();
/// let bm2 = correlated_paths[1].clone();
/// ```
///
/// # Details
///
/// This function generates two correlated Brownian motion paths using the provided correlation coefficient `rho`
/// and the number of time steps `n`. The total time `t` is optional and defaults to 1.0 if not provided. The function
/// ensures that `rho` is within the valid range of [-1, 1] and panics otherwise. The generated paths are stored in a
/// 2D array and returned as a tuple of two 1D arrays.

#[derive(Default, Builder)]
#[builder(setter(into))]
pub struct Cbms {
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
}

pub fn cbms(params: &Cbms) -> [Array1<f64>; 2] {
  let Cbms { rho, n, t } = *params;
  assert!(
    !(-1.0..=1.0).contains(&rho),
    "Correlation coefficient must be in [-1, 1]"
  );

  let mut bms = Array2::<f64>::zeros((2, n + 1));
  let [cgn1, cgn2] = cgns(&Cgns { rho, n, t });

  for i in 1..(n + 1) {
    bms[[0, i]] = bms[[0, i - 1]] + cgn1[i - 1];
    bms[[1, i]] = bms[[1, i - 1]] + rho * cgn1[i - 1] + (1.0 - rho.powi(2)).sqrt() * cgn2[i - 1];
  }

  [bms.row(0).into_owned(), bms.row(1).into_owned()]
}
