use crate::{
  noises::{fgn::FgnFft, gn},
  utils::Generator,
};
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
pub fn correlated_bms(rho: f64, n: usize, t: Option<f64>) -> [Array1<f64>; 2] {
  if !(-1.0..=1.0).contains(&rho) {
    panic!("Correlation coefficient must be in [-1, 1]");
  }

  let mut bms = Array2::<f64>::zeros((n, 2));

  let gn1 = gn::gn(n, Some(t.unwrap_or(1.0)));
  let gn2 = gn::gn(n, Some(t.unwrap_or(1.0)));

  for i in 1..n {
    bms[[i, 0]] = bms[[i - 1, 0]] + gn1[i - 1];
    bms[[i, 1]] = rho * gn1[i - 1] + (1.0 - rho.powi(2)).sqrt() * gn2[i - 1];
  }

  [bms.column(0).into_owned(), bms.column(1).into_owned()]
}

/// Generates two correlated fractional Brownian motion (fBM) paths.
///
/// # Parameters
///
/// - `hurst1`: Hurst parameter for the first fBM, must be in (0, 1).
/// - `hurst2`: Hurst parameter for the second fBM, must be in (0, 1).
/// - `rho`: Correlation coefficient between the two fBMs, must be in [-1, 1].
/// - `n`: Number of time steps.
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `[Array1<f64>; 2]` where each vector represents a generated fBM path.
///
/// # Panics
///
/// Panics if `rho` is not in the range [-1, 1].
/// Panics if either `hurst1` or `hurst2` is not in the range (0, 1).
///
/// # Example
///
/// ```
/// let correlated_fbms = correlated_fbms(0.75, 0.75, 0.5, 1000, Some(1.0));
/// let fbm1 = correlated_fbms[0];
/// let fbm2 = correlated_fbms[1];
/// ```
pub fn correlated_fbms(
  hurst1: f64,
  hurst2: f64,
  rho: f64,
  n: usize,
  t: Option<f64>,
) -> [Array1<f64>; 2] {
  if !(-1.0..=1.0).contains(&rho) || !(0.0..1.0).contains(&hurst1) || !(0.0..1.0).contains(&hurst2)
  {
    panic!("Correlation coefficient must be in [-1, 1] and Hurst parameters must be in (0, 1)");
  }

  let mut fbms = Array2::<f64>::zeros((n, 2));

  let fgn1 = FgnFft::new(hurst1, n - 1, t, None).sample();
  let fgn2 = FgnFft::new(hurst2, n - 1, t, None).sample();

  for i in 1..n {
    fbms[[i, 0]] = fbms[[i - 1, 0]] + fgn1[i - 1];
    fbms[[i, 1]] = rho * fgn2[i - 1] + (1.0 - rho.powi(2)).sqrt() * fgn2[i - 1];
  }

  [fbms.column(0).to_owned(), fbms.column(1).to_owned()]
}
