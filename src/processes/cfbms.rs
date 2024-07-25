use ndarray::{Array1, Array2};

use crate::{noises::fgn::FgnFft, utils::Generator};

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

#[derive(Default)]
pub struct CorrelatedFbms {
  pub hurst1: f64,
  pub hurst2: f64,
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
}

pub fn correlated_fbms(params: &CorrelatedFbms) -> [Array1<f64>; 2] {
  let CorrelatedFbms {
    hurst1,
    hurst2,
    rho,
    n,
    t,
  } = *params;

  assert!(
    hurst1 > 0.0 && hurst1 < 1.0,
    "Hurst parameter for the first fBM must be in (0, 1)"
  );
  assert!(
    hurst2 > 0.0 && hurst2 < 1.0,
    "Hurst parameter for the second fBM must be in (0, 1)"
  );
  assert!(
    rho >= -1.0 && rho <= 1.0,
    "Correlation coefficient must be in [-1, 1]"
  );

  let mut fbms = Array2::<f64>::zeros((n, 2));

  let fgn1 = FgnFft::new(hurst1, n - 1, t, None).sample();
  let fgn2 = FgnFft::new(hurst2, n - 1, t, None).sample();

  for i in 1..n {
    fbms[[i, 0]] = fbms[[i - 1, 0]] + fgn1[i - 1];
    fbms[[i, 1]] = rho * fgn2[i - 1] + (1.0 - rho.powi(2)).sqrt() * fgn2[i - 1];
  }

  [fbms.column(0).to_owned(), fbms.column(1).to_owned()]
}
