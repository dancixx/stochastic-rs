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
pub struct Cfbms {
  pub hurst1: f64,
  pub hurst2: Option<f64>,
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
}

pub fn correlated_fbms(params: &Cfbms) -> [Array1<f64>; 2] {
  let Cfbms {
    hurst1,
    hurst2,
    rho,
    n,
    t,
  } = *params;

  assert!(
    !(0.0..=1.0).contains(&hurst1),
    "Hurst parameter for the first fBM must be in (0, 1)"
  );

  if let Some(hurst2) = hurst2 {
    assert!(
      !(0.0..=1.0).contains(&hurst2),
      "Hurst parameter for the second fBM must be in (0, 1)"
    );
  }
  assert!(
    !(-1.0..=1.0).contains(&rho),
    "Correlation coefficient must be in [-1, 1]"
  );

  let mut fbms = Array2::<f64>::zeros((n, 2));

  let fgn1 = FgnFft::new(hurst1, n - 1, t, None).sample();
  let fgn2 = FgnFft::new(hurst2.unwrap_or(hurst1), n - 1, t, None).sample();

  for i in 1..n {
    fbms[[0, i]] = fbms[[0, i - 1]] + fgn1[i - 1];
    fbms[[1, i]] = fbms[[1, i - 1]] + rho * fgn2[i - 1] + (1.0 - rho.powi(2)).sqrt() * fgn2[i - 1];
  }

  [fbms.row(0).to_owned(), fbms.row(1).to_owned()]
}
