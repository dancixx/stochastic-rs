use derive_builder::Builder;
use ndarray::{Array1, Array2};

use crate::noises::gn;

/// Generates two correlated Gaussian noise (GN) paths.
///
/// # Parameters
///
/// - `rho`: Correlation coefficient between the two GNs, must be in [-1, 1].
/// - `n`: Number of time steps.
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `[Array1<f64>; 2]` where each vector represents a generated GN path.
///
/// # Panics
///
/// Panics if `rho` is not in the range [-1, 1].
///
/// # Example
///
/// ```
/// let params = Cgns {
///     rho: 0.5,
///     n: 1000,
///     t: Some(1.0),
/// };
/// let correlated_gn_paths = cgns(&params);
/// let gn1 = correlated_gn_paths[0].clone();
/// let gn2 = correlated_gn_paths[1].clone();
/// ```

#[derive(Default, Builder)]
#[builder(setter(into))]
pub struct Cgns {
  pub rho: f64,
  pub n: usize,
  pub t: Option<f64>,
}

pub fn cgns(params: &Cgns) -> [Array1<f64>; 2] {
  let Cgns { rho, n, t } = *params;
  assert!(
    (-1.0..=1.0).contains(&rho),
    "Correlation coefficient must be in [-1, 1]"
  );

  let mut cgns = Array2::<f64>::zeros((2, n + 1));
  let gn1 = gn::gn(n, Some(t.unwrap_or(1.0)));
  let gn2 = gn::gn(n, Some(t.unwrap_or(1.0)));

  for i in 1..(n + 1) {
    cgns[[0, i]] = gn1[i - 1];
    cgns[[1, i]] = rho * gn1[i - 1] + (1.0 - rho.powi(2)).sqrt() * gn2[i - 1];
  }

  [cgns.row(0).into_owned(), cgns.row(1).into_owned()]
}
