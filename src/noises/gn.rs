use ndarray::Array1;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

/// Generates a path of Gaussian noise (GN).
///
/// The GN process is commonly used in simulations requiring white noise or random perturbations.
///
/// # Parameters
///
/// - `n`: Number of time steps.
/// - `t`: Total time (optional, defaults to 1.0).
///
/// # Returns
///
/// A `Array1<f64>` representing the generated Gaussian noise path.
///
/// # Example
///
/// ```
/// let gn_path = gn(1000, Some(1.0));
/// ```
pub fn gn(n: usize, t: Option<f64>) -> Array1<f64> {
  let sqrt_dt = (t.unwrap_or(1.0) / n as f64).sqrt();
  Array1::random(n, Normal::new(0.0, sqrt_dt).unwrap())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_gn() {
    let n = 10;
    let t = 1.0;
    let gn = gn(n, Some(t));
    assert_eq!(gn.len(), n);
  }
}
