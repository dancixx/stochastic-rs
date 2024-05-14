use ndarray::Array1;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

pub fn gn(n: usize, t: Option<f32>) -> Vec<f32> {
  let sqrt_dt = (t.unwrap_or(1.0) / n as f32).sqrt();
  let gn = Array1::random(n, Normal::new(0.0, sqrt_dt).unwrap());

  gn.to_vec()
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
