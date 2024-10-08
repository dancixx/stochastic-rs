use impl_new_derive::ImplNew;
use ndarray::{s, Array1};
use ndarray_rand::RandomExt;
use pyo3::{
  types::{PyModule, PyModuleMethods},
  Bound, PyResult,
};
use rand_distr::Normal;

use crate::stochastic::Sampling;

#[derive(ImplNew)]
#[pyo3::pyclass(name = "BM")]
pub struct BM {
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl Sampling<f64> for BM {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut bm = Array1::<f64>::zeros(self.n);
    bm.slice_mut(s![1..]).assign(&gn);

    for i in 1..self.n {
      bm[i] += bm[i - 1];
    }

    bm
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}

/// Python bindings
#[pyo3::pymodule]
fn bm(m: Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<BM>()?;
  Ok(())
}
