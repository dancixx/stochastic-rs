use ndarray::{s, Array1};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::Sampling;

#[derive(Default)]
pub struct Bm {
  pub n: usize,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl Bm {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      n: params.n,
      t: params.t,
      m: params.m,
    }
  }
}

impl Sampling<f64> for Bm {
  fn sample(&self) -> Array1<f64> {
    let gn = Array1::random(
      self.n,
      Normal::new(0.0, (self.t.unwrap_or(1.0) / self.n as f64).sqrt()).unwrap(),
    );
    let mut bm = Array1::<f64>::zeros(self.n);
    bm.slice_mut(s![1..]).assign(&gn);

    for i in 1..self.n {
      bm[i] += bm[i - 1];
    }

    bm
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
