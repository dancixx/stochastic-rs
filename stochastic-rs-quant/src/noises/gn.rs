use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use rayon::prelude::*;

use crate::quant::traits::Sampling;

pub struct GN;

impl GN {
  #[must_use]
  #[inline(always)]
  pub fn new() -> Self {
    Self
  }
}

impl Sampling<f64> for GN {
  fn sample(&self, n: usize, _x_0: f64, t_0: f64, t: f64) -> Array1<f64> {
    let sqrt_dt = ((t - t_0) / n as f64).sqrt();
    Array1::random(n, Normal::new(0.0, sqrt_dt).unwrap())
  }

  fn sample_parallel(
    &self,
    m: usize,
    n: usize,
    x_0: f64,
    t_0: f64,
    t: f64,
  ) -> ndarray::Array2<f64> {
    let mut xs = ndarray::Array2::<f64>::zeros((m, n));

    xs.axis_iter_mut(ndarray::Axis(0))
      .into_par_iter()
      .for_each(|mut x| {
        x.assign(&self.sample(n, x_0, t_0, t));
      });

    xs
  }
}
